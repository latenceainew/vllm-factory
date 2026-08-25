"""Pooler: uni-encoder prompt + word extraction, LSTM, GLiNER Scorer.

Implements FactoryPooler protocol — zero vLLM imports.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from vllm_factory.optional_deps import require
from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states
from vllm_factory.pooling.shape_prefix import pack_shape_prefixed_tensor


class GLiNERRerankPooler(nn.Module):
    """Builds label prompts from sequence hidden states, LSTM-smooths words, runs scorer."""

    def __init__(self, model):
        super().__init__()
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_rnn", model.rnn)
        object.__setattr__(self, "_scorer_proj_token", model.scorer_proj_token)
        object.__setattr__(self, "_scorer_proj_label", model.scorer_proj_label)
        object.__setattr__(self, "_scorer_out_mlp", model.scorer_out_mlp)
        object.__setattr__(self, "_model_config", model.vllm_config.model_config)

    # ── FactoryPooler protocol ───────────────────────────────────────────

    def get_tasks(self) -> set[str]:
        return {"embed", "plugin"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        try:
            sequences = split_hidden_states(hidden_states, ctx.seq_lengths)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("Rerank pooler warmup fallback: %s", e)
            dummy = torch.zeros(4, device=hidden_states.device, dtype=hidden_states.dtype)
            return [dummy]

        if not ctx.extra_kwargs:
            return [
                torch.zeros(4, device=hidden_states.device, dtype=torch.float32) for _ in sequences
            ]

        cfg = self._model_config.hf_config
        class_token_index = int(cfg.class_token_index)
        embed_ent_token = bool(cfg.embed_ent_token)

        outputs: List[torch.Tensor] = []
        dev = hidden_states.device

        for i, tok in enumerate(sequences):
            add = ctx.extra_kwargs[i] if i < len(ctx.extra_kwargs) else {}
            prompt_ids = ctx.prompt_token_ids[i] if i < len(ctx.prompt_token_ids) else None
            if not add or "words_mask" not in add:
                outputs.append(torch.zeros(4, device=dev, dtype=torch.float32))
                continue

            wmask = self._to_tensor(add["words_mask"], device=dev, dtype=torch.long)
            text_length = int(add["text_lengths"])
            input_ids_value = add.get("input_ids", prompt_ids)
            if input_ids_value is None:
                outputs.append(torch.zeros(4, device=dev, dtype=torch.float32))
                continue
            input_ids = self._to_tensor(input_ids_value, device=dev, dtype=torch.long)
            attn = self._to_tensor(
                add.get("attention_mask", [1] * int(input_ids.numel())),
                device=dev,
                dtype=torch.long,
            )

            if input_ids.dim() == 1:
                input_ids_b = input_ids.unsqueeze(0)
                attn_b = attn.unsqueeze(0) if attn.dim() == 1 else attn
                wmask_b = wmask.unsqueeze(0) if wmask.dim() == 1 else wmask
            else:
                input_ids_b, attn_b, wmask_b = input_ids, attn, wmask

            L = tok.size(0)
            if input_ids_b.size(1) != L:
                outputs.append(torch.zeros(4, device=dev, dtype=torch.float32))
                continue

            tok_b = tok.unsqueeze(0)
            tl = torch.tensor([text_length], device=dev, dtype=torch.long)

            gliner_utils = require(
                "gliner.modeling.utils", "gliner", purpose="GLiNER rerank pooling"
            )
            prompts, p_mask, words_pre, w_mask = (
                gliner_utils.extract_prompt_features_and_word_embeddings(
                    class_token_index,
                    tok_b,
                    input_ids_b,
                    attn_b,
                    tl,
                    wmask_b,
                    embed_ent_token=embed_ent_token,
                )
            )

            words = self._run_lstm(words_pre, w_mask)
            scores = self._run_scorer(words, prompts)
            scores = scores.squeeze(0)
            W, C, S = scores.shape
            flat = pack_shape_prefixed_tensor([W, C, S], scores)
            outputs.append(flat)

        return outputs

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _to_tensor(x, device, dtype=None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype) if dtype else x.to(device=device)
        t = torch.tensor(x, device=device)
        return t.to(dtype) if dtype else t

    def _run_lstm(self, word_embs, mask_1d):
        """word_embs (1, W, H), mask (1, W) with 1 valid / 0 pad."""
        lengths = mask_1d.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            word_embs, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self._rnn(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return unpacked

    def _run_scorer(self, word_embs, label_embs):
        B, W, H = word_embs.shape
        C = label_embs.shape[1]
        token_rep = self._scorer_proj_token(word_embs)
        token_rep = token_rep.view(B, W, 1, 2, H)
        label_rep = self._scorer_proj_label(label_embs)
        label_rep = label_rep.view(B, 1, C, 2, H)
        token_rep = token_rep.expand(-1, -1, C, -1, -1).permute(3, 0, 1, 2, 4)
        label_rep = label_rep.expand(-1, W, -1, -1, -1).permute(3, 0, 1, 2, 4)
        cat = torch.cat([token_rep[0], label_rep[0], token_rep[1] * label_rep[1]], dim=-1)
        return self._scorer_out_mlp(cat)
