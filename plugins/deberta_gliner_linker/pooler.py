"""
GLiNER-Linker Token Pooler — token-level entity linking scoring.

Mirrors GLiNER BiEncoderTokenModel forward AFTER the text encoder: word
extraction then scorer.

Implements FactoryPooler protocol — zero vLLM imports.

Input metadata comes via PoolerContext.extra_kwargs per sequence:
    - words_mask: [L] int, 1-indexed word assignments
    - text_lengths: int, number of words
    - label_texts: list[str], entity labels to encode (if not using precomputed)
    - labels_embeds: (C, D) tensor, precomputed label embeddings (optional)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from gliner.modeling.utils import extract_spans_from_tokens

from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states
from vllm_factory.pooling.shape_prefix import pack_shape_prefixed_tensor


class GLiNERLinkerPooler(nn.Module):
    """Token-level pooler for GLiNER-Linker bi-encoder.

    Receives text encoder hidden states, extracts word embeddings, runs the
    scorer against label embeddings, returns flattened logits (no LSTM on this path).
    """

    def __init__(self, model):
        """Initialize with references to individual model components.

        Stores component references (not the parent model) to avoid
        circular references that cause RecursionError on model.eval().
        """
        super().__init__()
        object.__setattr__(self, "_labels_encoder", model.labels_encoder)
        object.__setattr__(self, "_span_rep_layer", model.span_rep_layer)
        object.__setattr__(self, "_scorer_proj_token", model.scorer_proj_token)
        object.__setattr__(self, "_scorer_proj_label", model.scorer_proj_label)
        object.__setattr__(self, "_scorer_out_mlp", model.scorer_out_mlp)
        object.__setattr__(self, "_model_config", model.vllm_config.model_config)
        object.__setattr__(self, "_tokenizer", None)
        object.__setattr__(self, "_label_cache", {})

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

            logging.getLogger(__name__).warning("Pooler warmup fallback: %s", e)
            dummy = torch.zeros(4, device=hidden_states.device, dtype=hidden_states.dtype)
            return [dummy]

        if not ctx.extra_kwargs:
            return [
                torch.zeros(4, device=hidden_states.device, dtype=torch.float32) for _ in sequences
            ]

        outputs: List[torch.Tensor] = []
        dev = hidden_states.device
        H = hidden_states.shape[-1]

        for i, tok in enumerate(sequences):
            add = ctx.extra_kwargs[i] if i < len(ctx.extra_kwargs) else {}

            if not add or "words_mask" not in add:
                outputs.append(torch.zeros(4, device=dev, dtype=torch.float32))
                continue

            wmask = self._to_tensor(add["words_mask"], device=dev, dtype=torch.long)
            text_length = int(add["text_lengths"])

            we, _ = self._extract_word_embeddings(tok, wmask, text_length, dev, H)

            label_key = add.get("labels_key")
            le = None
            if "labels_embeds" in add:
                le = self._to_tensor(add["labels_embeds"], device=dev, dtype=we.dtype)
                if le.dim() == 2:
                    le = le.unsqueeze(0)
                if label_key:
                    self._label_cache[label_key] = le
            elif label_key:
                cached = self._label_cache.get(label_key)
                if cached is not None:
                    le = cached.to(device=dev, dtype=we.dtype)
                    self._label_cache[label_key] = le
            elif "label_texts" in add:
                le_flat = self._encode_labels(add["label_texts"], dev)
                le = le_flat.unsqueeze(0).to(we.dtype)
            if le is None:
                outputs.append(torch.zeros(4, device=dev, dtype=torch.float32))
                continue

            threshold = float(add.get("threshold", 0.5))
            scores = self._run_scorer(we, le)
            span_idx, span_mask = extract_spans_from_tokens(
                scores, labels=None, threshold=threshold
            )
            span_rep = self._span_rep_layer(we, span_idx * span_mask.unsqueeze(-1).long())
            span_logits = torch.einsum("BND,BCD->BNC", span_rep, le)

            scores = scores.squeeze(0)
            span_idx = span_idx.squeeze(0)
            span_mask = span_mask.squeeze(0)
            span_logits = span_logits.squeeze(0)

            W, C, S = scores.shape
            N = int(span_idx.shape[0])
            flat = pack_shape_prefixed_tensor(
                [W, C, S, N], scores, span_idx, span_mask, span_logits
            )
            outputs.append(flat)

        return outputs

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _to_tensor(x, device, dtype=None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype) if dtype else x.to(device=device)
        t = torch.tensor(x, device=device)
        return t.to(dtype) if dtype else t

    def _extract_word_embeddings(self, tok_embs, wmask, text_length, device, H):
        W = int(text_length)
        we = tok_embs.new_zeros(1, W, H)
        pos = (wmask > 0).nonzero(as_tuple=False).flatten()
        if pos.numel() > 0:
            tgt = (wmask[pos] - 1).long()
            we[0, tgt] = tok_embs[pos]
        we_mask = torch.ones(1, W, dtype=torch.long, device=device)
        return we, we_mask

    @torch.no_grad()
    def _encode_labels(self, label_texts: List[str], device) -> torch.Tensor:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            object.__setattr__(
                self,
                "_tokenizer",
                AutoTokenizer.from_pretrained(self._model_config.model, use_fast=True),
            )

        all_embs = []
        for label in label_texts:
            enc = self._tokenizer(label, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            output = self._labels_encoder(input_ids=input_ids)
            hs = output.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hs.size()).float()
            mean = (hs * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
            all_embs.append(mean.squeeze(0))

        return torch.stack(all_embs, dim=0)

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
        scores = self._scorer_out_mlp(cat)
        return scores
