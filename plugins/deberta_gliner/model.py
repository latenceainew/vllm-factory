"""
DeBERTa v2 GLiNER — Custom vLLM-optimized DeBERTa v2 encoder + GLiNER span extraction pooler.

Backbone: models.deberta_v2.DebertaV2EncoderModel (Flash DeBERTa Triton kernel + vLLM parallel layers)
Pooler:   poolers.gliner (LSTM + SpanMarker + einsum scoring)
Weights:  GLiNER HF checkpoint with token_rep_layer.bert_layer.model.* prefix

vLLM 0.15.x compatible:
- num_hidden_layers=0 in config to skip KV cache allocation
- Custom encoder uses its own attention (Flash DeBERTa Triton kernel)
- Forward returns 2D (total_tokens, hidden_size) for the pooler
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import DebertaV2Config
from vllm.config import VllmConfig

from plugins.modernbert_gliner_rerank.pooler import GLiNERRerankPooler
from poolers.gliner import GLiNERSpanPooler
from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

from .config import GLiNERDebertaV2Config

logger = logging.getLogger(__name__)

# Load the custom DeBERTa v2 encoder with Flash DeBERTa Triton kernel
_ENCODER_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "deberta_v2" / "deberta_v2_encoder.py"
)


def _import_deberta_v2_encoder():
    spec = importlib.util.spec_from_file_location("deberta_v2_encoder", str(_ENCODER_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("deberta_v2_encoder", mod)
    spec.loader.exec_module(mod)
    return mod


_encoder_mod = _import_deberta_v2_encoder()
DebertaV2EncoderModel = _encoder_mod.DebertaV2EncoderModel


class GLiNERDebertaV2Model(nn.Module):
    """Custom vLLM-optimized DeBERTa v2 encoder + GLiNER span extraction pooler.

    Architecture:
        input_ids → DebertaV2EncoderModel (Flash DeBERTa Triton + vLLM parallel layers)
                  → projection(encoder_hidden → gliner_hidden)
                  → GLiNERSpanPooler (LSTM + SpanMarker + einsum)
                  → span scores
    """

    is_pooling_model = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg: GLiNERDebertaV2Config = vllm_config.model_config.hf_config
        self.config = cfg
        self.vllm_config = vllm_config

        self.encoder_hidden_size = cfg.encoder_hidden_size
        self.pooler_hidden_size = cfg.gliner_hidden_size

        # Build DeBERTa v2 config for the custom encoder
        encoder_cfg = DebertaV2Config(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.encoder_hidden_size,
            num_hidden_layers=cfg.encoder_num_hidden_layers,
            num_attention_heads=cfg.encoder_num_attention_heads,
            intermediate_size=cfg.encoder_intermediate_size,
            hidden_act=cfg.encoder_hidden_act,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=cfg.encoder_max_position_embeddings,
            type_vocab_size=cfg.encoder_type_vocab_size,
            layer_norm_eps=cfg.encoder_layer_norm_eps,
            relative_attention=cfg.encoder_relative_attention,
            max_relative_positions=cfg.encoder_max_relative_positions,
            position_buckets=cfg.encoder_position_buckets,
            pos_att_type=cfg.encoder_pos_att_type,
            share_att_key=cfg.encoder_share_att_key,
            norm_rel_ebd=cfg.encoder_norm_rel_ebd,
            position_biased_input=cfg.encoder_position_biased_input,
            pad_token_id=cfg.encoder_pad_token_id,
        )

        # 1. Backbone — custom DeBERTa v2 with Flash DeBERTa Triton kernel
        self.model = DebertaV2EncoderModel(config=encoder_cfg)

        # 2. Projection (encoder_hidden → gliner_hidden) if dimensions differ
        if self.encoder_hidden_size != self.pooler_hidden_size:
            self.projection = nn.Linear(self.encoder_hidden_size, self.pooler_hidden_size)
        else:
            self.projection = None

        self.is_token_level = cfg.span_mode == "token_level"
        if self.is_token_level:
            self.rnn = nn.LSTM(
                input_size=self.pooler_hidden_size,
                hidden_size=self.pooler_hidden_size // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            hidden_size = self.pooler_hidden_size
            self.scorer_proj_token = nn.Linear(hidden_size, hidden_size * 2)
            self.scorer_proj_label = nn.Linear(hidden_size, hidden_size * 2)
            self.scorer_out_mlp = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size * 4),
                nn.Dropout(0.0),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, 3),
            )
            self._business_pooler = GLiNERRerankPooler(self)
        else:
            pooler_cfg = _make_pooler_config(cfg)
            self._business_pooler = GLiNERSpanPooler(pooler_cfg)
        self.pooler = VllmPoolerAdapter(self._business_pooler, requires_token_ids=True)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Required by vLLM's pooling runner for embedding lookup."""
        return self.model.embeddings.word_embeddings(input_ids)

    def sample(self, logits: torch.Tensor, sampling_metadata):
        """Override sampling for pooling models — return empty outputs."""
        try:
            from vllm.sequence import SamplerOutput

            return SamplerOutput(outputs=[])
        except ImportError:
            return None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run DeBERTa v2 encoder + projection, return hidden states for pooler."""
        with torch.no_grad():
            hs = self._run_encoder(
                input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds
            )

        # Custom encoder returns tensor directly; flatten to 2D
        if hs.dim() == 3:
            hs = hs.squeeze(0)

        if self.projection is not None:
            hs = self.projection(hs)

        return hs

    def _run_encoder(
        self,
        *,
        input_ids: Optional[torch.LongTensor],
        positions: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if input_ids is not None and input_ids.dim() == 1 and positions is not None:
            pos = positions.to(device=input_ids.device)
            starts = (pos == 0).nonzero(as_tuple=False).flatten().tolist()
            if len(starts) > 1:
                ends = starts[1:] + [input_ids.shape[0]]
                lengths = [end - start for start, end in zip(starts, ends)]
                max_len = max(lengths)
                batch_size = len(lengths)
                pad_id = int(getattr(self.config, "encoder_pad_token_id", 0))
                batched_ids = input_ids.new_full((batch_size, max_len), pad_id)
                batched_positions = pos.new_zeros((batch_size, max_len))
                attention_mask = torch.zeros(
                    (batch_size, max_len),
                    dtype=torch.long,
                    device=input_ids.device,
                )
                for row, (start, end, length) in enumerate(zip(starts, ends, lengths)):
                    batched_ids[row, :length] = input_ids[start:end]
                    batched_positions[row, :length] = pos[start:end]
                    attention_mask[row, :length] = 1

                encoded = self.model(
                    input_ids=batched_ids,
                    attention_mask=attention_mask,
                    position_ids=batched_positions,
                )
                return torch.cat(
                    [encoded[row, :length] for row, length in enumerate(lengths)],
                    dim=0,
                )

        return self.model(input_ids=input_ids, position_ids=positions, inputs_embeds=inputs_embeds)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Map GLiNER HF checkpoint weights to backbone + pooler.

        HF prefix mapping:
            token_rep_layer.bert_layer.model.*  →  backbone (self.model)
            token_rep_layer.projection.*        →  projection
            rnn.*                               →  pooler.rnn.*
            span_rep_layer.span_rep_layer.*     →  pooler.span_rep.*
            prompt_rep_layer.*                  →  pooler.prompt_proj.*

        The custom encoder's load_weights() expects HF DeBERTa keys with
        'deberta.' prefix and handles the .attention.self. → .attention.self_attn.
        mapping internally.
        """
        backbone_prefix = "token_rep_layer.bert_layer.model."
        projection_prefix = "token_rep_layer.projection."
        rnn_prefix = "rnn."
        span_rep_prefix = "span_rep_layer.span_rep_layer."
        prompt_rep_prefix = "prompt_rep_layer."

        backbone_weights = []
        pooler_state = {}
        projection_state = {}

        pooler_keys = set(self._business_pooler.state_dict().keys())
        rnn_state = {}
        scorer_state = {}

        for hf_name, tensor in weights:
            if hf_name.startswith(backbone_prefix):
                hf_key = hf_name[len(backbone_prefix) :]
                backbone_weights.append(("deberta." + hf_key, tensor))

            elif hf_name.startswith(projection_prefix):
                stripped = hf_name[len(projection_prefix) :]
                projection_state[stripped] = tensor

            elif self.is_token_level and hf_name.startswith("rnn.lstm."):
                rnn_state[hf_name[len("rnn.lstm.") :]] = tensor

            elif self.is_token_level and hf_name.startswith("scorer."):
                scorer_state[hf_name[len("scorer.") :]] = tensor

            else:
                vllm_key = hf_name
                if hf_name.startswith(rnn_prefix):
                    vllm_key = hf_name
                elif hf_name.startswith(span_rep_prefix):
                    vllm_key = hf_name.replace(span_rep_prefix, "span_rep.")
                elif hf_name.startswith(prompt_rep_prefix):
                    vllm_key = hf_name.replace(prompt_rep_prefix, "prompt_proj.")

                if vllm_key in pooler_keys:
                    pooler_state[vllm_key] = tensor

        # Load backbone via custom encoder's load_weights (handles key remapping)
        self.model.load_weights(backbone_weights)
        logger.info("[GLiNERDebertaV2] Loaded backbone: %s weight tensors", len(backbone_weights))

        # Load projection
        if self.projection is not None and projection_state:
            self.projection.load_state_dict(projection_state)
            device = next(self.model.parameters()).device
            dtype = self.vllm_config.model_config.dtype
            self.projection.to(device=device, dtype=dtype)
            logger.info(
                "[GLiNERDebertaV2] Loaded projection: %s",
                list(projection_state.keys()),
            )

        # Load pooler (use business pooler directly to avoid _inner. prefix mismatch)
        if pooler_state:
            self._business_pooler.load_state_dict(pooler_state, strict=False)
            device = next(self.model.parameters()).device
            dtype = self.vllm_config.model_config.dtype
            self._business_pooler.to(device=device, dtype=dtype)
            logger.info(
                "[GLiNERDebertaV2] Loaded pooler: %s/%s keys",
                len(pooler_state),
                len(pooler_keys),
            )
        else:
            logger.warning("[GLiNERDebertaV2] No pooler weights loaded!")

        if self.is_token_level:
            device = next(self.model.parameters()).device
            dtype = self.vllm_config.model_config.dtype
            if rnn_state:
                self.rnn.load_state_dict(rnn_state, strict=False)
                self.rnn.to(device=device, dtype=dtype)
                logger.info("[GLiNERDebertaV2] Loaded token-level LSTM")
            if scorer_state:
                for key, tensor in scorer_state.items():
                    tensor = tensor.to(device=device, dtype=dtype)
                    if key.startswith("proj_token."):
                        getattr(self.scorer_proj_token, key[len("proj_token.") :]).data.copy_(
                            tensor
                        )
                    elif key.startswith("proj_label."):
                        getattr(self.scorer_proj_label, key[len("proj_label.") :]).data.copy_(
                            tensor
                        )
                    elif key.startswith("out_mlp."):
                        parts = key.split(".")
                        idx = int(parts[1])
                        attr_name = parts[2]
                        getattr(self.scorer_out_mlp[idx], attr_name).data.copy_(tensor)
                self.scorer_proj_token.to(device=device, dtype=dtype)
                self.scorer_proj_label.to(device=device, dtype=dtype)
                self.scorer_out_mlp.to(device=device, dtype=dtype)
                logger.info("[GLiNERDebertaV2] Loaded token-level scorer")

        return set(name for name, _ in self.named_parameters())


def _make_pooler_config(cfg: GLiNERDebertaV2Config):
    """Build a config-like namespace with attributes the GLiNERSpanPooler expects."""

    class PoolerConfig:
        pass

    pc = PoolerConfig()
    pc.hidden_size = cfg.gliner_hidden_size
    pc.gliner_hidden_size = cfg.gliner_hidden_size
    pc.gliner_dropout = cfg.gliner_dropout
    pc.max_width = cfg.max_width
    pc.class_token_index = cfg.class_token_index
    pc.sep_token_index = cfg.sep_token_index
    pc.ent_token = cfg.ent_token
    pc.sep_token = cfg.sep_token
    pc.has_rnn = cfg.has_rnn
    pc.embed_ent_token = cfg.embed_ent_token
    pc.subtoken_pooling = cfg.subtoken_pooling
    pc.max_len = cfg.max_len
    pc.span_mode = cfg.span_mode
    return pc
