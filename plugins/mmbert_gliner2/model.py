"""
mmBERT-GLiNER2 — ModernBERT encoder + GLiNER2 multi-task extraction pooler.

Backbone: models.modernbert.ModernBertModel (fused Triton kernels + vLLM parallel layers)
Pooler:   poolers.gliner2.GLiNER2Pooler (SpanRep + CountLSTM + classifier + count_pred)
Weights:  encoder.* → backbone, {span_rep,count_embed,classifier,count_pred}.* → pooler
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch import nn
from vllm.config import VllmConfig

from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

logger = logging.getLogger(__name__)

_ENCODER_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "modernbert" / "modernbert_encoder.py"
)


def _import_modernbert_encoder():
    spec = importlib.util.spec_from_file_location("modernbert_encoder", str(_ENCODER_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("modernbert_encoder", mod)
    spec.loader.exec_module(mod)
    return mod


_encoder_mod = _import_modernbert_encoder()
ModernBertModel = _encoder_mod.ModernBertModel


def _patch_modernbert_config(cfg):
    """Ensure all attributes the custom ModernBertModel encoder expects."""
    for attr, default in (
        ("embedding_dropout", 0.0),
        ("mlp_dropout", 0.0),
        ("attention_dropout", 0.0),
        ("classifier_dropout", 0.0),
        ("norm_bias", False),
        ("attention_bias", False),
        ("mlp_bias", False),
        ("classifier_bias", False),
    ):
        if not hasattr(cfg, attr):
            setattr(cfg, attr, default)
    if not hasattr(cfg, "norm_eps"):
        cfg.norm_eps = float(getattr(cfg, "layer_norm_eps", 1e-5))
    if not hasattr(cfg, "layer_types"):
        cfg.layer_types = [
            "full_attention" if i % cfg.global_attn_every_n_layers == 0 else "sliding_attention"
            for i in range(cfg.encoder_num_layers)
        ]
    if not hasattr(cfg, "rope_parameters"):
        cfg.rope_parameters = {
            "full_attention": {
                "rope_theta": cfg.global_rope_theta,
                "rope_type": "default",
            },
            "sliding_attention": {
                "rope_theta": cfg.local_rope_theta,
                "rope_type": "default",
            },
        }
    return cfg


class GLiNER2ModernBertModel(nn.Module):
    """ModernBERT encoder + GLiNER2 multi-task pooler for vLLM.

    Architecture:
        input_ids → ModernBertModel (fused Triton kernels)
                  → (total_tokens, hidden_size)
                  → GLiNER2Pooler (SpanRep + CountLSTM + classifier + count_pred)
                  → span scores + classification logits
    """

    is_pooling_model = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        cfg = _patch_modernbert_config(cfg)
        self.config = cfg
        self.vllm_config = vllm_config

        saved_nhl = cfg.num_hidden_layers
        saved_nah = cfg.num_attention_heads
        cfg.num_hidden_layers = cfg.encoder_num_layers
        cfg.num_attention_heads = cfg.encoder_num_attention_heads

        self.encoder = ModernBertModel(
            vllm_config=vllm_config,
            prefix=f"{prefix}.encoder" if prefix else "encoder",
        )

        cfg.num_hidden_layers = saved_nhl
        cfg.num_attention_heads = saved_nah

        hidden_size = int(cfg.hidden_size)
        max_width = int(getattr(cfg, "max_width", 12))
        counting_layer = getattr(cfg, "counting_layer", "count_lstm_v2")

        from poolers.gliner2 import GLiNER2Pooler

        self._business_pooler = GLiNER2Pooler(
            hidden_size=hidden_size,
            max_width=max_width,
            counting_layer=counting_layer,
        )
        self.pooler = VllmPoolerAdapter(self._business_pooler, requires_token_ids=True)
        self.pooler.to(vllm_config.model_config.dtype)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder.embeddings.tok_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.encoder(
            input_ids=input_ids,
            position_ids=positions.unsqueeze(0)
            if positions is not None and positions.dim() == 1
            else positions,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = (
            output.last_hidden_state
            if hasattr(output, "last_hidden_state")
            else output
            if isinstance(output, torch.Tensor)
            else output[0]
        )
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)
        return hidden_states

    def sample(self, logits: torch.Tensor, sampling_metadata):
        try:
            from vllm.sequence import SamplerOutput

            return SamplerOutput(outputs=[])
        except ImportError:
            return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Map checkpoint weights to backbone + pooler.

        Weight prefixes in hivetrace/gliner-guard-uniencoder:
            encoder.*                    → self.encoder (ModernBERT backbone)
            span_rep.*                   → pooler.span_rep
            count_embed.*                → pooler.count_embed (CountLSTM)
            classifier.*                 → pooler.classifier
            count_pred.*                 → pooler.count_pred
        """
        encoder_prefix = "encoder."
        pooler_prefixes = ("span_rep.", "count_embed.", "classifier.", "count_pred.")

        backbone_state = {}
        pooler_state = {}

        vllm_backbone = self.encoder.state_dict()
        pooler_keys = set(self._business_pooler.state_dict().keys())

        for hf_name, tensor in weights:
            if hf_name.startswith(encoder_prefix):
                local_key = hf_name[len(encoder_prefix) :]
                if local_key in vllm_backbone:
                    if "tok_embeddings.weight" in local_key:
                        target_shape = vllm_backbone[local_key].shape
                        if tensor.shape[0] < target_shape[0]:
                            pad = torch.zeros(
                                target_shape[0] - tensor.shape[0],
                                tensor.shape[1],
                                dtype=tensor.dtype,
                                device=tensor.device,
                            )
                            tensor = torch.cat([tensor, pad], dim=0)
                        elif tensor.shape[0] > target_shape[0]:
                            tensor = tensor[: target_shape[0], :]
                    backbone_state[local_key] = tensor
            elif any(hf_name.startswith(p) for p in pooler_prefixes):
                if hf_name in pooler_keys:
                    pooler_state[hf_name] = tensor

        self.encoder.load_state_dict(backbone_state, strict=False)
        logger.info(
            "[mmBERT-GLiNER2] Loaded backbone: %s/%s keys",
            len(backbone_state),
            len(vllm_backbone),
        )

        if pooler_state:
            missing = pooler_keys - pooler_state.keys()
            unexpected = pooler_state.keys() - pooler_keys
            if missing or unexpected:
                raise RuntimeError(
                    "mmBERT-GLiNER2 pooler weight-load mismatch — "
                    f"counting_layer={getattr(self.config, 'counting_layer', '?')!r} "
                    f"missing={sorted(missing)!r} unexpected={sorted(unexpected)!r}. "
                    "This indicates a pooler variant / checkpoint mismatch."
                )
            self._business_pooler.load_state_dict(pooler_state, strict=False)
            device = next(self.encoder.parameters()).device
            dtype = self.vllm_config.model_config.dtype
            self._business_pooler.to(device=device, dtype=dtype)
            logger.info(
                "[mmBERT-GLiNER2] Loaded pooler: %s/%s keys (counting_layer=%s)",
                len(pooler_state),
                len(pooler_keys),
                getattr(self.config, "counting_layer", "?"),
            )
        else:
            raise RuntimeError(
                "mmBERT-GLiNER2 pooler weights were empty after load_weights. "
                "Refusing to serve a random-init head."
            )

        return set(name for name, _ in self.named_parameters())
