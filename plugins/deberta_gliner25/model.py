"""GLiNER 2.5 vLLM model: DeBERTa v2 encoder + boundary pooler.

Pooler is imported inside ``__init__`` so ``register()`` stays safe without
the ``gliner2`` extra.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import ClassVar, Iterable, Tuple

import torch
import torch.nn as nn
from transformers import DebertaV2Config
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsLoRA

from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

from .config import GLiNER25Config
from .packing import pad_batch, sequence_lengths

logger = logging.getLogger(__name__)

_ENCODER_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "deberta_v2" / "deberta_v2_encoder.py"
)

_HEAD_PREFIXES = (
    "boundary_head.",
    "record_decoder.",
    "relation_scorer.",
    "classifier.",
)


def _import_deberta_v2_encoder():
    spec = importlib.util.spec_from_file_location("deberta_v2_encoder", str(_ENCODER_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("deberta_v2_encoder", mod)
    spec.loader.exec_module(mod)
    return mod


_encoder_mod = _import_deberta_v2_encoder()
DebertaV2EncoderModel = _encoder_mod.DebertaV2EncoderModel
_ENCODER_PACKED_MODULES_MAPPING: dict[str, list[str]] = _encoder_mod.PACKED_MODULES_MAPPING
_ENCODER_EMBEDDING_MODULES: dict[str, str] = _encoder_mod.EMBEDDING_MODULES


class GLiNER25VLLMModel(nn.Module, SupportsLoRA):
    """Boundary GLiNER 2.5: Flash DeBERTa encoder + gliner2 heads."""

    is_pooling_model = True
    supports_lora: ClassVar[bool] = True
    packed_modules_mapping: ClassVar[dict[str, list[str]]] = {
        f"encoder.{k}": [f"encoder.{n}" for n in v]
        for k, v in _ENCODER_PACKED_MODULES_MAPPING.items()
    }
    embedding_modules: ClassVar[dict[str, str]] = {
        f"encoder.{k}": v for k, v in _ENCODER_EMBEDDING_MODULES.items()
    }

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg: GLiNER25Config = vllm_config.model_config.hf_config
        self.config = cfg
        self.vllm_config = vllm_config

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
        self.encoder = DebertaV2EncoderModel(config=encoder_cfg)
        self._encoder_pad_id = int(cfg.encoder_pad_token_id or 0)

        from poolers.gliner25 import GLiNER25BoundaryPooler

        self._business_pooler = GLiNER25BoundaryPooler(
            hidden_size=cfg.encoder_hidden_size,
            boundary_head=cfg.boundary_head,
            tokenizer_name=vllm_config.model_config.model,
            max_model_len=getattr(vllm_config.model_config, "max_model_len", None),
        )
        self.pooler = VllmPoolerAdapter(self._business_pooler, requires_token_ids=True)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder.embeddings.word_embeddings(input_ids)

    def sample(self, logits: torch.Tensor, sampling_metadata):
        try:
            from vllm.sequence import SamplerOutput

            return SamplerOutput(outputs=[])
        except ImportError:
            return None

    def forward(
        self,
        input_ids: torch.Tensor = None,
        positions: torch.Tensor = None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the encoder over the scheduled batch and return flat hidden states.

        vLLM hands a single flat token tensor holding every scheduled sequence
        end to end. Feeding that straight to a bidirectional encoder would let
        the sequences attend to each other, so it is reshaped into one padded
        row per sequence before the forward and flattened back afterwards.

        Args:
            input_ids: Flat token ids for the whole batch, shape (total_tokens,).
            positions: Per-sequence position ids, restarting at 0 each sequence.
            intermediate_tensors: Unused; present for the vLLM model contract.
            inputs_embeds: Unused; embeddings are taken from ``input_ids``.
            **kwargs: Unused extras passed by the runner.

        Returns:
            Hidden states of shape (total_tokens, hidden_size), in the same
            token order as ``input_ids``.
        """
        flat = input_ids.view(-1) if input_ids.dim() > 1 else input_ids
        lengths = sequence_lengths(flat, positions)

        with torch.no_grad():
            if len(lengths) == 1:
                hs = self.encoder(input_ids=flat[: lengths[0]].unsqueeze(0))
            else:
                ids, mask = pad_batch(flat, lengths, self._encoder_pad_id)
                hs = self.encoder(input_ids=ids, attention_mask=mask)

        packed = torch.cat([hs[row, :length] for row, length in enumerate(lengths)], dim=0)
        # The runner pads the token count; give back a row per slot it sent.
        total = int(flat.numel())
        if packed.shape[0] == total:
            return packed
        out = packed.new_zeros((total, packed.shape[1]))
        out[: packed.shape[0]] = packed
        return out

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load encoder.* via the DeBERTa encoder; remaining prefixes into the pooler."""
        pooler_keys = set(self._business_pooler.state_dict().keys())
        backbone_weights = []
        pooler_loaded = {}

        for hf_name, tensor in weights:
            if hf_name.startswith("encoder."):
                hf_key = hf_name[len("encoder.") :]
                if "word_embeddings.weight" in hf_key:
                    vocab_size = getattr(self.config, "vocab_size", None)
                    if vocab_size and tensor.shape[0] != vocab_size:
                        if tensor.shape[0] > vocab_size:
                            tensor = tensor[:vocab_size]
                        else:
                            extra = vocab_size - tensor.shape[0]
                            tensor = torch.cat(
                                [tensor, torch.randn(extra, tensor.shape[1]) * 0.02],
                                dim=0,
                            )
                backbone_weights.append(("deberta." + hf_key, tensor))
                continue
            if any(hf_name.startswith(prefix) for prefix in _HEAD_PREFIXES):
                if hf_name in pooler_keys:
                    pooler_loaded[hf_name] = tensor

        self.encoder.load_weights(backbone_weights)
        logger.info("[GLiNER25] Loaded encoder: %s tensors", len(backbone_weights))

        if not pooler_loaded:
            raise RuntimeError(
                "GLiNER25 pooler weights were empty after load_weights. "
                "Refusing to serve a random-init boundary head."
            )
        missing = pooler_keys - pooler_loaded.keys()
        unexpected = pooler_loaded.keys() - pooler_keys
        if missing or unexpected:
            raise RuntimeError(
                "GLiNER25 pooler weight-load mismatch — "
                f"missing={sorted(missing)!r} unexpected={sorted(unexpected)!r}."
            )
        self._business_pooler.load_state_dict(pooler_loaded, strict=False)
        device = next(self.encoder.parameters()).device
        dtype = self.vllm_config.model_config.dtype
        self._business_pooler.to(device=device, dtype=dtype)
        logger.info("[GLiNER25] Loaded pooler: %s/%s keys", len(pooler_loaded), len(pooler_keys))
        return {name for name, _ in self.named_parameters()}
