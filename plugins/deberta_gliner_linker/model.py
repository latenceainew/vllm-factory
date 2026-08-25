"""
GLiNER-Linker — Full bi-encoder pipeline for entity linking.

Backbone: Two custom vLLM-optimized DebertaEncoderModel v1 instances (text + labels encoder)
          with Flash DeBERTa Triton kernel + vLLM parallel layers
Pooler:   word extraction + Scorer (bilinear + MLP), matching GLiNER BiEncoderTokenModel
          (checkpoint includes unused LSTM weights; GLiNER inference does not apply them)
Weights:  GLiNER bi-encoder checkpoint with both encoder prefixes

Architecture:
    Text  → DebertaEncoderModel (text encoder)  → hidden states
    Labels → DebertaEncoderModel (labels encoder) → mean-pooled embeddings
    hidden states → word-level gather (words_mask)  ─┐
    label embeddings ───────────────────────────────┤
                                                    ↓
    Scorer(words, labels) → (W, C, 3) logits
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import DebertaConfig
from vllm.config import VllmConfig

from vllm_factory.optional_deps import require
from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

from .config import GLiNERLinkerConfig

logger = logging.getLogger(__name__)

# Load the custom DeBERTa v1 encoder with Flash DeBERTa Triton kernel
_ENCODER_PATH = Path(__file__).resolve().parents[2] / "models" / "deberta" / "deberta_encoder.py"


def _import_deberta_encoder():
    spec = importlib.util.spec_from_file_location("deberta_encoder", str(_ENCODER_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("deberta_encoder", mod)
    spec.loader.exec_module(mod)
    return mod


_encoder_mod = _import_deberta_encoder()
DebertaEncoderModel = _encoder_mod.DebertaEncoderModel


class GLiNERLinkerModel(nn.Module):
    """Full bi-encoder GLiNER-Linker model.

    The model forward runs the TEXT encoder (custom DeBERTa v1).
    The pooler (GLiNERLinkerPooler) handles word extraction, label encoding (or
    precomputed embeddings), and the scorer head → (W, C, 3) logits.
    """

    is_pooling_model = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg: GLiNERLinkerConfig = vllm_config.model_config.hf_config
        self.config = cfg
        self.vllm_config = vllm_config

        # Build DeBERTa v1 config (same architecture for both encoders).
        gliner_cfg_path = Path(vllm_config.model_config.model) / "gliner_config.json"
        if gliner_cfg_path.exists():
            with gliner_cfg_path.open() as fh:
                gliner_cfg = json.load(fh)
            encoder_cfg = DebertaConfig(**gliner_cfg["encoder_config"])
        else:
            encoder_cfg = DebertaConfig(
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
                position_biased_input=cfg.encoder_position_biased_input,
                pad_token_id=cfg.encoder_pad_token_id,
                pos_att_type=cfg.encoder_pos_att_type,
            )

        # TEXT encoder — runs in model forward, produces hidden states
        self.text_encoder = DebertaEncoderModel(config=encoder_cfg)

        # LABELS encoder — used by pooler for online label encoding
        self.labels_encoder = DebertaEncoderModel(config=encoder_cfg)

        # LSTM — native GLiNER bi-encoder inference applies this to word embeddings.
        H = cfg.encoder_hidden_size
        self.rnn = nn.LSTM(
            input_size=H,
            hidden_size=H // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.rnn.eval()

        # Span representation head used when represent_spans=True.
        span_rep_mod = require(
            "gliner.modeling.span_rep", "gliner", purpose="GLiNER linker span representation"
        )
        SpanRepLayer = span_rep_mod.SpanRepLayer
        self.span_rep_layer = SpanRepLayer(
            span_mode=getattr(cfg, "span_mode", "token_level"),
            hidden_size=H,
            max_width=getattr(cfg, "max_width", 12),
            dropout=float(getattr(cfg, "dropout", 0.0)),
        )

        # Scorer — bilinear interaction + MLP → 3 scores per token-label pair
        self.scorer_proj_token = nn.Linear(H, H * 2)
        self.scorer_proj_label = nn.Linear(H, H * 2)
        self.scorer_out_mlp = nn.Sequential(
            nn.Linear(H * 3, H * 4),
            nn.Dropout(0.0),
            nn.ReLU(),
            nn.Linear(H * 4, 3),
        )

        logger.info(
            "Initialized full pipeline: hidden=%d, layers=%d",
            H,
            cfg.encoder_num_hidden_layers,
        )

        # Pooler mirrors GLiNER's full bi-encoder inference path.
        from .pooler import GLiNERLinkerPooler

        self.pooler = VllmPoolerAdapter(GLiNERLinkerPooler(self), requires_token_ids=True)

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Required by vLLM's pooling runner to embed token IDs."""
        return self.text_encoder.embeddings.word_embeddings(input_ids)

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
        """Run TEXT encoder (custom DeBERTa v1), return hidden states for pooler."""
        attn_flat: Optional[torch.Tensor] = kwargs.get("attention_mask")
        if attn_flat is not None and not isinstance(attn_flat, torch.Tensor):
            attn_flat = torch.tensor(attn_flat, device=input_ids.device, dtype=torch.long)

        if input_ids is not None and input_ids.dim() == 1:
            if positions is not None and len(positions) > 0:
                boundaries = (positions == 0).nonzero(as_tuple=True)[0]
                if len(boundaries) > 1:
                    seqs = []
                    attn_seqs: list[Optional[torch.Tensor]] = []
                    for i in range(len(boundaries)):
                        start = boundaries[i].item()
                        end = (
                            boundaries[i + 1].item() if i + 1 < len(boundaries) else len(input_ids)
                        )
                        seqs.append(input_ids[start:end])
                        if attn_flat is not None:
                            if attn_flat.shape[0] != input_ids.shape[0]:
                                raise ValueError(
                                    "attention_mask length must match flat input_ids "
                                    f"({attn_flat.shape[0]} vs {input_ids.shape[0]})"
                                )
                            attn_seqs.append(attn_flat[start:end])
                        else:
                            attn_seqs.append(None)

                    max_len = max(s.size(0) for s in seqs)
                    pad_id = self.config.encoder_pad_token_id
                    batched_ids = torch.full(
                        (len(seqs), max_len),
                        pad_id,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                    attn_mask = torch.zeros(
                        (len(seqs), max_len),
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                    seq_lens = []
                    for i, s in enumerate(seqs):
                        L = s.size(0)
                        batched_ids[i, :L] = s
                        am = attn_seqs[i]
                        if am is not None:
                            attn_mask[i, :L] = am.to(dtype=torch.long)
                        else:
                            attn_mask[i, :L] = 1
                        seq_lens.append(L)

                    with torch.no_grad():
                        hs = self.text_encoder(
                            input_ids=batched_ids,
                            attention_mask=attn_mask,
                        )
                    # Custom encoder returns tensor: (B, max_len, H) or (B*max_len, H)
                    if hs.dim() == 2:
                        hs = hs.view(len(seqs), max_len, -1)

                    all_hidden = []
                    for i, L in enumerate(seq_lens):
                        all_hidden.append(hs[i, :L])
                    return torch.cat(all_hidden, dim=0)

        with torch.no_grad():
            if attn_flat is not None:
                if input_ids is None:
                    raise ValueError("attention_mask requires input_ids")
                if attn_flat.shape[0] != input_ids.shape[0]:
                    raise ValueError(
                        "attention_mask length must match input_ids "
                        f"({attn_flat.shape[0]} vs {input_ids.shape[0]})"
                    )
                if input_ids.dim() == 1:
                    hs = self.text_encoder(
                        input_ids=input_ids.unsqueeze(0),
                        attention_mask=attn_flat.unsqueeze(0).to(dtype=torch.long),
                    )
                else:
                    hs = self.text_encoder(
                        input_ids=input_ids,
                        attention_mask=attn_flat.to(dtype=torch.long),
                    )
            else:
                hs = self.text_encoder(input_ids=input_ids)

        # Custom encoder returns tensor directly; flatten to 2D
        if hs.dim() == 3:
            hs = hs.squeeze(0)
        return hs

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load all GLiNER bi-encoder checkpoint weights.

        Prefix mapping:
            token_rep_layer.bert_layer.model.*     → self.text_encoder
            token_rep_layer.labels_encoder.model.* → self.labels_encoder
            rnn.lstm.*                             → self.rnn
            scorer.proj_token.*                    → self.scorer_proj_token
            scorer.proj_label.*                    → self.scorer_proj_label
            scorer.out_mlp.*                       → self.scorer_out_mlp

        Both custom encoders' load_weights() expect HF DeBERTa keys with
        'deberta.' prefix and handle the .attention.self. → .attention.self_attn.
        mapping internally.
        """
        text_prefix = "token_rep_layer.bert_layer.model."
        labels_prefix = "token_rep_layer.labels_encoder.model."

        text_weights = []
        labels_weights = []
        rnn_state = {}
        scorer_state = {}
        span_rep_state = {}

        for hf_name, tensor in weights:
            if hf_name.startswith(text_prefix):
                key = hf_name[len(text_prefix) :]
                text_weights.append(("deberta." + key, tensor))

            elif hf_name.startswith(labels_prefix):
                key = hf_name[len(labels_prefix) :]
                labels_weights.append(("deberta." + key, tensor))

            elif hf_name.startswith("rnn.lstm."):
                key = hf_name[len("rnn.lstm.") :]
                rnn_state[key] = tensor

            elif hf_name.startswith("scorer."):
                key = hf_name[len("scorer.") :]
                scorer_state[key] = tensor
            elif hf_name.startswith("span_rep_layer."):
                key = hf_name[len("span_rep_layer.") :]
                span_rep_state[key] = tensor

        # Load text encoder via custom encoder's load_weights
        self.text_encoder.load_weights(text_weights)
        logger.info("Text encoder: %d weight tensors", len(text_weights))

        # Load labels encoder via custom encoder's load_weights
        self.labels_encoder.load_weights(labels_weights)
        logger.info("Labels encoder: %d weight tensors", len(labels_weights))

        # Load LSTM
        if rnn_state:
            self.rnn.load_state_dict(rnn_state, strict=False)
            logger.info("LSTM: %d keys", len(rnn_state))

        if span_rep_state:
            self.span_rep_layer.load_state_dict(span_rep_state, strict=False)
            logger.info("SpanRep: %d keys", len(span_rep_state))

        # Load scorer
        if scorer_state:
            loaded = 0
            for key, tensor in scorer_state.items():
                if key.startswith("proj_token."):
                    attr_key = key[len("proj_token.") :]
                    getattr(self.scorer_proj_token, attr_key).data.copy_(tensor)
                    loaded += 1
                elif key.startswith("proj_label."):
                    attr_key = key[len("proj_label.") :]
                    getattr(self.scorer_proj_label, attr_key).data.copy_(tensor)
                    loaded += 1
                elif key.startswith("out_mlp."):
                    parts = key.split(".")
                    idx = int(parts[1])
                    attr_name = parts[2]
                    getattr(self.scorer_out_mlp[idx], attr_name).data.copy_(tensor)
                    loaded += 1
            logger.info("Scorer: %d keys", loaded)

        return set(name for name, _ in self.named_parameters())
