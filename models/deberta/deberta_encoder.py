# SPDX-License-Identifier: Apache-2.0
"""
DeBERTa v1 Encoder for vLLM — Inference-only, vLLM-optimized.

Implements the DeBERTa v1 architecture with:
- Disentangled self-attention (c2p + p2c relative position bias)
- Custom DebertaLayerNorm (epsilon inside sqrt, NOT nn.LayerNorm)
- vLLM parallel linear layers (ColumnParallelLinear, RowParallelLinear)
- VocabParallelEmbedding for token embeddings
- FlashDeBERTa fused Triton kernel for attention + disentangled bias
  (with PyTorch SDPA fallback for non-Triton environments)

Weight loading supports HuggingFace DeBERTa checkpoints
(e.g., microsoft/deberta-base, deberta-large, deberta-xlarge).
"""

from typing import ClassVar, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DebertaConfig
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA

try:
    from kernels.flash_deberta_attention import HAS_TRITON, flash_deberta_attention
    HAS_FLASH_DEBERTA = HAS_TRITON
except ImportError:
    HAS_FLASH_DEBERTA = False


# ============================================================================
# LoRA registration metadata
# ============================================================================
#
# vLLM discovers adaptable linears by walking `named_modules()` and matching
# against the LoRARequest's `target_modules`. All projections in this encoder
# are already `ColumnParallelLinear`/`RowParallelLinear`, so vLLM's LoRA
# manager can inject adapters into them without any further structural change.
#
# The mapping + class membership below satisfy the `SupportsLoRA` Protocol
# check in `vllm.model_executor.models.interfaces.supports_lora(...)`.
#
# DeBERTa v1 uses a single fused `in_proj` (Q/K/V concatenated) exactly as HF
# transformers does, so no packing rewrite is needed: PEFT adapters trained
# against `target_modules=["in_proj"]` map 1:1. The relative-position linears
# (`pos_proj`, `pos_q_proj`) are standard single-linear LoRA targets. Likewise
# for `attention.output.dense`, `intermediate.dense`, and `output.dense`.
#
# `embedding_modules` is intentionally empty; GLiNER2/DeBERTa PEFT recipes do
# not adapt the token embedding matrix, and keeping this empty avoids pulling
# the vocab-parallel embedding into the LoRA path.

PACKED_MODULES_MAPPING: dict[str, list[str]] = {}
EMBEDDING_MODULES: dict[str, str] = {}


# ============================================================================
# DebertaLayerNorm — epsilon INSIDE the sqrt (different from nn.LayerNorm)
# ============================================================================

class DebertaLayerNorm(nn.Module):
    """LayerNorm module with epsilon inside the square root.

    This produces DIFFERENT numerical results from nn.LayerNorm which has
    epsilon outside the sqrt. Must be preserved exactly for parity.
    """

    def __init__(self, size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_type = hidden_states.dtype
        hidden_states = hidden_states.float()
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.to(input_type)
        y = self.weight * hidden_states + self.bias
        return y


# ============================================================================
# Relative Position Utilities
# ============================================================================

@torch.jit.script
def build_relative_position(query_size: int, key_size: int, device: torch.device):
    """Build relative position tensor [1, query_size, key_size]."""
    q_ids = torch.arange(query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    return rel_pos_ids.unsqueeze(0)


@torch.jit.script
def c2p_dynamic_expand(c2p_pos: torch.Tensor, query_layer: torch.Tensor,
                       relative_pos: torch.Tensor) -> torch.Tensor:
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1),
                           query_layer.size(2), relative_pos.size(-1)])


@torch.jit.script
def p2c_dynamic_expand(c2p_pos: torch.Tensor, query_layer: torch.Tensor,
                       key_layer: torch.Tensor) -> torch.Tensor:
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1),
                           key_layer.size(-2), key_layer.size(-2)])


@torch.jit.script
def pos_dynamic_expand(pos_index: torch.Tensor, p2c_att: torch.Tensor,
                       key_layer: torch.Tensor) -> torch.Tensor:
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


# ============================================================================
# Disentangled Self-Attention
# ============================================================================

class DisentangledSelfAttention(nn.Module):
    """DeBERTa v1 disentangled self-attention with vLLM parallel layers.

    Uses a fused in_proj for Q/K/V with separate q_bias and v_bias.
    Computes c2p (content-to-position) and p2c (position-to-content)
    relative attention bias from learned rel_embeddings.
    """

    def __init__(self, config: DebertaConfig, prefix: str = ""):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) not divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # DeBERTa v1: fused in_proj for Q/K/V (no bias), separate q_bias, v_bias
        self.in_proj = ColumnParallelLinear(
            config.hidden_size,
            self.all_head_size * 3,
            bias=False,
            prefix=f"{prefix}.in_proj",
        )
        self.q_bias = nn.Parameter(torch.zeros(self.all_head_size))
        self.v_bias = nn.Parameter(torch.zeros(self.all_head_size))

        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, "relative_attention", False)
        self.talking_head = getattr(config, "talking_head", False)

        if self.talking_head:
            self.head_logits_proj = nn.Linear(
                config.num_attention_heads, config.num_attention_heads, bias=False
            )
            self.head_weights_proj = nn.Linear(
                config.num_attention_heads, config.num_attention_heads, bias=False
            )

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)

            if "c2p" in self.pos_att_type:
                self.pos_proj = ColumnParallelLinear(
                    config.hidden_size,
                    self.all_head_size,
                    bias=False,
                    prefix=f"{prefix}.pos_proj",
                )
            if "p2c" in self.pos_att_type:
                self.pos_q_proj = ColumnParallelLinear(
                    config.hidden_size,
                    self.all_head_size,
                    bias=True,
                    prefix=f"{prefix}.pos_q_proj",
                )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # TP info
        self.tp_size = get_tensor_model_parallel_world_size()
        self.heads_per_partition = self.num_attention_heads // self.tp_size

        # Flash kernel availability (disable for talking_head which needs
        # intermediate attention matrix access)
        self.use_flash_kernel = HAS_FLASH_DEBERTA and not self.talking_head

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.heads_per_partition, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        relative_pos: Optional[torch.Tensor] = None,
        rel_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape

        # QKV projection via fused in_proj
        qp, _ = self.in_proj(hidden_states)
        query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)

        # Add biases
        query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
        value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])

        # ── Flash DeBERTa path: fused Triton kernel ──────────────
        if self.use_flash_kernel and self.relative_attention and rel_embeddings is not None:
            context_layer = self._flash_forward(
                query_layer, key_layer, value_layer,
                attention_mask, rel_embeddings,
            )
        else:
            # ── PyTorch fallback path ────────────────────────────
            context_layer = self._pytorch_forward(
                query_layer, key_layer, value_layer,
                attention_mask, relative_pos, rel_embeddings,
            )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer

    # ── Fused Triton kernel path ─────────────────────────────────

    def _flash_forward(
        self,
        query_layer: torch.Tensor,   # (B, H, M, D)
        key_layer: torch.Tensor,     # (B, H, N, D)
        value_layer: torch.Tensor,   # (B, H, N, D)
        attention_mask: torch.Tensor,
        rel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Flash attention with fused disentangled position bias."""
        rel_embeddings = self.pos_dropout(rel_embeddings)
        M = query_layer.size(2)
        N = key_layer.size(2)
        att_span = min(max(M, N), self.max_relative_positions)

        # Slice rel_embeddings to the needed span
        rel_emb = rel_embeddings[
            self.max_relative_positions - att_span : self.max_relative_positions + att_span, :
        ].unsqueeze(0)  # (1, 2*att_span, hidden_size)

        scale_factor = 1 + len(self.pos_att_type)
        sm_scale = 1.0 / (self.attention_head_size * scale_factor) ** 0.5

        # Pre-compute pos_key: Q @ pos_proj(rel_emb)^T → (B, H, M, 2*att_span)
        pos_key = None
        if "c2p" in self.pos_att_type:
            pos_key_layer_out, _ = self.pos_proj(rel_emb)
            pos_key_layer = self.transpose_for_scores(pos_key_layer_out)  # (1, H, 2*att_span, D)
            pos_key = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))  # (B, H, M, 2*att_span)

        # Pre-compute pos_query: K @ pos_q_proj(rel_emb)^T → (B, H, N, 2*att_span)
        # NOTE: Do NOT pre-scale pos_query_layer — the kernel handles scaling via sm_scale
        pos_query = None
        if "p2c" in self.pos_att_type:
            pos_query_layer_out, _ = self.pos_q_proj(rel_emb)
            pos_query_layer = self.transpose_for_scores(pos_query_layer_out)  # (1, H, 2*att_span, D)
            pos_query = torch.matmul(
                key_layer, pos_query_layer.transpose(-1, -2).to(dtype=key_layer.dtype)
            )  # (B, H, N, 2*att_span)

        # Build seq_lengths from attention_mask
        if attention_mask.dim() == 4:
            # (B, 1, 1, N) or (B, 1, L, L) mask
            seq_lengths = attention_mask[:, 0, 0, :].sum(dim=-1).int()
        elif attention_mask.dim() == 2:
            seq_lengths = attention_mask.sum(dim=-1).int()
        else:
            seq_lengths = torch.full((query_layer.size(0),), M,
                                     dtype=torch.int32, device=query_layer.device)

        context_layer = flash_deberta_attention(
            q=query_layer,
            k=key_layer,
            v=value_layer,
            seq_lengths=seq_lengths,
            pos_key=pos_key,
            pos_query=pos_query,
            causal=False,
            sm_scale=sm_scale,
            position_buckets=att_span,
            max_relative_distance=self.max_relative_positions,
            use_log_bucket=False,  # v1 uses linear positions, not log-bucket
        )
        return context_layer

    # ── PyTorch fallback path ────────────────────────────────────

    def _pytorch_forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        relative_pos: Optional[torch.Tensor],
        rel_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Original PyTorch attention (exact HF parity)."""
        scale_factor = 1 + len(self.pos_att_type)
        scale = torch.sqrt(torch.tensor(
            self.attention_head_size * scale_factor, dtype=torch.float
        )).to(dtype=query_layer.dtype)
        query_layer = query_layer / scale
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # Disentangled relative position bias
        if self.relative_attention and rel_embeddings is not None:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self._disentangled_att_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )
            if rel_att is not None:
                attention_scores = attention_scores + rel_att

        # Talking head (pre-softmax)
        if self.talking_head and hasattr(self, 'head_logits_proj') and self.head_logits_proj is not None:
            attention_scores = self.head_logits_proj(
                attention_scores.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)

        # Apply mask
        attention_mask = attention_mask.bool()
        attention_scores = attention_scores.masked_fill(
            ~attention_mask, torch.finfo(query_layer.dtype).min
        )

        # Softmax + dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Talking head (post-softmax)
        if self.talking_head and hasattr(self, 'head_weights_proj') and self.head_weights_proj is not None:
            attention_probs = self.head_weights_proj(
                attention_probs.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)

        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer

    def _disentangled_att_bias(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        relative_pos: Optional[torch.Tensor],
        rel_embeddings: torch.Tensor,
        scale_factor: int,
    ) -> Optional[torch.Tensor]:
        if relative_pos is None:
            q_len = query_layer.size(2)
            k_len = key_layer.size(2)
            relative_pos = build_relative_position(q_len, k_len, query_layer.device)

        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)

        att_span = min(
            max(query_layer.size(2), key_layer.size(2)),
            self.max_relative_positions
        )
        relative_pos = relative_pos.long()

        rel_embeddings = rel_embeddings[
            self.max_relative_positions - att_span : self.max_relative_positions + att_span, :
        ].unsqueeze(0)

        score = 0

        # content->position
        if "c2p" in self.pos_att_type:
            pos_key_layer_out, _ = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer_out)
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(
                c2p_att, dim=-1,
                index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos)
            )
            score = score + c2p_att

        # position->content
        if "p2c" in self.pos_att_type:
            pos_query_layer_out, _ = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer_out)
            pos_query_layer = pos_query_layer / torch.sqrt(torch.tensor(
                pos_query_layer.size(-1) * scale_factor, dtype=torch.float
            )).to(dtype=pos_query_layer.dtype)

            if query_layer.size(2) != key_layer.size(2):
                r_pos = build_relative_position(
                    query_layer.size(2), key_layer.size(2), query_layer.device
                )
            else:
                r_pos = relative_pos

            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.matmul(
                key_layer, pos_query_layer.transpose(-1, -2).to(dtype=key_layer.dtype)
            )
            p2c_att = torch.gather(
                p2c_att, dim=-1,
                index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)
            ).transpose(-1, -2)

            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
                p2c_att = torch.gather(
                    p2c_att, dim=2,
                    index=pos_dynamic_expand(pos_index, p2c_att, key_layer)
                )

            score = score + p2c_att

        return score


# ============================================================================
# Self-Output (projection + LayerNorm + residual)
# ============================================================================

class DebertaSelfOutput(nn.Module):
    def __init__(self, config: DebertaConfig, prefix: str = ""):
        super().__init__()
        self.dense = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            prefix=f"{prefix}.dense",
        )
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# ============================================================================
# Attention (self-attention + output projection)
# ============================================================================

class DebertaAttention(nn.Module):
    def __init__(self, config: DebertaConfig, prefix: str = ""):
        super().__init__()
        self.self_attn = DisentangledSelfAttention(config, prefix=f"{prefix}.self")
        self.output = DebertaSelfOutput(config, prefix=f"{prefix}.output")

    def forward(self, hidden_states, attention_mask,
                relative_pos=None, rel_embeddings=None):
        self_output = self.self_attn(
            hidden_states, attention_mask,
            relative_pos=relative_pos, rel_embeddings=rel_embeddings,
        )
        attention_output = self.output(self_output, hidden_states)
        return attention_output


# ============================================================================
# MLP (Intermediate + Output)
# ============================================================================

class DebertaIntermediate(nn.Module):
    def __init__(self, config: DebertaConfig, prefix: str = ""):
        super().__init__()
        self.dense = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            prefix=f"{prefix}.dense",
        )
        self.act_fn = get_act_fn(config.hidden_act)

    def forward(self, hidden_states):
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class DebertaOutput(nn.Module):
    def __init__(self, config: DebertaConfig, prefix: str = ""):
        super().__init__()
        self.dense = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            prefix=f"{prefix}.dense",
        )
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# ============================================================================
# Encoder Layer
# ============================================================================

class DebertaLayer(nn.Module):
    def __init__(self, config: DebertaConfig, prefix: str = ""):
        super().__init__()
        self.attention = DebertaAttention(config, prefix=f"{prefix}.attention")
        self.intermediate = DebertaIntermediate(config, prefix=f"{prefix}.intermediate")
        self.output = DebertaOutput(config, prefix=f"{prefix}.output")

    def forward(self, hidden_states, attention_mask,
                relative_pos=None, rel_embeddings=None):
        attention_output = self.attention(
            hidden_states, attention_mask,
            relative_pos=relative_pos, rel_embeddings=rel_embeddings,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# ============================================================================
# Embeddings
# ============================================================================

class DebertaEmbeddings(nn.Module):
    """DeBERTa v1 embeddings: word + position + token_type + LayerNorm."""

    def __init__(self, config: DebertaConfig):
        super().__init__()
        getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size, self.embedding_size,
        )

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if self.position_biased_input:
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, self.embedding_size
            )
        else:
            self.position_embeddings = None

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, self.embedding_size
            )
        else:
            self.token_type_embeddings = None

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        else:
            self.embed_proj = None

        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None,
                mask=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings = embeddings + position_embeddings
        if self.token_type_embeddings is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        if self.embed_proj is not None:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)
            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


# ============================================================================
# Encoder
# ============================================================================

class DebertaEncoder(nn.Module):
    """DeBERTa v1 encoder with relative position bias."""

    def __init__(self, config: DebertaConfig, prefix: str = ""):
        super().__init__()
        self.layer = nn.ModuleList([
            DebertaLayer(config, prefix=f"{prefix}.layer.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.relative_attention = getattr(config, "relative_attention", False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.rel_embeddings = nn.Embedding(
                self.max_relative_positions * 2, config.hidden_size
            )

    def get_rel_embedding(self):
        return self.rel_embeddings.weight if self.relative_attention else None

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended * extended.squeeze(-2).unsqueeze(-1)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def get_rel_pos(self, hidden_states):
        if self.relative_attention:
            q_len = hidden_states.size(1)
            relative_pos = build_relative_position(q_len, q_len, hidden_states.device)
            return relative_pos
        return None

    def forward(self, hidden_states, attention_mask):
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states)
        rel_embeddings = self.get_rel_embedding()

        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states, attention_mask,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )

        return hidden_states


# ============================================================================
# Top-Level Model
# ============================================================================

class DebertaEncoderModel(nn.Module, SupportsLoRA):
    """DeBERTa v1 encoder model for vLLM — returns hidden states only.

    Declares `SupportsLoRA` so vLLM's LoRA manager can inject adapters into
    the encoder's `ColumnParallelLinear`/`RowParallelLinear` projections
    (``in_proj``, ``pos_proj``, ``pos_q_proj``, ``attention.output.dense``,
    ``intermediate.dense``, ``output.dense``). See the module-level
    ``PACKED_MODULES_MAPPING`` / ``EMBEDDING_MODULES`` constants for the
    rationale.
    """

    supports_lora: ClassVar[bool] = True
    packed_modules_mapping: ClassVar[dict[str, list[str]]] = PACKED_MODULES_MAPPING
    embedding_modules: ClassVar[dict[str, str]] = EMBEDDING_MODULES

    def __init__(self, vllm_config: VllmConfig = None, config: DebertaConfig = None,
                 prefix: str = ""):
        super().__init__()

        if vllm_config is not None:
            config = vllm_config.model_config.hf_config
        assert config is not None, "Must provide either vllm_config or config"

        self.config = config
        self.embeddings = DebertaEmbeddings(config)
        self.encoder = DebertaEncoder(config, prefix=f"{prefix}encoder" if prefix else "encoder")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Handle flat vLLM input
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if inputs_embeds is not None and inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        if input_ids is not None:
            input_shape = input_ids.shape
            device = input_ids.device
        else:
            input_shape = inputs_embeds.shape[:2]
            device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        encoder_output = self.encoder(embedding_output, attention_mask)
        return encoder_output

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from HuggingFace DeBERTa checkpoint.

        HF prefix: deberta.embeddings.*, deberta.encoder.*
        We strip 'deberta.' prefix and map to our structure.
        """
        # Build mapping from our param names
        params_dict = dict(self.named_parameters())

        # Also track buffers (e.g., position_ids)
        buffers_dict = dict(self.named_buffers())

        for name, loaded_weight in weights:
            # Strip 'deberta.' prefix from HF checkpoint
            param_name = name
            if param_name.startswith("deberta."):
                param_name = param_name[len("deberta."):]

            # Map HF attention names to our names
            # HF: encoder.layer.N.attention.self.X -> our: encoder.layer.N.attention.self_attn.X
            param_name = param_name.replace(".attention.self.", ".attention.self_attn.")

            if param_name in params_dict:
                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            elif param_name in buffers_dict:
                buffers_dict[param_name].copy_(loaded_weight)


__all__ = [
    "DebertaEncoderModel",
    "DebertaEncoder",
    "DebertaLayerNorm",
    "PACKED_MODULES_MAPPING",
    "EMBEDDING_MODULES",
]
