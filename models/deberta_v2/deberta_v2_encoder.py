# SPDX-License-Identifier: Apache-2.0
"""
DeBERTa v2/v3 Encoder for vLLM — Inference-only, vLLM-optimized.

Implements the DeBERTa v2 architecture with:
- Disentangled self-attention (c2p + p2c with log-bucket position encoding)
- Separate query_proj / key_proj / value_proj (not fused in_proj like v1)
- Standard nn.LayerNorm (not custom DebertaLayerNorm like v1)
- Optional ConvLayer after first encoder block
- Optional norm_rel_ebd (LayerNorm on relative embeddings)
- position_buckets for log-bucket relative positions
- FlashDeBERTa fused Triton kernel for attention + disentangled bias
  (with PyTorch fallback for exact HF parity)
- vLLM parallel linear layers (ColumnParallelLinear, RowParallelLinear)
- VocabParallelEmbedding for token embeddings

Weight loading supports HuggingFace DeBERTa v2/v3 checkpoints
(e.g., microsoft/deberta-v2-base, microsoft/deberta-v3-base).
"""

from typing import ClassVar, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DebertaV2Config
from transformers.activations import ACT2FN
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
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
# DeBERTa v2/v3 uses separate `query_proj`, `key_proj`, `value_proj` linears —
# matching HF transformers naming exactly — so PEFT adapters with
# `target_modules=["query_proj", "key_proj", "value_proj"]` (the GLiNER2
# convention) map 1:1 onto our `ColumnParallelLinear` projections with no
# packing rewrite. `pos_key_proj` / `pos_query_proj` (when
# ``share_att_key=False``) and the `attention.output.dense`,
# `intermediate.dense`, `output.dense` row/column projections are standard
# single-linear LoRA targets.
#
# `embedding_modules` is intentionally empty; GLiNER2 adapters do not touch
# the token embedding matrix, and keeping it empty avoids pulling the vocab-
# parallel embedding into the LoRA path.

PACKED_MODULES_MAPPING: dict[str, list[str]] = {}
EMBEDDING_MODULES: dict[str, str] = {}


# ============================================================================
# Relative Position Utilities (v2: with log-bucket)
# ============================================================================

@torch.jit.script
def make_log_bucket_position(relative_pos: torch.Tensor, bucket_size: int,
                             max_position: int) -> torch.Tensor:
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1).type_as(relative_pos),
        torch.abs(relative_pos),
    )
    log_pos = (
        torch.ceil(
            torch.log(abs_pos / mid) /
            torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)
        ) + mid
    )
    bucket_pos = torch.where(
        abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign
    )
    return bucket_pos


def build_relative_position_v2(query_size: int, key_size: int,
                                device: torch.device,
                                bucket_size: int = -1,
                                max_position: int = -1) -> torch.Tensor:
    """Build relative position tensor [1, query_size, key_size] with optional log-bucket."""
    q_ids = torch.arange(query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.to(torch.long)
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


@torch.jit.script
def scaled_size_sqrt(query_layer: torch.Tensor, scale_factor: int) -> torch.Tensor:
    return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)


# ============================================================================
# Disentangled Self-Attention (v2: separate Q/K/V, position_buckets)
# ============================================================================

class DisentangledSelfAttentionV2(nn.Module):
    """DeBERTa v2 disentangled self-attention with vLLM parallel layers.

    Uses separate query_proj/key_proj/value_proj (with bias).
    Supports position_buckets, share_att_key, and log-bucket relative positions.
    """

    def __init__(self, config: DebertaV2Config, prefix: str = ""):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) not divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(config, "attention_head_size", _attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # DeBERTa v2: separate Q/K/V with bias
        self.query_proj = ColumnParallelLinear(
            config.hidden_size, self.all_head_size, bias=True,
            prefix=f"{prefix}.query_proj",
        )
        self.key_proj = ColumnParallelLinear(
            config.hidden_size, self.all_head_size, bias=True,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ColumnParallelLinear(
            config.hidden_size, self.all_head_size, bias=True,
            prefix=f"{prefix}.value_proj",
        )

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = ColumnParallelLinear(
                        config.hidden_size, self.all_head_size, bias=True,
                        prefix=f"{prefix}.pos_key_proj",
                    )
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = ColumnParallelLinear(
                        config.hidden_size, self.all_head_size, bias=True,
                        prefix=f"{prefix}.pos_query_proj",
                    )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.tp_size = get_tensor_model_parallel_world_size()
        self.heads_per_partition = self.num_attention_heads // self.tp_size

        # Flash kernel availability
        self.use_flash_kernel = HAS_FLASH_DEBERTA

    def transpose_for_scores(self, x):
        """(B, L, all_head_size) -> (B*heads, L, head_dim) for bmm compat with HF."""
        new_x_shape = x.size()[:-1] + (self.heads_per_partition, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def _transpose_4d(self, x):
        """(B, L, all_head_size) -> (B, H, L, D) for flash kernel."""
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

        # Separate Q/K/V projections
        query_layer_out, _ = self.query_proj(hidden_states)
        key_layer_out, _ = self.key_proj(hidden_states)
        value_layer_out, _ = self.value_proj(hidden_states)

        # ── Flash DeBERTa path: fused Triton kernel ──────────────
        if self.use_flash_kernel and self.relative_attention and rel_embeddings is not None:
            context_layer = self._flash_forward(
                query_layer_out, key_layer_out, value_layer_out,
                attention_mask, rel_embeddings,
            )
        else:
            # ── PyTorch fallback path ────────────────────────────
            context_layer = self._pytorch_forward(
                query_layer_out, key_layer_out, value_layer_out,
                attention_mask, relative_pos, rel_embeddings,
            )

        return context_layer

    # ── Fused Triton kernel path ─────────────────────────────────

    def _flash_forward(
        self,
        query_layer_out: torch.Tensor,   # (B, L, all_head_size)
        key_layer_out: torch.Tensor,     # (B, L, all_head_size)
        value_layer_out: torch.Tensor,   # (B, L, all_head_size)
        attention_mask: torch.Tensor,
        rel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Flash attention with fused disentangled position bias."""
        # Reshape to 4D (B, H, L, D) for flash kernel
        query_layer = self._transpose_4d(query_layer_out)
        key_layer = self._transpose_4d(key_layer_out)
        value_layer = self._transpose_4d(value_layer_out)

        rel_embeddings = self.pos_dropout(rel_embeddings)
        att_span = self.pos_ebd_size
        rel_emb = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)

        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        sm_scale = 1.0 / (self.attention_head_size * scale_factor) ** 0.5

        # Pre-compute pos_key/pos_query using 4D layout
        pos_key = None
        pos_query = None

        if self.share_att_key:
            if "c2p" in self.pos_att_type:
                pos_key_layer_out, _ = self.key_proj(rel_emb)
                pos_key_layer = self._transpose_4d(pos_key_layer_out)  # (1, H, 2*att_span, D)
                pos_key = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            if "p2c" in self.pos_att_type:
                pos_query_layer_out, _ = self.query_proj(rel_emb)
                pos_query_layer = self._transpose_4d(pos_query_layer_out)
                # NOTE: Do NOT pre-scale — kernel handles scaling via sm_scale
                pos_query = torch.matmul(
                    key_layer, pos_query_layer.transpose(-1, -2).to(dtype=key_layer.dtype)
                )
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer_out, _ = self.pos_key_proj(rel_emb)
                pos_key_layer = self._transpose_4d(pos_key_layer_out)
                pos_key = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            if "p2c" in self.pos_att_type:
                pos_query_layer_out, _ = self.pos_query_proj(rel_emb)
                pos_query_layer = self._transpose_4d(pos_query_layer_out)
                # NOTE: Do NOT pre-scale — kernel handles scaling via sm_scale
                pos_query = torch.matmul(
                    key_layer, pos_query_layer.transpose(-1, -2).to(dtype=key_layer.dtype)
                )

        # Build seq_lengths from attention_mask
        M = query_layer.size(2)
        if attention_mask.dim() == 4:
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
            position_buckets=self.position_buckets if self.position_buckets > 0 else att_span,
            max_relative_distance=self.max_relative_positions,
            use_log_bucket=(self.position_buckets > 0),  # v2 uses log-bucket when position_buckets set
        )

        # Reshape back: (B, H, L, D) -> (B, L, all_head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer

    # ── PyTorch fallback path ────────────────────────────────────

    def _pytorch_forward(
        self,
        query_layer_out: torch.Tensor,
        key_layer_out: torch.Tensor,
        value_layer_out: torch.Tensor,
        attention_mask: torch.Tensor,
        relative_pos: Optional[torch.Tensor],
        rel_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Original PyTorch attention (exact HF parity)."""
        query_layer = self.transpose_for_scores(query_layer_out)
        key_layer = self.transpose_for_scores(key_layer_out)
        value_layer = self.transpose_for_scores(value_layer_out)

        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1

        scale = scaled_size_sqrt(query_layer, scale_factor)
        attention_scores = torch.bmm(
            query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype)
        )

        # Disentangled relative position bias
        if self.relative_attention and rel_embeddings is not None:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self._disentangled_att_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )
            if rel_att is not None:
                attention_scores = attention_scores + rel_att

        # Reshape for mask application: (B*H, L, L) -> (B, H, L, L)
        attention_scores = attention_scores.view(
            -1, self.heads_per_partition, attention_scores.size(-2), attention_scores.size(-1)
        )

        # Apply mask
        attention_mask = attention_mask.bool()
        attention_scores = attention_scores.masked_fill(
            ~attention_mask, torch.finfo(query_layer.dtype).min
        )

        # Softmax + dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Context via bmm
        context_layer = torch.bmm(
            attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)),
            value_layer,
        )
        context_layer = (
            context_layer.view(-1, self.heads_per_partition, context_layer.size(-2), context_layer.size(-1))
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)
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
            q_len = query_layer.size(1)  # bmm layout: (B*H, L, D)
            k_len = key_layer.size(1)
            relative_pos = build_relative_position_v2(
                q_len, k_len, query_layer.device,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )

        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.to(device=query_layer.device, dtype=torch.long)

        rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)

        if self.share_att_key:
            pos_query_layer_out, _ = self.query_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer_out)
            pos_query_layer = pos_query_layer.repeat(
                query_layer.size(0) // self.heads_per_partition, 1, 1
            )
            pos_key_layer_out, _ = self.key_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer_out)
            pos_key_layer = pos_key_layer.repeat(
                query_layer.size(0) // self.heads_per_partition, 1, 1
            )
        else:
            pos_key_layer = None
            pos_query_layer = None
            if "c2p" in self.pos_att_type:
                pos_key_layer_out, _ = self.pos_key_proj(rel_embeddings)
                pos_key_layer = self.transpose_for_scores(pos_key_layer_out)
                pos_key_layer = pos_key_layer.repeat(
                    query_layer.size(0) // self.heads_per_partition, 1, 1
                )
            if "p2c" in self.pos_att_type:
                pos_query_layer_out, _ = self.pos_query_proj(rel_embeddings)
                pos_query_layer = self.transpose_for_scores(pos_query_layer_out)
                pos_query_layer = pos_query_layer.repeat(
                    query_layer.size(0) // self.heads_per_partition, 1, 1
                )

        score = 0

        # content->position
        if "c2p" in self.pos_att_type and pos_key_layer is not None:
            scale = scaled_size_sqrt(pos_key_layer, scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(
                c2p_att, dim=-1,
                index=c2p_pos.squeeze(0).expand(
                    [query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]
                ),
            )
            score = score + c2p_att / scale.to(dtype=c2p_att.dtype)

        # position->content
        if "p2c" in self.pos_att_type and pos_query_layer is not None:
            scale = scaled_size_sqrt(pos_query_layer, scale_factor)
            if key_layer.size(1) != query_layer.size(1):
                r_pos = build_relative_position_v2(
                    key_layer.size(1), key_layer.size(1), key_layer.device,
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                )
            else:
                r_pos = relative_pos

            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(
                p2c_att, dim=-1,
                index=p2c_pos.squeeze(0).expand(
                    [query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]
                ),
            ).transpose(-1, -2)

            score = score + p2c_att / scale.to(dtype=p2c_att.dtype)

        return score


# ============================================================================
# Self-Output (projection + LayerNorm + residual + dropout)
# ============================================================================

class DebertaV2SelfOutput(nn.Module):
    def __init__(self, config: DebertaV2Config, prefix: str = ""):
        super().__init__()
        self.dense = RowParallelLinear(
            config.hidden_size, config.hidden_size, bias=True,
            prefix=f"{prefix}.dense",
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# ============================================================================
# Attention (self-attention + output projection)
# ============================================================================

class DebertaV2Attention(nn.Module):
    def __init__(self, config: DebertaV2Config, prefix: str = ""):
        super().__init__()
        self.self_attn = DisentangledSelfAttentionV2(config, prefix=f"{prefix}.self")
        self.output = DebertaV2SelfOutput(config, prefix=f"{prefix}.output")

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

class DebertaV2Intermediate(nn.Module):
    def __init__(self, config: DebertaV2Config, prefix: str = ""):
        super().__init__()
        self.dense = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, bias=True,
            prefix=f"{prefix}.dense",
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class DebertaV2Output(nn.Module):
    def __init__(self, config: DebertaV2Config, prefix: str = ""):
        super().__init__()
        self.dense = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=True,
            prefix=f"{prefix}.dense",
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# ============================================================================
# ConvLayer (optional, DeBERTa v2 specific)
# ============================================================================

class ConvLayer(nn.Module):
    """Optional convolutional layer in DeBERTa v2 encoder."""

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        kernel_size = getattr(config, "conv_kernel_size", 3)
        groups = getattr(config, "conv_groups", 1)
        self.conv_act = getattr(config, "conv_act", "tanh")
        self.conv = nn.Conv1d(
            config.hidden_size, config.hidden_size, kernel_size,
            padding=(kernel_size - 1) // 2, groups=groups,
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, residual_states, input_mask):
        out = (
            self.conv(hidden_states.permute(0, 2, 1).contiguous())
            .permute(0, 2, 1)
            .contiguous()
        )
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = ACT2FN[self.conv_act](self.dropout(out))

        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input).to(layer_norm_input)

        if input_mask is not None:
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)
            input_mask = input_mask.to(output.dtype)
            output = output * input_mask

        return output


# ============================================================================
# Encoder Layer
# ============================================================================

class DebertaV2Layer(nn.Module):
    def __init__(self, config: DebertaV2Config, prefix: str = ""):
        super().__init__()
        self.attention = DebertaV2Attention(config, prefix=f"{prefix}.attention")
        self.intermediate = DebertaV2Intermediate(config, prefix=f"{prefix}.intermediate")
        self.output = DebertaV2Output(config, prefix=f"{prefix}.output")

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

class DebertaV2Embeddings(nn.Module):
    """DeBERTa v2 embeddings: word + position + token_type + LayerNorm."""

    def __init__(self, config: DebertaV2Config):
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

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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

class DebertaV2Encoder(nn.Module):
    """DeBERTa v2 encoder with relative position bias and optional ConvLayer."""

    def __init__(self, config: DebertaV2Config, prefix: str = ""):
        super().__init__()
        self.layer = nn.ModuleList([
            DebertaV2Layer(config, prefix=f"{prefix}.layer.{i}")
            for i in range(config.num_hidden_layers)
        ])

        self.relative_attention = getattr(config, "relative_attention", False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2
            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2
            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        self.norm_rel_ebd = [
            x.strip()
            for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")
        ]
        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True
            )

        self.conv = (
            ConvLayer(config)
            if getattr(config, "conv_kernel_size", 0) > 0
            else None
        )

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and "layer_norm" in self.norm_rel_ebd:
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

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
            position_buckets = getattr(self, 'position_buckets', -1)
            max_relative_positions = getattr(self, 'max_relative_positions', -1)
            relative_pos = build_relative_position_v2(
                q_len, q_len, hidden_states.device,
                bucket_size=position_buckets,
                max_position=max_relative_positions,
            )
            return relative_pos
        return None

    def forward(self, hidden_states, attention_mask, input_mask=None):
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states)
        rel_embeddings = self.get_rel_embedding()

        all_hidden_states = [hidden_states]

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states, attention_mask,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )
            # ConvLayer after first layer
            if i == 0 and self.conv is not None and input_mask is not None:
                hidden_states = self.conv(hidden_states, all_hidden_states[-1], input_mask)
            all_hidden_states.append(hidden_states)

        return hidden_states


# ============================================================================
# Top-Level Model
# ============================================================================

class DebertaV2EncoderModel(nn.Module, SupportsLoRA):
    """DeBERTa v2/v3 encoder model for vLLM — returns hidden states only.

    Declares `SupportsLoRA` so vLLM's LoRA manager can inject adapters into
    the encoder's parallel linears (``query_proj``, ``key_proj``,
    ``value_proj``, ``pos_key_proj``, ``pos_query_proj``,
    ``attention.output.dense``, ``intermediate.dense``, ``output.dense``). See
    the module-level ``PACKED_MODULES_MAPPING`` / ``EMBEDDING_MODULES``
    constants for the rationale.
    """

    supports_lora: ClassVar[bool] = True
    packed_modules_mapping: ClassVar[dict[str, list[str]]] = PACKED_MODULES_MAPPING
    embedding_modules: ClassVar[dict[str, str]] = EMBEDDING_MODULES

    def __init__(self, vllm_config: VllmConfig = None, config: DebertaV2Config = None,
                 prefix: str = ""):
        super().__init__()

        if vllm_config is not None:
            config = vllm_config.model_config.hf_config
        assert config is not None, "Must provide either vllm_config or config"

        self.config = config
        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config, prefix=f"{prefix}encoder" if prefix else "encoder")

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

        # For ConvLayer: pass input_mask (2D) separately
        input_mask = attention_mask
        if input_mask.dim() > 2:
            input_mask = None  # ConvLayer needs 2D mask

        encoder_output = self.encoder(
            embedding_output, attention_mask, input_mask=input_mask
        )
        return encoder_output

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from HuggingFace DeBERTa v2/v3 checkpoint.

        HF prefix: deberta.embeddings.*, deberta.encoder.*
        We strip 'deberta.' prefix and map to our structure.
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())

        for name, loaded_weight in weights:
            param_name = name
            # Strip 'deberta.' prefix from HF checkpoint
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
    "DebertaV2EncoderModel",
    "DebertaV2Encoder",
    "PACKED_MODULES_MAPPING",
    "EMBEDDING_MODULES",
]
