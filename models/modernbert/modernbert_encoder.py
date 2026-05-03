"""
ModernBERT with Full vLLM Parallel Layers + Fused GLU Triton Kernel
(Corrected & Optimized)
"""

import copy
import os
import sys
from pathlib import Path as _Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# CRITICAL FIX: Disable Flash Attention 2.8.3
# ============================================================================
os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'sdpa'
if 'flash_attn' in sys.modules:
    del sys.modules['flash_attn']
sys.modules['flash_attn'] = None

from vllm.config import VllmConfig  # noqa: E402
from vllm.distributed import get_tensor_model_parallel_world_size  # noqa: E402
from vllm.model_executor.layers.linear import (  # noqa: E402
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding  # noqa: E402
from vllm.model_executor.model_loader.weight_utils import default_weight_loader  # noqa: E402

# Import fused Triton kernels from project-root kernels/ package.
# Each kernel has a HAS_FUSED_* flag so the model falls back to PyTorch if
# Triton is unavailable (e.g. CPU-only or older GPU).
_project_root = str(_Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from kernels.fused_glu_mlp import fused_gelu_mul_dropout
    HAS_FUSED_GLU = True
except ImportError:
    HAS_FUSED_GLU = False

try:
    from kernels.fused_dropout_residual import fused_dropout_residual
    HAS_FUSED_DROPOUT = True
except ImportError:
    HAS_FUSED_DROPOUT = False

try:
    from kernels.fused_layernorm import FusedLayerNorm
    HAS_FUSED_LAYERNORM = True
except ImportError:
    HAS_FUSED_LAYERNORM = False

try:
    from kernels.fused_rope_global import fused_rope_global_apply
    HAS_FUSED_ROPE = True
except ImportError:
    HAS_FUSED_ROPE = False

# ============================================================================
# OPTIMIZATION FLAGS
# ============================================================================
USE_FUSED_GLU = True
USE_FUSED_DROPOUT = True
USE_FUSED_LAYERNORM = True
USE_FUSED_ROPE = True

print(f"\n{'='*80}")
print("[ModernBERT Parallel] Optimization Flags:")
print(f"  USE_FUSED_GLU:       {USE_FUSED_GLU}")
print(f"  USE_FUSED_DROPOUT:   {USE_FUSED_DROPOUT}")
print(f"  USE_FUSED_LAYERNORM: {USE_FUSED_LAYERNORM}")
print(f"  USE_FUSED_ROPE:      {USE_FUSED_ROPE}")
print(f"{'='*80}\n")

try:
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
    from transformers.modeling_outputs import BaseModelOutput
    from transformers.models.modernbert.modeling_modernbert import (
        ModernBertConfig,
        ModernBertRotaryEmbedding,
    )
    HAS_MODERNBERT = True
except ImportError:
    HAS_MODERNBERT = False

# ============================================================================
# EMBEDDINGS
# ============================================================================

class VllmModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        if USE_FUSED_LAYERNORM and HAS_FUSED_LAYERNORM:
            self.norm = FusedLayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

        self.drop = nn.Dropout(config.embedding_dropout)

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is not None:
            hidden_states = self.drop(self.norm(inputs_embeds))
        else:
            hidden_states = self.tok_embeddings(input_ids)
            hidden_states = self.drop(self.norm(hidden_states))
        return hidden_states

# ============================================================================
# MLP (Corrected to use ColumnParallelLinear)
# ============================================================================

class ParallelModernBertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # FIX: Use ColumnParallelLinear for the Input Projection (Wi)
        # This ensures the matrix multiplication is actually split across GPUs.
        # vLLM automatically handles the chunking of the GLU weights.
        self.Wi = ColumnParallelLinear(
            config.hidden_size,
            int(config.intermediate_size) * 2,
            bias=config.mlp_bias
        )

        # Output projection
        self.Wo = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias
        )

        self.act = nn.GELU(approximate="none")
        self.act_fn = config.hidden_activation if hasattr(config, 'hidden_activation') else 'gelu'
        self.dropout_p = config.mlp_dropout
        self.drop = nn.Dropout(config.mlp_dropout)

    def forward(self, hidden_states):
        # Wi projection (Returns parallel chunks)
        x, _ = self.Wi(hidden_states)
        input_proj, gate_proj = x.chunk(2, dim=-1)

        # GLU activation (Fused Kernel)
        if USE_FUSED_GLU and HAS_FUSED_GLU and self.training:
            # We flatten input for the kernel to handle strides safely
            x = fused_gelu_mul_dropout(
                input_proj,
                gate_proj,
                act_fn=self.act_fn,
                dropout_p=self.dropout_p,
                training=self.training,
            )
        else:
            x = self.act(input_proj) * gate_proj
            x = self.drop(x)

        # Wo projection (RowParallel reduces the result across GPUs)
        x, _ = self.Wo(x)
        return x

# ============================================================================
# ATTENTION (Corrected for Dual vLLM RoPE)
# ============================================================================

class ParallelModernBertAttention(nn.Module):
    def __init__(self, config, layer_id, rotary_emb_global, rotary_emb_local):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = config.num_attention_heads
        self.num_heads_per_partition = self.num_heads // tp_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads_per_partition

        self.is_global = (layer_id % config.global_attn_every_n_layers == 0)

        # FIX: Use the passed vLLM Optimized RoPE instance
        self.rotary_emb = rotary_emb_global if self.is_global else rotary_emb_local

        self.Wqkv = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.num_heads,
            bias=config.attention_bias,
        )
        self.Wo = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias
        )
        self.out_drop = nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states, attention_mask, sliding_window_mask, position_ids, output_attentions=False):
        batch_size, seq_len, _ = hidden_states.shape

        # 1. QKV
        qkv, _ = self.Wqkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape (Batch, Seq, Heads, Dim)
        q = q.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)

        # 3. Apply HF RoPE (Standard Implementation)
        # Replaces vLLM kernel which had numerical issues with ModernBERT
        layer_type = "full_attention" if self.is_global else "sliding_attention"
        cos, sin = self.rotary_emb(v, position_ids=position_ids, layer_type=layer_type)
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        # 4. SDPA
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        mask = attention_mask if self.is_global else sliding_window_mask

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.out_drop.p if self.training else 0.0,
            is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        output, _ = self.Wo(attn_output)

        return self.out_drop(output), None

    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embedding to queries and keys."""
        # Use Fused Kernel if available (Fastest)
        if USE_FUSED_ROPE and HAS_FUSED_ROPE and q.is_cuda:
            return fused_rope_global_apply(q, k, cos, sin)

        # Fallback to PyTorch (Standard Implementation)
        # Expand cos/sin for all heads: (B, S, D) -> (B, S, 1, D)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

# ============================================================================
# ENCODER LAYER
# ============================================================================

class ParallelModernBertEncoderLayer(nn.Module):
    # FIX: Accept the RoPE instances in __init__
    def __init__(self, config, layer_id, rotary_emb_global, rotary_emb_local):
        super().__init__()
        self.config = config

        # Norms
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            if USE_FUSED_LAYERNORM and HAS_FUSED_LAYERNORM:
                self.attn_norm = FusedLayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
            else:
                self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

        if USE_FUSED_LAYERNORM and HAS_FUSED_LAYERNORM:
            self.mlp_norm = FusedLayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        else:
            self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

        # Pass RoPE down to attention
        self.attn = ParallelModernBertAttention(config, layer_id, rotary_emb_global, rotary_emb_local)
        self.mlp = ParallelModernBertMLP(config)

        self.drop = nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states, attention_mask, sliding_window_mask, position_ids, **kwargs):
        # Attention
        attn_input = self.attn_norm(hidden_states)
        attn_outputs = self.attn(
            attn_input,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids
        )
        attn_output = attn_outputs[0]

        # Residual + Dropout
        if USE_FUSED_DROPOUT and HAS_FUSED_DROPOUT:
            hidden_states = fused_dropout_residual(attn_output, hidden_states, self.config.attention_dropout, self.training)
        else:
            hidden_states = hidden_states + self.drop(attn_output)

        # MLP
        mlp_output = self.mlp(self.mlp_norm(hidden_states))

        # Residual + Dropout
        if USE_FUSED_DROPOUT and HAS_FUSED_DROPOUT:
            hidden_states = fused_dropout_residual(mlp_output, hidden_states, self.config.attention_dropout, self.training)
        else:
            hidden_states = hidden_states + self.drop(mlp_output)

        return (hidden_states,)

# ============================================================================
# MODEL
# ============================================================================

class ModernBertModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        if not HAS_MODERNBERT:
            raise ImportError("ModernBERT from transformers is required.")

        config = vllm_config.model_config.hf_config
        self.config = config
        self.prefix = prefix

        # Force SDPA
        config._attn_implementation = 'sdpa'
        if hasattr(config, 'use_flash_attention_2'):
            config.use_flash_attention_2 = False

        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            full_attention = rope_parameters.get("full_attention")
            sliding_attention = rope_parameters.get("sliding_attention")
            if not hasattr(config, "global_rope_theta") and isinstance(full_attention, dict):
                config.global_rope_theta = full_attention.get("rope_theta", 160000.0)
            if not hasattr(config, "local_rope_theta") and isinstance(sliding_attention, dict):
                config.local_rope_theta = sliding_attention.get("rope_theta", 10000.0)
        if not hasattr(config, "global_rope_theta"):
            config.global_rope_theta = getattr(config, "rope_theta", 160000.0)
        if not hasattr(config, "local_rope_theta"):
            config.local_rope_theta = getattr(config, "rope_theta", 10000.0)

        self.embeddings = VllmModernBertEmbeddings(config)

        # FIX: Initialize DUAL HF RoPE Instances with CORRECT Thetas
        # Replaces vLLM RoPE which caused accuracy issues
        conf_global = copy.copy(config)
        conf_global.rope_theta = config.global_rope_theta
        self.rope_global = ModernBertRotaryEmbedding(conf_global)

        conf_local = copy.copy(config)
        conf_local.rope_theta = config.local_rope_theta
        self.rope_local = ModernBertRotaryEmbedding(conf_local)

        # Pass RoPE instances down to layers
        self.layers = nn.ModuleList([
            ParallelModernBertEncoderLayer(config, i, self.rope_global, self.rope_local)
            for i in range(config.num_hidden_layers)
        ])

        if USE_FUSED_LAYERNORM and HAS_FUSED_LAYERNORM:
            self.final_norm = FusedLayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        else:
            self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

        self.gradient_checkpointing = False

        print(f"\n{'='*80}")
        print("[ModernBERT Parallel] vLLM Optimized Layers Loaded")
        print("  ✓ Dual HF RoPE Implementation (Standard)")
        print("  ✓ ColumnParallelLinear (MLP Input)")
        print("  ✓ Block-diagonal mask (cross-sequence isolation)")
        print(f"{'='*80}\n")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[tuple[torch.Tensor, ...], BaseModelOutput, torch.Tensor]:

        # vLLM Flat Input Handling
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if inputs_embeds is not None and inputs_embeds.dim() == 2:
             inputs_embeds = inputs_embeds.unsqueeze(0)

        # Batch/Seq checks
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
        else:
            batch_size, seq_len = inputs_embeds.shape[:2]
            device = inputs_embeds.device

        # Match PyLate + superpod: mask MASK tokens (50284) in attention.
        # PyLate confirmed: attention_mask=0 for MASK padding tokens.
        if input_ids is not None:
            PAD_TOKEN_ID = 50284
            token_mask = (input_ids != PAD_TOKEN_ID).to(torch.bool)
        else:
            token_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        if position_ids is None:
             position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Build block-diagonal mask from position_ids to prevent cross-sequence
        # attention when vLLM concatenates multiple sequences into one flat tensor.
        attention_mask = token_mask  # (batch, seq)

        # Create Masks (passing position_ids for block-diagonal construction)
        attention_mask_4d, sliding_window_mask = self._update_attention_mask(
            attention_mask, position_ids
        )

        # Embeddings
        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        # Layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.final_norm(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)

    def _build_seq_block_mask(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Build a block-diagonal boolean mask from position_ids.

        When vLLM batches multiple sequences, it concatenates them into a flat
        1D tensor. Each sequence's positions start from 0. We detect boundaries
        where positions reset (decrease) and assign segment IDs, then build a
        mask where only tokens in the same segment can attend to each other.

        Args:
            position_ids: (1, seq_len) tensor of position indices

        Returns:
            (seq_len, seq_len) boolean mask: True where tokens CAN attend
        """
        pos = position_ids.squeeze(0)  # (seq_len,)
        seq_len = pos.shape[0]

        # Detect sequence boundaries: position resets (pos[i] <= pos[i-1])
        # First token is always start of a sequence
        boundaries = torch.zeros(seq_len, dtype=torch.bool, device=pos.device)
        boundaries[0] = True
        if seq_len > 1:
            boundaries[1:] = pos[1:] <= pos[:-1]

        # Convert boundaries to segment IDs via cumsum
        segment_ids = boundaries.cumsum(0)  # (seq_len,)

        # Block-diagonal mask: tokens attend only within same segment
        return segment_ids.unsqueeze(1) == segment_ids.unsqueeze(0)  # (seq_len, seq_len)

    def _update_attention_mask(self, attention_mask, position_ids=None):
        dtype = self.embeddings.tok_embeddings.weight.dtype
        seq_len = attention_mask.shape[1]

        # Start from the standard 4D causal-free attention mask
        global_attention_mask = _prepare_4d_attention_mask(attention_mask, dtype)

        # Apply block-diagonal mask to prevent cross-sequence attention
        if position_ids is not None:
            block_mask = self._build_seq_block_mask(position_ids)
            # block_mask is (seq_len, seq_len), expand to (1, 1, seq_len, seq_len)
            block_mask_4d = block_mask.unsqueeze(0).unsqueeze(0)
            # Where block_mask is False (cross-sequence), set to -inf
            global_attention_mask = global_attention_mask.masked_fill(
                block_mask_4d.logical_not(), torch.finfo(dtype).min
            )

        # Sliding window mask: combine block-diagonal with distance constraint
        rows = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
        distance = torch.abs(rows - rows.T)
        window_mask = (distance <= self.config.local_attention // 2).unsqueeze(0).unsqueeze(0).to(attention_mask.device)
        sliding_window_mask = global_attention_mask.masked_fill(
            window_mask.logical_not(), torch.finfo(dtype).min
        )
        return global_attention_mask, sliding_window_mask

    def load_weights(self, weights):
        # Standard vLLM loading logic
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            param_name = name
            if self.prefix and name.startswith(f"{self.prefix}."):
                param_name = name[len(self.prefix)+1:]
            if param_name in params_dict:
                param = params_dict[param_name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)

__all__ = ["ModernBertModel"]
