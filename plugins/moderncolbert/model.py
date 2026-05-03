"""
ModernColBERT — Custom ModernBERT encoder + ColBERT multi-vector pooler.

Backbone: models.modernbert (custom encoder with fused Triton kernels)
Pooler:   ColBERT projection (nn.Linear → L2 norm), token-level
Weights:  Main encoder from HF checkpoint + projection from 1_Dense/model.safetensors

vLLM 0.15.x compatible:
- @attn_type("encoder_only") for bidirectional attention (no KV cache)
- @default_pooling_type(tok_pooling_type="ALL") for token-level embeddings
- embed_input_ids() for the pooling runner
"""

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

# Load the shared ModernBERT encoder by absolute file path.
# This avoids sys.path manipulation and works in all contexts:
# editable .pth install, vLLM worker subprocess, direct script execution.
# Mirrors the superpod colbert_plugin self-contained encoder approach.
_ENCODER_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "modernbert" / "modernbert_encoder.py"
)


def _import_modernbert_encoder():
    spec = importlib.util.spec_from_file_location("modernbert_encoder", str(_ENCODER_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("modernbert_encoder", mod)
    spec.loader.exec_module(mod)
    return mod


_modernbert_encoder_mod = _import_modernbert_encoder()
ModernBertModel = _modernbert_encoder_mod.ModernBertModel

from vllm.config import VllmConfig  # noqa: E402
from vllm.model_executor.model_loader.weight_utils import default_weight_loader  # noqa: E402
from vllm.model_executor.models.interfaces_base import attn_type, default_pooling_type  # noqa: E402
from vllm.sequence import IntermediateTensors  # noqa: E402

from vllm_factory.pooling.protocol import PassthroughPooler  # noqa: E402
from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter  # noqa: E402

from .config import ModernColBERTConfig  # noqa: E402

logger = logging.getLogger(__name__)


_LEGACY_PROJECTION_MARKERS = (
    "colbert_linear",
    "projection",
    "projector",
    "1_Dense",
    "2_Dense",
    "3_Dense",
)


class _ResidualDense(nn.Module):
    """PyLate Dense projection with optional residual branch."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_residual = bool(use_residual)
        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.residual = (
            nn.Linear(self.in_features, self.out_features, bias=False)
            if self.use_residual and self.in_features != self.out_features
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.linear(hidden_states)
        if self.use_residual:
            residual = hidden_states if self.residual is None else self.residual(hidden_states)
            projected = projected + residual
        return projected


@attn_type("encoder_only")
@default_pooling_type(tok_pooling_type="ALL")
class ModernBertForColBERT(nn.Module):
    """Custom ModernBERT encoder + ColBERT projection + L2 normalization.

    Architecture:
        input_ids → ModernBertModel → (total_tokens, hidden_size)
                  → colbert_linear  → (total_tokens, colbert_dim)
                  → L2 normalize    → (total_tokens, colbert_dim)
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config

        # Ensure ColBERT parameters exist on config
        if not isinstance(config, ModernColBERTConfig):
            config.colbert_dim = getattr(config, "colbert_dim", 128)
            config.query_length = getattr(config, "query_length", 256)
            config.document_length = getattr(config, "document_length", 8192)

        self.config = config
        self.model_path = vllm_config.model_config.model
        self.colbert_dim = getattr(config, "colbert_dim", 128)
        self.hidden_size = config.hidden_size
        self._dense_specs = self._load_dense_specs()

        # 1. Backbone — custom ModernBERT with Triton kernels
        self.model = ModernBertModel(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )

        # 2. PyLate projection stack. LateOn ships three Dense modules:
        #    768 -> 1536 (+ residual), 1536 -> 768 (+ residual), 768 -> 128.
        #    Loading only the final projection is not parity-preserving.
        self.projection_layers = nn.ModuleList()
        if self._dense_specs:
            for spec in self._dense_specs:
                layer = _ResidualDense(
                    int(spec["in_features"]),
                    int(spec["out_features"]),
                    bias=bool(spec.get("bias", False)),
                    use_residual=bool(spec.get("use_residual", False)),
                )
                self.projection_layers.append(layer)
            self.colbert_dim = int(self._dense_specs[-1]["out_features"])
            config.colbert_dim = self.colbert_dim
        else:
            self.projection_layers.append(
                _ResidualDense(config.hidden_size, self.colbert_dim, bias=False)
            )

        # 3. vLLM pooler for token-level embedding task
        self.pooler = VllmPoolerAdapter(
            PassthroughPooler(),
            pooler_config=vllm_config.model_config.pooler_config,
        )

        self._projection_loaded = False

    def _resolve_hf_file(self, filename: str) -> Path | None:
        local_file = Path(self.model_path) / filename
        if local_file.exists():
            return local_file
        try:
            from huggingface_hub import hf_hub_download

            return Path(
                hf_hub_download(
                    repo_id=str(self.model_path),
                    filename=filename,
                    token=os.environ.get("HF_TOKEN"),
                )
            )
        except Exception as exc:
            logger.debug("Could not resolve %s from %s: %s", filename, self.model_path, exc)
            return None

    def _load_dense_specs(self) -> list[dict]:
        modules_file = self._resolve_hf_file("modules.json")
        if modules_file is None:
            return []
        try:
            modules = json.loads(modules_file.read_text())
        except Exception as exc:
            logger.warning("Could not read ColBERT modules.json: %s", exc)
            return []

        specs: list[dict] = []
        for module in modules:
            path = str(module.get("path") or "")
            module_type = str(module.get("type") or "")
            if not path.endswith("_Dense") and "Dense" not in module_type:
                continue
            config_file = self._resolve_hf_file(f"{path}/config.json")
            if config_file is None:
                continue
            try:
                spec = json.loads(config_file.read_text())
            except Exception as exc:
                logger.warning("Could not read ColBERT Dense config %s: %s", path, exc)
                continue
            spec["path"] = path
            specs.append(spec)
        return specs

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Required by vLLM pooling runner for embeddings lookup."""
        return self.model.embeddings.tok_embeddings(input_ids)

    def sample(self, logits: torch.Tensor, sampling_metadata):
        try:
            from vllm.sequence import SamplerOutput

            return SamplerOutput(outputs=[])
        except ImportError:
            return None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode → project → L2 normalize.

        Returns: (total_tokens, colbert_dim) tensor of unit-length embeddings.
        """
        if not self._projection_loaded:
            self._ensure_projection_loaded()

        # vLLM sends 1D tensors; encoder expects 2D
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if positions is not None and positions.dim() == 1:
            positions = positions.unsqueeze(0)

        output = self.model(
            input_ids=input_ids,
            position_ids=positions,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = (
            output.last_hidden_state
            if hasattr(output, "last_hidden_state")
            else output
            if isinstance(output, torch.Tensor)
            else output[0]
        )

        # Flatten to (total_tokens, hidden_size) for token-level pooling
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Project through the complete PyLate Dense stack + L2 normalize.
        projected = hidden_states
        for layer in self.projection_layers:
            if next(layer.parameters()).dtype != projected.dtype:
                layer.to(projected.dtype)
            projected = layer(projected)
        projected = projected / projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return projected

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        """Load encoder weights + ColBERT projection from 1_Dense/model.safetensors."""
        weights_list = list(weights)
        loaded = set()
        params = dict(self.named_parameters())

        model_weights = []
        projection_weights = []

        for name, param in weights_list:
            # SentenceTransformers Dense modules live in subdirectories and are
            # loaded explicitly below. Legacy/custom ColBERT checkpoints may
            # still stream a single projection weight; keep those candidates so
            # one-layer models remain compatible.
            if any(k in name for k in (*_LEGACY_PROJECTION_MARKERS, "projection_layers")):
                projection_weights.append((name, param))
            else:
                model_weights.append((name, param))

        # Encoder weights
        for name, loaded_weight in model_weights:
            mapped = name
            if not name.startswith("model."):
                mapped = "model." + name
            if name.startswith("encoder."):
                mapped = "model." + name[8:]

            if mapped in params:
                weight_loader = getattr(params[mapped], "weight_loader", default_weight_loader)
                weight_loader(params[mapped], loaded_weight)
                loaded.add(mapped)

        if self._load_projection_from_file():
            loaded.update(name for name in params if name.startswith("projection_layers."))
        elif self._load_projection_from_stream(projection_weights):
            loaded.update(name for name in params if name.startswith("projection_layers."))

        # Mark constructor-initialized params as loaded for vLLM 0.19+ validation
        for name in dict(self.named_parameters()):
            loaded.add(name)
        return loaded

    def _ensure_projection_loaded(self):
        """Safety net: load projection on first forward if vLLM skipped it."""
        if not self._projection_loaded:
            self._load_projection_from_file()

    def _load_projection_from_file(self) -> bool:
        """Load the full PyLate Dense stack from *_Dense/model.safetensors.

        Returns True if the weight was loaded successfully, False otherwise.
        """
        from safetensors import safe_open

        if not self._dense_specs:
            self._dense_specs = [{"path": "1_Dense"}]

        loaded_ok = True
        for idx, layer in enumerate(self.projection_layers):
            path = str(self._dense_specs[idx].get("path", f"{idx + 1}_Dense"))
            dense_file = self._resolve_hf_file(f"{path}/model.safetensors")
            if dense_file is None or not dense_file.exists():
                logger.warning("%s/model.safetensors not found for ColBERT projection", path)
                loaded_ok = False
                continue

            logger.info("Loading ColBERT projection layer %s from %s", idx, dense_file)
            with safe_open(str(dense_file), framework="pt", device="cpu") as f:
                if "linear.weight" in f.keys():
                    tensor = f.get_tensor("linear.weight")
                    if layer.linear.weight.shape == tensor.shape:
                        layer.linear.weight.data.copy_(tensor)
                    else:
                        logger.warning(
                            "Shape mismatch for %s linear.weight: expected %s got %s",
                            path,
                            tuple(layer.linear.weight.shape),
                            tuple(tensor.shape),
                        )
                        loaded_ok = False
                if layer.residual is not None and "residual.weight" in f.keys():
                    tensor = f.get_tensor("residual.weight")
                    if layer.residual.weight.shape == tensor.shape:
                        layer.residual.weight.data.copy_(tensor)
                    else:
                        logger.warning(
                            "Shape mismatch for %s residual.weight: expected %s got %s",
                            path,
                            tuple(layer.residual.weight.shape),
                            tuple(tensor.shape),
                        )
                        loaded_ok = False

        self._projection_loaded = loaded_ok
        return loaded_ok

    def _load_projection_from_stream(
        self,
        projection_weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> bool:
        """Load legacy one-layer projection weights from the vLLM stream."""

        loaded_ok = False
        unmatched = list(projection_weights)
        for layer_idx, layer in enumerate(self.projection_layers):
            for attr_name in ("linear", "residual"):
                module = getattr(layer, attr_name, None)
                if module is None:
                    continue
                target = module.weight
                for weight_idx, (name, tensor) in list(enumerate(unmatched)):
                    if tensor.dim() != 2 or target.shape != tensor.shape:
                        continue
                    target.data.copy_(tensor)
                    unmatched.pop(weight_idx)
                    loaded_ok = True
                    logger.info(
                        "Loaded ColBERT projection layer %s %s.weight from checkpoint stream key %s",
                        layer_idx,
                        attr_name,
                        name,
                    )
                    break

        self._projection_loaded = loaded_ok
        return loaded_ok
