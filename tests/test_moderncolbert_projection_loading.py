from __future__ import annotations

import pytest
import torch
import torch.nn as nn

try:
    from plugins.moderncolbert.model import ModernBertForColBERT, _ResidualDense
except ImportError as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"ModernColBERT projection tests require importable vLLM plugin stack ({exc!r})",
        allow_module_level=True,
    )


def _projection_only_model(*layers: _ResidualDense) -> ModernBertForColBERT:
    model = object.__new__(ModernBertForColBERT)
    nn.Module.__init__(model)
    model.projection_layers = nn.ModuleList(layers)
    model._projection_loaded = False
    return model


def test_load_projection_from_stream_supports_legacy_single_layer_weight() -> None:
    model = _projection_only_model(_ResidualDense(4, 2, bias=False))
    weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    assert model._load_projection_from_stream([("colbert_linear.weight", weight)])
    assert torch.equal(model.projection_layers[0].linear.weight, weight)
    assert model._projection_loaded is True


def test_load_projection_from_stream_supports_residual_dense_stack() -> None:
    model = _projection_only_model(
        _ResidualDense(4, 6, bias=False, use_residual=True),
        _ResidualDense(6, 2, bias=False),
    )
    first_linear = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    first_residual = torch.arange(24, 48, dtype=torch.float32).reshape(6, 4)
    second_linear = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    assert model._load_projection_from_stream(
        [
            ("1_Dense.linear.weight", first_linear),
            ("1_Dense.residual.weight", first_residual),
            ("2_Dense.linear.weight", second_linear),
        ]
    )

    assert torch.equal(model.projection_layers[0].linear.weight, first_linear)
    assert torch.equal(model.projection_layers[0].residual.weight, first_residual)
    assert torch.equal(model.projection_layers[1].linear.weight, second_linear)
    assert model._projection_loaded is True
