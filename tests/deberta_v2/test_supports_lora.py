"""SupportsLoRA protocol-compliance tests for the DeBERTa v2/v3 encoder.

Requires vLLM to be importable (CPU-only environments skip these). On a
supported image they verify:

1. ``DebertaV2EncoderModel`` satisfies vLLM's ``SupportsLoRA`` protocol.
2. The required class attributes are present and well-formed.
3. The module-level mapping constants are the same objects as the class
   attributes (so the plugin layer can re-export them consistently).
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm", reason="SupportsLoRA tests require vLLM")

try:
    from vllm.model_executor.models.interfaces import supports_lora  # noqa: F401
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"vLLM importable but interfaces unavailable ({exc!r})",
        allow_module_level=True,
    )


def test_deberta_v2_encoder_model_satisfies_supports_lora_protocol():
    from vllm.model_executor.models.interfaces import supports_lora

    from models.deberta_v2.deberta_v2_encoder import DebertaV2EncoderModel

    assert supports_lora(DebertaV2EncoderModel), (
        "DebertaV2EncoderModel must satisfy vllm.SupportsLoRA — check that "
        "supports_lora / packed_modules_mapping / embedding_modules are "
        "declared as class attributes."
    )
    assert DebertaV2EncoderModel.supports_lora is True


def test_deberta_v2_encoder_lora_attrs_are_well_formed():
    from models.deberta_v2.deberta_v2_encoder import DebertaV2EncoderModel

    packed = DebertaV2EncoderModel.packed_modules_mapping
    assert isinstance(packed, dict)
    for k, v in packed.items():
        assert isinstance(k, str) and isinstance(v, list)
        assert all(isinstance(n, str) for n in v)

    emb = DebertaV2EncoderModel.embedding_modules
    assert isinstance(emb, dict)
    for k, v in emb.items():
        assert isinstance(k, str) and isinstance(v, str)


def test_deberta_v2_encoder_module_constants_exposed():
    from models.deberta_v2.deberta_v2_encoder import (
        EMBEDDING_MODULES,
        PACKED_MODULES_MAPPING,
        DebertaV2EncoderModel,
    )

    assert DebertaV2EncoderModel.packed_modules_mapping is PACKED_MODULES_MAPPING
    assert DebertaV2EncoderModel.embedding_modules is EMBEDDING_MODULES
