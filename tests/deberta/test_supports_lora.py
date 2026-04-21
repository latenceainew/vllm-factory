"""SupportsLoRA protocol-compliance tests for the DeBERTa v1 encoder.

These tests require vLLM to be importable (the encoder pulls in
``vllm.config`` / ``vllm.model_executor``), so they skip gracefully when
running in a CPU-only environment without vLLM. On a GPU image with vLLM
installed they verify:

1. ``DebertaEncoderModel`` satisfies vLLM's ``SupportsLoRA`` protocol via
   ``vllm.model_executor.models.interfaces.supports_lora``.
2. The class exposes the required ``packed_modules_mapping`` and
   ``embedding_modules`` attributes.
3. The module-level constants ``PACKED_MODULES_MAPPING`` /
   ``EMBEDDING_MODULES`` are well-formed (string keys, consistent value
   shapes) so the plugin layer can safely re-export them with prefixing.
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


def test_deberta_encoder_model_satisfies_supports_lora_protocol():
    from vllm.model_executor.models.interfaces import supports_lora

    from models.deberta.deberta_encoder import DebertaEncoderModel

    assert supports_lora(DebertaEncoderModel), (
        "DebertaEncoderModel must satisfy vllm.SupportsLoRA — check that "
        "supports_lora / packed_modules_mapping / embedding_modules are "
        "declared as class attributes."
    )
    assert DebertaEncoderModel.supports_lora is True


def test_deberta_encoder_lora_attrs_are_well_formed():
    from models.deberta.deberta_encoder import DebertaEncoderModel

    packed = DebertaEncoderModel.packed_modules_mapping
    assert isinstance(packed, dict)
    for k, v in packed.items():
        assert isinstance(k, str) and isinstance(v, list)
        assert all(isinstance(n, str) for n in v)

    emb = DebertaEncoderModel.embedding_modules
    assert isinstance(emb, dict)
    for k, v in emb.items():
        assert isinstance(k, str) and isinstance(v, str)


def test_deberta_encoder_module_constants_exposed():
    from models.deberta.deberta_encoder import (
        EMBEDDING_MODULES,
        PACKED_MODULES_MAPPING,
        DebertaEncoderModel,
    )

    # Class constants must be the same objects the module exports so the
    # plugin layer can re-export them with an ``encoder.`` prefix without
    # drifting out of sync.
    assert DebertaEncoderModel.packed_modules_mapping is PACKED_MODULES_MAPPING
    assert DebertaEncoderModel.embedding_modules is EMBEDDING_MODULES
