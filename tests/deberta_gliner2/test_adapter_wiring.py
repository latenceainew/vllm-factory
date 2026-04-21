"""Tests for the per-request LoRA adapter wiring on deberta_gliner2_io.

Covers:

* ``GLiNER2Input.adapter`` parse / validation contract in ``factory_parse``.
* Round-tripping of the adapter into the post-process meta and observability
  payload (so ``/infer`` shims and bench harnesses can verify which adapter
  each request targeted).
* The ``build_lora_requests`` helper that offline callers feed into
  ``LLM.encode(..., lora_request=...)``.

No vLLM forward pass is exercised here — cross-request LoRA batching is a
vLLM scheduler/Punica concern and is covered by vLLM's own test suite.  The
plugin's only responsibility is to parse + surface ``adapter`` so that each
``LoRARequest`` attached to ``add_request(...)`` is well-formed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm", reason="adapter wiring tests require vLLM imports")

try:
    from vllm.config import VllmConfig  # noqa: F401
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"vLLM importable but config unavailable ({exc!r})",
        allow_module_level=True,
    )


@pytest.fixture()
def parse_data():
    """Return ``DeBERTaGLiNER2IOProcessor.factory_parse`` bound to a
    tokenizer-free instance.

    The tokenizer is only used by ``factory_pre_process``; ``factory_parse``
    is a pure validation path, so we deliberately bypass ``__init__`` to
    avoid pulling a real HF model checkpoint into CPU-only CI.
    """
    from plugins.deberta_gliner2.io_processor import DeBERTaGLiNER2IOProcessor

    processor = DeBERTaGLiNER2IOProcessor.__new__(DeBERTaGLiNER2IOProcessor)
    return processor.factory_parse


def _base_payload(**overrides):
    payload = {
        "text": "Apple released iPhone 15 in Cupertino.",
        "labels": ["company", "product", "location"],
    }
    payload.update(overrides)
    return payload


def test_adapter_absent_parses_as_none(parse_data):
    parsed = parse_data(_base_payload())
    assert parsed.adapter is None


def test_adapter_explicit_none_parses_as_none(parse_data):
    parsed = parse_data(_base_payload(adapter=None))
    assert parsed.adapter is None


def test_adapter_empty_string_parses_as_none(parse_data):
    # JSON producers routinely emit ``""`` for optional string fields; treat
    # it as "base model" to avoid surprising routing behaviour.
    parsed = parse_data(_base_payload(adapter="   "))
    assert parsed.adapter is None


@pytest.mark.parametrize(
    "adapter",
    [
        "adapter_0",
        "sql-lora",
        "team/adapter:v1",
        "org.adapter-name_42",
        "a" * 128,
    ],
)
def test_adapter_accepts_well_formed_names(parse_data, adapter):
    parsed = parse_data(_base_payload(adapter=adapter))
    assert parsed.adapter == adapter


@pytest.mark.parametrize(
    "adapter",
    [
        "with space",
        "with\nnewline",
        "tab\there",
        "unicode\u00e9",
        "a" * 129,
        "semi;colon",
        "pipe|char",
        "back\\slash",
    ],
)
def test_adapter_rejects_malformed_names(parse_data, adapter):
    with pytest.raises(ValueError, match="adapter"):
        parse_data(_base_payload(adapter=adapter))


@pytest.mark.parametrize("adapter", [123, 1.0, [], {}, True])
def test_adapter_rejects_non_string_types(parse_data, adapter):
    with pytest.raises(ValueError, match="must be a string"):
        parse_data(_base_payload(adapter=adapter))


def test_build_lora_request_none_returns_none():
    from plugins.deberta_gliner2.lora import build_lora_request

    assert build_lora_request(None, {"a": (1, "/tmp/a")}) is None


def test_build_lora_request_resolves_known_adapter():
    from vllm.lora.request import LoRARequest

    from plugins.deberta_gliner2.lora import build_lora_request

    req = build_lora_request("sql-lora", {"sql-lora": (7, "/tmp/sql")})
    assert isinstance(req, LoRARequest)
    assert req.lora_name == "sql-lora"
    assert req.lora_int_id == 7
    assert req.lora_path == "/tmp/sql"


def test_build_lora_request_unknown_raises():
    from plugins.deberta_gliner2.lora import build_lora_request

    with pytest.raises(KeyError, match="sql-lora"):
        build_lora_request("sql-lora", {})


def test_build_lora_request_unknown_with_allow_unknown_returns_none():
    from plugins.deberta_gliner2.lora import build_lora_request

    assert build_lora_request("sql-lora", {}, allow_unknown=True) is None


def test_build_lora_request_rejects_reserved_int_id():
    # vLLM's LoRA manager treats ``lora_int_id == 0`` as the base model;
    # handing out id 0 would silently mask an adapter.  Reject early.
    from plugins.deberta_gliner2.lora import build_lora_request

    with pytest.raises(ValueError, match="non-positive"):
        build_lora_request("bad", {"bad": (0, "/tmp/bad")})


def test_build_lora_requests_preserves_order_and_nulls():
    from vllm.lora.request import LoRARequest

    from plugins.deberta_gliner2.io_processor import GLiNER2Input
    from plugins.deberta_gliner2.lora import build_lora_requests

    inputs = [
        GLiNER2Input(text="a", schema={"entities": {"x": ""}}, adapter="adapter_a"),
        GLiNER2Input(text="b", schema={"entities": {"x": ""}}, adapter=None),
        GLiNER2Input(text="c", schema={"entities": {"x": ""}}, adapter="adapter_b"),
    ]
    registry = {"adapter_a": (1, "/tmp/a"), "adapter_b": (2, "/tmp/b")}

    requests = build_lora_requests(inputs, registry)

    assert len(requests) == 3
    assert isinstance(requests[0], LoRARequest) and requests[0].lora_name == "adapter_a"
    assert requests[1] is None
    assert isinstance(requests[2], LoRARequest) and requests[2].lora_name == "adapter_b"
    # Distinct int_ids are a hard requirement — the scheduler uses them as
    # the LoRA identity key when counting toward ``max_loras``.
    assert requests[0].lora_int_id != requests[2].lora_int_id
