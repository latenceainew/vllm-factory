"""Unit tests for per-field threshold plumbing in GLiNER2Pooler.

CPU-only, no vLLM runtime. Stubs out vllm and vllm_factory so we can
import the pooler module, then tests decode methods directly with fake
span_scores tensors.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

# Stub out vllm and vllm_factory before importing the pooler.
_STUBS = {}
for pkg_name in [
    "vllm",
    "vllm.config",
    "vllm.distributed",
    "vllm_factory",
    "vllm_factory.pooling",
    "vllm_factory.pooling.protocol",
    "vllm_factory.pooling.vllm_adapter",
]:
    if pkg_name not in sys.modules:
        mod = types.ModuleType(pkg_name)
        mod.__path__ = []
        mod.__package__ = pkg_name.rsplit(".", 1)[0] if "." in pkg_name else pkg_name
        _STUBS[pkg_name] = mod
        sys.modules[pkg_name] = mod

# Provide stub classes expected by imports — but only on modules we stubbed
# ourselves. When the real module is already loaded (CI installs vllm), it
# provides the real classes and must not be monkeypatched.
if "vllm.config" in _STUBS:
    _STUBS["vllm.config"].PoolerConfig = type("PoolerConfig", (), {})
if "vllm_factory.pooling.protocol" in _STUBS:
    _STUBS["vllm_factory.pooling.protocol"].PoolerContext = type("PoolerContext", (), {})
    _STUBS["vllm_factory.pooling.protocol"].split_hidden_states = lambda *a, **kw: None
if "vllm_factory.pooling.vllm_adapter" in _STUBS:
    _STUBS["vllm_factory.pooling.vllm_adapter"].VllmPoolerAdapter = type(
        "VllmPoolerAdapter", (), {}
    )

# Now import the pooler
_POOLER_PATH = Path(__file__).resolve().parents[2] / "poolers" / "gliner2.py"
_spec = importlib.util.spec_from_file_location("gliner2_pooler_test", str(_POOLER_PATH))
_pooler_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pooler_mod)

GLiNER2Pooler = _pooler_mod.GLiNER2Pooler

# Remove the stubs again: the pooler module above holds its own references,
# and leaving e.g. the empty-__path__ "vllm_factory.pooling" stub in
# sys.modules poisons every later real import of vllm_factory.pooling.* in
# this pytest process (CI-only ModuleNotFoundError for
# vllm_factory.pooling.shape_prefix — locally these later tests skip because
# vllm isn't installed, so the pollution went unnoticed).
for _name, _mod in _STUBS.items():
    if sys.modules.get(_name) is _mod:
        del sys.modules[_name]


@pytest.fixture
def pooler():
    """Create a GLiNER2Pooler instance without __init__ (decode methods are stateless)."""
    p = GLiNER2Pooler.__new__(GLiNER2Pooler)
    return p


def _make_span_scores(n_instances, n_fields, text_len, max_width=12):
    """Create fake span_scores: (n_instances, n_fields, text_len, max_width).

    This matches predict_spans output shape from the GLiNER2 pooler.
    _decode_entities does ``span_scores[0, :, -text_len:]`` which on shape
    (1, n_fields, text_len, max_width) yields (n_fields, text_len, max_width).
    Then scores[idx] → (text_len, max_width), fed to _find_spans which uses
    torch.where to get (start_positions, width_indices).
    """
    return torch.zeros(n_instances, n_fields, text_len, max_width)


class TestDecodeEntitiesThresholds:
    def test_global_threshold_filters(self, pooler):
        """Spans below global threshold are excluded."""
        text_len = 2
        scores = _make_span_scores(1, 2, text_len)
        # person: span at position 0, width 0 → single token "alice"
        scores[0, 0, 0, 0] = 0.8
        # org: span at position 1, width 0 → single token "works."
        scores[0, 1, 1, 0] = 0.3

        text = "alice works."
        text_tokens = ["alice", "works."]
        start_map = [0, 6]
        end_map = [5, 12]

        result = pooler._decode_entities(
            scores,
            ["person", "org"],
            text_len,
            text_tokens,
            text,
            start_map,
            end_map,
            0.5,
        )
        assert len(result["entities"]["person"]) == 1
        assert len(result["entities"]["org"]) == 0

    def test_per_field_threshold_overrides(self, pooler):
        """Per-field threshold allows lower-confidence span through."""
        text_len = 2
        scores = _make_span_scores(1, 2, text_len)
        scores[0, 0, 0, 0] = 0.8
        scores[0, 1, 1, 0] = 0.3  # below global 0.5 but above per-field 0.2

        text = "alice works."
        text_tokens = ["alice", "works."]
        start_map = [0, 6]
        end_map = [5, 12]

        result = pooler._decode_entities(
            scores,
            ["person", "org"],
            text_len,
            text_tokens,
            text,
            start_map,
            end_map,
            0.5,
            per_field_thresholds=[0.5, 0.2],
        )
        assert len(result["entities"]["person"]) == 1
        assert len(result["entities"]["org"]) == 1

    def test_per_field_threshold_excludes(self, pooler):
        """Per-field threshold can be stricter than global."""
        text_len = 2
        scores = _make_span_scores(1, 2, text_len)
        scores[0, 0, 0, 0] = 0.6  # above global 0.5 but below per-field 0.9
        scores[0, 1, 1, 0] = 0.8

        text = "alice works."
        text_tokens = ["alice", "works."]
        start_map = [0, 6]
        end_map = [5, 12]

        result = pooler._decode_entities(
            scores,
            ["person", "org"],
            text_len,
            text_tokens,
            text,
            start_map,
            end_map,
            0.5,
            per_field_thresholds=[0.9, 0.5],
        )
        assert len(result["entities"]["person"]) == 0
        assert len(result["entities"]["org"]) == 1

    def test_none_per_field_uses_global(self, pooler):
        """When per_field_thresholds is None, global threshold is used."""
        text_len = 2
        scores = _make_span_scores(1, 1, text_len)
        scores[0, 0, 0, 0] = 0.6

        text = "alice works."
        text_tokens = ["alice", "works."]
        start_map = [0, 6]
        end_map = [5, 12]

        result = pooler._decode_entities(
            scores,
            ["person"],
            text_len,
            text_tokens,
            text,
            start_map,
            end_map,
            0.5,
            per_field_thresholds=None,
        )
        assert len(result["entities"]["person"]) == 1


class TestDecodeStructuresThresholds:
    def test_per_field_structure_thresholds(self, pooler):
        """Structure fields use per-field thresholds."""
        text_len = 3
        scores = _make_span_scores(1, 2, text_len)
        scores[0, 0, 0, 0] = 0.6  # date at pos 0
        scores[0, 1, 1, 0] = 0.4  # memo at pos 1 — below 0.5 but above 0.2

        text = "jan 15 notes."
        text_tokens = ["jan", "15", "notes."]
        start_map = [0, 4, 7]
        end_map = [3, 6, 13]

        result = pooler._decode_structures(
            scores,
            1,
            ["date", "memo"],
            text_len,
            text_tokens,
            text,
            start_map,
            end_map,
            0.5,
            "invoice",
            {},
            per_field_thresholds=[0.3, 0.2],
        )
        assert result["type"] == "json_structures"
        assert len(result["instances"]) == 1
        inst = result["instances"][0]
        assert inst["date"] is not None
        assert inst["memo"] is not None

    def test_global_fallback_for_structures(self, pooler):
        """Without per_field_thresholds, global threshold is used."""
        text_len = 2
        scores = _make_span_scores(1, 1, text_len)
        scores[0, 0, 0, 0] = 0.4  # below 0.5

        text = "jan 15."
        text_tokens = ["jan", "15."]
        start_map = [0, 4]
        end_map = [3, 7]

        result = pooler._decode_structures(
            scores,
            1,
            ["date"],
            text_len,
            text_tokens,
            text,
            start_map,
            end_map,
            0.5,
            "invoice",
            {},
            per_field_thresholds=None,
        )
        assert result["instances"] == []
