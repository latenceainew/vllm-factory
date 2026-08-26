"""Construct the boundary pooler against local gliner2.layers.

The pooler is loaded by file path (see the ``pooler_mod`` fixture) so
poolers/__init__.py never imports ColBERT.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

# Pick up a source checkout when gliner2 is not installed: $GLINER2_SRC, else a
# sibling clone of this repo.
_GLINER2_ROOT = Path(
    os.environ.get("GLINER2_SRC") or Path(__file__).resolve().parents[3] / "GLiNER2"
)
if _GLINER2_ROOT.is_dir() and str(_GLINER2_ROOT) not in sys.path:
    sys.path.insert(0, str(_GLINER2_ROOT))

pytest.importorskip("gliner2", reason="boundary pooler construct needs gliner2")


def test_boundary_pooler_constructs_and_exposes_checkpoint_prefixes(
    pooler_mod: ModuleType,
):
    pooler = pooler_mod.GLiNER25BoundaryPooler(hidden_size=32, boundary_head={"dropout": 0.0})
    keys = pooler.state_dict().keys()
    assert any(k.startswith("classifier.") for k in keys)
    assert any(k.startswith("boundary_head.") for k in keys)
    if pooler.enable_records:
        assert any(k.startswith("record_decoder.") for k in keys)
    if pooler.enable_relations:
        assert any(k.startswith("relation_scorer.") for k in keys)
    assert pooler.get_tasks() == {"embed", "classify", "plugin"}


def test_decode_host_borrows_modules_without_module_init(pooler_mod: ModuleType) -> None:
    """Warmup used to crash: assigning nn.Modules onto object.__new__ host."""
    pooler = pooler_mod.GLiNER25BoundaryPooler(hidden_size=32, boundary_head={"dropout": 0.0})
    host = pooler._decode_host({"text_states": None})
    assert host.boundary_head is pooler.boundary_head
    assert host.classifier is pooler.classifier
    assert host.strict_extraction is True
    assert host._encode_core({})["text_states"] is None
