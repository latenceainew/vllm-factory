"""Boundary decode formats through gliner2, not a four-key Pioneer grouping."""

import json
from pathlib import Path

from gliner2.inference.runtime import format_results

from plugins.deberta_gliner25.processor import schema_format_args


def test_format_results_merges_entity_list_of_dicts():
    decoded = {
        "entities": [
            {"person": [{"text": "Ada", "start": 0, "end": 3, "confidence": 0.9}]}
        ]
    }
    out = format_results(decoded, include_confidence=True)
    assert out["entities"]["person"][0]["text"] == "Ada"
    assert "classifications" not in out
    assert "structures" not in out
    assert "relations" not in out


def test_schema_format_args_reads_tasks_and_relations():
    rels, tasks = schema_format_args(
        {
            "classifications": [{"task": "topic", "labels": ["a"]}],
            "relations": {"works_at": "", "located_in": ""},
        }
    )
    assert tasks == ["topic"]
    assert rels == ["works_at", "located_in"]


def test_format_results_accepts_json_decoded_classification_pair():
    out = format_results(
        {"topic": ["finance", 0.79]},
        include_confidence=True,
        classification_tasks=["topic"],
    )
    assert out["topic"] == {"label": "finance", "confidence": 0.79}


def test_format_results_lifts_classification_from_schema_tasks():
    decoded = {
        "entities": {},
        "topic": ("technology", 0.8),
    }
    out = format_results(
        decoded,
        include_confidence=True,
        classification_tasks=["topic"],
    )
    assert out["topic"] == {"label": "technology", "confidence": 0.8}


def test_saved_pooling_payload_formats_to_entity_dict():
    fixture = Path(__file__).parent / "fixtures" / "raw_pooling.json"
    decoded = json.loads(fixture.read_text())
    assert isinstance(decoded["entities"], list)
    out = format_results(decoded, include_confidence=True)
    assert isinstance(out["entities"], dict)
    assert out["entities"]["person"][0]["text"] == "John Smith"
    assert "classifications" not in out or out["classifications"] == {}
    decoded = {
        "entities": {},
        "topic": ("technology", 0.8),
    }
    out = format_results(
        decoded,
        include_confidence=True,
        classification_tasks=["topic"],
    )
    assert out["topic"] == {"label": "technology", "confidence": 0.8}
