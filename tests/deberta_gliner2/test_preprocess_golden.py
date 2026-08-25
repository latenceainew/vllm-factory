"""Golden-output contract for the serving preprocess path.

HTTP schemas go through ``normalize_gliner2_schema`` then ``preprocess``,
matching ``io_processor.factory_parse``. The SchemaTransformer-backed
tokenizer half must keep these fields stable. Awkward cases (per-field
thresholds, overflow truncation, empty/malformed schemas, mixed tasks) are
included so a silent behaviour change cannot hide behind the happy path.
"""

from __future__ import annotations

import pytest

from plugins.deberta_gliner2.processor import normalize_gliner2_schema, preprocess


class _DummyTokenizer:
    """Deterministic tokenizer: each word is one token, id = len(word)."""

    def tokenize(self, token: str) -> list[str]:
        return [token] if token else []

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [len(t) for t in tokens]


def _ids_and_tasks(text, schema, **kwargs):
    normalized = normalize_gliner2_schema(schema)
    out = preprocess(_DummyTokenizer(), text, normalized, **kwargs)
    return {
        "input_ids": out["input_ids"],
        "task_types": out["task_types"],
        "text_tokens": out["text_tokens"],
        "schema_count": out["schema_count"],
        "threshold_meta": out["threshold_meta"],
        "start_mapping": out["start_mapping"],
        "end_mapping": out["end_mapping"],
    }


def test_preprocess_entities_classifications_relations_structures():
    schema = {
        "entities": {"person": "People", "org": ""},
        "classifications": [
            {"task": "sentiment", "labels": ["pos", "neg"], "cls_threshold": 0.6}
        ],
        "relations": {"works_at": {"description": "Employment", "threshold": 0.25}},
        "structures": {
            "invoice": {
                "fields": [
                    {"name": "date", "dtype": "str", "threshold": 0.8},
                    {"name": "memo"},
                ]
            }
        },
    }
    got = _ids_and_tasks("Alice joined Acme Corp in 2020", schema)
    assert got["schema_count"] == 4
    assert "entities" in got["task_types"]
    assert got["threshold_meta"]["relations"]["works_at"] == 0.25
    assert got["text_tokens"]
    assert got["input_ids"]
    assert got["start_mapping"]
    assert got["end_mapping"][-1] <= len("Alice joined Acme Corp in 2020.") + 1


def test_preprocess_per_field_thresholds_and_truncation():
    schema = {"entities": {"person": {"description": "x", "threshold": 0.9}}}
    got = _ids_and_tasks(
        "one two three four five six seven eight nine ten",
        schema,
        max_model_len=20,
        truncate_overflow_text=True,
    )
    assert got["threshold_meta"]["entities"]["person"] == 0.9
    assert len(got["text_tokens"]) < 10 or len(got["input_ids"]) <= 20


def test_preprocess_empty_schema_raises():
    with pytest.raises(ValueError, match="at least one task"):
        normalize_gliner2_schema({})


def test_preprocess_malformed_classification_raises():
    with pytest.raises(ValueError):
        normalize_gliner2_schema({"classifications": [{"task": "x", "labels": []}]})
