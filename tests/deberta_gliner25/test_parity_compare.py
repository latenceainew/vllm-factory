"""The GPU parity check must actually compare spans, not just count them."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "deberta_gliner25"
    / "parity_test.py"
)
_spec = importlib.util.spec_from_file_location("gliner25_parity_test", str(_PATH))
parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parity)

compare_outputs = parity.compare_outputs


def _result(person: list[dict], topic: str = "technology") -> dict:
    return {
        "entities": {"person": person},
        "topic": {"label": topic, "confidence": 0.91},
    }


_JOHN = {"text": "John Smith", "confidence": 0.93, "start": 0, "end": 10}


def test_identical_results_are_parity():
    assert compare_outputs(_result([_JOHN]), _result([_JOHN])) == []


def test_different_span_text_is_a_mismatch():
    other = {**_JOHN, "text": "John"}

    problems = compare_outputs(_result([_JOHN]), _result([other]))

    assert len(problems) == 1
    assert "entities[person]" in problems[0]


def test_shifted_offsets_are_a_mismatch():
    shifted = {**_JOHN, "start": 4, "end": 14}

    assert compare_outputs(_result([_JOHN]), _result([shifted]))


def test_extra_entity_is_a_mismatch():
    extra = {"text": "Jensen Huang", "confidence": 0.88, "start": 20, "end": 32}

    assert compare_outputs(_result([_JOHN]), _result([_JOHN, extra]))


def test_dropped_entity_type_is_a_mismatch():
    problems = compare_outputs(_result([_JOHN]), {"topic": {"label": "technology"}})

    assert any("entities[person]" in problem for problem in problems)


def test_any_non_empty_extraction_is_not_enough():
    """The old check passed on any non-empty entities dict."""
    wrong = {"text": "NVIDIA Corporation", "confidence": 0.9, "start": 20, "end": 38}

    assert compare_outputs(_result([_JOHN]), _result([wrong]))


def test_small_confidence_drift_is_tolerated():
    drifted = {**_JOHN, "confidence": 0.95}

    assert compare_outputs(_result([_JOHN]), _result([drifted])) == []


def test_large_confidence_drift_is_a_mismatch():
    drifted = {**_JOHN, "confidence": 0.55}

    problems = compare_outputs(_result([_JOHN]), _result([drifted]))

    assert len(problems) == 1
    assert "confidence" in problems[0]


def test_different_classification_label_is_a_mismatch():
    problems = compare_outputs(_result([_JOHN]), _result([_JOHN], topic="finance"))

    assert any("classification[topic]" in problem for problem in problems)


def test_bare_string_records_compare_by_text():
    reference = {"entities": {"person": ["John Smith"]}}

    assert compare_outputs(reference, {"entities": {"person": ["John Smith"]}}) == []
    assert compare_outputs(reference, {"entities": {"person": ["Jane Doe"]}})


def test_four_key_grouping_is_a_payload_mismatch():
    formatted = {"entities": {"person": [_JOHN]}}
    grouped = {
        "entities": {"person": [_JOHN]},
        "classifications": {},
        "structures": {},
        "relations": {},
    }

    problems = compare_outputs(formatted, grouped)

    assert any(problem.startswith("keys:") for problem in problems)
