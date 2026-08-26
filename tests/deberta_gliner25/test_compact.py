"""Compact extra_kwargs and multi-text /pooling payload parsing."""

from plugins.deberta_gliner25.processor import (
    compact_boundary_extra,
    is_compact_extra,
    iter_boundary_items,
)


def test_iter_single_dict():
    items = iter_boundary_items({"text": "Ada", "schema": {"entities": ["person"]}})
    assert len(items) == 1
    assert items[0]["text"] == "Ada"


def test_iter_text_list():
    items = iter_boundary_items(
        {"text": ["Ada", "Bob"], "schema": {"entities": ["person"]}}
    )
    assert [item["text"] for item in items] == ["Ada", "Bob"]
    assert items[0]["schema"] == {"entities": ["person"]}


def test_iter_list_of_dicts():
    items = iter_boundary_items(
        [
            {"text": "Ada", "schema": {"entities": ["person"]}},
            {"text": "Bob", "schema": {"entities": ["person"]}},
        ]
    )
    assert [item["text"] for item in items] == ["Ada", "Bob"]


def test_compact_extra_has_no_routing_tensors():
    extra = compact_boundary_extra(
        "Ada visited Berlin.",
        {"entities": ["person", "location"]},
        threshold=0.5,
        max_len=3968,
    )
    assert is_compact_extra(extra)
    assert "text_word_indices" not in extra
    assert extra["text"] == "Ada visited Berlin."
    assert extra["max_len"] == 3968


def test_legacy_routing_extra_is_not_compact():
    extra = {
        "text_word_indices": [1, 2, 3],
        "text": "Ada",
        "schema": {},
    }
    assert not is_compact_extra(extra)
