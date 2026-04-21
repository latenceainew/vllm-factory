"""Unit tests for plugins.deberta_gliner2.processor schema normalization.

No GPU, no vLLM runtime. Exercises normalize_gliner2_schema and the
normalize_*_schema helpers for both pre-existing shapes (regression guard)
and the widened shapes introduced by the combined-schema PR.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest

from plugins.deberta_gliner2.processor import (
    normalize_classifications_schema,
    normalize_entities_schema,
    normalize_gliner2_schema,
    normalize_relations_schema,
    normalize_structures_schema,
)


# ======================================================================
# Fix A — relations dict-of-dicts
# ======================================================================


class TestRelationsDictOfDicts:
    """Repro + regression guard for the dict-value relations bug.

    Before Fix A, passing {"works_at": {"description": "Employment"}} would
    stringify the dict value, embedding "{'description': 'Employment'}" into
    the schema prompt.
    """

    def test_relation_dict_of_dicts_description(self):
        """Dict value with 'description' key extracts the description string."""
        relations, meta = normalize_relations_schema(
            {"works_at": {"description": "Employment relationship"}}
        )
        assert relations == [{"works_at": {"head": "", "tail": ""}}]
        assert meta["works_at"]["description"] == "Employment relationship"

    def test_relation_dict_of_dicts_threshold(self):
        """Dict value with 'threshold' key is preserved in meta."""
        relations, meta = normalize_relations_schema(
            {"works_at": {"description": "Employment", "threshold": 0.25}}
        )
        assert meta["works_at"]["description"] == "Employment"
        assert meta["works_at"]["threshold"] == 0.25

    def test_relation_dict_of_dicts_threshold_only(self):
        """Dict value with only threshold, no description."""
        relations, meta = normalize_relations_schema(
            {"works_at": {"threshold": 0.3}}
        )
        assert meta["works_at"]["description"] == ""
        assert meta["works_at"]["threshold"] == 0.3

    def test_relation_string_value_unchanged(self):
        """Pre-existing string-value dict still works."""
        relations, meta = normalize_relations_schema(
            {"works_at": "Employment relationship"}
        )
        assert relations == [{"works_at": {"head": "", "tail": ""}}]
        assert meta["works_at"]["description"] == "Employment relationship"

    def test_relation_list_unchanged(self):
        """Pre-existing list shape still works."""
        relations, meta = normalize_relations_schema(["works_at", "reports_to"])
        assert relations == [
            {"works_at": {"head": "", "tail": ""}},
            {"reports_to": {"head": "", "tail": ""}},
        ]
        assert meta == {}

    def test_relation_through_normalize_gliner2_schema(self):
        """End-to-end: dict-of-dicts via the top-level normalizer."""
        schema = normalize_gliner2_schema({
            "relations": {"works_at": {"description": "Employment relationship"}}
        })
        assert schema["_meta"]["relations"]["works_at"]["description"] == "Employment relationship"

    def test_relation_invalid_threshold(self):
        """Threshold outside [0, 1] raises."""
        with pytest.raises(ValueError, match="[Tt]hreshold"):
            normalize_relations_schema({"works_at": {"threshold": 1.5}})

    def test_relation_invalid_threshold_negative(self):
        with pytest.raises(ValueError, match="[Tt]hreshold"):
            normalize_relations_schema({"works_at": {"threshold": -0.1}})


# ======================================================================
# Entities — pre-existing shapes unchanged + widened dict-of-dicts
# ======================================================================


class TestEntitiesNormalization:
    def test_list_of_names(self):
        result = normalize_entities_schema(["person", "org"])
        assert result == OrderedDict([("person", ""), ("org", "")])

    def test_dict_str_values(self):
        result = normalize_entities_schema({"person": "People", "org": "Companies"})
        assert result == OrderedDict([("person", "People"), ("org", "Companies")])

    def test_dict_of_dicts_with_description(self):
        result = normalize_entities_schema(
            {"person": {"description": "Person names"}}
        )
        # No threshold → plain OrderedDict return
        assert result == OrderedDict([("person", "Person names")])

    def test_dict_of_dicts_with_threshold(self):
        entities, meta = normalize_entities_schema(
            {"person": {"description": "People", "threshold": 0.9}}
        )
        assert entities == OrderedDict([("person", "People")])
        assert meta == {"person": {"threshold": 0.9}}

    def test_dict_of_dicts_threshold_only(self):
        entities, meta = normalize_entities_schema(
            {"email": {"threshold": 0.8}}
        )
        assert entities == OrderedDict([("email", "")])
        assert meta == {"email": {"threshold": 0.8}}

    def test_dict_of_dicts_mixed(self):
        """Mix of string values and dict values with threshold."""
        entities, meta = normalize_entities_schema(
            {"person": "People", "email": {"threshold": 0.8}}
        )
        assert entities == OrderedDict([("person", "People"), ("email", "")])
        assert meta == {"person": {"threshold": None}, "email": {"threshold": 0.8}}

    def test_none_returns_empty(self):
        result = normalize_entities_schema(None)
        assert result == OrderedDict()

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="[Tt]hreshold"):
            normalize_entities_schema({"person": {"threshold": 2.0}})


# ======================================================================
# Structures — pre-existing shape + threshold on fields
# ======================================================================


class TestStructuresNormalization:
    def test_basic_structure(self):
        json_structures, meta = normalize_structures_schema({
            "employee": {
                "fields": [
                    {"name": "name", "dtype": "str"},
                    {"name": "title"},
                ]
            }
        })
        assert len(json_structures) == 1
        assert "employee" in json_structures[0]
        assert list(json_structures[0]["employee"].keys()) == ["name", "title"]

    def test_field_threshold_preserved(self):
        json_structures, meta = normalize_structures_schema({
            "invoice": {
                "fields": [
                    {"name": "date", "dtype": "str", "threshold": 0.8},
                    {"name": "memo", "threshold": 0.2},
                ]
            }
        })
        assert meta["invoice"][0]["threshold"] == 0.8
        assert meta["invoice"][1]["threshold"] == 0.2

    def test_field_without_threshold(self):
        """Fields without threshold have None in meta."""
        json_structures, meta = normalize_structures_schema({
            "summary": {"fields": [{"name": "product"}]}
        })
        assert meta["summary"][0]["threshold"] is None

    def test_invalid_field_threshold(self):
        with pytest.raises(ValueError, match="[Tt]hreshold"):
            normalize_structures_schema({
                "invoice": {"fields": [{"name": "date", "threshold": 1.5}]}
            })


# ======================================================================
# Classifications — cls_threshold + multi_label
# ======================================================================


class TestClassificationsNormalization:
    def test_basic_classification(self):
        result = normalize_classifications_schema([
            {"task": "sentiment", "labels": ["positive", "negative"]}
        ])
        assert result[0]["task"] == "sentiment"
        assert result[0]["labels"] == ["positive", "negative"]

    def test_cls_threshold_preserved(self):
        result = normalize_classifications_schema([
            {"task": "sentiment", "labels": ["pos", "neg"], "cls_threshold": 0.6}
        ])
        assert result[0]["cls_threshold"] == 0.6

    def test_multi_label_preserved(self):
        result = normalize_classifications_schema([
            {"task": "topics", "labels": ["tech", "finance"], "multi_label": True}
        ])
        assert result[0]["multi_label"] is True

    def test_cls_threshold_and_multi_label(self):
        result = normalize_classifications_schema([{
            "task": "topics",
            "labels": ["tech", "finance"],
            "multi_label": True,
            "cls_threshold": 0.4,
        }])
        assert result[0]["cls_threshold"] == 0.4
        assert result[0]["multi_label"] is True

    def test_no_cls_threshold_absent(self):
        """When cls_threshold is not provided, it's absent from the config."""
        result = normalize_classifications_schema([
            {"task": "sentiment", "labels": ["pos", "neg"]}
        ])
        assert "cls_threshold" not in result[0]

    def test_invalid_cls_threshold(self):
        with pytest.raises(ValueError, match="[Tt]hreshold"):
            normalize_classifications_schema([{
                "task": "sentiment",
                "labels": ["pos", "neg"],
                "cls_threshold": 1.5,
            }])


# ======================================================================
# normalize_gliner2_schema — end-to-end _meta shape
# ======================================================================


class TestNormalizeGliner2SchemaMeta:
    def test_meta_entities_with_thresholds(self):
        schema = normalize_gliner2_schema({
            "entities": {
                "person": {"description": "People", "threshold": 0.9},
                "email": {"threshold": 0.8},
                "org": "Companies",
            }
        })
        assert schema["_meta"]["entities"]["person"]["threshold"] == 0.9
        assert schema["_meta"]["entities"]["email"]["threshold"] == 0.8
        assert schema["_meta"]["entities"]["org"]["threshold"] is None

    def test_meta_entities_list_all_none(self):
        schema = normalize_gliner2_schema({"entities": ["person", "org"]})
        for name in ["person", "org"]:
            assert schema["_meta"]["entities"][name]["threshold"] is None

    def test_meta_relations_with_thresholds(self):
        schema = normalize_gliner2_schema({
            "relations": {
                "works_at": {"description": "Employment", "threshold": 0.25},
                "reports_to": "Reporting chain",
            }
        })
        assert schema["_meta"]["relations"]["works_at"]["threshold"] == 0.25
        assert schema["_meta"]["relations"]["reports_to"]["threshold"] is None

    def test_meta_structures_with_thresholds(self):
        schema = normalize_gliner2_schema({
            "structures": {
                "invoice": {
                    "fields": [
                        {"name": "date", "threshold": 0.8},
                        {"name": "memo"},
                    ]
                }
            }
        })
        fields_meta = schema["_meta"]["json_structures"]["invoice"]
        assert fields_meta[0]["threshold"] == 0.8
        assert fields_meta[1]["threshold"] is None

    def test_meta_classifications(self):
        schema = normalize_gliner2_schema({
            "classifications": [{
                "task": "sentiment",
                "labels": ["pos", "neg"],
                "cls_threshold": 0.6,
                "multi_label": False,
            }]
        })
        assert schema["_meta"]["classifications"]["sentiment"]["cls_threshold"] == 0.6
        assert schema["_meta"]["classifications"]["sentiment"]["multi_label"] is False

    def test_meta_classifications_defaults(self):
        schema = normalize_gliner2_schema({
            "classifications": [{"task": "topic", "labels": ["a", "b"]}]
        })
        assert schema["_meta"]["classifications"]["topic"]["cls_threshold"] is None
        assert schema["_meta"]["classifications"]["topic"]["multi_label"] is False

    def test_pre_existing_entity_list_shape_unchanged(self):
        """Regression: list entities produce the same normalized schema as before."""
        schema = normalize_gliner2_schema({"entities": ["person", "org"]})
        assert schema["entities"] == OrderedDict([("person", ""), ("org", "")])
        assert schema["classifications"] == []
        assert schema["relations"] == []
        assert schema["json_structures"] == []

    def test_pre_existing_entity_dict_shape_unchanged(self):
        schema = normalize_gliner2_schema({"entities": {"person": "People"}})
        assert schema["entities"] == OrderedDict([("person", "People")])

    def test_pre_existing_relations_list_shape_unchanged(self):
        schema = normalize_gliner2_schema({"relations": ["works_at"]})
        assert schema["relations"] == [{"works_at": {"head": "", "tail": ""}}]
