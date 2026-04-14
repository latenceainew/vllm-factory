"""GLiNER2 vLLM Processor — handles preprocessing and postprocessing.

Ports the SchemaTransformer logic from gliner2.processor for vLLM use.
Preprocessing: Schema + text → tokenized input with mapped_indices.
Postprocessing: Raw output tensor → structured extraction results.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import OrderedDict
from typing import Any, Dict, List

# ==================================================================
# Text Splitter (mirrors gliner2.processor.WhitespaceTokenSplitter)
# ==================================================================

WORD_PATTERN = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)


def split_words(text: str, lower: bool = True):
    t = text.lower() if lower else text
    return [(m.group(), m.start(), m.end()) for m in WORD_PATTERN.finditer(t)]


# ==================================================================
# Special Tokens
# ==================================================================

SEP_STRUCT = "[SEP_STRUCT]"
SEP_TEXT = "[SEP_TEXT]"
P_TOKEN = "[P]"
C_TOKEN = "[C]"
E_TOKEN = "[E]"
R_TOKEN = "[R]"
L_TOKEN = "[L]"
EXAMPLE_TOKEN = "[EXAMPLE]"
OUTPUT_TOKEN = "[OUTPUT]"
DESC_TOKEN = "[DESCRIPTION]"

SPECIAL_TOKENS = [
    SEP_STRUCT,
    SEP_TEXT,
    P_TOKEN,
    C_TOKEN,
    E_TOKEN,
    R_TOKEN,
    L_TOKEN,
    EXAMPLE_TOKEN,
    OUTPUT_TOKEN,
    DESC_TOKEN,
]


def _empty_schema() -> dict[str, Any]:
    return {
        "json_structures": [],
        "classifications": [],
        "entities": OrderedDict(),
        "relations": [],
        "_meta": {
            "json_structures": {},
            "relations": {},
        },
    }


def build_special_token_ids(tokenizer) -> dict[str, int]:
    special_ids: dict[str, int] = {}
    for tok in SPECIAL_TOKENS:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tok))
        if ids:
            special_ids[tok] = ids[0]
    return special_ids


def build_tokenization_cache(tokenizer) -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    for tok in SPECIAL_TOKENS + ["(", ")", ",", "|"]:
        cache[tok] = tokenizer.tokenize(tok)
    return cache


def _tokenize_cached(
    tokenizer,
    token: str,
    tokenization_cache: dict[str, list[str]] | None = None,
):
    if tokenization_cache is None:
        return tokenizer.tokenize(token)
    cached = tokenization_cache.get(token)
    if cached is not None:
        return cached
    cached = tokenizer.tokenize(token)
    tokenization_cache[token] = cached
    return cached


def get_schema_cache_key(schema: dict[str, Any]) -> str:
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# ==================================================================
# Schema Processing (inference-only, no training augmentation)
# ==================================================================


def normalize_entities_schema(entities: Any) -> OrderedDict[str, str]:
    if entities is None:
        return OrderedDict()
    if isinstance(entities, list):
        return OrderedDict((str(label), "") for label in entities)
    if isinstance(entities, dict):
        return OrderedDict((str(label), str(desc or "")) for label, desc in entities.items())
    raise ValueError("'entities' must be a list or dict")


def normalize_classifications_schema(classifications: Any) -> list[dict[str, Any]]:
    if classifications is None:
        return []
    if not isinstance(classifications, list):
        raise ValueError("'classifications' must be a list")

    normalized = []
    for item in classifications:
        if not isinstance(item, dict):
            raise ValueError("Each classification config must be a dict")
        task = item.get("task")
        labels = item.get("labels")
        if not task:
            raise ValueError("Each classification config must include a task")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"Classification task '{task}' must include non-empty labels")

        config = dict(item)
        config["task"] = str(task)
        config["labels"] = [str(label) for label in labels]
        if "label_descriptions" in config and config["label_descriptions"] is not None:
            if not isinstance(config["label_descriptions"], dict):
                raise ValueError("'label_descriptions' must be a dict when provided")
            config["label_descriptions"] = {
                str(label): str(desc or "") for label, desc in config["label_descriptions"].items()
            }
        normalized.append(config)
    return normalized


def normalize_relations_schema(
    relations: Any,
) -> tuple[list[dict[str, dict[str, str]]], dict[str, str]]:
    if relations is None:
        return [], {}
    if isinstance(relations, list):
        return [{str(name): {"head": "", "tail": ""}} for name in relations], {}
    if isinstance(relations, dict):
        return (
            [{str(name): {"head": "", "tail": ""}} for name in relations],
            {str(name): str(desc or "") for name, desc in relations.items()},
        )
    raise ValueError("'relations' must be a list or dict")


def normalize_structures_schema(
    structures: Any,
) -> tuple[list[dict[str, OrderedDict[str, str]]], dict[str, list[dict[str, Any]]]]:
    if structures is None:
        return [], {}
    if not isinstance(structures, dict):
        raise ValueError("'structures' must be a dict")

    json_structures = []
    meta = {}

    for parent, spec in structures.items():
        fields = spec.get("fields") if isinstance(spec, dict) else None
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"Structure '{parent}' must include a non-empty fields list")

        normalized_fields = OrderedDict()
        field_meta = []
        for field in fields:
            if not isinstance(field, dict):
                raise ValueError(f"Structure '{parent}' fields must be dict objects")
            name = field.get("name")
            if not name:
                raise ValueError(f"Structure '{parent}' field is missing name")
            name = str(name)
            normalized_fields[name] = ""
            field_meta.append(
                {
                    "name": name,
                    "dtype": str(field.get("dtype") or ""),
                    "description": str(field.get("description") or ""),
                    "choices": [str(choice) for choice in field.get("choices", [])],
                }
            )

        parent_name = str(parent)
        json_structures.append({parent_name: normalized_fields})
        meta[parent_name] = field_meta

    return json_structures, meta


def normalize_gliner2_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise ValueError("'schema' must be a dict")

    normalized = _empty_schema()
    normalized["entities"] = normalize_entities_schema(schema.get("entities"))
    normalized["classifications"] = normalize_classifications_schema(schema.get("classifications"))

    relations, relation_meta = normalize_relations_schema(schema.get("relations"))
    normalized["relations"] = relations
    normalized["_meta"]["relations"] = relation_meta

    json_structures, json_meta = normalize_structures_schema(schema.get("structures"))
    normalized["json_structures"] = json_structures
    normalized["_meta"]["json_structures"] = json_meta

    if not (
        normalized["entities"]
        or normalized["classifications"]
        or normalized["relations"]
        or normalized["json_structures"]
    ):
        raise ValueError("GLiNER2 schema must include at least one task section")

    return normalized


def build_schema_for_entities(entity_types: List[str]) -> Dict:
    """Build schema dict for entity extraction."""
    return normalize_gliner2_schema({"entities": entity_types})


def build_schema_for_relations(relation_types: List[str]) -> Dict:
    """Build schema dict for relation extraction."""
    return normalize_gliner2_schema({"relations": relation_types})


def build_schema_for_json(structures: Dict[str, List[str]]) -> Dict:
    """Build schema dict for JSON structure extraction."""
    legacy = {
        parent: {"fields": [{"name": field, "dtype": ""} for field in fields]}
        for parent, fields in structures.items()
    }
    return normalize_gliner2_schema({"structures": legacy})


def build_schema_for_classification(tasks: Dict) -> Dict:
    """Build schema dict for classification tasks."""
    classifications = []
    for task_name, labels in tasks.items():
        if isinstance(labels, dict):
            config = dict(labels)
            config["task"] = task_name
            config.setdefault("true_label", ["N/A"])
        else:
            config = {"task": task_name, "labels": labels, "true_label": ["N/A"]}
        classifications.append(config)
    return normalize_gliner2_schema({"classifications": classifications})


# ==================================================================
# Schema → Token Sequence
# ==================================================================


def _transform_schema(
    parent, fields, child_prefix, prompt=None, label_descriptions=None, example_mode="none"
):
    """Transform a single schema into a token sequence."""
    prompt_str = parent
    if prompt:
        prompt_str = f"{parent}: {prompt}"

    if example_mode in ("descriptions", "both") and label_descriptions:
        for label, desc in label_descriptions.items():
            if label in fields:
                prompt_str += f" {DESC_TOKEN} {label}: {desc}"

    tokens = ["(", P_TOKEN, prompt_str, "("]
    for field in fields:
        tokens.extend([child_prefix, field])
    tokens.extend([")", ")"])
    return tokens


def _count_tokenized_length(
    tokenizer,
    tokens: list[str],
    tokenization_cache: dict[str, list[str]] | None = None,
) -> int:
    return sum(len(_tokenize_cached(tokenizer, token, tokenization_cache)) for token in tokens)


def _truncate_text_to_token_budget(
    tokenizer,
    schema_tokens_list: list[list[str]],
    text_tokens: list[str],
    start_mapping: list[int],
    end_mapping: list[int],
    max_model_len: int | None,
    truncate_overflow_text: bool = False,
    tokenization_cache: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[int], list[int]]:
    if max_model_len is None:
        return text_tokens, start_mapping, end_mapping

    schema_budget_tokens = []
    for idx, struct in enumerate(schema_tokens_list):
        schema_budget_tokens.extend(struct)
        if idx < len(schema_tokens_list) - 1:
            schema_budget_tokens.append(SEP_STRUCT)
    schema_budget_tokens.append(SEP_TEXT)

    schema_len = _count_tokenized_length(tokenizer, schema_budget_tokens, tokenization_cache)
    if schema_len >= max_model_len:
        raise ValueError(
            "GLiNER2 schema is too large for the configured max_model_len; "
            "reduce schema size or increase max_model_len"
        )

    total_len = schema_len + _count_tokenized_length(tokenizer, text_tokens, tokenization_cache)
    if total_len <= max_model_len:
        return text_tokens, start_mapping, end_mapping

    if not truncate_overflow_text:
        raise ValueError(
            "GLiNER2 request exceeds the configured max_model_len; "
            "reduce text or schema size, increase max_model_len, or set "
            "'truncate_overflow_text' to true"
        )

    kept = 0
    used = schema_len
    for idx, token in enumerate(text_tokens):
        token_len = len(_tokenize_cached(tokenizer, token, tokenization_cache))
        if used + token_len > max_model_len:
            break
        used += token_len
        kept = idx + 1

    if kept == 0:
        raise ValueError(
            "GLiNER2 schema leaves no room for text tokens within max_model_len; "
            "reduce schema size or increase max_model_len"
        )

    return text_tokens[:kept], start_mapping[:kept], end_mapping[:kept]


def infer_schemas_from_dict(schema: Dict) -> Dict:
    """Infer schema token sequences and task types from schema dict (inference mode)."""
    schemas = []
    labels = []
    types = []
    meta = schema.get("_meta", {})
    structure_meta = meta.get("json_structures", {})
    relation_meta = meta.get("relations", {})

    for item in schema.get("json_structures", []):
        for parent, fields in item.items():
            field_names = list(fields.keys())
            field_defs = structure_meta.get(parent, [])
            field_descs = (
                {fd["name"]: fd.get("description", "") for fd in field_defs} if field_defs else {}
            )
            mode = "descriptions" if any(field_descs.values()) else "none"
            schemas.append(
                _transform_schema(
                    parent,
                    field_names,
                    C_TOKEN,
                    label_descriptions=field_descs,
                    example_mode=mode,
                )
            )
            count = sum(1 for value in fields.values() if value != "")
            labels.append([max(1, count), []])
            types.append("json_structures")

    if schema.get("entities"):
        entity_fields = list(schema["entities"].keys())
        entity_descriptions = dict(schema["entities"])
        mode = "descriptions" if any(entity_descriptions.values()) else "none"
        schemas.append(
            _transform_schema(
                "entities",
                entity_fields,
                E_TOKEN,
                label_descriptions=entity_descriptions,
                example_mode=mode,
            )
        )
        labels.append([1, []])
        types.append("entities")

    relation_fields = []
    for item in schema.get("relations", []):
        for parent in item:
            relation_fields.append(parent)
    if relation_fields:
        mode = (
            "descriptions"
            if any(relation_meta.get(field, "") for field in relation_fields)
            else "none"
        )
        schemas.append(
            _transform_schema(
                "relations",
                relation_fields,
                R_TOKEN,
                label_descriptions=relation_meta,
                example_mode=mode,
            )
        )
        labels.append([1, []])
        types.append("relations")

    for cls_item in schema.get("classifications", []):
        task = cls_item["task"]
        cls_labels = cls_item["labels"]
        descs = cls_item.get("label_descriptions", {})
        mode = "descriptions" if descs else "none"
        schemas.append(
            _transform_schema(
                task,
                cls_labels,
                L_TOKEN,
                prompt=cls_item.get("prompt"),
                label_descriptions=descs,
                example_mode=mode,
            )
        )
        labels.append([])
        types.append("classifications")

    return {"schemas": schemas, "task_types": types, "structure_labels": labels}


# ==================================================================
# Input Formatting
# ==================================================================


def format_input_with_mapping(
    tokenizer,
    schema_tokens_list,
    text_tokens,
    tokenization_cache: dict[str, list[str]] | None = None,
):
    """Format schema + text into token IDs with segment mappings."""
    combined = []
    for struct in schema_tokens_list:
        combined.extend(struct)
        combined.append(SEP_STRUCT)
    if combined:
        combined.pop()
    combined.append(SEP_TEXT)
    combined.extend(text_tokens)

    subwords = []
    mappings = []
    num_schemas = len(schema_tokens_list)
    text_schema_idx = num_schemas
    current_schema = 0
    found_sep = False

    for orig_idx, token in enumerate(combined):
        if token == SEP_TEXT:
            seg_type = "sep"
            schema_idx = text_schema_idx
            found_sep = True
        elif not found_sep:
            seg_type = "schema"
            schema_idx = current_schema
            if token == SEP_STRUCT:
                current_schema += 1
        else:
            seg_type = "text"
            schema_idx = text_schema_idx

        sub_tokens = _tokenize_cached(tokenizer, token, tokenization_cache)
        subwords.extend(sub_tokens)
        mappings.extend([(seg_type, orig_idx, schema_idx)] * len(sub_tokens))

    input_ids = tokenizer.convert_tokens_to_ids(subwords)
    return {
        "input_ids": input_ids,
        "mapped_indices": mappings,
    }


def preprocess_schema(
    schema: Dict,
    schema_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cache_key = get_schema_cache_key(schema)
    if schema_cache is not None and cache_key in schema_cache:
        cached = copy.deepcopy(schema_cache[cache_key])
        if hasattr(schema_cache, "move_to_end"):
            schema_cache.move_to_end(cache_key)
        cached["schema_cache_hit"] = True
        return cached

    processed = infer_schemas_from_dict(schema)
    schema_tokens_list = processed["schemas"]
    task_types = processed["task_types"]

    result = {
        "schema_tokens_list": schema_tokens_list,
        "task_types": task_types,
        "schema_count": len(schema_tokens_list),
        "schema_cache_hit": False,
    }
    if schema_cache is not None:
        schema_cache[cache_key] = copy.deepcopy(result)
        if hasattr(schema_cache, "move_to_end"):
            schema_cache.move_to_end(cache_key)
            while len(schema_cache) > 256:
                schema_cache.popitem(last=False)
    return result


# ==================================================================
# Main Preprocessing
# ==================================================================


def preprocess(
    tokenizer,
    text: str,
    schema: Dict,
    token_pooling: str = "first",
    max_model_len: int | None = None,
    truncate_overflow_text: bool = False,
    special_token_ids: dict[str, int] | None = None,
    tokenization_cache: dict[str, list[str]] | None = None,
    schema_cache: dict[str, dict[str, Any]] | None = None,
):
    """Preprocess text + schema for vLLM inference.

    Returns a dict with all data needed for vLLM embed call.
    """
    if text and not text.endswith((".", "!", "?")):
        text = text + "."
    elif not text:
        text = "."

    schema_state = preprocess_schema(
        schema,
        schema_cache=schema_cache,
    )

    word_triples = split_words(text, lower=True)
    text_tokens = [w[0] for w in word_triples]
    start_mapping = [w[1] for w in word_triples]
    end_mapping = [w[2] for w in word_triples]

    text_tokens, start_mapping, end_mapping = _truncate_text_to_token_budget(
        tokenizer,
        schema_state["schema_tokens_list"],
        text_tokens,
        start_mapping,
        end_mapping,
        max_model_len,
        truncate_overflow_text,
        tokenization_cache,
    )
    if end_mapping:
        text = text[: end_mapping[-1]]

    fmt = format_input_with_mapping(
        tokenizer,
        schema_state["schema_tokens_list"],
        text_tokens,
        tokenization_cache,
    )

    return {
        "input_ids": fmt["input_ids"],
        "mapped_indices": fmt["mapped_indices"],
        "schema_tokens_list": schema_state["schema_tokens_list"],
        "task_types": schema_state["task_types"],
        "text_tokens": text_tokens,
        "schema_count": schema_state["schema_count"],
        "schema_cache_hit": schema_state["schema_cache_hit"],
        "original_text": text,
        "start_mapping": start_mapping,
        "end_mapping": end_mapping,
        "special_token_ids": special_token_ids or build_special_token_ids(tokenizer),
        "token_pooling": token_pooling,
        "schema_dict": schema,
    }


# ==================================================================
# Postprocessing
# ==================================================================


def decode_output(raw_output, schema: Dict, task_types: List[str] = None) -> Dict:
    """Decode raw vLLM output tensor back into structured results.

    The output is a JSON-encoded byte tensor prepended by length.
    """
    if isinstance(raw_output, list):
        data = raw_output
    elif hasattr(raw_output, "tolist"):
        data = raw_output.tolist()
    else:
        data = list(raw_output)

    length = int(data[0])
    byte_data = bytes([int(b) for b in data[1 : length + 1]])
    results = json.loads(byte_data.decode("utf-8"))

    classifications = schema.get("classifications", []) if isinstance(schema, dict) else []
    classification_config = {
        item.get("task"): item
        for item in classifications
        if isinstance(item, dict) and item.get("task")
    }
    for key, value in results.items():
        if not isinstance(value, dict) or value.get("type") != "classification":
            continue
        config = classification_config.get(key, {})
        if config:
            value["multi_label"] = bool(config.get("multi_label", False))

    return results


def _format_entity_record(
    item: dict[str, Any], include_confidence: bool, include_spans: bool
) -> Any:
    if not isinstance(item, dict):
        return item

    if not include_confidence and not include_spans:
        return item.get("text")

    record = {"text": item.get("text")}
    if include_confidence and "confidence" in item:
        record["confidence"] = item["confidence"]
    if include_spans:
        if "start" in item:
            record["start"] = item["start"]
        if "end" in item:
            record["end"] = item["end"]
    return record


def _strip_nested_metadata(value: Any, include_confidence: bool, include_spans: bool) -> Any:
    if isinstance(value, dict):
        if "text" in value:
            record = {"text": value["text"]}
            if include_confidence and "confidence" in value:
                record["confidence"] = value["confidence"]
            if include_spans:
                if "start" in value:
                    record["start"] = value["start"]
                if "end" in value:
                    record["end"] = value["end"]
            return record if (include_confidence or include_spans) else value["text"]
        return {
            key: _strip_nested_metadata(item, include_confidence, include_spans)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_strip_nested_metadata(item, include_confidence, include_spans) for item in value]
    return value


def format_results(
    results: Dict,
    threshold: float | None = None,
    include_confidence: bool = False,
    include_spans: bool = False,
) -> Dict:
    """Format raw results into user-friendly output."""
    formatted = {}
    relations = {}

    for key, value in results.items():
        if not isinstance(value, dict) or "type" not in value:
            formatted[key] = value
            continue

        result_type = value["type"]

        if result_type == "classification":
            import torch

            logits = value.get("logits", [])
            labels = value.get("labels", [])
            multi_label = bool(value.get("multi_label", False))
            if multi_label:
                probs = torch.sigmoid(torch.tensor(logits))
                selected = [
                    (label, probs[idx].item())
                    for idx, label in enumerate(labels)
                    if probs[idx].item() >= (0.5 if threshold is None else threshold)
                ]
                if include_confidence:
                    formatted[key] = [
                        {"label": label, "confidence": score} for label, score in selected
                    ]
                else:
                    formatted[key] = [label for label, _ in selected]
            else:
                probs = torch.softmax(torch.tensor(logits), dim=-1)
                best = int(probs.argmax().item())
                best_score = probs[best].item()
                if threshold is not None and best_score < threshold:
                    formatted[key] = None
                elif include_confidence:
                    formatted[key] = {"label": labels[best], "confidence": best_score}
                else:
                    formatted[key] = labels[best]

        elif result_type == "entities":
            entity_results = {}
            for label, spans in value.get("entities", {}).items():
                records = []
                for item in spans:
                    record = _format_entity_record(item, include_confidence, include_spans)
                    if record is not None:
                        records.append(record)
                entity_results[label] = records
            formatted["entities"] = entity_results

        elif result_type == "relations":
            relation_items = []
            for inst in value.get("instances", []):
                filtered_instance = {
                    field: _strip_nested_metadata(field_value, include_confidence, include_spans)
                    for field, field_value in inst.items()
                    if field_value is not None
                }
                if filtered_instance:
                    relation_items.append(filtered_instance)
            relations[key] = relation_items

        elif result_type == "json_structures":
            formatted[key] = []
            for inst in value.get("instances", []):
                filtered_instance = {
                    field: _strip_nested_metadata(field_value, include_confidence, include_spans)
                    for field, field_value in inst.items()
                }
                if filtered_instance:
                    formatted[key].append(filtered_instance)

    if relations:
        formatted["relation_extraction"] = relations

    return formatted
