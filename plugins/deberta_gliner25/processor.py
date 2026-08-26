"""Boundary serving processor: GLiNER2 collate + Pioneer response keys."""

from __future__ import annotations

from typing import Any

from plugins.deberta_gliner2.processor import decode_output
from vllm_factory.optional_deps import require

_COMPACT_FLAG = "_compact"
# Schema markers, separators and specials share the prompt with the text.
_SCHEMA_TOKEN_HEADROOM = 128
_FIT_ATTEMPTS = 3


def boundary_transformer(tokenizer: Any) -> Any:
    """Build a SchemaTransformer bound to ``tokenizer``.

    Args:
        tokenizer: Tokenizer the transformer encodes with.

    Returns:
        A GLiNER2 SchemaTransformer. Construction is cheap but not free, so
        callers that collate per request hold onto the instance.
    """
    processor_mod = require(
        "gliner2.processor", "gliner2", purpose="GLiNER25 preprocess"
    )
    return processor_mod.SchemaTransformer(tokenizer=tokenizer, token_pooling="first")


def collate_word_cap(max_model_len: int | None) -> int | None:
    """Word budget for truncation, leaving room for the schema tokens.

    Args:
        max_model_len: Encoded-token ceiling, or None when unbounded.

    Returns:
        Word cap to hand GLiNER2 collate, or None when unbounded. Words are
        not tokens, so this is a starting point that ``preprocess_boundary``
        tightens while the encoded prompt still overflows.
    """
    if not max_model_len:
        return None
    return max(1, int(max_model_len) - _SCHEMA_TOKEN_HEADROOM)


def iter_boundary_items(data: Any) -> list[dict[str, Any]]:
    """Normalize a /pooling ``data`` payload into per-text dicts.

    Accepts a single item, ``text: list``, or a list of items. Unwraps a
    vLLM request object or ``{"data": ...}`` wrapper when present.
    """
    if hasattr(data, "data"):
        data = data.data
    elif isinstance(data, dict) and "data" in data and not _is_item_dict(data):
        data = data["data"]
    if isinstance(data, list):
        if not data:
            raise ValueError("'data' must be a non-empty list")
        return [_as_item_dict(item) for item in data]
    if isinstance(data, dict):
        text = data.get("text")
        if isinstance(text, list):
            if not text:
                raise ValueError("'text' must be a non-empty list")
            shared = {key: value for key, value in data.items() if key != "text"}
            return [{**shared, "text": item} for item in text]
        return [data]
    raise ValueError("Expected request data dict")


def _is_item_dict(data: dict[str, Any]) -> bool:
    return "text" in data or "schema" in data or "labels" in data


def _as_item_dict(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Each batch item must be a dict")
    return item


def compact_boundary_extra(
    text: str,
    schema: dict[str, Any],
    *,
    threshold: float = 0.5,
    include_confidence: bool = False,
    include_spans: bool = False,
    max_len: int | None = None,
) -> dict[str, Any]:
    """Metadata the GPU worker needs to re-collate routing tensors.

    This is what crosses vLLM V1 ZMQ — not the collate index tensors. The
    worker must reproduce the frontend's token layout, so ``max_len`` records
    the cap that was actually applied.

    Args:
        text: Input text to extract from.
        schema: Normalized GLiNER2 schema.
        threshold: Score threshold carried through to decode.
        include_confidence: Whether decode should return scores.
        include_spans: Whether decode should return character spans.
        max_len: Word cap applied during collate, or None when unbounded.

    Returns:
        Compact ``extra_kwargs`` for one sequence.
    """
    extra: dict[str, Any] = {
        _COMPACT_FLAG: True,
        "text": text,
        "schema": schema,
        "threshold": threshold,
        "include_confidence": include_confidence,
        "include_spans": include_spans,
    }
    if max_len is not None:
        extra["max_len"] = int(max_len)
    return extra


def is_compact_extra(extra: dict[str, Any]) -> bool:
    """True when extras are reconstructable text/schema, not routing tensors."""
    if extra.get(_COMPACT_FLAG):
        return True
    return "text" in extra and "schema" in extra and "text_word_indices" not in extra


def preprocess_boundary(
    tokenizer,
    text: str,
    schema: dict[str, Any],
    *,
    threshold: float = 0.5,
    include_confidence: bool = False,
    include_spans: bool = False,
    truncate_overflow_text: bool = False,
    word_cap: int | None = None,
    max_model_len: int | None = None,
    transformer: Any = None,
) -> dict[str, Any]:
    """Run GLiNER2 boundary collate; stash compact extras for the pooler.

    Collate drops words past its cap without reporting it, so the cap is only
    applied when the caller allowed truncation. The cap counts words while the
    engine limit counts tokens, so an allowed truncation retries with a
    tighter cap until the encoded prompt fits.

    Args:
        tokenizer: Tokenizer used when ``transformer`` is not supplied.
        text: Input text to extract from.
        schema: Normalized GLiNER2 schema.
        threshold: Score threshold carried through to decode.
        include_confidence: Whether decode should return scores.
        include_spans: Whether decode should return character spans.
        truncate_overflow_text: Whether text past the budget may be dropped.
        word_cap: Word budget used when truncating; None leaves text intact.
        max_model_len: Encoded-token ceiling the engine can schedule.
        transformer: SchemaTransformer to collate with, built if omitted.

    Returns:
        Dict with ``input_ids``, compact ``extra_kwargs`` and ``schema_dict``.

    Raises:
        ValueError: The encoded prompt is above ``max_model_len`` and either
            truncation was not requested or it did not converge.
    """
    host = transformer if transformer is not None else boundary_transformer(tokenizer)
    cap = word_cap if truncate_overflow_text else None
    for _ in range(_FIT_ATTEMPTS):
        batch = host.collate_fn_inference(
            [(text, schema)],
            architecture="boundary",
            error_policy="raise",
            max_len=cap,
        )
        input_ids = batch.input_ids[0].detach().cpu().tolist()
        if not max_model_len or len(input_ids) <= max_model_len:
            return {
                "input_ids": input_ids,
                "extra_kwargs": compact_boundary_extra(
                    text,
                    schema,
                    threshold=threshold,
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                    max_len=cap,
                ),
                "schema_dict": schema,
            }
        if not truncate_overflow_text:
            raise ValueError(
                f"encoded prompt is {len(input_ids)} tokens, above the "
                f"{max_model_len}-token limit; shorten 'text', raise "
                "--max-model-len, or set 'truncate_overflow_text'"
            )
        cap = _tighter_word_cap(batch, len(input_ids), max_model_len)
    raise ValueError(
        f"'text' did not fit under {max_model_len} tokens after {_FIT_ATTEMPTS} tries"
    )


def _tighter_word_cap(batch: Any, tokens: int, budget: int) -> int:
    """Scale the word cap down by the ratio the encoded prompt overflowed by."""
    kept = len(batch.text_tokens[0])
    return max(1, int(kept * budget / tokens) - 1)


def collate_compact_extras(
    tokenizer: Any,
    extras: list[dict[str, Any]],
    *,
    transformer: Any = None,
) -> Any:
    """Collate compact extras into one GLiNER2 ``PreprocessedBatch``.

    Args:
        tokenizer: Tokenizer used when ``transformer`` is not supplied.
        extras: Compact extras sharing one word cap, in sequence order.
        transformer: SchemaTransformer to collate with, built if omitted.

    Returns:
        One batch whose per-row token layout matches what the frontend encoded.
    """
    host = transformer if transformer is not None else boundary_transformer(tokenizer)
    rows = [(item["text"], item["schema"]) for item in extras]
    word_cap = extras[0].get("max_len") if extras else None
    if word_cap is not None:
        word_cap = int(word_cap)
    return host.collate_fn_inference(
        rows,
        architecture="boundary",
        error_policy="raise",
        max_len=word_cap,
    )


def reshape_boundary_output(sample: dict[str, Any]) -> dict[str, Any]:
    """Map GLiNER2 decode keys onto Pioneer ``entities/classifications/structures/relations``."""
    entities = sample.get("entities", {})
    if isinstance(entities, list) and entities and isinstance(entities[0], dict):
        merged: dict[str, Any] = {}
        for item in entities:
            merged.update(item)
        entities = merged
    classifications: dict[str, Any] = {}
    structures: dict[str, Any] = {}
    relations: dict[str, Any] = {}
    for key, value in sample.items():
        if key == "entities":
            continue
        if _is_relation_payload(value):
            relations[key] = value
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            structures[key] = value
        else:
            classifications[key] = value
    return {
        "entities": entities or {},
        "classifications": classifications,
        "structures": structures,
        "relations": relations,
    }


def _is_relation_payload(value: Any) -> bool:
    if not isinstance(value, list) or not value or not isinstance(value[0], dict):
        return False
    keys = set(value[0])
    return bool(keys & {"head", "tail", "subject", "object", "src", "dst"})


def decode_boundary_output(
    raw_output, schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Unpack the JSON byte tensor produced by the boundary pooler."""
    return decode_output(raw_output, schema or {})


def schema_format_args(schema: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    """Derive ``requested_relations`` and ``classification_tasks`` from a schema.

    Args:
        schema: GLiNER2 extract schema, or None.

    Returns:
        ``(requested_relations, classification_tasks)``.
    """
    schema = schema or {}
    classifications = schema.get("classifications") or []
    tasks = [
        item["task"]
        for item in classifications
        if isinstance(item, dict) and "task" in item
    ]
    relations = schema.get("relations") or {}
    if isinstance(relations, dict):
        rels = list(relations)
    elif isinstance(relations, list):
        rels = [item if isinstance(item, str) else str(item) for item in relations]
    else:
        rels = []
    return rels, tasks
