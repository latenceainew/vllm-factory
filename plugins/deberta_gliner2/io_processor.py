"""
IOProcessor plugin for deberta_gliner2 — server-side GLiNER2 extraction
via vLLM's native IOProcessor pipeline.

Uses the schema-based preprocessing from deberta_gliner2.processor instead of
the GLiNERPreprocessor/GLiNERDecoder used by other GLiNER plugins.

Supports four task types: entities, classification, relations, json.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  deberta_gliner2_io

Request format (online POST /pooling):
    {"data": {"text": "...", "labels": [...],
              "schema": {...},
              "threshold": 0.5,
              "include_confidence": false,
              "include_spans": false},
     "model": "...", "task": "plugin"}

Schema shapes (all optional per-field thresholds fall back to request-level
``threshold`` when omitted):

    entities — list or dict:
        ["person", "org"]
        {"person": "Description", "org": ""}
        {"person": {"description": "People", "threshold": 0.9}}

    classifications — list of dicts:
        [{"task": "sentiment", "labels": ["pos", "neg"],
          "multi_label": false, "cls_threshold": 0.6}]

    relations — list or dict:
        ["works_at", "reports_to"]
        {"works_at": {"description": "Employment", "threshold": 0.25}}

    structures — dict of structure definitions:
        {"invoice": {"fields": [
            {"name": "date", "dtype": "str", "threshold": 0.8},
            {"name": "memo", "threshold": 0.2}
        ]}}

Request format (offline):
    llm.encode({"data": {"text": "...", "schema": {...}}})
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict

from transformers import AutoTokenizer
from vllm.config import VllmConfig

from plugins.deberta_gliner2.processor import (
    build_special_token_ids,
    build_tokenization_cache,
    decode_output,
    format_results,
    normalize_gliner2_schema,
    preprocess,
)
from vllm_factory.io.base import FactoryIOProcessor, PoolingRequestOutput, PromptType, TokensPrompt

logger = logging.getLogger(__name__)


@dataclass
class GLiNER2Input:
    text: str
    schema: Dict = field(default_factory=dict)
    threshold: float = 0.5
    include_confidence: bool = False
    include_spans: bool = False
    truncate_overflow_text: bool = False


class DeBERTaGLiNER2IOProcessor(FactoryIOProcessor):
    """IOProcessor for deberta_gliner2 — schema-based extraction with DeBERTa backbone.

    Data flow:
        IOProcessorRequest(data={text, schema, threshold, include_confidence, include_spans})
        → factory_parse   → GLiNER2Input (with normalized schema)
        → factory_pre_process → TokensPrompt (+ stash extra_kwargs and metadata)
        → merge_pooling_params → PoolingParams(task="plugin", extra_kwargs=...)
        → engine.encode    → PoolingRequestOutput
        → factory_post_process → dict (decoded + formatted results)
    """

    pooling_task = "plugin"

    @staticmethod
    def _text_length_bucket(token_count: int) -> str:
        if token_count <= 32:
            return "0_32"
        if token_count <= 128:
            return "33_128"
        if token_count <= 512:
            return "129_512"
        return "513_plus"

    @staticmethod
    def _task_shape(task_types: Sequence[str]) -> str:
        return ",".join(task_types) if task_types else "none"

    def _log_observability(self, request_meta: Dict[str, Any] | None) -> None:
        if not request_meta or not logger.isEnabledFor(logging.INFO):
            return

        try:
            obs = request_meta.get("_observability")
            if not isinstance(obs, dict):
                return

            request_started_at = obs.get("request_started_at")
            preprocess_elapsed_ms = obs.get("preprocess_elapsed_ms")
            schema_cache_hit = obs.get("schema_cache_hit")
            schema_count = obs.get("schema_count")
            text_token_count = obs.get("text_token_count")
            text_length_bucket = obs.get("text_length_bucket")
            task_types = obs.get("task_types")
            request_id = obs.get("request_id", "_unknown")

            if not isinstance(request_started_at, (int, float)):
                return
            if not isinstance(preprocess_elapsed_ms, (int, float)):
                return
            if not isinstance(schema_count, int):
                return
            if not isinstance(text_token_count, int):
                return
            if not isinstance(text_length_bucket, str):
                return
            if not isinstance(task_types, str):
                return

            total_elapsed_ms = (time.perf_counter() - request_started_at) * 1000.0
            logger.info(
                "GLiNER2 request complete request_id=%s preprocess_ms=%.3f total_ms=%.3f "
                "schema_cache_hit=%s schema_count=%d text_token_count=%d text_length_bucket=%s "
                "task_types=%s",
                request_id,
                preprocess_elapsed_ms,
                total_elapsed_ms,
                schema_cache_hit,
                schema_count,
                text_token_count,
                text_length_bucket,
                task_types,
            )
        except Exception:
            try:
                logger.exception("GLiNER2 observability logging failed")
            except Exception:
                return

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)

        model_id = vllm_config.model_config.model
        self._max_model_len = getattr(vllm_config.model_config, "max_model_len", None)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )
        self._tokenization_cache = build_tokenization_cache(self._tokenizer)
        self._special_token_ids = build_special_token_ids(self._tokenizer)
        self._schema_preprocess_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    @staticmethod
    def _coerce_bool(value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"'{field_name}' must be a boolean")

    # ------------------------------------------------------------------
    # FactoryIOProcessor implementation
    # ------------------------------------------------------------------

    def factory_parse(self, data: Any) -> GLiNER2Input:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError("Expected request data dict")

        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("'text' is required")

        threshold = data.get("threshold", 0.5)
        try:
            threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("'threshold' must be a number") from exc
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("'threshold' must be between 0 and 1")

        include_confidence = self._coerce_bool(
            data.get("include_confidence", False), "include_confidence"
        )
        include_spans = self._coerce_bool(data.get("include_spans", False), "include_spans")
        truncate_overflow_text = self._coerce_bool(
            data.get("truncate_overflow_text", False), "truncate_overflow_text"
        )

        raw_schema = data.get("schema")
        labels = data.get("labels")

        if raw_schema is not None:
            schema = normalize_gliner2_schema(raw_schema)
        elif labels is not None:
            schema = normalize_gliner2_schema({"entities": labels})
        else:
            raise ValueError("Request must include schema or labels")

        return GLiNER2Input(
            text=text,
            schema=schema,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            truncate_overflow_text=truncate_overflow_text,
        )

    def factory_pre_process(
        self,
        parsed_input: GLiNER2Input,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        request_started_at = time.perf_counter()
        result = preprocess(
            self._tokenizer,
            parsed_input.text,
            parsed_input.schema,
            max_model_len=self._max_model_len,
            truncate_overflow_text=parsed_input.truncate_overflow_text,
            special_token_ids=self._special_token_ids or None,
            tokenization_cache=self._tokenization_cache,
            schema_cache=self._schema_preprocess_cache,
        )
        preprocess_elapsed_ms = (time.perf_counter() - request_started_at) * 1000.0

        ids_list = result["input_ids"]

        gliner_data = {
            "mapped_indices": result["mapped_indices"],
            "schema_count": result["schema_count"],
            "special_token_ids": result["special_token_ids"],
            "token_pooling": result["token_pooling"],
            "schema_dict": result["schema_dict"],
            "task_types": result["task_types"],
            "schema_tokens_list": result["schema_tokens_list"],
            "text_tokens": result["text_tokens"],
            "original_text": result["original_text"],
            "start_mapping": result["start_mapping"],
            "end_mapping": result["end_mapping"],
            "threshold": parsed_input.threshold,
            "threshold_meta": result.get("threshold_meta"),
        }

        postprocess_meta = {
            "schema_dict": result["schema_dict"],
            "task_types": result["task_types"],
            "schema_tokens_list": result["schema_tokens_list"],
            "text_tokens": result["text_tokens"],
            "original_text": result["original_text"],
            "start_mapping": result["start_mapping"],
            "end_mapping": result["end_mapping"],
            "threshold": parsed_input.threshold,
            "include_confidence": parsed_input.include_confidence,
            "include_spans": parsed_input.include_spans,
            "_observability": {
                "request_id": request_id or "_offline",
                "request_started_at": request_started_at,
                "preprocess_elapsed_ms": preprocess_elapsed_ms,
                "schema_cache_hit": result["schema_cache_hit"],
                "schema_count": result["schema_count"],
                "text_token_count": len(result["text_tokens"]),
                "text_length_bucket": self._text_length_bucket(len(result["text_tokens"])),
                "task_types": self._task_shape(result["task_types"]),
            },
        }

        self._stash(extra_kwargs=gliner_data, request_id=request_id, meta=postprocess_meta)

        return TokensPrompt(prompt_token_ids=ids_list)

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: Any,
    ) -> Dict:
        try:
            if not model_output or request_meta is None:
                return {}

            output = model_output[0]
            raw = output.outputs.data
            if raw is None:
                return {}

            results = decode_output(
                raw,
                schema=request_meta["schema_dict"],
                task_types=request_meta["task_types"],
            )

            return format_results(
                results,
                threshold=request_meta.get("threshold", 0.5),
                include_confidence=request_meta.get("include_confidence", False),
                include_spans=request_meta.get("include_spans", False),
            )
        finally:
            self._log_observability(request_meta)


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.deberta_gliner2.io_processor.DeBERTaGLiNER2IOProcessor"
