"""IO processor for deberta_gliner25 — same HTTP shape as the span plugin."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict

from transformers import AutoTokenizer
from vllm.config import VllmConfig

from gliner2.inference.runtime import format_results as gliner2_format_results

from plugins.deberta_gliner2.processor import normalize_gliner2_schema
from plugins.deberta_gliner25.processor import (
    boundary_transformer,
    collate_word_cap,
    decode_boundary_output,
    iter_boundary_items,
    preprocess_boundary,
    schema_format_args,
)
from vllm_factory.io.base import (
    FactoryIOProcessor,
    PoolingRequestOutput,
    PromptType,
    TokensPrompt,
)

logger = logging.getLogger(__name__)

_ADAPTER_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-:/]{1,128}$")


@dataclass
class GLiNER25Input:
    text: str
    schema: Dict = field(default_factory=dict)
    threshold: float = 0.5
    include_confidence: bool = False
    include_spans: bool = False
    truncate_overflow_text: bool = False
    adapter: str | None = None


class DeBERTaGLiNER25IOProcessor(FactoryIOProcessor):
    """Boundary IO processor. HTTP contract matches deberta_gliner2."""

    pooling_task = "plugin"

    def __init__(self, vllm_config: VllmConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        model_id = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )
        self._transformer = boundary_transformer(self._tokenizer)
        raw_max = getattr(vllm_config.model_config, "max_model_len", None)
        self._max_model_len = int(raw_max) if raw_max else None
        self._word_cap = collate_word_cap(self._max_model_len)

    @staticmethod
    def _coerce_bool(value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"'{field_name}' must be a boolean")

    @staticmethod
    def _coerce_adapter(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("'adapter' must be a string or null")
        stripped = value.strip()
        if not stripped:
            return None
        if not _ADAPTER_NAME_RE.match(stripped):
            raise ValueError(
                f"'adapter' must match ^[A-Za-z0-9_.\\-:/]{{1,128}}$ — got {value!r}"
            )
        return stripped

    def _parse_one(self, data: dict[str, Any]) -> GLiNER25Input:
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
        raw_schema = data.get("schema")
        labels = data.get("labels")
        if raw_schema is not None:
            schema = normalize_gliner2_schema(raw_schema)
        elif labels is not None:
            schema = normalize_gliner2_schema({"entities": labels})
        else:
            raise ValueError("Request must include schema or labels")
        return GLiNER25Input(
            text=text,
            schema=schema,
            threshold=threshold,
            include_confidence=self._coerce_bool(
                data.get("include_confidence", False), "include_confidence"
            ),
            include_spans=self._coerce_bool(
                data.get("include_spans", False), "include_spans"
            ),
            truncate_overflow_text=self._coerce_bool(
                data.get("truncate_overflow_text", False), "truncate_overflow_text"
            ),
            adapter=self._coerce_adapter(data.get("adapter")),
        )

    def factory_parse(self, data: Any) -> GLiNER25Input | list[GLiNER25Input]:
        """Validate a request body holding one item or a batch of them.

        Args:
            data: Raw ``/pooling`` data field: one item, ``text: list``, or a
                list of items.

        Returns:
            One parsed input, or a list of them for a batch. The shape decides
            whether the rest of the pipeline runs single or batched.
        """
        items = iter_boundary_items(data)
        parsed = [self._parse_one(item) for item in items]
        if len(parsed) == 1:
            return parsed[0]
        return parsed

    def _pre_one(
        self, parsed_input: GLiNER25Input
    ) -> tuple[TokensPrompt, dict[str, Any], dict[str, Any]]:
        """Collate one parsed request into a prompt, extras and decode meta.

        Args:
            parsed_input: One validated request item.

        Returns:
            The prompt to schedule, the pooler's compact extras, and the
            metadata ``factory_post_process`` decodes that sequence with.
        """
        result = preprocess_boundary(
            self._tokenizer,
            parsed_input.text,
            parsed_input.schema,
            threshold=parsed_input.threshold,
            include_confidence=parsed_input.include_confidence,
            include_spans=parsed_input.include_spans,
            truncate_overflow_text=parsed_input.truncate_overflow_text,
            word_cap=self._word_cap,
            max_model_len=self._max_model_len,
            transformer=self._transformer,
        )
        postprocess_meta = {
            "schema_dict": parsed_input.schema,
            "threshold": parsed_input.threshold,
            "include_confidence": parsed_input.include_confidence,
            "include_spans": parsed_input.include_spans,
            "adapter": parsed_input.adapter,
        }
        prompt = TokensPrompt(prompt_token_ids=result["input_ids"])
        return prompt, result["extra_kwargs"], postprocess_meta

    def factory_pre_process(
        self,
        parsed_input: GLiNER25Input | list[GLiNER25Input],
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        """Collate every item and stash its extras and decode metadata.

        A batch stashes extras under ``_per_seq`` so each scheduled prompt can
        be given its own ``PoolingParams``; one shared params object would make
        every sequence decode against the first item's schema.

        Args:
            parsed_input: One parsed input, or a batch of them.
            request_id: vLLM request id used to key the stash.

        Returns:
            One prompt, or one prompt per batch item in the same order.
        """
        items = parsed_input if isinstance(parsed_input, list) else [parsed_input]
        prompts: list[TokensPrompt] = []
        extras: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []
        for item in items:
            prompt, extra, meta = self._pre_one(item)
            prompts.append(prompt)
            extras.append(extra)
            metas.append(meta)
        if len(prompts) == 1:
            self._stash(extra_kwargs=extras[0], request_id=request_id, meta=metas[0])
            return prompts[0]
        self._stash(
            extra_kwargs={"_per_seq": extras},
            request_id=request_id,
            meta=metas,
        )
        return prompts

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: Any,
    ) -> Dict | list[Dict]:
        """Decode each pooled output with the metadata of its own sequence.

        Args:
            model_output: Pooler outputs, one per scheduled prompt.
            request_meta: The metadata stashed by ``factory_pre_process``: one
                dict, or a list of them for a batch.

        Returns:
            One formatted result, or a list of them for a batch, in request
            order.
        """
        if not model_output or request_meta is None:
            return [] if isinstance(request_meta, list) else {}
        metas = request_meta if isinstance(request_meta, list) else [request_meta]
        results: list[Dict] = []
        for output, meta in zip(model_output, metas, strict=True):
            raw = output.outputs.data
            if raw is None:
                results.append({})
                continue
            schema = meta.get("schema_dict") or {}
            decoded = decode_boundary_output(raw, schema)
            requested_relations, classification_tasks = schema_format_args(schema)
            formatted = gliner2_format_results(
                decoded,
                include_confidence=meta.get("include_confidence", False),
                requested_relations=requested_relations,
                classification_tasks=classification_tasks,
            )
            if isinstance(formatted, dict) and meta.get("adapter") is not None:
                formatted.setdefault("adapter", meta["adapter"])
            results.append(formatted if isinstance(formatted, dict) else {})
        if len(results) == 1:
            return results[0]
        return results


def get_processor_cls() -> str:
    return "plugins.deberta_gliner25.io_processor.DeBERTaGLiNER25IOProcessor"
