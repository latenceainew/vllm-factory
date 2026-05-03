"""
IOProcessor plugin for moderncolbert — ColBERT multi-vector embeddings via
vLLM's native IOProcessor pipeline.

Handles text queries and document inputs with [Q]/[D] prefix insertion,
returning multi-vector embeddings as base64-encoded flattened arrays.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  moderncolbert_io

Request format (online POST /pooling) — single text:
    Query: {"data": {"text": "What is ML?", "is_query": true}, "model": "...", "task": "plugin"}
    Doc:   {"data": {"text": "ML is ...", "is_query": false}, "model": "...", "task": "plugin"}

Request format — batched (preferred for high-throughput callers):
    {"data": {"text": ["q1", "q2", ...],
              "is_query": [true, true, ...]},
     "model": "...", "task": "plugin"}

The batched form decomposes to N prompts in a single ``factory_pre_process``
call so vLLM's continuous batcher fuses them into one engine step and the
HTTP-side overhead drops from O(N) round trips to O(1).

Request format (offline):
    llm.encode({"data": {"text": "What is ML?", "is_query": true}})
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, List

import torch
from transformers import AutoTokenizer
from vllm.config import VllmConfig

from vllm_factory.io.base import (
    FactoryIOProcessor,
    PoolingRequestOutput,
    PromptType,
    TokensPrompt,
)

QUERY_PREFIX_ID = 50368  # [Q] with trailing space
DOC_PREFIX_ID = 50369  # [D] with trailing space


@dataclass
class ModernColBERTInput:
    """Validated embedding request after :meth:`factory_parse`.

    A single call always carries one or more ``texts`` (size 1 for the
    legacy single-text request shape). The ``is_query_per_text`` list is
    either user-provided or broadcast from the request-level ``is_query``
    flag so each prompt picks the correct ``[Q]`` / ``[D]`` prefix.
    """

    texts: List[str] = field(default_factory=list)
    is_query_per_text: List[bool] = field(default_factory=list)
    batched: bool = False


class ModernColBERTIOProcessor(FactoryIOProcessor):
    """IOProcessor for ModernColBERT — multi-vector late-interaction embeddings.

    Data flow:
        IOProcessorRequest(data={text, is_query})
        → factory_parse        → ModernColBERTInput (list of N >= 1 texts)
        → factory_pre_process  → Sequence[TokensPrompt] (with [Q]/[D] prefix per text)
        → merge_pooling_params → PoolingParams(task="plugin")
        → engine.encode        → Sequence[PoolingRequestOutput]
        → factory_post_process → base64 string (single) OR list[str] (batched)
    """

    pooling_task = "token_embed"

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)

        model_id = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )

    def factory_parse(self, data: Any) -> ModernColBERTInput:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'text' key, got {type(data)}")

        if "text" not in data:
            raise ValueError("Request data must contain a 'text' key")

        text_field = data["text"]
        is_query_field = data.get("is_query", True)

        # Detect batched vs single shape. We treat "list/tuple of strings"
        # as batched and "string" as single; lists with mixed types are an
        # explicit error so callers don't silently mis-pack their batch.
        if isinstance(text_field, (list, tuple)):
            texts = [str(t) for t in text_field]
            if not all(isinstance(t, str) for t in text_field):
                raise ValueError("All elements of 'text' must be strings when batched")
            if isinstance(is_query_field, (list, tuple)):
                if len(is_query_field) != len(texts):
                    raise ValueError(
                        "'is_query' list length must match 'text' length when both are lists"
                    )
                is_query_per_text = [bool(x) for x in is_query_field]
            else:
                is_query_per_text = [bool(is_query_field)] * len(texts)
            batched = True
        else:
            texts = [str(text_field)]
            if isinstance(is_query_field, (list, tuple)):
                if len(is_query_field) != 1:
                    raise ValueError(
                        "'is_query' list must have length 1 when 'text' is a single string"
                    )
                is_query_per_text = [bool(is_query_field[0])]
            else:
                is_query_per_text = [bool(is_query_field)]
            batched = False

        if not texts:
            raise ValueError("Empty 'text' batch")

        return ModernColBERTInput(
            texts=texts,
            is_query_per_text=is_query_per_text,
            batched=batched,
        )

    def factory_pre_process(
        self,
        parsed_input: ModernColBERTInput,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        prompts: list[TokensPrompt] = []
        for text, is_query in zip(parsed_input.texts, parsed_input.is_query_per_text):
            max_len = 256 if is_query else 8192
            prefix_id = QUERY_PREFIX_ID if is_query else DOC_PREFIX_ID

            tokens = self._tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_len - 1,
                padding=False,
                return_tensors=None,
            )
            ids = list(tokens["input_ids"])
            input_ids = [int(ids[0]), int(prefix_id), *[int(t) for t in ids[1:]]]
            prompts.append(TokensPrompt(prompt_token_ids=input_ids))

        # Stash the batched flag so factory_post_process knows whether to
        # return a single string (legacy shape) or a list (batched shape).
        self._stash(
            request_id=request_id,
            meta={"batched": parsed_input.batched, "n": len(prompts)},
        )

        if not parsed_input.batched and len(prompts) == 1:
            return prompts[0]
        return prompts

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: Any,
    ) -> Any:
        import base64

        if not model_output:
            return [] if (request_meta or {}).get("batched") else ""

        encoded: list[str] = []
        for output in model_output:
            raw = output.outputs.data
            if raw is None:
                encoded.append("")
                continue
            if not isinstance(raw, torch.Tensor):
                raw = torch.as_tensor(raw)
            encoded.append(
                base64.b64encode(raw.cpu().contiguous().to(torch.float32).numpy().tobytes()).decode(
                    "ascii"
                )
            )

        if (request_meta or {}).get("batched") or len(encoded) > 1:
            return encoded
        return encoded[0]


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.moderncolbert.io_processor.ModernColBERTIOProcessor"
