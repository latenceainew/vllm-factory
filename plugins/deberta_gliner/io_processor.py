"""
IOProcessor plugin for deberta_gliner — server-side GLiNER NER via vLLM's
native IOProcessor pipeline.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  deberta_gliner_io

Request format (online POST /pooling):
    {"data": {"text": "...", "labels": ["person", "org"], "threshold": 0.5},
     "model": "...", "task": "plugin"}

Request format (offline):
    llm.encode({"data": {"text": "...", "labels": [...]}})
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer
from vllm.config import VllmConfig

from forge.gliner_postprocessor import GLiNERDecoder, get_final_entities
from forge.gliner_preprocessor import GLiNERPreprocessor
from vllm_factory.io.base import FactoryIOProcessor, PoolingRequestOutput, PromptType, TokensPrompt


@dataclass
class GLiNERInput:
    """Validated NER request after parse_request."""

    text: str
    labels: list[str]
    threshold: float = 0.5
    flat_ner: bool = False
    multi_label: bool = False


class DeBERTaGLiNERIOProcessor(FactoryIOProcessor):
    """IOProcessor for deberta_gliner — GLiNER NER with DeBERTa backbone.

    Data flow:
        IOProcessorRequest(data={text, labels, ...})
        → factory_parse   → GLiNERInput
        → factory_pre_process → TokensPrompt (+ stash extra_kwargs and metadata)
        → merge_pooling_params → PoolingParams(task="plugin", extra_kwargs=gliner_data)
        → engine.encode    → PoolingRequestOutput
        → factory_post_process → list[dict] (decoded entities)
    """

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        from plugins.deberta_gliner_linker.vllm_pooling_attention_mask import (
            apply_pooling_attention_mask_patch,
        )

        apply_pooling_attention_mask_patch()

        model_id = vllm_config.model_config.model
        config = vllm_config.model_config.hf_config
        self._is_token_level = config.span_mode == "token_level"
        if self._is_token_level:
            from gliner import GLiNER
            from gliner.data_processing.collator import TokenDataCollator

            gliner = GLiNER.from_pretrained(model_id)
            data_processor = gliner.data_processor
            self._words_splitter = data_processor.words_splitter
            self._token_decoder = gliner.decoder
            self._token_collator = TokenDataCollator(
                gliner.config,
                data_processor=data_processor,
                return_tokens=True,
                return_id_to_classes=True,
                prepare_labels=False,
            )
            del gliner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                trust_remote_code=True,
            )
            self._preprocessor = GLiNERPreprocessor(
                underlying_tokenizer=self._tokenizer,
                config=config,
                device="cpu",
                include_attention_mask=True,
            )
            self._decoder = GLiNERDecoder()

    # ------------------------------------------------------------------
    # FactoryIOProcessor implementation
    # ------------------------------------------------------------------

    def factory_parse(self, data: Any) -> GLiNERInput:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'text' and 'labels' keys, got {type(data)}")

        labels = data.get("labels", [])
        if not labels:
            raise ValueError("'labels' list must not be empty")

        return GLiNERInput(
            text=data.get("text", ""),
            labels=labels,
            threshold=float(data.get("threshold", 0.5)),
            flat_ner=bool(data.get("flat_ner", False)),
            multi_label=bool(data.get("multi_label", False)),
        )

    def factory_pre_process(
        self,
        parsed_input: GLiNERInput,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        if self._is_token_level:
            return self._token_level_pre_process(parsed_input, request_id)

        result = self._preprocessor(parsed_input.text, parsed_input.labels, device="cpu")
        enc = result["model_inputs"]
        meta = result["postprocessing_metadata"]

        input_ids = enc["input_ids"][0]
        words_mask = enc["words_mask"][0]
        text_lengths = enc["text_lengths"][0].item()

        ids_list = input_ids.tolist()
        mask_list = words_mask.tolist()
        attn_list = enc["attention_mask"][0].tolist()

        gliner_data = {
            "input_ids": ids_list,
            "words_mask": mask_list,
            "text_lengths": text_lengths,
            "attention_mask": attn_list,
            "span_idx": enc["span_idx"][0].tolist(),
            "span_mask": enc["span_mask"][0].tolist(),
        }

        postprocess_meta = {
            "text": parsed_input.text,
            "labels": parsed_input.labels,
            "threshold": parsed_input.threshold,
            "flat_ner": parsed_input.flat_ner,
            "multi_label": parsed_input.multi_label,
            "tokens": meta["tokens"],
            "word_positions": meta["word_positions"],
            "id_to_classes": meta["id_to_classes"],
        }

        self._stash(extra_kwargs=gliner_data, request_id=request_id, meta=postprocess_meta)

        return TokensPrompt(prompt_token_ids=ids_list)

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: Any,
    ) -> list[dict[str, Any]]:
        if not model_output or request_meta is None:
            return []

        output = model_output[0]
        raw = output.outputs.data
        if raw is None:
            return []

        scores = torch.as_tensor(raw) if not isinstance(raw, torch.Tensor) else raw

        if request_meta.get("token_level"):
            return self._token_level_post_process(scores, request_meta)

        if scores.dim() == 1 and scores.numel() > 3:
            L = int(scores[0].item())
            K = int(scores[1].item())
            C = int(scores[2].item())
            logits = scores[3:].reshape(1, L, K, C)
        elif scores.dim() == 3:
            logits = scores.unsqueeze(0)
        else:
            return []

        decoded = self._decoder.decode(
            tokens=request_meta["tokens"],
            id_to_classes=request_meta["id_to_classes"],
            logits=logits,
            flat_ner=request_meta.get("flat_ner", False),
            threshold=request_meta.get("threshold", 0.5),
            multi_label=request_meta.get("multi_label", False),
        )

        entities_batch = get_final_entities(
            decoded_outputs=decoded,
            word_positions=request_meta["word_positions"],
            original_texts=[request_meta["text"]],
        )

        return entities_batch[0] if entities_batch else []

    def _token_level_pre_process(
        self,
        parsed_input: GLiNERInput,
        request_id: str | None,
    ) -> PromptType:
        words: list[str] = []
        word_starts: list[int] = []
        word_ends: list[int] = []
        for token, start, end in self._words_splitter(parsed_input.text):
            words.append(token)
            word_starts.append(start)
            word_ends.append(end)

        batch = self._token_collator(
            [{"tokenized_text": words, "ner": None}],
            entity_types=parsed_input.labels,
        )

        input_ids = batch["input_ids"][0].detach().cpu()
        words_mask = batch["words_mask"][0].detach().cpu()
        attention_mask = batch["attention_mask"][0].detach().cpu()
        text_lengths = batch["text_lengths"]
        if text_lengths.dim() == 2:
            text_length = int(text_lengths[0, 0].item())
        else:
            text_length = int(text_lengths[0].item())

        ids_list = input_ids.tolist()
        postprocess_meta = {
            "token_level": True,
            "text": parsed_input.text,
            "words": words,
            "word_starts": word_starts,
            "word_ends": word_ends,
            "labels": parsed_input.labels,
            "threshold": parsed_input.threshold,
            "flat_ner": parsed_input.flat_ner,
            "multi_label": parsed_input.multi_label,
        }
        self._stash(
            extra_kwargs={
                "input_ids": ids_list,
                "words_mask": words_mask.tolist(),
                "attention_mask": attention_mask.tolist(),
                "text_lengths": text_length,
            },
            request_id=request_id,
            meta=postprocess_meta,
        )

        return TokensPrompt(prompt_token_ids=ids_list)

    def _token_level_post_process(
        self,
        scores: torch.Tensor,
        request_meta: Any,
    ) -> list[dict[str, Any]]:
        if scores.dim() == 1 and scores.numel() > 3:
            word_count = int(scores[0].item())
            class_count = int(scores[1].item())
            score_dim = int(scores[2].item())
            logits = scores[3:].reshape(1, word_count, class_count, score_dim)
        elif scores.dim() == 3:
            logits = scores.unsqueeze(0)
        else:
            return []

        id_to_classes = {i + 1: label for i, label in enumerate(request_meta["labels"])}
        spans = self._token_decoder.decode(
            tokens=[request_meta["words"]],
            id_to_classes=id_to_classes,
            model_output=logits,
            flat_ner=request_meta.get("flat_ner", False),
            threshold=request_meta.get("threshold", 0.5),
            multi_label=request_meta.get("multi_label", False),
        )

        source_text = request_meta["text"]
        entities: list[dict[str, Any]] = []
        for span in spans[0]:
            word_start = span.start
            word_end = span.end
            char_start = (
                request_meta["word_starts"][word_start]
                if word_start < len(request_meta["word_starts"])
                else 0
            )
            char_end = (
                request_meta["word_ends"][word_end]
                if word_end < len(request_meta["word_ends"])
                else len(source_text)
            )
            entities.append(
                {
                    "start": char_start,
                    "end": char_end,
                    "text": source_text[char_start:char_end],
                    "label": span.entity_type,
                    "score": round(span.score, 4),
                }
            )
        return entities


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.deberta_gliner.io_processor.DeBERTaGLiNERIOProcessor"
