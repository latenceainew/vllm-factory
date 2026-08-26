"""GLiNER 2.5 boundary pooler.

Holds real ``gliner2`` head modules (never a copy) under the checkpoint
prefixes ``boundary_head`` / ``record_decoder`` / ``relation_scorer`` /
``classifier``. vLLM hidden states replace the encoder call; decode is
delegated to ``BoundaryExtractor._extract_from_batch``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import fields
from typing import Any

import torch
import torch.nn as nn

from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states

logger = logging.getLogger(__name__)


def _require(module: str, purpose: str):
    from vllm_factory.optional_deps import require

    return require(module, "gliner2", purpose=purpose)


def _decode_key(extra: dict[str, Any]) -> tuple[bool, bool, float, int]:
    """Settings that must match for sequences to decode in one collate."""
    return (
        bool(extra.get("include_confidence", False)),
        bool(extra.get("include_spans", False)),
        float(extra.get("threshold", 0.5)),
        int(extra.get("max_len") or 0),
    )


def _can_batch_compact(extras: list[dict[str, Any]]) -> bool:
    """True when every extra is compact and shares one decode key.

    Args:
        extras: Per-sequence extras in scheduled order.

    Returns:
        Whether the batch can be re-collated in one call. ``_extract_from_batch``
        takes a single threshold and flag set, and the word cap has to match or
        the re-collated rows would not line up with the encoded tokens.
    """
    if not extras or any(not extra for extra in extras):
        return False
    from plugins.deberta_gliner25.processor import is_compact_extra

    if not all(is_compact_extra(extra) for extra in extras):
        return False
    first = _decode_key(extras[0])
    return all(_decode_key(extra) == first for extra in extras)


class GLiNER25BoundaryPooler(nn.Module):
    """Boundary heads + serving split/pack. Does not subclass gliner2 types."""

    def __init__(
        self,
        hidden_size: int,
        boundary_head: dict[str, Any] | None = None,
        tokenizer_name: str | None = None,
        max_model_len: int | None = None,
    ):
        super().__init__()
        cfg = dict(boundary_head or {})
        layers = _require("gliner2.layers", "GLiNER25 classifier")
        model_mod = _require("gliner2.models.boundary.model", "GLiNER25 boundary head")
        config_mod = _require("gliner2.configuration", "GLiNER25 boundary settings")
        records_mod = _require("gliner2.models.boundary.records", "GLiNER25 record decoder")
        relations_mod = _require("gliner2.models.boundary.relations", "GLiNER25 relation scorer")

        settings_cls = config_mod.BoundaryHeadSettings
        allowed = {item.name for item in fields(settings_cls)}
        self.boundary_settings = settings_cls(
            **{key: value for key, value in cfg.items() if key in allowed}
        )
        self.hidden_size = hidden_size
        self.enable_records = bool(self.boundary_settings.enable_records)
        self.enable_relations = bool(self.boundary_settings.enable_relations)
        self._tokenizer_name = tokenizer_name
        self._max_model_len = max_model_len
        self._tokenizer = None
        self._transformer = None

        self.classifier = layers.create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size * 2],
            output_dim=1,
            dropout=cfg.get("dropout", 0.1),
            activation="relu",
            add_layer_norm=False,
        )
        self.boundary_head = model_mod.BoundaryHead(
            hidden_size,
            self.boundary_settings,
            query_dim=hidden_size,
            build_candidate_states=self.enable_records,
        )
        if self.enable_records:
            self.record_decoder = records_mod.RecordHead(
                hidden_size,
                self.boundary_settings.record_dim,
                self.boundary_settings.record_instance_queries,
            )
        if self.enable_relations:
            self.relation_pair_generator = relations_mod.TypedRelationPairGenerator(
                relations_mod.RelationProposalSettings(
                    heads_per_relation=self.boundary_settings.relation_heads_per_type,
                    tails_per_relation=self.boundary_settings.relation_tails_per_type,
                    pair_cap=self.boundary_settings.relation_pair_cap,
                    argument_threshold=self.boundary_settings.relation_argument_proposal_threshold,
                )
            )
            self.relation_scorer = relations_mod.SparseRelationScorer(
                hidden_size,
                dropout=self.boundary_settings.dropout,
                relation_query_dim=(
                    2 * hidden_size
                    if self.boundary_settings.directional_relation_states
                    else hidden_size
                ),
                use_biaffine_content=self.boundary_settings.relation_biaffine_content,
            )

    def _get_transformer(self) -> Any:
        if self._transformer is None:
            if not self._tokenizer_name:
                raise RuntimeError(
                    "GLiNER25 pooler has no tokenizer_name; cannot reconstruct compact extras"
                )
            from transformers import AutoTokenizer

            from plugins.deberta_gliner25.processor import boundary_transformer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name, use_fast=True, trust_remote_code=True
            )
            self._transformer = boundary_transformer(self._tokenizer)
        return self._transformer

    def get_tasks(self) -> set[str]:
        return {"embed", "classify", "plugin"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        """Decode every sequence in one scheduled batch.

        Args:
            hidden_states: Concatenated encoder output for the whole batch.
            ctx: Scheduled batch context — sequence lengths and per-prompt
                extras, in the same order.

        Returns:
            One packed JSON payload per scheduled sequence. The count is the
            contract: a batch holds several callers, so returning fewer pairs
            results with the wrong requests.
        """
        extras = list(ctx.extra_kwargs)
        device = hidden_states.device
        try:
            sequences = split_hidden_states(hidden_states, ctx.seq_lengths)
        except (IndexError, TypeError, RuntimeError):
            logger.exception("[GLiNER25] cannot split hidden states; returning empty")
            return _empty_payloads(len(extras) or 1, device)

        extras = extras[: len(sequences)]
        extras.extend({} for _ in range(len(sequences) - len(extras)))
        if _can_batch_compact(extras):
            return self._process_batch(sequences, extras)

        outputs: list[torch.Tensor | None] = []
        for token_embs, extra in zip(sequences, extras, strict=True):
            if not extra:
                outputs.append(_pack_json({}, token_embs.device))
                continue
            outputs.append(self._process_one(token_embs, extra))
        return outputs

    def _process_batch(
        self,
        sequences: list[torch.Tensor],
        extras: list[dict[str, Any]],
    ) -> list[torch.Tensor | None]:
        """Decode a whole scheduled batch through one collate and one head pass.

        Args:
            sequences: Per-sequence hidden states, in scheduled order.
            extras: Compact extras sharing one decode key, same order.

        Returns:
            One packed JSON payload per sequence. Falls back to per-sequence
            decode if the re-collated rows do not match the encoded lengths,
            which would otherwise gather the wrong words.
        """
        device = sequences[0].device
        dtype = sequences[0].dtype
        batch = self._collate_compact(extras).to(device, dtype=dtype)
        orig_lens = [int(x) for x in (getattr(batch, "original_lengths", None) or [])]
        seq_lens = [int(seq.shape[0]) for seq in sequences]
        if orig_lens and orig_lens != seq_lens:
            logger.warning(
                "[GLiNER25] collate lengths %s != encoder lengths %s; per-seq fallback",
                orig_lens,
                seq_lens,
            )
            return [self._process_one(seq, extra) for seq, extra in zip(sequences, extras)]
        max_t = max(seq_lens)
        padded = sequences[0].new_zeros(len(sequences), max_t, sequences[0].shape[-1])
        for i, seq in enumerate(sequences):
            padded[i, : seq.shape[0]] = seq
        core = self._core_from_hidden(padded, batch)
        host = self._decode_host(core)
        threshold = float(extras[0].get("threshold", 0.5))
        include_confidence = bool(extras[0].get("include_confidence", False))
        include_spans = bool(extras[0].get("include_spans", False))
        metadata = [{} for _ in extras]
        samples = host._extract_from_batch(
            batch, threshold, metadata, include_confidence, include_spans
        )
        return [_pack_json(sample if sample else {}, device) for sample in samples]

    def _collate_compact(self, extras: list[dict[str, Any]]) -> Any:
        from plugins.deberta_gliner25.processor import collate_compact_extras

        return collate_compact_extras(self._tokenizer, extras, transformer=self._get_transformer())

    def _process_one(self, token_embs: torch.Tensor, extra: dict[str, Any]) -> torch.Tensor:
        """Decode one sequence.

        Args:
            token_embs: Hidden states for this sequence, shape (tokens, hidden).
            extra: This sequence's compact extras.

        Returns:
            The packed JSON payload for this sequence.

        Raises:
            ValueError: ``extra`` is not compact, which means the sequence was
                handed the whole request's ``_per_seq`` payload instead of its
                own slice.
        """
        from plugins.deberta_gliner25.processor import is_compact_extra

        if not is_compact_extra(extra):
            raise ValueError(f"expected compact extra_kwargs, got keys {sorted(extra)[:8]}")
        batch = self._collate_compact([extra]).to(token_embs.device, dtype=token_embs.dtype)
        core = self._core_from_hidden(token_embs.unsqueeze(0), batch)
        host = self._decode_host(core)
        samples = host._extract_from_batch(
            batch,
            float(extra.get("threshold", 0.5)),
            [{}],
            bool(extra.get("include_confidence", False)),
            bool(extra.get("include_spans", False)),
        )
        sample = samples[0] if samples else {}
        return _pack_json(sample, token_embs.device)

    def _decode_host(self, core: dict[str, Any]) -> Any:
        engine = _require("gliner2.models.boundary.engine", "GLiNER25 decode host")
        # ``object.__new__`` skips ``nn.Module.__init__``, so attribute
        # assignment must not go through ``Module.__setattr__`` (it raises
        # ``cannot assign module before Module.__init__() call``).
        host = object.__new__(engine.BoundaryExtractor)
        for name, value in (
            ("boundary_head", self.boundary_head),
            ("boundary_settings", self.boundary_settings),
            ("classifier", self.classifier),
            ("enable_records", self.enable_records),
            ("enable_relations", self.enable_relations),
            ("record_decoder", getattr(self, "record_decoder", None)),
            ("relation_scorer", getattr(self, "relation_scorer", None)),
            ("relation_pair_generator", getattr(self, "relation_pair_generator", None)),
            ("hidden_size", self.hidden_size),
            ("strict_extraction", True),
            ("_encode_core", lambda _batch: core),
        ):
            object.__setattr__(host, name, value)
        return host

    def _core_from_hidden(
        self, token_embeddings: torch.Tensor, batch: Any
    ) -> dict[str, Any]:
        """Gather word/query/cls states from vLLM hidden states (fast routing)."""
        h = token_embeddings.shape[-1]

        def gather_routed(indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            safe_idx = indices.clamp(0, token_embeddings.shape[1] - 1)
            states = token_embeddings.gather(1, safe_idx.unsqueeze(-1).expand(-1, -1, h))
            return states * mask.unsqueeze(-1).to(states.dtype)

        text_states = gather_routed(batch.text_word_indices, batch.text_word_mask)
        query_states = gather_routed(batch.query_marker_indices, batch.query_marker_mask)
        text_mask = batch.text_word_mask
        query_mask = batch.query_marker_mask
        cls_states = gather_routed(batch.cls_marker_indices, batch.cls_marker_mask)

        ext_specs: list[list[dict[str, Any]]] = []
        cls_specs: list[list[dict[str, Any]]] = []
        rel_specs: list[list[dict[str, Any]]] = []
        word_offsets: list[int] = []
        relations_mod = _require("gliner2.models.boundary.relations", "relation specs")
        relation_spec_cls = relations_mod.RelationTypeSpec

        for i in range(len(batch)):
            layout = batch.query_layouts[i]
            specs_i = [
                {
                    "group_index": spec.task_index,
                    "field_index": spec.role_index,
                    "task_type": spec.task_type,
                    "task_name": spec.task_name,
                    "field_name": spec.role_name,
                }
                for spec in layout.queries
            ]
            ext_specs.append(specs_i)
            text_len_i = (
                len(batch.start_mappings[i])
                if batch.start_mappings
                else int(batch.text_word_counts[i])
            )
            word_offsets.append(max(int(batch.text_word_counts[i]) - text_len_i, 0))

            cls_i: list[dict[str, Any]] = []
            cls_offset = 0
            for group_index in range(batch.schema_counts[i]):
                if batch.task_types[i][group_index] != "classifications":
                    continue
                choice_count = max(len(batch.schema_special_indices[i][group_index]) - 1, 0)
                if choice_count:
                    schema_tokens = batch.schema_tokens_list[i][group_index]
                    cls_i.append(
                        {
                            "group_index": group_index,
                            "task_name": schema_tokens[2] if len(schema_tokens) > 2 else "",
                            "schema_tokens": schema_tokens,
                            "choice_states": cls_states[i, cls_offset : cls_offset + choice_count],
                            "group_embs": torch.cat(
                                (
                                    cls_states.new_zeros((1, h)),
                                    cls_states[i, cls_offset : cls_offset + choice_count],
                                )
                            ),
                        }
                    )
                cls_offset += choice_count
            cls_specs.append(cls_i)

            rel_i: list[dict[str, Any]] = []
            groups: dict[int, list[int]] = {}
            for query_id, spec in enumerate(specs_i):
                groups.setdefault(spec["group_index"], []).append(query_id)
            for group_index, role_ids_list in groups.items():
                if batch.task_types[i][group_index] != "relations":
                    continue
                role_ids = tuple(role_ids_list)
                if len(role_ids) < 2:
                    continue
                head_id, tail_id = role_ids[:2]
                max_q = query_states.shape[1] - 1
                head_id = min(head_id, max_q)
                tail_id = min(tail_id, max_q)
                role_states = query_states[i, [head_id, tail_id]]
                relation_state = (
                    torch.cat((role_states[0], role_states[1]), dim=-1)
                    if self.boundary_settings.directional_relation_states
                    else role_states.mean(dim=0)
                )
                rel_i.append(
                    {
                        "group_index": group_index,
                        "relation_type": specs_i[head_id]["task_name"],
                        "spec": relation_spec_cls(
                            specs_i[head_id]["task_name"],
                            head_query_ids=(head_id,),
                            tail_query_ids=(tail_id,),
                        ),
                        "query_state": relation_state,
                    }
                )
            rel_specs.append(rel_i)

        return {
            "text_states": text_states,
            "text_mask": text_mask,
            "text_lengths": text_mask.sum(-1).long(),
            "query_states": query_states,
            "query_mask": query_mask,
            "ext_specs": ext_specs,
            "cls_specs": cls_specs,
            "rel_specs": rel_specs,
            "word_offsets": word_offsets,
        }


def _pack_json(sample: dict[str, Any], device: torch.device) -> torch.Tensor:
    payload = json.dumps(sample, default=str).encode("utf-8")
    values = [float(len(payload)), *[float(byte) for byte in payload]]
    return torch.tensor(values, device=device, dtype=torch.float32)


def _empty_payloads(count: int, device: torch.device) -> list[torch.Tensor | None]:
    """``count`` decodable empty results, one per sequence in the batch."""
    return [_pack_json({}, device) for _ in range(count)]
