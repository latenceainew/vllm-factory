# GLiNER2 Pooler — Schema-based multi-task information extraction.
#
# Head modules come from gliner2.layers (optional extra). This file owns
# serving: splitting vLLM hidden states, extra_kwargs plumbing, and packing.

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List

import torch
import torch.nn as nn

from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states


def _gliner2_layers():
    """Import gliner2.layers; raises with the extra hint if it is missing."""
    from vllm_factory.optional_deps import require

    return require("gliner2.layers", "gliner2", purpose="GLiNER2 pooling")


# ==================================================================
# Main GLiNER2 Pooler
# ==================================================================

class GLiNER2Pooler(nn.Module):
    """GLiNER2 pooler: SpanRepLayer + CountLSTM + classifier + count_pred.

    This handles the head computation (everything after the encoder backbone).
    """


    def __init__(self, hidden_size: int, max_width: int = 8,
                 counting_layer: str = "count_lstm"):
        super().__init__()
        layers = _gliner2_layers()
        self.hidden_size = hidden_size
        self.max_width = max_width
        self.counting_layer = counting_layer

        # gliner2.layers.SpanRepLayer(hidden_size, max_width, span_mode, **kwargs)
        self.span_rep = layers.SpanRepLayer(
            hidden_size, max_width, "markerV0", dropout=0.1,
        )
        self.classifier = layers.create_mlp(
            input_dim=hidden_size, intermediate_dims=[hidden_size * 2],
            output_dim=1, dropout=0., activation="relu", add_layer_norm=False,
        )
        self.count_pred = layers.create_mlp(
            input_dim=hidden_size, intermediate_dims=[hidden_size * 2],
            output_dim=20, dropout=0., activation="relu", add_layer_norm=False,
        )
        if counting_layer == "count_lstm":
            self.count_embed = layers.CountLSTM(hidden_size)
        elif counting_layer == "count_lstm_v2":
            self.count_embed = layers.CountLSTMv2(hidden_size=hidden_size)
        elif counting_layer == "count_lstm_moe":
            self.count_embed = layers.CountLSTMoE(hidden_size=hidden_size)
        else:
            raise ValueError(
                f"Unsupported counting_layer {counting_layer!r}; expected one of "
                "'count_lstm', 'count_lstm_v2', 'count_lstm_moe'."
            )

    # ── FactoryPooler protocol ───────────────────────────────────────────

    def get_tasks(self) -> set[str]:
        return {"embed", "classify", "plugin"}

    def compute_span_rep(self, token_embs: torch.Tensor) -> Dict[str, Any]:
        """Compute span representations from token embeddings.

        Returns span_rep of shape (text_len, max_width, D).
        """
        text_length = len(token_embs)
        device = token_embs.device

        spans_idx = []
        for i in range(text_length):
            for j in range(self.max_width):
                if i + j < text_length:
                    spans_idx.append((i, i + j))
                else:
                    spans_idx.append((0, 0))  # safe padding

        spans_idx = torch.tensor([spans_idx], dtype=torch.long, device=device)

        span_rep = self.span_rep(token_embs.unsqueeze(0), spans_idx)
        # Native gliner2 returns (B, L, K, D); the deleted port returned (B, L*K, D).
        if span_rep.dim() == 4:
            span_rep = span_rep.squeeze(0)
        else:
            span_rep = span_rep.squeeze(0).view(text_length, self.max_width, -1)

        return {"span_rep": span_rep}

    def predict_spans(self, token_embs: torch.Tensor, schema_embs: torch.Tensor):
        """Predict spans for a structure/entity/relation schema.

        Args:
            token_embs: (text_len, D) text embeddings
            schema_embs: (num_fields+1, D) stacked schema embeddings ([P] + fields)

        Returns:
            span_scores: sigmoid scores, shape (count, num_fields, text_len, max_width)
            pred_count: predicted count
        """
        # Count prediction from [P] token (first schema embedding)
        count_logits = self.count_pred(schema_embs[0].unsqueeze(0))
        pred_count = int(count_logits.argmax(dim=1).item())

        if pred_count <= 0 or token_embs.numel() == 0:
            return None, 0

        # Span representations
        span_info = self.compute_span_rep(token_embs)

        # Count-aware structure projection
        struct_proj = self.count_embed(schema_embs[1:], pred_count)

        # Score: einsum('lkd,bpd->bplk')
        span_scores = torch.sigmoid(
            torch.einsum("lkd,bpd->bplk", span_info["span_rep"], struct_proj)
        )

        return span_scores, pred_count

    def classify(self, schema_embs: torch.Tensor):
        """Classification from schema embeddings.

        Args:
            schema_embs: (num_labels+1, D) — [P] + label embeddings

        Returns:
            logits: (num_labels,) raw logits
        """
        cls_embeds = schema_embs[1:]
        logits = self.classifier(cls_embeds).squeeze(-1)
        return logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        try:
            sequences = split_hidden_states(hidden_states, ctx.seq_lengths)
        except Exception:
            dummy = torch.zeros(
                4, device=hidden_states.device, dtype=hidden_states.dtype
            )
            return [dummy]

        outputs: List[torch.Tensor] = []

        for i, tok in enumerate(sequences):
            dev = tok.device
            add = ctx.extra_kwargs[i] if i < len(ctx.extra_kwargs) else {}
            prompt_ids = ctx.prompt_token_ids[i] if i < len(ctx.prompt_token_ids) else None

            if not add:
                outputs.append(torch.zeros(4, device=dev, dtype=torch.float32))
                continue

            if prompt_ids is not None and "input_ids" not in add:
                add = {**add, "input_ids": prompt_ids}

            result = self._process_single(tok, add)
            outputs.append(result)

        return outputs

    def _process_single(self, tok_embs: torch.Tensor, kwargs: dict) -> torch.Tensor:
        """Process a single sequence through the GLiNER2 head.

        Returns a serialized tensor with results for all schemas.
        """
        import json
        device = tok_embs.device

        mappings = kwargs["mapped_indices"]
        schema_tokens_list = kwargs["schema_tokens_list"]
        task_types = kwargs["task_types"]
        text_tokens = kwargs["text_tokens"]
        schema_count = kwargs["schema_count"]
        original_text = kwargs["original_text"]
        start_mapping = kwargs["start_mapping"]
        end_mapping = kwargs["end_mapping"]
        threshold = kwargs.get("threshold", 0.5)
        schema_dict = kwargs.get("schema_dict", {})
        token_pooling = kwargs.get("token_pooling", "first")
        threshold_meta = kwargs.get("threshold_meta", {})

        seq_len = tok_embs.shape[0]
        hidden = tok_embs.shape[-1]

        # Extract schema embeddings and text embeddings from mappings
        special_ids = kwargs.get("special_token_ids", {})
        p_id = special_ids.get("[P]")
        c_id = special_ids.get("[C]")
        e_id = special_ids.get("[E]")
        r_id = special_ids.get("[R]")
        l_id = special_ids.get("[L]")
        special_set = {p_id, c_id, e_id, r_id, l_id} - {None}

        input_ids = kwargs.get("input_ids", None)
        if input_ids is not None:
            if isinstance(input_ids, list):
                input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
            else:
                input_ids_t = input_ids.to(device)
        else:
            input_ids_t = None

        # Extract per-schema embeddings and text word embeddings
        schema_embs_list = [[] for _ in range(schema_count)]
        word_embs = []
        bucket = []
        last_orig = None

        for j in range(min(seq_len, len(mappings))):
            seg_type, orig_idx, schema_idx = mappings[j]
            emb = tok_embs[j]

            if seg_type == "schema":
                # Only keep special token embeddings
                if input_ids_t is not None and j < len(input_ids_t):
                    tid = int(input_ids_t[j].item())
                    if tid in special_set:
                        if schema_idx < schema_count:
                            schema_embs_list[schema_idx].append(emb)
            elif seg_type == "text":
                if last_orig is not None and orig_idx != last_orig and bucket:
                    word_embs.append(self._aggregate(bucket, token_pooling))
                    bucket = []
                bucket.append(emb)
                last_orig = orig_idx

        if bucket:
            word_embs.append(self._aggregate(bucket, token_pooling))

        if word_embs:
            text_embs = torch.stack(word_embs)
        else:
            text_embs = torch.empty(0, hidden, device=device)

        len(text_tokens)
        # Use word count from text tokens (lowercased), not word_embs
        # text_embs might not have exactly text_len entries due to prefix
        text_len_actual = text_embs.shape[0]

        # Process each schema
        results = {}
        for si, (schema_tokens, task_type) in enumerate(zip(schema_tokens_list, task_types)):
            if not schema_embs_list[si] or len(schema_tokens) < 4:
                continue

            schema_name = schema_tokens[2].split(" [DESCRIPTION] ")[0]
            embs = torch.stack(schema_embs_list[si])

            if task_type == "classifications":
                logits = self.classify(embs)
                results[schema_name] = {
                    "type": "classification",
                    "logits": logits.detach().cpu().tolist(),
                    "labels": self._extract_field_names(schema_tokens),
                }
            else:
                span_scores, pred_count = self.predict_spans(text_embs, embs)
                if span_scores is None or pred_count <= 0:
                    field_names = self._extract_field_names(schema_tokens)
                    if schema_name == "entities":
                        results[schema_name] = {"type": task_type, "entities": {}}
                    elif task_type == "relations":
                        results[schema_name] = {"type": task_type, "instances": []}
                    else:
                        results[schema_name] = {"type": task_type, "instances": []}
                    continue

                field_names = self._extract_field_names(schema_tokens)
                decoded = self._decode_spans(
                    span_scores, pred_count, field_names, task_type,
                    schema_name, text_len_actual, text_tokens,
                    original_text, start_mapping, end_mapping,
                    threshold, schema_dict, threshold_meta,
                )
                results[schema_name] = decoded

        # Serialize results to JSON bytes then to float tensor
        result_json = json.dumps(results, default=str)
        result_bytes = result_json.encode("utf-8")
        result_tensor = torch.tensor(
            [float(b) for b in result_bytes],
            device=device, dtype=torch.float32,
        )
        # Prepend length
        length = torch.tensor([float(len(result_bytes))], device=device, dtype=torch.float32)
        return torch.cat([length, result_tensor])

    @staticmethod
    def _aggregate(pieces: List[torch.Tensor], mode: str = "first") -> torch.Tensor:
        if mode == "first":
            return pieces[0]
        stack = torch.stack(pieces)
        if mode == "mean":
            return stack.mean(dim=0)
        if mode == "max":
            return stack.max(dim=0).values
        return pieces[0]

    @staticmethod
    def _extract_field_names(schema_tokens: List[str]) -> List[str]:
        """Extract field names from schema token list."""
        field_names = []
        for j in range(len(schema_tokens) - 1):
            if schema_tokens[j] in ("[E]", "[C]", "[R]", "[L]"):
                field_names.append(schema_tokens[j + 1])
        return field_names

    def _decode_spans(
        self, span_scores, pred_count, field_names, task_type,
        schema_name, text_len, text_tokens, original_text,
        start_mapping, end_mapping, threshold, schema_dict,
        threshold_meta=None,
    ) -> dict:
        """Decode span scores into structured results."""
        if threshold_meta is None:
            threshold_meta = {}

        if schema_name == "entities":
            entity_thresholds = threshold_meta.get("entities", {})
            per_field = [
                entity_thresholds.get(name) or threshold
                for name in field_names
            ]
            return self._decode_entities(
                span_scores, field_names, text_len, text_tokens,
                original_text, start_mapping, end_mapping, threshold,
                per_field_thresholds=per_field,
            )
        elif task_type == "relations":
            rel_threshold = (
                threshold_meta.get("relations", {}).get(schema_name) or threshold
            )
            return self._decode_relations(
                span_scores, pred_count, field_names, text_len,
                text_tokens, original_text, start_mapping, end_mapping,
                rel_threshold, schema_name,
            )
        else:
            struct_thresholds = threshold_meta.get("json_structures", {}).get(schema_name, [])
            per_field = [
                (struct_thresholds[i] if i < len(struct_thresholds) and struct_thresholds[i] is not None else threshold)
                for i in range(len(field_names))
            ]
            return self._decode_structures(
                span_scores, pred_count, field_names, text_len,
                text_tokens, original_text, start_mapping, end_mapping,
                threshold, schema_name, schema_dict,
                per_field_thresholds=per_field,
            )

    def _find_spans(self, scores, threshold, text_len, text,
                    start_map, end_map):
        """Find valid spans above threshold."""
        valid = torch.where(scores >= threshold)
        starts, widths = valid

        spans = []
        for start, width in zip(starts.tolist(), widths.tolist()):
            end = start + width + 1
            if 0 <= start < text_len and end <= text_len:
                try:
                    char_start = start_map[start]
                    char_end = end_map[end - 1]
                    text_span = text[char_start:char_end].strip()
                except (IndexError, KeyError):
                    continue
                if text_span:
                    conf = scores[start, width].item()
                    spans.append((text_span, conf, char_start, char_end))
        return spans

    def _format_spans(self, spans):
        """Format spans with overlap removal."""
        if not spans:
            return []
        sorted_spans = sorted(spans, key=lambda x: x[1], reverse=True)
        selected = []
        for text, conf, start, end in sorted_spans:
            overlap = any(not (end <= s[2] or start >= s[3]) for s in selected)
            if not overlap:
                selected.append((text, conf, start, end))
        return [{"text": s[0], "confidence": s[1], "start": s[2], "end": s[3]} for s in selected]

    def _decode_entities(self, span_scores, field_names, text_len,
                         text_tokens, text, start_map, end_map, threshold,
                         per_field_thresholds=None):
        scores = span_scores[0, :, -text_len:]
        entity_results = OrderedDict()
        for idx, name in enumerate(field_names):
            t = per_field_thresholds[idx] if per_field_thresholds else threshold
            spans = self._find_spans(scores[idx], t, text_len, text, start_map, end_map)
            entity_results[name] = self._format_spans(spans)
        return {"type": "entities", "entities": entity_results}

    def _decode_relations(self, span_scores, count, field_names, text_len,
                          text_tokens, text, start_map, end_map, threshold, schema_name):
        instances = []
        for inst in range(count):
            scores = span_scores[inst, :, -text_len:]
            values = {}
            for fidx, fname in enumerate(field_names):
                spans = self._find_spans(scores[fidx], threshold, text_len, text, start_map, end_map)
                if spans:
                    values[fname] = {"text": spans[0][0], "confidence": spans[0][1]}
                else:
                    values[fname] = None
            if all(v is not None for v in values.values()):
                instances.append(values)
        return {"type": "relations", "instances": instances}

    def _decode_structures(self, span_scores, count, field_names, text_len,
                           text_tokens, text, start_map, end_map, threshold,
                           schema_name, schema_dict, per_field_thresholds=None):
        instances = []
        for inst in range(count):
            scores = span_scores[inst, :, -text_len:]
            instance = OrderedDict()
            for fidx, fname in enumerate(field_names):
                t = per_field_thresholds[fidx] if per_field_thresholds else threshold
                spans = self._find_spans(scores[fidx], t, text_len, text, start_map, end_map)
                if spans:
                    instance[fname] = {"text": spans[0][0], "confidence": spans[0][1]}
                else:
                    instance[fname] = None
            if any(v is not None for v in instance.values()):
                instances.append(instance)
        return {"type": "json_structures", "instances": instances}
