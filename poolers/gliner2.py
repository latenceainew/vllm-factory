# GLiNER2 Pooler — Schema-based multi-task information extraction.
#
# Architecture:
#     SpanRepLayer (markerV0) → CountLSTM (GRU + pos embs) → classifier MLP → count_pred MLP
#     Produces:
#       - Span scores: einsum('lkd,bpd->bplk', span_rep, count_embed_out)
#       - Classification logits: classifier(schema_embs)
#       - Count logits: count_pred(prompt_emb)
#
# Supports 4 task types: entities, json_structures, relations, classifications
# Uses the same SpanRepLayer from gliner.modeling.span_rep as GLiNER v1
#
# Implements FactoryPooler protocol — zero vLLM imports.

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states

# ==================================================================
# Utility: MLP builder (mirrors gliner2.layers.create_mlp)
# ==================================================================

def create_mlp(input_dim, intermediate_dims, output_dim,
               dropout=0.1, activation="gelu", add_layer_norm=False):
    activation_mapping = {
        "relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU,
    }
    layers = []
    in_dim = input_dim
    for dim in intermediate_dims:
        layers.append(nn.Linear(in_dim, dim))
        if add_layer_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(activation_mapping[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


# ==================================================================
# CountLSTM (mirrors gliner2.layers.CountLSTM)
# ==================================================================

class CountLSTM(nn.Module):
    """Count-aware unrolling: GRU + positional embeddings → count copies of schema embeddings."""

    def __init__(self, hidden_size, max_count=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.projector = create_mlp(
            input_dim=hidden_size * 2,
            intermediate_dims=[hidden_size * 4],
            output_dim=hidden_size,
            dropout=0., activation="relu", add_layer_norm=False,
        )

    def forward(self, pc_emb: torch.Tensor, gold_count_val: int) -> torch.Tensor:
        """pc_emb: (M, D) field embeddings → (count, M, D) count-aware embeddings."""
        M, D = pc_emb.shape
        gold_count_val = min(gold_count_val, self.max_count)
        count_indices = torch.arange(gold_count_val, device=pc_emb.device)
        pos_seq = self.pos_embedding(count_indices).unsqueeze(1).expand(gold_count_val, M, D)
        h0 = pc_emb.unsqueeze(0)
        output, _ = self.gru(pos_seq, h0)
        return self.projector(torch.cat([output, pc_emb.unsqueeze(0).expand_as(output)], dim=-1))


# ==================================================================
# DownscaledTransformer (mirrors gliner2.layers.DownscaledTransformer)
# ==================================================================

class DownscaledTransformer(nn.Module):
    """Bottleneck transformer used as the projector inside ``CountLSTMv2``.

    Projects the GRU output into a small hidden space, runs a two-layer
    ``nn.TransformerEncoder``, concatenates the transformer output with the
    original input, and projects back to ``input_size``. State-dict keys
    match the native ``gliner2.layers.DownscaledTransformer`` exactly so
    checkpoints produced with ``counting_layer == "count_lstm_v2"`` load
    without any key remapping.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.in_projector = nn.Linear(input_size, hidden_size)

        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)

        self.out_projector = create_mlp(
            input_dim=hidden_size + input_size,
            intermediate_dims=[input_size, input_size],
            output_dim=input_size,
            dropout=0.,
            activation="relu",
            add_layer_norm=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x
        x = self.in_projector(x)
        x = self.transformer(x)
        x = torch.cat([x, original_x], dim=-1)
        x = self.out_projector(x)
        return x


# ==================================================================
# CountLSTMv2 (mirrors gliner2.layers.CountLSTMv2)
# ==================================================================

class CountLSTMv2(nn.Module):
    """Count-aware unrolling with a ``DownscaledTransformer`` projector.

    Drop-in replacement for ``CountLSTM`` used by GLiNER2 checkpoints whose
    extractor config sets ``counting_layer == "count_lstm_v2"`` (e.g.
    ``fastino/gliner2-base-v1``). Parameter names match upstream so
    ``state_dict`` loads without remapping.
    """

    def __init__(self, hidden_size: int, max_count: int = 20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.transformer = DownscaledTransformer(
            input_size=hidden_size,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
        )

    def forward(self, pc_emb: torch.Tensor, gold_count_val: int) -> torch.Tensor:
        M, D = pc_emb.shape
        gold_count_val = min(gold_count_val, self.max_count)
        full_idx = torch.arange(self.max_count, device=pc_emb.device)
        count_idx = full_idx[:gold_count_val]
        pos_seq = self.pos_embedding(count_idx).unsqueeze(1).expand(-1, M, -1)
        output, _ = self.gru(pos_seq, pc_emb.unsqueeze(0))
        pc_broadcast = pc_emb.unsqueeze(0).expand_as(output)
        return self.transformer(output + pc_broadcast)


# ==================================================================
# CountLSTMoE (mirrors gliner2.layers.CountLSTMoE)
# ==================================================================

class CountLSTMoE(nn.Module):
    """Count-aware unrolling with a packed-expert MoE projector.

    Drop-in replacement for ``CountLSTM`` used by GLiNER2 checkpoints whose
    extractor config sets ``counting_layer == "count_lstm_moe"``. Parameter
    names (``w1``/``b1``/``w2``/``b2``/``router.*``) match upstream so
    ``state_dict`` loads without remapping.
    """

    def __init__(self, hidden_size: int, max_count: int = 20,
                 n_experts: int = 4, ffn_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        self.n_experts = n_experts

        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)

        inner = hidden_size * ffn_mult
        self.w1 = nn.Parameter(torch.empty(n_experts, hidden_size, inner))
        self.b1 = nn.Parameter(torch.zeros(n_experts, inner))
        self.w2 = nn.Parameter(torch.empty(n_experts, inner, hidden_size))
        self.b2 = nn.Parameter(torch.zeros(n_experts, hidden_size))
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

        self.dropout = nn.Dropout(dropout)

        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, pc_emb: torch.Tensor, gold_count_val: int) -> torch.Tensor:
        M, D = pc_emb.shape
        L = min(gold_count_val, self.max_count)
        idx = torch.arange(L, device=pc_emb.device)
        pos_seq = self.pos_embedding(idx).unsqueeze(1).expand(L, M, D)
        h, _ = self.gru(pos_seq, pc_emb.unsqueeze(0))
        gates = self.router(h)
        x = torch.einsum("lmd,edh->lmeh", h, self.w1) + self.b1
        x = F.gelu(x)
        x = self.dropout(x)
        x = torch.einsum("lmeh,ehd->lmed", x, self.w2) + self.b2
        return (gates.unsqueeze(-1) * x).sum(dim=2)


# ==================================================================
# SpanRepLayer (from gliner.modeling.span_rep — reused)
# ==================================================================

class SpanMarkerV0(nn.Module):
    """Span marker using start/end projections."""

    def __init__(self, hidden_size, max_width, dropout=0.1):
        super().__init__()
        self.max_width = max_width
        self.hidden_size = hidden_size

        def _proj(h_in, h_out=None):
            h_out = h_out or h_in
            return nn.Sequential(
                nn.Linear(h_in, h_out * 4), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(h_out * 4, h_out),
            )

        self.project_start = _proj(hidden_size)
        self.project_end = _proj(hidden_size)
        self.out_project = _proj(hidden_size * 2, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        """h: (B,L,D), span_idx: (B,S,2) → (B,S,D)."""
        B, L, D = h.shape
        start_rep = self.project_start(h)
        end_rep = self.project_end(h)
        idx_start = span_idx[:, :, 0].clamp(0, L - 1)
        idx_end = span_idx[:, :, 1].clamp(0, L - 1)
        start_span = start_rep.gather(1, idx_start.unsqueeze(-1).expand(-1, -1, D))
        end_span = end_rep.gather(1, idx_end.unsqueeze(-1).expand(-1, -1, D))
        cat = torch.cat([start_span, end_span], dim=-1).relu()
        return self.out_project(cat)


class SpanRepLayer(nn.Module):
    """Wraps SpanMarkerV0 with the same interface as gliner.modeling.span_rep.SpanRepLayer."""

    def __init__(self, span_mode="markerV0", hidden_size=768, max_width=8, dropout=0.1):
        super().__init__()
        self.span_rep_layer = SpanMarkerV0(hidden_size, max_width, dropout)

    def forward(self, x: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        return self.span_rep_layer(x, span_idx)


# ==================================================================
# Text Splitter (same as gliner2.processor.WhitespaceTokenSplitter)
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
    """Split text into (word, start, end) tuples."""
    if lower:
        text_lower = text.lower()
    else:
        text_lower = text
    return [(m.group(), m.start(), m.end()) for m in WORD_PATTERN.finditer(text_lower)]


def split_words_with_original(text: str):
    """Split lowercased but return original-case char positions."""
    text_lower = text.lower()
    return [(m.group(), m.start(), m.end()) for m in WORD_PATTERN.finditer(text_lower)]


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
        self.hidden_size = hidden_size
        self.max_width = max_width
        self.counting_layer = counting_layer

        # Span representation
        self.span_rep = SpanRepLayer(
            span_mode="markerV0", hidden_size=hidden_size,
            max_width=max_width, dropout=0.1,
        )

        # Classifier for classification tasks
        self.classifier = create_mlp(
            input_dim=hidden_size, intermediate_dims=[hidden_size * 2],
            output_dim=1, dropout=0., activation="relu", add_layer_norm=False,
        )

        # Count prediction (0-19)
        self.count_pred = create_mlp(
            input_dim=hidden_size, intermediate_dims=[hidden_size * 2],
            output_dim=20, dropout=0., activation="relu", add_layer_norm=False,
        )

        # Count embedding — mirror gliner2/model.py so every checkpoint
        # variant produced by the native repo (count_lstm / count_lstm_v2 /
        # count_lstm_moe) loads into a pooler whose state_dict keys match
        # exactly. Silently defaulting to CountLSTM regardless of config
        # value would let non-count_lstm checkpoints load with a mostly
        # random-init count_embed (load_state_dict strict=False drops the
        # mismatched keys) and produce semantically-wrong output.
        if counting_layer == "count_lstm":
            self.count_embed = CountLSTM(hidden_size)
        elif counting_layer == "count_lstm_v2":
            self.count_embed = CountLSTMv2(hidden_size=hidden_size)
        elif counting_layer == "count_lstm_moe":
            self.count_embed = CountLSTMoE(hidden_size=hidden_size)
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

        span_rep = self.span_rep(
            token_embs.unsqueeze(0), spans_idx,
        ).squeeze(0)  # (L*K, D)

        # Reshape to (L, K, D) for einsum('lkd,bpd->bplk')
        span_rep = span_rep.view(text_length, self.max_width, -1)

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
