"""Unit tests for decode_output + format_results cls_threshold behaviour.

CPU-only, no vLLM runtime. Tests that per-classification cls_threshold
is attached by decode_output and honoured by format_results.
"""

from __future__ import annotations

import json

import torch

from plugins.deberta_gliner2.processor import (
    decode_output,
    format_results,
)


def _encode_results(results: dict) -> list[float]:
    """Fake the pooler's JSON-bytes-to-float-tensor serialization."""
    raw = json.dumps(results).encode("utf-8")
    return [float(len(raw))] + [float(b) for b in raw]


class TestDecodeOutputClsThreshold:
    def test_cls_threshold_attached_from_meta(self):
        """cls_threshold from _meta is attached to classification records."""
        raw_results = {
            "sentiment": {
                "type": "classification",
                "logits": [1.0, -1.0],
                "labels": ["positive", "negative"],
            }
        }
        raw = _encode_results(raw_results)
        schema = {
            "classifications": [
                {"task": "sentiment", "labels": ["positive", "negative"]}
            ],
            "_meta": {
                "classifications": {
                    "sentiment": {"cls_threshold": 0.8, "multi_label": False}
                }
            },
        }
        decoded = decode_output(raw, schema)
        assert decoded["sentiment"]["cls_threshold"] == 0.8

    def test_cls_threshold_attached_from_config(self):
        """cls_threshold from classification config is attached."""
        raw_results = {
            "sentiment": {
                "type": "classification",
                "logits": [1.0, -1.0],
                "labels": ["positive", "negative"],
            }
        }
        raw = _encode_results(raw_results)
        schema = {
            "classifications": [
                {"task": "sentiment", "labels": ["positive", "negative"],
                 "cls_threshold": 0.7}
            ],
        }
        decoded = decode_output(raw, schema)
        assert decoded["sentiment"]["cls_threshold"] == 0.7

    def test_no_cls_threshold_when_absent(self):
        """When cls_threshold is not provided anywhere, it's absent."""
        raw_results = {
            "sentiment": {
                "type": "classification",
                "logits": [1.0, -1.0],
                "labels": ["positive", "negative"],
            }
        }
        raw = _encode_results(raw_results)
        schema = {
            "classifications": [
                {"task": "sentiment", "labels": ["positive", "negative"]}
            ],
        }
        decoded = decode_output(raw, schema)
        assert "cls_threshold" not in decoded["sentiment"]


class TestFormatResultsClsThreshold:
    def test_single_label_below_cls_threshold_returns_none(self):
        """Single-label classification below cls_threshold returns None."""
        results = {
            "sentiment": {
                "type": "classification",
                "logits": [0.5, -0.5],
                "labels": ["positive", "negative"],
                "cls_threshold": 0.95,
            }
        }
        formatted = format_results(results, threshold=0.3)
        assert formatted["sentiment"] is None

    def test_single_label_above_cls_threshold(self):
        """Single-label classification above cls_threshold returns the label."""
        results = {
            "sentiment": {
                "type": "classification",
                "logits": [2.0, -2.0],
                "labels": ["positive", "negative"],
                "cls_threshold": 0.5,
            }
        }
        formatted = format_results(results, threshold=0.99)
        # cls_threshold (0.5) overrides global threshold (0.99)
        assert formatted["sentiment"] == "positive"

    def test_multi_label_uses_cls_threshold(self):
        """Multi-label classification uses cls_threshold, not global."""
        results = {
            "topics": {
                "type": "classification",
                "logits": [1.0, 0.1, -1.0],
                "labels": ["tech", "finance", "sports"],
                "multi_label": True,
                "cls_threshold": 0.3,
            }
        }
        formatted = format_results(results, threshold=0.99)
        assert "tech" in formatted["topics"]

    def test_no_cls_threshold_uses_global(self):
        """Without cls_threshold, the global threshold is used."""
        results = {
            "sentiment": {
                "type": "classification",
                "logits": [0.5, -0.5],
                "labels": ["positive", "negative"],
            }
        }
        formatted_low = format_results(results, threshold=0.3)
        assert formatted_low["sentiment"] == "positive"

        formatted_high = format_results(results, threshold=0.99)
        assert formatted_high["sentiment"] is None
