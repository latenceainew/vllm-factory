# DeBERTa-GLiNER2

Schema-driven entity extraction, classification, relations via GLiNER2.

**Model:** [fastino/gliner2-large-v1](https://huggingface.co/fastino/gliner2-large-v1)
**Architecture:** DeBERTa v3-large backbone, GLiNER2 schema processor (entity/classification/relation/JSON)
**Performance:** Speedup vs vanilla GLiNER2, scales with batch size — run `benchmark_gliner2.py` on your hardware
**Parity:** Entity F1 = 1.0000, Classification ✅, Relations ✅, JSON ✅

## Usage

```python
from vllm import LLM

llm = LLM("fastino/gliner2-large-v1", trust_remote_code=True)
# GLiNER2 supports simple labels, schema requests, and mixed schema output
```

## Request Contract

- `labels` stays supported for the simple entity-extraction path.
- `schema` supports mixed-task requests.
- If both are present, `schema` wins.
- One request can mix `entities`, `classifications`, `relations`, and `structures`.
- `threshold`, `include_confidence`, and `include_spans` are request-level flags.
- `labels` is the simplest path for benchmarks and parity tests.

Simple labels payload:

```json
{
  "data": {
    "text": "Apple released iPhone 15 in Cupertino.",
    "labels": ["company", "product", "location"]
  }
}
```

Mixed schema payload:

```json
{
  "data": {
    "text": "Tim Cook announced the iPhone 15 at Apple Park.",
    "schema": {
      "entities": {"person": "Person names", "product": "Product names"},
      "classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"]}],
      "relations": ["announced_at"],
      "structures": {"summary": {"fields": [{"name": "product", "dtype": "str"}]}}
    },
    "threshold": 0.5,
    "include_confidence": true,
    "include_spans": true
  }
}
```

Per-field threshold payload (all thresholds optional, fall back to request-level `threshold`):

```json
{
  "data": {
    "text": "John Smith works at NVIDIA in Santa Clara. His email is john@nvidia.com.",
    "schema": {
      "entities": {
        "person": {"description": "Person names", "threshold": 0.3},
        "email": {"threshold": 0.9}
      },
      "classifications": [
        {"task": "topic", "labels": ["tech", "finance"], "cls_threshold": 0.6}
      ],
      "relations": {
        "works_at": {"description": "Employment", "threshold": 0.25}
      },
      "structures": {
        "employee": {"fields": [
          {"name": "name", "dtype": "str", "threshold": 0.8},
          {"name": "title", "threshold": 0.2}
        ]}
      }
    },
    "threshold": 0.5,
    "include_confidence": true
  }
}
```

Response notes:

- `entities` returns per-label lists.
- single-label classification returns one label or `null` if filtered by `threshold`.
- multi-label classification returns a list.
- `relation_extraction` contains relation results.
- JSON structures return per-structure instance lists.

## Serve

Requires a prepared model directory (see `forge/model_prep.py`).

```bash
vllm serve /tmp/gliner2-vllm \
  --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill --port 8200
```

## Verify

```bash
python plugins/deberta_gliner2/parity_test.py
```

