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

### LoRA adapters

`GLiNER2VLLMModel` declares vLLM's `SupportsLoRA` protocol, and the
underlying DeBERTa v2/v3 backbone registers its parallel linears
(`query_proj`, `key_proj`, `value_proj`, `pos_key_proj`, `pos_query_proj`,
`attention.output.dense`, `intermediate.dense`, `output.dense`) as adapter
targets. PEFT adapters produced against the GLiNER2 backbone with the
default `target_modules=["query_proj", "key_proj", "value_proj"]` recipe
map 1:1 onto our layer names — no packing rewrite is needed.

```bash
vllm serve /tmp/gliner2-vllm \
  --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --enable-lora --max-loras 32 --max-lora-rank 64 \
  --port 8200
```

Per-request adapter switching is plumbed via two complementary paths, both
respecting vLLM's own LoRA selection contract — vLLM's v1 scheduler and
Punica SGMV kernels already handle cross-request LoRA batching (up to
`--max-loras` distinct adapters in a single forward pass) once each
request carries a `LoRARequest`. The plugin's job is to make that easy
from both online and offline entry points.

#### Request payload: optional `adapter` field

The IOProcessor accepts an optional `adapter` string on the request body:

```json
{
  "data": {
    "text": "...",
    "labels": ["..."],
    "adapter": "sql-lora"
  },
  "model": "sql-lora",
  "task": "plugin"
}
```

Validated against `^[A-Za-z0-9_.\-:/]{1,128}$`; `null`, missing, or
whitespace-only values mean **base model**. The adapter name is stashed
in post-process meta (`request_meta["adapter"]`) and emitted on the
observability log line, and — when the final response is a JSON object —
echoed back to the caller as `adapter` for end-to-end tracing.

#### Online path (`vllm serve` + `/pooling`)

vLLM resolves `LoRARequest` from `request.model` in
`_maybe_get_adapters(ctx)` **before** the IOProcessor runs. Register
adapters at startup (or use the runtime `/v1/load_lora_adapter` endpoint
or a `LoRAResolver` plugin), then have any HTTP shim in front of
`/pooling` mirror `data.adapter` into `model`:

```bash
vllm serve /tmp/gliner2-vllm \
  --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --enable-lora --max-loras 32 --max-lora-rank 64 \
  --lora-modules sql-lora=/adapters/sql-lora finance=/adapters/finance \
  --port 8200
```

Clients (and the POC Modal `/infer` shim) set both `data.adapter` and the
top-level `model` to the same value so vLLM routes correctly and the
plugin can validate / echo the selection.

#### Offline path (`LLM.encode(...)`)

`LLM.encode(..., lora_request=...)` accepts either a single `LoRARequest`
(broadcast to every engine input) or a list (zipped one-to-one with the
plugin's rendered prompts). The plugin ships a small helper that maps
parsed inputs through a caller-owned adapter registry:

```python
from vllm import LLM
from plugins.deberta_gliner2.io_processor import DeBERTaGLiNER2IOProcessor
from plugins.deberta_gliner2.lora import build_lora_requests

registry = {
    "sql-lora": (1, "/adapters/sql-lora"),
    "finance": (2, "/adapters/finance"),
}

parsed = [DeBERTaGLiNER2IOProcessor.factory_parse.__func__(..., data) for data in batch]
lora_requests = build_lora_requests(parsed, registry)
# N.B. `lora_int_id` MUST be unique per adapter — vLLM uses it as the
# identity key for `--max-loras` batching.

outputs = llm.encode(
    [{"data": data} for data in batch],
    lora_request=lora_requests,
    pooling_task="plugin",
)
```

#### Cross-request LoRA batching

No plugin code participates in batching. Once each request carries its
own `LoRARequest`, vLLM's scheduler (`vllm/v1/core/sched/scheduler.py`)
groups requests with distinct `lora_int_id`s into a single step up to
`--max-loras`, and the Punica SGMV kernels apply the correct delta per
token in one fused forward pass. The pooler head (`span_rep` /
`classifier` / `count_pred` / `count_embed`) is intentionally **not**
adapter-eligible — only the DeBERTa backbone linears registered in
[PR #11](https://github.com/ddickmann/vllm-factory/pull/11) receive LoRA.

## Verify

```bash
python plugins/deberta_gliner2/parity_test.py
```

