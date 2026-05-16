# Multi-Instance Serving (beta)

Encoder models are memory-bound, not compute-bound. A single vLLM instance on a 24 GB GPU might use only a fraction of the available compute while the continuous-batching scheduler idles between requests. By running multiple vLLM instances on the same GPU — each with its own scheduler — throughput scales nearly linearly until compute saturation.

`vllm-factory-serve` automates this: it launches N identical backends, partitions GPU memory, and places a thin async dispatcher in front that distributes requests across them. Clients see a single endpoint.

## Quick start

```bash
# 1. Install (if not already)
pip install -e ".[gliner]"
pip install vllm

# 2. Prepare a GLiNER model (one-time)
vllm-factory-prep --model VAGOsolutions/SauerkrautLM-GLiNER \
  --output /tmp/sauerkraut-gliner-vllm

# 3. Serve with 4 instances
vllm-factory-serve /tmp/sauerkraut-gliner-vllm \
  --num-instances 4 \
  --max-batch-size 32 \
  --dtype bfloat16 \
  --enforce-eager \
  --io-processor-plugin mmbert_gliner_io
```

That's it. The dispatcher listens on port 8000 (default). Clients use the same `POST /pooling` API as with `vllm serve` — no code changes required.

```bash
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/tmp/sauerkraut-gliner-vllm",
    "data": {
      "text": "Apple Inc. announced a partnership with OpenAI in San Francisco.",
      "labels": ["company", "location", "person"],
      "threshold": 0.3
    }
  }'
```

## How it works

```
                          ┌─────────────────────┐
                          │  Dispatcher (:8000)  │
                          │  async reverse proxy │
                          │ round-robin/affinity │
                          │    + semaphores      │
                          └──┬───────┬───────┬───┘
                             │       │       │
               ┌─────────────┘       │       └─────────────┐
               ▼                     ▼                     ▼
       ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
       │ vLLM (:9100) │     │ vLLM (:9101) │     │ vLLM (:9102) │
       │  scheduler   │     │  scheduler   │     │  scheduler   │
       │  KV cache    │     │  KV cache    │     │  KV cache    │
       └──────────────┘     └──────────────┘     └──────────────┘
                          ┌─────────────────────┐
                          │     Single GPU       │
                          │   memory split N     │
                          └─────────────────────┘
```

1. **`--num-instances 1` (default)** — identical to `vllm serve`. No proxy, no overhead. This is a hard compatibility guarantee.
2. **`--num-instances N` (N > 1)** — launches N vLLM backends on ports 9100..9100+N-1, each with `gpu_memory_utilization ≈ 0.92 / N`. A lightweight `aiohttp` dispatcher binds to the user-facing port (default 8000).
3. **Per-backend concurrency cap** — each backend is guarded by an `asyncio.Semaphore(max_batch_size)`. When all backends are at capacity, new requests queue in FIFO order. The dispatcher never drops requests.
4. **Backend selection** — round-robin with capacity preference by default. With `--enable-request-affinity`, JSON requests also get a stable request-shape key so repeated schema/labels-heavy calls can prefer the same backend-local cache when capacity is available. If the pinned backend is full, the dispatcher falls back to the normal round-robin-capacity path.
5. **Health** — `GET /health` on the dispatcher returns 200 if at least one backend is healthy.
6. **Stats** — `GET /stats` returns dispatcher-local routing counters and per-backend available slots so affinity behavior can be measured during benchmarks and live runs.

## Runtime observability

The dispatcher now exposes two lightweight runtime surfaces:

- **Response headers** on proxied requests:
  - `X-VLLM-Factory-Backend-Index`
  - `X-VLLM-Factory-Affinity-State` (`disabled`, `miss`, `hit`, or `fallback`)
  - `X-VLLM-Factory-Schema-Size` when the request body exposes `schema` or `labels`
- **`GET /stats`** on the dispatcher:
  - total routed requests
  - per-backend request counts
  - affinity hit / miss / fallback counters
  - current affinity cache occupancy
  - current available slots per backend

Example:

```bash
curl -s http://localhost:8000/stats | jq
```

## CLI reference

```
vllm-factory-serve MODEL [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `MODEL` | *(required)* | HuggingFace model ID or local path |
| `--num-instances N` | 1 | Number of vLLM backend instances |
| `--max-batch-size N` | 32 | Per-backend max concurrent requests (also sets `--max-num-seqs`) |
| `--port P` | 8000 | User-facing port |
| `--port-start P` | 9100 | First internal backend port (backends use P, P+1, ..., P+N-1) |
| `--gpu-memory-utilization F` | auto | Per-instance GPU memory fraction. Auto-scaled as `0.92/N` when omitted |
| `--dtype` | auto | Model dtype (`bfloat16` recommended) |
| `--enforce-eager` | off | Disable CUDA graph compilation |
| `--io-processor-plugin` | — | vLLM IOProcessor plugin name |
| `--max-model-len` | — | Override max sequence length |
| `--cuda-devices` | "0" | CUDA_VISIBLE_DEVICES for backend(s) |
| `--enable-request-affinity` | off | Prefer backend-locality for repeated JSON request shapes in multi-instance mode |

Extra flags after `--` are forwarded to each `vllm serve` backend.

## Examples

### NER with GLiNER (ModernBERT, 150M)

```bash
vllm-factory-prep --model VAGOsolutions/SauerkrautLM-GLiNER \
  --output /tmp/sauerkraut-gliner-vllm

vllm-factory-serve /tmp/sauerkraut-gliner-vllm \
  --num-instances 4 \
  --io-processor-plugin mmbert_gliner_io \
  --dtype bfloat16 --enforce-eager
```

### Schema extraction with GLiNER2 (DeBERTa v3, 304M)

```bash
vllm-factory-prep --model fastino/gliner2-large-v1 \
  --output /tmp/gliner2-vllm

vllm-factory-serve /tmp/gliner2-vllm \
  --num-instances 2 \
  --enable-request-affinity \
  --io-processor-plugin deberta_gliner2_io \
  --dtype bfloat16 --enforce-eager
```

### Multilingual NER with mT5 (800M)

```bash
vllm-factory-prep --model knowledgator/gliner-x-large \
  --output /tmp/gliner-x-large-vllm

vllm-factory-serve /tmp/gliner-x-large-vllm \
  --num-instances 2 \
  --io-processor-plugin mt5_gliner_io \
  --dtype bfloat16 --enforce-eager
```

### Embedding models (no prep required)

```bash
vllm-factory-serve unsloth/embeddinggemma-300m \
  --num-instances 4 \
  --io-processor-plugin embeddinggemma_io \
  --dtype bfloat16
```

### ColBERT retrieval

```bash
vllm-factory-serve VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT \
  --num-instances 2 \
  --io-processor-plugin moderncolbert_io \
  --dtype bfloat16
```

## Choosing the right instance count

| GPU VRAM | Model size | Recommended instances |
|---|---|---|
| 24 GB | < 200M (ModernBERT, DeBERTa) | 4 |
| 24 GB | 300–500M (GLiNER2, LFM2) | 2–4 |
| 24 GB | 800M+ (mT5 GLiNER) | 2 |
| 48 GB | < 500M | 4–8 |
| 80 GB | < 500M | 4–8 |

Start with 2 instances and increase until throughput plateaus or latency spikes. The sweet spot depends on model size, sequence length, and GPU architecture.

**Rule of thumb**: if your single-instance GPU utilization (check `nvidia-smi`) shows memory mostly full but SM activity < 50%, more instances will help.

## Benchmark results

Measured on NVIDIA RTX A5000 (24 GB), 200 requests, NuNER dataset, bfloat16, `--max-batch-size 32`, vLLM 0.19.0.

| Model | Backbone | Params | 1 instance | 2 instances | 4 instances | Speedup |
|:------|:---------|-------:|-----------:|------------:|------------:|:-------:|
| DeBERTa GLiNER2 | DeBERTa v3 | 304M | 133 req/s | 164 req/s | 229 req/s | **1.72x** |
| MMBert GLiNER | ModernBERT | 150M | 159 req/s | 255 req/s | 313 req/s | **1.98x** |
| MT5 GLiNER X-Large | mT5 | 800M | 142 req/s | 204 req/s | 236 req/s | **1.66x** |

Smaller models benefit more — they are more memory-bound and leave more compute headroom for additional instances.

## Docker

```dockerfile
FROM vllm/vllm-openai:latest

COPY . /app/vllm-factory
WORKDIR /app/vllm-factory

RUN pip install -e ".[gliner]"

# Prepare model at build time
RUN vllm-factory-prep --model VAGOsolutions/SauerkrautLM-GLiNER \
    --output /models/sauerkraut-gliner-vllm

EXPOSE 8000
CMD ["vllm-factory-serve", "/models/sauerkraut-gliner-vllm", \
     "--num-instances", "4", "--max-batch-size", "32", \
     "--dtype", "bfloat16", "--enforce-eager", \
     "--io-processor-plugin", "mmbert_gliner_io"]
```

## Python client example

```python
import aiohttp
import asyncio

async def main():
    url = "http://localhost:8000/pooling"
    payload = {
        "model": "/tmp/sauerkraut-gliner-vllm",
        "data": {
            "text": "Tesla CEO Elon Musk announced the Cybertruck launch in Austin, Texas.",
            "labels": ["person", "company", "location", "product"],
            "threshold": 0.3,
        },
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            result = await resp.json()
            for entity in result["data"]:
                print(f"  {entity['label']}: {entity['text']} ({entity['score']:.2f})")

asyncio.run(main())
```

## Troubleshooting

**Server fails to start with "Engine core initialization failed"**
GPU memory is too constrained for the requested number of instances. Reduce `--num-instances` or pass a lower `--gpu-memory-utilization` (e.g., `0.20` per instance for 4 instances on a busy GPU).

**Port already in use**
Another process is using port 8000 or the backend port range (9100+). Either stop the other process or use `--port` and `--port-start` to pick different ports.

**Throughput doesn't improve with more instances**
The model may be compute-bound rather than memory-bound. This is typical for larger models (> 1B params) or when GPU SM utilization is already high with a single instance. Try 2 instances first before going higher.

**Health check returns 503**
No backend is healthy. Check server logs in the terminal for startup errors. Common causes: missing model weights (run `vllm-factory-prep` first), insufficient GPU memory, or incompatible vLLM version.

## Architecture details

The multi-instance feature is implemented in three files:

- **`forge/dispatcher.py`** — async HTTP reverse proxy built on `aiohttp.web`. It forwards any path (`/pooling`, `/v1/embeddings`, `/health`, `/stats`, etc.). When request affinity is disabled, request bodies stay opaque. When affinity is enabled, the dispatcher inspects JSON request shape to derive a stable routing key and emits lightweight routing stats and headers. Per-backend concurrency is managed with `asyncio.Semaphore`.
- **`forge/multi_instance.py`** — orchestrator that creates N `ModelServer` instances with scaled GPU memory, starts them sequentially (to avoid CUDA allocation races), and launches the dispatcher.
- **`forge/serve_cli.py`** — CLI entry point. When `--num-instances 1`, delegates directly to `ModelServer` with zero overhead. When N > 1, uses `MultiInstanceServer`.

No existing code is modified. The single-instance path is identical to `vllm serve`.
