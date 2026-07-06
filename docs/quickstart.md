# Quickstart

## 1. Clone and Install

> **Critical: vLLM must be the last package installed.** This ensures it pins all shared dependencies (especially `transformers`) to compatible versions.

```bash
git clone https://github.com/latenceainew/vllm-factory.git
cd vllm-factory

# Install vllm-factory + GLiNER deps (skip [gliner] if you only need embedding/ColBERT models)
pip install -e ".[gliner]"

# Install vLLM — ALWAYS LAST
pip install "vllm==0.15.1"

# Apply pooling patch (one-time, enables extra_kwargs passthrough)
python -m forge.patches.pooling_extra_kwargs
```

Or use the Makefile (handles ordering automatically):

```bash
make install
```

## 2. Pick and Serve a Model

Every model is served with `vllm serve` + an IOProcessor plugin that handles all pre/post-processing server-side.

### ColBERT retrieval (simplest)

```bash
vllm serve VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin moderncolbert_io
```

```bash
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"model":"VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT",
       "data":{"text":"European Central Bank monetary policy"}}'
```

### GLiNER NER (requires model prep)

```bash
# One-time model preparation
vllm-factory-prep --model VAGOsolutions/SauerkrautLM-GLiNER \
  --output /tmp/sauerkraut-gliner-vllm

# Serve with IOProcessor
vllm serve /tmp/sauerkraut-gliner-vllm \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin mmbert_gliner_io
```

```bash
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"model":"/tmp/sauerkraut-gliner-vllm",
       "data":{"text":"Apple announced a partnership with OpenAI.",
               "labels":["company","product"],
               "threshold":0.3}}'
```

### Dense embeddings

```bash
vllm serve unsloth/embeddinggemma-300m \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching \
  --io-processor-plugin embeddinggemma_io
```

```bash
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"model":"unsloth/embeddinggemma-300m",
       "data":{"text":"What is the knapsack problem?"}}'
```

## 3. Run Parity Tests

Every plugin ships with end-to-end parity tests that start a real `vllm serve`, send HTTP requests, and compare against reference outputs.

```bash
# Test a single plugin
python scripts/serve_parity_test.py --plugin moderncolbert

# Test all 12 plugins
python scripts/serve_parity_test.py
```

## 4. Run Examples

```bash
# Start a server (see examples/ for server commands)
python examples/embedding_search.py
python examples/colbert_retrieval.py
python examples/gliner_ner.py
```

## Next Steps

- **[Server Deployment](server_guide.md)** — production config, Docker, multi-model
- **[Building a Plugin](building_a_plugin.md)** — create your own encoder + pooler plugin
- **[All 12 Plugins](../README.md#plugins)** — full model table with serve commands
