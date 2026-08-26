# Plugin Integration Guide

Central reference for building vLLM Factory plugins. Every plugin lives in
`plugins/<name>/` and follows a consistent structure.

---

## Decision Tree — Which Pattern Do I Need?

```
Does your model already exist in vLLM?
├── YES → Do you need a custom encoder with Triton kernels?
│   ├── YES → Use a custom model from models/ (see moderncolbert, mmbert_gliner)
│   └── NO  → Extend the built-in model directly (see colqwen3, collfm2)
└── NO  → Write a new model in models/ first (see models/modernbert/, models/mt5/)

Is your pooler reusable across multiple plugins?
├── YES → Put it in poolers/ and re-export from plugin (see poolers/gliner.py)
└── NO  → Put it in the plugin directory (task-specific, non-shared)

What kind of output do you need?
├── Multi-vector embeddings → PassthroughPooler + "ALL" pooling (see colqwen3)
├── Span extraction / NER   → Custom FactoryPooler with extra_kwargs (see mmbert_gliner)
├── Token classification    → Built-in vLLM pooler
└── Relation / pairwise     → Custom FactoryPooler with bi-affine head (plugin-specific)
```

---

## Plugin Structure

Each plugin needs these core files:

| File | Purpose | Template |
|------|---------|----------|
| `__init__.py` | Auto-register model + config with vLLM on import | [below](#initpy) |
| `config.py` | Extend a HF config with task-specific params | [below](#configpy) |
| `model.py` | Wire backbone + pooler + weight loading | [below](#modelpy) |
| `pooler.py` | Implement `FactoryPooler` protocol OR re-export shared pooler | [below](#poolerpy) |
| `io_processor.py` | Subclass `FactoryIOProcessor` for server-side I/O | [below](#io_processorpy) |
| `parity_test.py` | Validation against reference implementation | — |

### Key abstractions

| Abstraction | Module | Purpose |
|---|---|---|
| `FactoryIOProcessor` | `vllm_factory.io.base` | Base class all IO processors inherit from. Handles vLLM ABC delegation. |
| `FactoryPooler` | `vllm_factory.pooling.protocol` | Protocol (interface) for pooler business logic. **Zero vLLM imports.** |
| `PoolerContext` | `vllm_factory.pooling.protocol` | Stable data model passed to `FactoryPooler.forward()`. |
| `VllmPoolerAdapter` | `vllm_factory.pooling.vllm_adapter` | Bridges `FactoryPooler` to vLLM's Pooler ABC. Shared — not plugin-specific. |
| `PassthroughPooler` | `vllm_factory.pooling.protocol` | No-op pooler for models that handle everything in `model.forward()`. |

---

## Templates

### `__init__.py`

```python
"""My Plugin — registers with vLLM on import."""
from vllm import ModelRegistry
from transformers import AutoConfig
from .model import MyModel
from .config import MyConfig

def register():
    try: AutoConfig.register("my_model_type", MyConfig, exist_ok=True)
    except Exception: pass
    try: ModelRegistry.register_model("MyModel", MyModel)
    except (KeyError, ValueError): pass

register()
__all__ = ["MyModel", "MyConfig"]
```

### `config.py`

```python
from transformers import SomeBaseConfig  # e.g. Qwen2Config, MT5Config, ModernBertConfig

class MyConfig(SomeBaseConfig):
    model_type = "my_model_type"  # must match __init__.py registration

    def __init__(self, my_custom_param: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.my_custom_param = my_custom_param
```

### `model.py`

```python
from vllm.config import VllmConfig

class MyModel(nn.Module):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model = SomeBackbone(vllm_config=vllm_config, prefix=...)

        # Wire pooler via VllmPoolerAdapter
        from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter
        from .pooler import MyPooler
        self.pooler = VllmPoolerAdapter.wrap(MyPooler(...), vllm_config)

    def forward(self, input_ids, positions, **kwargs) -> torch.Tensor:
        return self.model(input_ids=input_ids, positions=positions, **kwargs)

    def load_weights(self, weights) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
```

### `pooler.py`

**Option A — Shared pooler (reusable):**
```python
from poolers.gliner import GLiNERSpanPooler
__all__ = ["GLiNERSpanPooler"]
```

**Option B — Custom FactoryPooler (plugin-specific):**
```python
import torch
from vllm_factory.pooling.protocol import FactoryPooler, PoolerContext, split_hidden_states

class MyCustomPooler:
    """Implements FactoryPooler protocol — zero vLLM imports."""

    def get_tasks(self) -> set[str]:
        return {"plugin"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        per_seq = split_hidden_states(hidden_states, ctx.seq_lengths)
        results = []
        for i, hs in enumerate(per_seq):
            extra = ctx.extra_kwargs[i] if ctx.extra_kwargs else {}
            output = self._process(hs, extra)
            results.append(output)
        return results

    def _process(self, hidden: torch.Tensor, extra: dict) -> torch.Tensor:
        # Your task-specific logic here
        return hidden.mean(dim=0)
```

### `io_processor.py`

```python
from vllm_factory.io.base import (
    FactoryIOProcessor,
    PoolingRequestOutput,
    TokensPrompt,
)
from vllm.config import VllmConfig
from vllm.pooling_params import PoolingParams

class MyIOProcessor(FactoryIOProcessor):
    """Server-side I/O for my plugin."""

    pooling_task = "plugin"  # or "token_embed" for embedding models

    def __init__(self, vllm_config: VllmConfig, **kwargs):
        super().__init__(vllm_config, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(
            vllm_config.model_config.model
        )

    def factory_pre_process(self, data: dict) -> tuple[TokensPrompt, PoolingParams, dict]:
        """Tokenize input and build extra_kwargs."""
        text = data.get("text", "")
        tokens = self._tokenizer(text, truncation=True, max_length=512)

        prompt = TokensPrompt(prompt_token_ids=tokens["input_ids"])
        params = PoolingParams()
        extra_kwargs = {"some_metadata": data.get("metadata")}
        return prompt, params, extra_kwargs

    def factory_post_process(self, output: PoolingRequestOutput, meta: dict) -> dict:
        """Convert raw model output to structured response."""
        embedding = output.outputs.embedding
        return {"embedding": embedding}


def get_processor_cls():
    return MyIOProcessor
```

---

## Registration Flow

```
pyproject.toml declares entry points:
  [project.entry-points."vllm.general_plugins"]
  my_plugin = "plugins.my_plugin:register"

  [project.entry-points."vllm.io_processor_plugins"]
  my_plugin_io = "plugins.my_plugin.io_processor:get_processor_cls"
    │
    ▼
vllm serve model-name --runner pooling --io-processor-plugin my_plugin_io
    │
    ▼
vLLM loads general_plugins → calls register() → model + config registered
vLLM loads io_processor_plugins → gets MyIOProcessor class
    │
    ▼
vLLM instantiates MyModel(vllm_config=...) → loads weights → serves
MyIOProcessor handles all /pooling request I/O
```

---

## Weight Loading Patterns

### Pattern 1: AutoWeightsLoader + WeightsMapper (simplest)

Use when your model extends a vLLM built-in and only needs prefix remapping.

```python
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_prefix={
        "custom_text_proj.": "projection.linear.",
    }
)

def load_weights(self, weights):
    loader = AutoWeightsLoader(self)
    return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
```

**Used by:** `colqwen3`, `collfm2`

### Pattern 2: Manual key mapping

Use when HF checkpoint key names differ structurally from vLLM's.

```python
def load_weights(self, weights):
    params = dict(self.model.named_parameters())
    for name, tensor in weights:
        target = self._map_key(name)
        if target and target in params:
            params[target].data.copy_(tensor)
```

**Used by:** `mmbert_gliner`, `mt5_gliner`

### Pattern 3: Separate head checkpoint

Use when pooler weights are in a separate file (e.g., `custom_head.pt`).

```python
head_path = Path(model_dir) / "custom_head.pt"
if head_path.exists():
    self.pooler.load_state_dict(torch.load(head_path), strict=False)
```

---

## Reference Plugins by Complexity

| Plugin | Backbone | Pooler | IO Processor | Complexity |
|--------|----------|--------|-------------|------------|
| `colqwen3` | vLLM built-in | PassthroughPooler | FactoryIOProcessor | Low |
| `collfm2` | vLLM built-in | PassthroughPooler | FactoryIOProcessor | Low |
| `moderncolbert` | Custom encoder | ColBERTPooler (shared) | FactoryIOProcessor | Medium |
| `mmbert_gliner` | Custom encoder | GLiNERSpanPooler (shared) | FactoryIOProcessor | Medium |
| `mt5_gliner` | Custom encoder | GLiNERSpanPooler (shared) | FactoryIOProcessor | High |
| `deberta_gliner_linker` | Custom encoder | LinkerPooler (custom) | FactoryIOProcessor | High |

---

## Parity Testing

Every plugin must prove its vLLM output matches the reference HuggingFace /
PyTorch implementation.

### Parity Thresholds

| Precision | min_cosine_sim | Notes |
|-----------|---------------|-------|
| BF16 (standard) | 0.999 | Standard for serving |
| NER models | recall = 1.0 | Every reference entity must be found |
| FP8 | 0.99 | Quantized models |

### What to Test

1. **Shape parity** — output dimensions match reference
2. **Value parity** — cosine similarity above threshold (embeddings) or entity recall = 1.0 (NER)
3. **Edge cases** — empty input, max-length input, batch of 1 vs many
4. **Weight loading** — all expected weights loaded (check `load_weights` return set)

---

## Optional extras (`gliner` / `gliner2`)

`vllm-factory` installs and **registers** without those packages. vLLM calls
every `vllm.general_plugins` `register()` at engine start, so:

- No `import gliner` / `import gliner2` at module scope on a plugin's
  registration path (`__init__.py` → `config.py` / `model.py`).
- Import those libraries inside `__init__` / method bodies via
  `vllm_factory.optional_deps.require(module, extra, purpose=...)`.
- `pip install vllm-factory[gliner2]` pulls `gliner2>=2.0.0`;
  `vllm-factory[all]` unions every extra.

`tests/test_plugin_registration_without_extras.py` blocks `gliner`/`gliner2`
with a `sys.meta_path` finder and asserts every `register()` still succeeds.

---

## Common Gotchas

| Issue | Solution |
|-------|----------|
| `KeyError: model not registered` | Ensure `register()` is called in `__init__.py` AND entry point is in `pyproject.toml` |
| `attention_bias mismatch` (Qwen3) | Use `Qwen3Model`, not `Qwen2Model` — they differ on `attention_bias` |
| `Weight shape mismatch` | Check `WeightsMapper` prefix order (longer prefixes first) |
| `NaN in classification` | Return hidden states from `forward()`, NOT logits — the pooler applies the head |
| `Unsupported task: 'plugin'` | Set `pooling_task = "token_embed"` on your IOProcessor for embedding-style models |
| `extra_kwargs not reaching pooler` | Ensure `factory_pre_process` returns extra_kwargs as the 3rd tuple element |
