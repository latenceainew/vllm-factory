# Contributing

## Quick Setup

> **Critical: vLLM must be installed last.** The Makefile handles this automatically.

```bash
git clone https://github.com/latenceainew/vllm-factory.git
cd vllm-factory
make install          # installs deps in correct order + applies patch
```

## Adding a Plugin

See [docs/PLUGIN_GUIDE.md](docs/PLUGIN_GUIDE.md) for the step-by-step guide.

Each plugin lives in `plugins/<name>/` and needs:

| File | Purpose |
|---|---|
| `__init__.py` | Model registration via `forge.registration.register_plugin()` |
| `config.py` | HuggingFace-compatible config (dimensions, layers) |
| `model.py` | Encoder forward path + `self.pooler` wiring |
| `io_processor.py` | IOProcessor — parse, pre-process, post-process, response |
| `parity_test.py` | Validation against reference implementation |
| `generate_reference.py` | Script to generate reference outputs |

The IOProcessor is registered in `pyproject.toml` under `[project.entry-points."vllm.io_processor_plugins"]`.

## Code Standards

- **Logging**: Use `logging.getLogger(__name__)` — no `print()` in production code
- **Type hints**: All public methods must have return type annotations
- **Error handling**: Catch specific exceptions, never bare `except Exception: pass`
- **IOProcessor-first**: All new plugins must include an IOProcessor for server-side pre/post-processing
- **Parity tests**: Every plugin must have end-to-end `vllm serve` parity tests

## Running Tests

```bash
make test-serve P=moderncolbert     # Single plugin (end-to-end via vllm serve)
make test-all                       # All 12 plugins
make test P=moderncolbert           # Offline parity test
make lint                           # Ruff check
make ci-test                        # Full CI checks
```

## Dependency Management

vLLM Factory uses a split-install approach to avoid dependency conflicts:

1. `pip install -e ".[gliner]"` — installs base deps + GLiNER
2. `pip install "vllm==0.15.1"` — installs vLLM **last**, pinning shared deps

Never add `vllm` to the `dependencies` list in `pyproject.toml`. It must always be installed as a separate step after all other packages.

## Roadmap and Onboarding

- Roadmap: [`ROADMAP.md`](ROADMAP.md)
- New contributor path: [`docs/contributor_first_week.md`](docs/contributor_first_week.md)
