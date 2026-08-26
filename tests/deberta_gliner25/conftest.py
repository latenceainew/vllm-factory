"""Conftest for deberta_gliner25 CPU-only tests.

Loads processor.py by file path so tests never import the plugin __init__
(which chain-imports model.py → vllm), and offers ``load_pooler_module`` for
tests that need poolers/gliner25.py where vLLM may be absent.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "deberta_gliner25"
_PROCESSOR_PATH = _PLUGIN_DIR / "processor.py"
_SPAN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "deberta_gliner2"

for pkg_name, pkg_path in [
    ("plugins", [str(_PLUGIN_DIR.parent)]),
    ("plugins.deberta_gliner25", [str(_PLUGIN_DIR)]),
    ("plugins.deberta_gliner2", [str(_SPAN_DIR)]),
]:
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = pkg_path
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

_span_proc = "plugins.deberta_gliner2.processor"
if _span_proc not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _span_proc, str(_SPAN_DIR / "processor.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_span_proc] = mod
    spec.loader.exec_module(mod)

_mod_name = "plugins.deberta_gliner25.processor"
if _mod_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(_mod_name, str(_PROCESSOR_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_mod_name] = mod
    spec.loader.exec_module(mod)

_ROOT = Path(__file__).resolve().parents[2]
# poolers/gliner25.py needs these; vllm_factory.pooling.__init__ pulls in the
# vLLM adapter, so the whole chain is unimportable without vllm installed.
_VLLM_MODULES = (
    "vllm",
    "vllm.config",
    "vllm_factory",
    "vllm_factory.pooling",
    "vllm_factory.pooling.protocol",
    "vllm_factory.pooling.vllm_adapter",
)


@contextlib.contextmanager
def stubbed_vllm() -> Iterator[None]:
    """Stub only the vLLM modules that are genuinely unimportable.

    Restores ``sys.modules`` on exit: a stub left behind shadows the real
    module for every test that runs afterwards, which is a failure the
    polluting test never sees itself.

    Yields:
        None, with the stubs installed for the duration of the block.
    """
    added: list[str] = []
    for name in _VLLM_MODULES:
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except ImportError:
            stub = ModuleType(name)
            stub.__path__ = []  # ty: ignore[unresolved-attribute]
            sys.modules[name] = stub
            added.append(name)
    protocol = sys.modules["vllm_factory.pooling.protocol"]
    if not hasattr(protocol, "PoolerContext"):
        protocol.PoolerContext = type("PoolerContext", (), {})
    if not hasattr(protocol, "split_hidden_states"):
        protocol.split_hidden_states = lambda *a, **kw: None
    try:
        yield
    finally:
        for name in reversed(added):
            sys.modules.pop(name, None)


def _ensure_optional_deps() -> None:
    """Make ``vllm_factory.optional_deps`` importable under a stubbed package."""
    name = "vllm_factory.optional_deps"
    if hasattr(sys.modules.get(name), "require"):
        return
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "vllm_factory" / "optional_deps.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def _load_pooler_module(name: str) -> ModuleType:
    """Exec poolers/gliner25.py by path, leaving sys.modules as it was.

    Args:
        name: Module name to register the loaded pooler under.

    Returns:
        The executed pooler module.
    """
    with stubbed_vllm():
        _ensure_optional_deps()
        spec = importlib.util.spec_from_file_location(
            name, str(_ROOT / "poolers" / "gliner25.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pooler_mod() -> ModuleType:
    """poolers/gliner25.py, loaded without importing poolers/__init__.py."""
    return _load_pooler_module("gliner25_pooler_under_test")
