"""Plugin ``register()`` must succeed when optional extras are missing.

vLLM imports every ``vllm.general_plugins`` entry point at engine start.
A module-level ``import gliner`` / ``import gliner2`` on that path makes a
bare ``pip install vllm-factory`` fail for every plugin.
"""

from __future__ import annotations

import importlib
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _BlockOptional:
    """Raise ``ModuleNotFoundError`` for ``gliner`` / ``gliner2`` (and submodules)."""

    BLOCKED = frozenset({"gliner", "gliner2"})

    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> ModuleSpec | None:
        root = fullname.split(".", 1)[0]
        if root in self.BLOCKED:
            raise ModuleNotFoundError(fullname)
        return None


def _general_plugin_modules() -> list[str]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    entries = data["project"]["entry-points"]["vllm.general_plugins"]
    modules = []
    for spec in entries.values():
        modules.append(spec.split(":", 1)[0])
    return modules


@pytest.fixture
def block_gliner_extras():
    finder = _BlockOptional()
    sys.meta_path.insert(0, finder)
    blocked = [name for name in list(sys.modules) if name.split(".", 1)[0] in _BlockOptional.BLOCKED]
    saved = {name: sys.modules.pop(name) for name in blocked}
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(saved)


def _caused_by_blocked_extra(exc: BaseException) -> bool:
    """True when the failure is a missing ``gliner`` / ``gliner2`` extra."""
    current: BaseException | None = exc
    while current is not None:
        name = getattr(current, "name", None) or ""
        root = str(name).split(".", 1)[0]
        if root in _BlockOptional.BLOCKED:
            return True
        message = str(current).lower()
        if "no module named 'gliner" in message or message.startswith("gliner2"):
            return True
        current = current.__cause__ or current.__context__
    return False


def _drop_plugin_stubs() -> None:
    """Remove empty ``plugins.*`` stubs injected by sibling conftests."""
    for name in list(sys.modules):
        if name != "plugins" and not name.startswith("plugins."):
            continue
        module = sys.modules[name]
        if getattr(module, "__file__", None) is None and not hasattr(module, "register"):
            del sys.modules[name]


def test_every_general_plugin_register_without_gliner_extras(block_gliner_extras):
    _drop_plugin_stubs()
    failures: list[str] = []
    for mod_name in _general_plugin_modules():
        try:
            module = importlib.import_module(mod_name)
            register = getattr(module, "register", None)
            if register is None:
                failures.append(f"{mod_name}: no register()")
                continue
            register()
        except ImportError as exc:
            if _caused_by_blocked_extra(exc):
                failures.append(f"{mod_name}: {exc}")
    assert failures == []
