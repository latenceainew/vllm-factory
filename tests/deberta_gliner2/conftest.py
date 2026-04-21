"""Conftest for deberta_gliner2 CPU-only tests.

Imports processor.py directly by file path to avoid loading the plugin's
__init__.py (which chain-imports model.py → vllm → CUDA, etc.).
The functions are injected as module-level attributes in this conftest
so test files can ``from conftest import normalize_gliner2_schema`` or
use fixtures, but we also register the module in sys.modules so a
normal ``from plugins.deberta_gliner2.processor import ...`` works.
"""

import importlib.util
import sys
import types
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "deberta_gliner2"
_PROCESSOR_PATH = _PLUGIN_DIR / "processor.py"

# Register stub packages so `from plugins.deberta_gliner2.processor import ...`
# works without triggering __init__.py.
for pkg_name, pkg_path in [
    ("plugins", [str(_PLUGIN_DIR.parent)]),
    ("plugins.deberta_gliner2", [str(_PLUGIN_DIR)]),
]:
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = pkg_path
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

# Load processor.py into the fake package.
_mod_name = "plugins.deberta_gliner2.processor"
if _mod_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(_mod_name, str(_PROCESSOR_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_mod_name] = mod
    spec.loader.exec_module(mod)
