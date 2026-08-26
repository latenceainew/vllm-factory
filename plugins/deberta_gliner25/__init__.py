"""GLiNER 2.5 plugin — DeBERTa v2 encoder + boundary pooler."""

from forge.registration import register_plugin

from .config import GLiNER25Config
from .model import GLiNER25VLLMModel


def register() -> None:
    register_plugin("gliner25", GLiNER25Config, "GLiNER25VLLMModel", GLiNER25VLLMModel)


register()

__all__ = ["GLiNER25VLLMModel", "GLiNER25Config"]
