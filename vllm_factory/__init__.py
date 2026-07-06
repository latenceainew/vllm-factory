"""vllm_factory — stable internal layers for vLLM Factory.

Subpackages:
    api      — FactoryRequest / FactoryResponse (version-stable public contract)
    compat   — vLLM capability detection, bridges, legacy patch, doctor
    registry — thin entry-point registration wrappers
    pooling  — pooler context / adapter layer decoupling pooler math from vLLM metadata
"""

__version__ = "0.2.3"
