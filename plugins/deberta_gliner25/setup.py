"""Setup for GLiNER 2.5 vLLM plugin."""

from setuptools import setup

setup(
    name="vllm-factory-deberta-gliner25",
    version="0.1.0",
    description="GLiNER 2.5 boundary plugin for vLLM",
    package_dir={"deberta_gliner25": "."},
    packages=["deberta_gliner25"],
    python_requires=">=3.11",
    install_requires=[
        "vllm==0.20.0",
        "torch>=2.0",
        "transformers>=4.40",
    ],
    entry_points={
        "vllm.general_plugins": [
            "deberta_gliner25 = deberta_gliner25:register",
        ],
        "vllm.io_processor_plugins": [
            "deberta_gliner25_io = deberta_gliner25.io_processor:get_processor_cls",
        ],
    },
)
