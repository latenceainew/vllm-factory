"""Plugin registry — declares models, serve flags, vanilla baselines, and datasets."""

from __future__ import annotations

import functools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_BASE_WORDS_EN = (
    "What is the significance of transformer architecture in modern deep learning "
    "applications including attention mechanisms self supervised pretraining and "
    "transfer learning for natural language processing computer vision and "
    "multimodal understanding tasks in production environments"
).split()

_BASE_WORDS_NER = (
    "Max Mustermann the lead research engineer at Siemens AG headquartered "
    "in Munich presented results on neural network optimization at the "
    "International Conference on Machine Learning in Vienna Austria on 15 "
    "January 2025 together with colleagues from Google DeepMind and OpenAI"
).split()


def _generate_texts(base_words: list[str], target_tokens: int, n: int = 100) -> list[str]:
    target_words = max(8, int(target_tokens * 0.75))
    texts = []
    for i in range(n):
        words = []
        while len(words) < target_words:
            words.extend(base_words)
        words = words[:target_words]
        words[0] = f"Sample{i}"
        texts.append(" ".join(words))
    return texts


def dataset_embedding(seq_len: int = 128) -> list[str]:
    return _generate_texts(_BASE_WORDS_EN, seq_len)


def dataset_colbert(seq_len: int = 128) -> list[str]:
    return _generate_texts(_BASE_WORDS_EN, seq_len)


def dataset_ner(seq_len: int = 128) -> list[dict]:
    texts = _generate_texts(_BASE_WORDS_NER, seq_len)
    return [
        {"text": t, "labels": ["person", "organization", "location", "date", "event"]}
        for t in texts
    ]


# ---------------------------------------------------------------------------
# SciFact corpus (BEIR) — document encoding benchmark for ColBERT models
# ---------------------------------------------------------------------------

_SCIFACT_DIR = Path("/tmp/beir_scifact/scifact")
_SCIFACT_SAMPLE_COUNT = 512


def _ensure_scifact():
    """Download SciFact corpus from BEIR if not already cached."""
    if (_SCIFACT_DIR / "corpus.jsonl").exists():
        return
    import io
    import zipfile

    import requests as req

    print("  [dataset] Downloading SciFact corpus from BEIR...")
    r = req.get(
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        timeout=120,
    )
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(_SCIFACT_DIR.parent)
    print(f"  [dataset] SciFact extracted to {_SCIFACT_DIR}")


@functools.lru_cache(maxsize=1)
def _load_scifact_docs(n: int = _SCIFACT_SAMPLE_COUNT) -> list[str]:
    """Load *n* random SciFact document texts (title + body)."""
    _ensure_scifact()
    docs: list[str] = []
    with open(_SCIFACT_DIR / "corpus.jsonl") as f:
        for line in f:
            row = json.loads(line)
            text = ((row.get("title") or "") + " " + (row.get("text") or "")).strip()
            if len(text) > 50:
                docs.append(text)
    if len(docs) > n:
        docs = random.Random(42).sample(docs, n)
    return docs


def dataset_scifact_colbert(seq_len: int = 512) -> list[dict]:
    """SciFact docs as ``{"text": ..., "is_query": False}`` for ModernColBERT / ColBERT-Zero."""
    return [{"text": d, "is_query": False} for d in _load_scifact_docs()]


def dataset_scifact_lfm2(seq_len: int = 512) -> list[dict]:
    """SciFact docs as ``{"text": ...}`` for LFM2-ColBERT (no is_query flag)."""
    return [{"text": d} for d in _load_scifact_docs()]


# ---------------------------------------------------------------------------
# DocVQA images — document image encoding benchmark for ColPali / ColEmbed
# ---------------------------------------------------------------------------

_DOCVQA_IMAGE_DIR = Path("/tmp/bench_docvqa")
_DOCVQA_SAMPLE_COUNT = 512


@functools.lru_cache(maxsize=1)
def _load_docvqa_image_paths(n: int = _DOCVQA_SAMPLE_COUNT) -> list[str]:
    """Load DocVQA images from disk, download via streaming if missing."""
    img_dir = _DOCVQA_IMAGE_DIR
    manifest = img_dir / "manifest.json"
    if manifest.exists():
        paths = json.loads(manifest.read_text())
        if len(paths) >= n and all(Path(p).exists() for p in paths[:5]):
            return paths[:n]

    from datasets import load_dataset

    print(f"  [dataset] Loading DocumentVQA via streaming ({n} images)...")
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="test", streaming=True)

    img_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        path = img_dir / f"{i}.png"
        if not path.exists():
            item["image"].save(str(path))
        paths.append(str(path))
        if (i + 1) % 100 == 0:
            print(f"  [dataset] Saved {i + 1}/{n} images...")

    manifest.write_text(json.dumps(paths))
    print(f"  [dataset] {len(paths)} images ready at {img_dir}")
    return paths


def dataset_docvqa_colpali(seq_len: int = 512) -> list[dict]:
    """DocVQA image paths as ``{"image": path}`` for ColPali-family models."""
    return [{"image": p} for p in _load_docvqa_image_paths()]


def dataset_docvqa_nemotron(seq_len: int = 512) -> list[dict]:
    """DocVQA image paths as ``{"image": path, "is_query": False}`` for Nemotron ColEmbed."""
    return [{"image": p, "is_query": False} for p in _load_docvqa_image_paths()]


_GLINER2_MODEL_ID = "fastino/gliner2-large-v1"
_GLINER2_SAMPLE_COUNT = 512
_GLINER2_MAX_SEQ_LEN = 768
_MMBERT_GLINER_MODEL_ID = "VAGOsolutions/SauerkrautLM-GLiNER"
_MMBERT_GLINER_LOCAL_DIR = "/tmp/sauerkraut-gliner-vllm"
_MT5_GLINER_MODEL_ID = "knowledgator/gliner-x-large"
_MT5_GLINER_LOCAL_DIR = "/tmp/gliner-x-large-vllm"
_GLINER_LINKER_MODEL_ID = "knowledgator/gliner-linker-large-v1.0"
_GLINER_RERANK_MODEL_ID = "knowledgator/gliner-linker-rerank-v1.0"


def _parse_nuner_row(row: dict) -> dict | None:
    text = row.get("input", row.get("text", ""))
    raw_output = row.get("output", "")
    if not text or not raw_output:
        return None

    labels_set: set[str] = set()
    raw_str = str(raw_output)
    for sep in (" <> ", " < "):
        if sep in raw_str:
            for chunk in raw_str.split("'"):
                chunk = chunk.strip()
                if sep in chunk:
                    label = chunk.split(sep, 1)[1].strip().rstrip("']").lower()
                    if label:
                        labels_set.add(label)
            break

    if not labels_set or len(text) <= 20:
        return None
    return {"text": text, "labels": sorted(labels_set)}


@functools.lru_cache(maxsize=4)
def _load_nuner_samples(
    n: int = _GLINER2_SAMPLE_COUNT,
) -> list[dict]:
    """Load a fixed-seed random NuNER sample for GLiNER2 benchmarking."""
    from datasets import load_dataset

    ds = load_dataset("numind/NuNER", split="entity[:8192]")
    samples = []
    for row in ds:
        sample = _parse_nuner_row(row)
        if sample is None:
            continue
        samples.append(sample)

    if len(samples) >= n:
        return random.Random(42).sample(samples, n)
    if samples:
        return samples

    return [
        {"text": t, "labels": ["person", "organization", "location"]}
        for t in _generate_texts(_BASE_WORDS_NER, 128, n=50)
    ]


def dataset_gliner2(seq_len: int = 128) -> list[dict]:
    return _load_nuner_samples()


def dataset_nuner_random(seq_len: int = 128) -> list[dict]:
    return _load_nuner_samples()


_FIXED_NER_LABELS = ["person", "organization", "location", "date", "event"]


def dataset_nuner_fixed_labels(seq_len: int = 128) -> list[dict]:
    """NuNER samples with a single fixed label set — avoids repeated label
    encoding in bi-encoder models like the GLiNER Linker."""
    samples = _load_nuner_samples()
    return [{"text": s["text"], "labels": _FIXED_NER_LABELS} for s in samples]


@dataclass
class PluginEntry:
    plugin_name: str
    model_id: str
    io_plugin: str
    serve_flags: list[str]
    vanilla_family: str
    dataset_fn: Callable = field(repr=False)
    seq_len: int = 128
    endpoint: str = "/v1/pooling"
    parity_metric: str = "cosine_sim"
    dataset_label: str = ""
    payload_key: str = "data"
    request_task: str | None = None
    vanilla_batch_size: int | None = None
    vanilla_kwargs: dict = field(default_factory=dict)
    prep_fn: Callable | None = field(default=None, repr=False)

    def get_dataset(self) -> list:
        return self.dataset_fn(self.seq_len)


def _colbert_flags() -> list[str]:
    return [
        "--enforce-eager",
        "--dtype", "bfloat16",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--gpu-memory-utilization", "0.7",
        "--max-model-len", "8192",
        "--max-num-batched-tokens", "8192",
    ]


def _ner_flags() -> list[str]:
    return [
        "--dtype", "bfloat16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--gpu-memory-utilization", "0.80",
    ]


def _embedding_flags() -> list[str]:
    return [
        "--dtype", "bfloat16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--gpu-memory-utilization", "0.80",
    ]


def _colpali_flags() -> list[str]:
    return [
        "--dtype", "bfloat16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--skip-mm-profiling",
        "--mm-processor-cache-gb", "1",
        "--limit-mm-per-prompt", '{"image": 1}',
        "--gpu-memory-utilization", "0.7",
        "--max-model-len", "8192",
        "--max-num-batched-tokens", "8192",
    ]


def _nemotron_flags() -> list[str]:
    return [
        "--dtype", "bfloat16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--skip-mm-profiling",
        "--mm-processor-cache-gb", "1",
        "--limit-mm-per-prompt", '{"image": 1}',
        "--gpu-memory-utilization", "0.80",
        "--max-model-len", "8192",
        "--max-num-batched-tokens", "8192",
    ]


def _prep_gliner2():
    """Run parity_test.py --prepare to create /tmp/gliner2-vllm model dir."""
    import subprocess
    import sys
    from pathlib import Path

    model_dir = Path("/tmp/gliner2-vllm")
    if (model_dir / "config.json").exists() and (model_dir / "model.safetensors").exists():
        print("  [prep] /tmp/gliner2-vllm already exists, skipping.")
        return

    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "plugins" / "deberta_gliner2" / "parity_test.py"
    print("  [prep] Preparing GLiNER2 model dir via parity_test.py --prepare ...")
    result = subprocess.run(
        [sys.executable, str(script), "--prepare"],
        cwd=str(repo_root),
    )
    if result.returncode != 0:
        raise RuntimeError("GLiNER2 model preparation failed")
    print("  [prep] GLiNER2 model dir ready.")


def _prep_mmbert_gliner():
    """Prepare the Sauerkraut GLiNER ModernBERT model dir for vLLM."""
    from forge.model_prep import prepare_gliner_model

    out = prepare_gliner_model(
        hf_model_id=_MMBERT_GLINER_MODEL_ID,
        plugin="mmbert_gliner",
        output_dir=_MMBERT_GLINER_LOCAL_DIR,
    )
    print(f"  [prep] MMBERT GLiNER model dir ready at {out}")


def _prep_mt5_gliner():
    """Prepare the knowledgator/gliner-x-large MT5 model dir for vLLM."""
    from forge.model_prep import prepare_gliner_model

    out = prepare_gliner_model(
        hf_model_id=_MT5_GLINER_MODEL_ID,
        plugin="mt5_gliner",
        output_dir=_MT5_GLINER_LOCAL_DIR,
    )
    print(f"  [prep] MT5 GLiNER model dir ready at {out}")


def _prep_gliner_linker():
    """Prepare the knowledgator/gliner-linker-large-v1.0 model dir for vLLM."""
    from plugins.deberta_gliner_linker import prepare_model_dir
    out = prepare_model_dir()
    print(f"  [prep] GLiNER Linker model dir ready at {out}")


def _prep_gliner_rerank():
    """Prepare the knowledgator/gliner-linker-rerank-v1.0 model dir for vLLM."""
    from plugins.modernbert_gliner_rerank import prepare_model_dir
    out = prepare_model_dir()
    print(f"  [prep] GLiNER Rerank model dir ready at {out}")


_MODERNCOLBERT_MODEL_ID = "VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT"
_COLBERT_ZERO_MODEL_ID = "lightonai/ColBERT-Zero"
# Canonical HF id uses capital O; resolves same as lightonai/lateon
_LATEON_MODEL_ID = "lightonai/LateOn"
_LFM2_COLBERT_MODEL_ID = "LiquidAI/LFM2-ColBERT-350M"
_COLLFM2_MODEL_ID = "VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1"
_COLQWEN3_MODEL_ID = "VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1"
_NEMOTRON_MODEL_ID = "nvidia/nemotron-colembed-vl-4b-v2"


REGISTRY: list[PluginEntry] = [
    # ---- Text retrieval (ColBERT) ---- #
    PluginEntry(
        plugin_name="moderncolbert",
        model_id=_MODERNCOLBERT_MODEL_ID,
        io_plugin="moderncolbert_io",
        serve_flags=_colbert_flags(),
        vanilla_family="pylate_colbert",
        dataset_fn=dataset_scifact_colbert,
        seq_len=512,
        endpoint="/pooling",
        parity_metric="cosine_sim",
        dataset_label="BEIR SciFact 512 docs",
        request_task="token_embed",
        # README / methodology: vs vanilla at batch_size=1 (not batched PyLate @ concurrency).
        vanilla_batch_size=1,
    ),
    PluginEntry(
        plugin_name="colbert_zero",
        model_id=_COLBERT_ZERO_MODEL_ID,
        io_plugin="moderncolbert_io",
        serve_flags=_colbert_flags(),
        vanilla_family="pylate_colbert",
        dataset_fn=dataset_scifact_colbert,
        seq_len=512,
        endpoint="/pooling",
        parity_metric="cosine_sim",
        dataset_label="BEIR SciFact 512 docs",
        request_task="token_embed",
        vanilla_kwargs={"prompt_name": "document"},
        vanilla_batch_size=1,
    ),
    PluginEntry(
        plugin_name="lfm2_colbert",
        model_id=_LFM2_COLBERT_MODEL_ID,
        io_plugin="lfm2_colbert_io",
        serve_flags=_colbert_flags(),
        vanilla_family="pylate_colbert",
        dataset_fn=dataset_scifact_lfm2,
        seq_len=512,
        endpoint="/pooling",
        parity_metric="cosine_sim",
        dataset_label="BEIR SciFact 512 docs",
        request_task="token_embed",
        vanilla_batch_size=1,
    ),
    PluginEntry(
        plugin_name="lateon",
        model_id=_LATEON_MODEL_ID,
        io_plugin="moderncolbert_io",
        serve_flags=_colbert_flags(),
        vanilla_family="pylate_colbert",
        dataset_fn=dataset_scifact_colbert,
        seq_len=512,
        endpoint="/pooling",
        parity_metric="cosine_sim",
        dataset_label="BEIR SciFact 512 docs",
        request_task="token_embed",
        vanilla_batch_size=1,
    ),
    # ---- Vision retrieval (ColPali / ColEmbed) ---- #
    PluginEntry(
        plugin_name="collfm2",
        model_id=_COLLFM2_MODEL_ID,
        io_plugin="collfm2_io",
        serve_flags=_colpali_flags(),
        vanilla_family="sauerkraut_colpali",
        dataset_fn=dataset_docvqa_colpali,
        seq_len=512,
        endpoint="/pooling",
        parity_metric="cosine_sim",
        dataset_label="DocumentVQA 512 images",
        request_task="token_embed",
        vanilla_kwargs={"model_class": "ColLFM2"},
    ),
    PluginEntry(
        plugin_name="colqwen3",
        model_id=_COLQWEN3_MODEL_ID,
        io_plugin="colqwen3_io",
        serve_flags=_colpali_flags(),
        vanilla_family="sauerkraut_colpali",
        dataset_fn=dataset_docvqa_colpali,
        seq_len=512,
        endpoint="/pooling",
        parity_metric="cosine_sim",
        dataset_label="DocumentVQA 512 images",
        request_task="token_embed",
        vanilla_kwargs={"model_class": "ColQwen3"},
    ),
    PluginEntry(
        plugin_name="nemotron_colembed",
        model_id=_NEMOTRON_MODEL_ID,
        io_plugin="nemotron_colembed_io",
        serve_flags=_nemotron_flags(),
        vanilla_family="nemotron_transformers",
        dataset_fn=dataset_docvqa_nemotron,
        seq_len=512,
        endpoint="/pooling",
        parity_metric="cosine_sim",
        dataset_label="DocumentVQA 512 images",
        request_task="token_embed",
    ),
    # ---- NER models ---- #
    PluginEntry(
        plugin_name="mt5_gliner",
        model_id=_MT5_GLINER_LOCAL_DIR,
        io_plugin="mt5_gliner_io",
        serve_flags=_ner_flags(),
        vanilla_family="gliner",
        dataset_fn=dataset_nuner_random,
        seq_len=_GLINER2_MAX_SEQ_LEN,
        endpoint="/pooling",
        parity_metric="entity_recall",
        dataset_label="NuNER random 512 samples",
        request_task="plugin",
        vanilla_batch_size=1,
        vanilla_kwargs={"hf_model_id": _MT5_GLINER_MODEL_ID},
        prep_fn=_prep_mt5_gliner,
    ),
    PluginEntry(
        plugin_name="embeddinggemma",
        model_id="unsloth/embeddinggemma-300m",
        io_plugin="embeddinggemma_io",
        serve_flags=_embedding_flags(),
        vanilla_family="sentence_transformers",
        dataset_fn=dataset_embedding,
        seq_len=128,
        endpoint="/v1/embeddings",
        payload_key="input",
        parity_metric="cosine_sim",
    ),
    PluginEntry(
        plugin_name="deberta_gliner2",
        model_id="/tmp/gliner2-vllm",
        io_plugin="deberta_gliner2_io",
        serve_flags=_ner_flags(),
        vanilla_family="gliner2",
        dataset_fn=dataset_gliner2,
        seq_len=_GLINER2_MAX_SEQ_LEN,
        endpoint="/pooling",
        parity_metric="entity_recall",
        dataset_label="NuNER random 512 samples",
        request_task="plugin",
        vanilla_batch_size=1,
        vanilla_kwargs={"hf_model_id": _GLINER2_MODEL_ID},
        prep_fn=_prep_gliner2,
    ),
    PluginEntry(
        plugin_name="mmbert_gliner",
        model_id=_MMBERT_GLINER_LOCAL_DIR,
        io_plugin="mmbert_gliner_io",
        serve_flags=_ner_flags(),
        vanilla_family="gliner",
        dataset_fn=dataset_nuner_random,
        seq_len=_GLINER2_MAX_SEQ_LEN,
        endpoint="/pooling",
        parity_metric="entity_recall",
        dataset_label="NuNER random 512 samples",
        request_task="plugin",
        vanilla_batch_size=1,
        vanilla_kwargs={"hf_model_id": _MMBERT_GLINER_MODEL_ID},
        prep_fn=_prep_mmbert_gliner,
    ),
    PluginEntry(
        plugin_name="deberta_gliner_linker",
        model_id=str(Path(__file__).resolve().parent.parent / "plugins" / "deberta_gliner_linker" / "_model_cache"),
        io_plugin="deberta_gliner_linker_io",
        serve_flags=_ner_flags(),
        vanilla_family="glinker",
        dataset_fn=dataset_nuner_fixed_labels,
        seq_len=_GLINER2_MAX_SEQ_LEN,
        endpoint="/pooling",
        parity_metric="entity_recall",
        dataset_label="NuNER 512 samples (fixed 5 labels)",
        request_task="plugin",
        vanilla_batch_size=1,
        vanilla_kwargs={"hf_model_id": _GLINER_LINKER_MODEL_ID, "layer": "l3"},
        prep_fn=_prep_gliner_linker,
    ),
    PluginEntry(
        plugin_name="modernbert_gliner_rerank",
        model_id=str(Path(__file__).resolve().parent.parent / "plugins" / "modernbert_gliner_rerank" / "_model_cache"),
        io_plugin="modernbert_gliner_rerank_io",
        serve_flags=_ner_flags(),
        vanilla_family="glinker",
        dataset_fn=dataset_nuner_fixed_labels,
        seq_len=_GLINER2_MAX_SEQ_LEN,
        endpoint="/pooling",
        parity_metric="entity_recall",
        dataset_label="NuNER 512 samples (fixed 5 labels)",
        request_task="plugin",
        vanilla_batch_size=1,
        vanilla_kwargs={"hf_model_id": _GLINER_RERANK_MODEL_ID, "layer": "l4"},
        prep_fn=_prep_gliner_rerank,
    ),
]


def get_entry(plugin_name: str) -> PluginEntry:
    for entry in REGISTRY:
        if entry.plugin_name == plugin_name:
            return entry
    available = [e.plugin_name for e in REGISTRY]
    raise KeyError(f"Unknown plugin {plugin_name!r}. Available: {available}")


def list_plugins() -> list[str]:
    return [e.plugin_name for e in REGISTRY]
