"""GLiNER 2.5 boundary parity test against AutoExtractor.

Two-phase design:
    Phase 1 (--prepare): AutoExtractor reference + vLLM model dir
    Phase 2 (--test):    vLLM inference + key-set / output comparison

Target: vllm==0.20.0, checkpoint fastino/gliner2.5-multi-v1 (334 head tensors).

Usage:
    python plugins/deberta_gliner25/parity_test.py --prepare
    python plugins/deberta_gliner25/parity_test.py --test
"""

from __future__ import annotations

import argparse
import json
import os
import time

MODEL = "fastino/gliner2.5-multi-v1"
LOCAL_MODEL_DIR = "/tmp/gliner25-multi-vllm"
REF_FILE = "/tmp/gliner25-multi-reference.json"

TEXT = (
    "John Smith works at NVIDIA Corporation in Santa Clara, California. "
    "His email is john.smith@nvidia.com and phone number is 555-123-4567. "
    "He is the VP of AI Research and reports to Jensen Huang."
)

SCHEMA = {
    "entities": {
        "person": "",
        "organization": "",
        "location": "",
        "email": "",
        "phone_number": "",
    },
    "classifications": [
        {
            "task": "topic",
            "labels": ["technology", "finance", "sports", "healthcare"],
        }
    ],
    "relations": {"works_at": "", "reports_to": ""},
    "structures": {
        "employee": {
            "fields": [
                {"name": "name", "dtype": "str"},
                {"name": "title", "dtype": "str"},
                {"name": "company", "dtype": "str"},
            ]
        }
    },
}

THRESHOLD = 0.5
EXPECTED_HEAD_TENSORS = 334
# bf16 on the vLLM path against fp32 eager in AutoExtractor.
CONFIDENCE_TOLERANCE = 0.05


def _entity_spans(payload: dict) -> dict[str, list[tuple]]:
    """Map each entity type to its extracted ``(text, start, end)`` spans."""
    spans: dict[str, list[tuple]] = {}
    for label, records in (payload.get("entities") or {}).items():
        found = []
        for record in records:
            if isinstance(record, dict):
                found.append(
                    (record.get("text"), record.get("start"), record.get("end"))
                )
            else:
                found.append((record, None, None))
        spans[label] = found
    return spans


def _entity_confidences(payload: dict) -> dict[tuple, float]:
    """Map each ``(type, text)`` to its confidence, where one was returned."""
    scores: dict[tuple, float] = {}
    for label, records in (payload.get("entities") or {}).items():
        for record in records:
            if isinstance(record, dict) and "confidence" in record:
                scores[label, record.get("text")] = float(record["confidence"])
    return scores


def _classifications(payload: dict) -> dict[str, str]:
    """Map each classification task to its winning label."""
    picked: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, dict) and "label" in value:
            picked[key] = str(value["label"])
        elif isinstance(value, str) and key != "text":
            picked[key] = value
    return picked


def compare_outputs(
    reference: dict, candidate: dict, *, tolerance: float = CONFIDENCE_TOLERANCE
) -> list[str]:
    """Diff a vLLM result against the AutoExtractor reference.

    Args:
        reference: Formatted AutoExtractor output.
        candidate: Formatted vLLM output for the same text and schema.
        tolerance: Largest confidence difference treated as agreement.

    Returns:
        One human-readable line per disagreement; empty means parity.
    """
    problems: list[str] = []
    ref_spans, got_spans = _entity_spans(reference), _entity_spans(candidate)
    for label in sorted(set(ref_spans) | set(got_spans)):
        expected, actual = ref_spans.get(label, []), got_spans.get(label, [])
        if expected != actual:
            problems.append(f"entities[{label}]: expected {expected}, got {actual}")

    ref_scores, got_scores = (
        _entity_confidences(reference),
        _entity_confidences(candidate),
    )
    for key in sorted(set(ref_scores) & set(got_scores)):
        delta = abs(ref_scores[key] - got_scores[key])
        if delta > tolerance:
            problems.append(
                f"confidence{list(key)}: {ref_scores[key]:.4f} vs "
                f"{got_scores[key]:.4f} (delta {delta:.4f} > {tolerance})"
            )

    ref_cls, got_cls = _classifications(reference), _classifications(candidate)
    for task in sorted(set(ref_cls) | set(got_cls)):
        if ref_cls.get(task) != got_cls.get(task):
            problems.append(
                f"classification[{task}]: expected {ref_cls.get(task)!r}, got {got_cls.get(task)!r}"
            )

    skip = {"text"}
    ref_keys = set(reference) - skip
    got_keys = set(candidate) - skip
    if ref_keys != got_keys:
        problems.append(f"keys: expected {sorted(ref_keys)}, got {sorted(got_keys)}")
    return problems


def phase_prepare(
    model_name: str = MODEL,
    local_model_dir: str = LOCAL_MODEL_DIR,
    ref_file: str = REF_FILE,
) -> None:
    from gliner2 import AutoExtractor

    from forge.model_prep import prepare_gliner25_model

    print("=" * 60)
    print(f"PHASE 1: AutoExtractor reference ({model_name})")
    print("=" * 60)

    extractor = AutoExtractor.from_pretrained(model_name)
    extractor.eval()
    reference = extractor.extract(
        TEXT,
        SCHEMA,
        threshold=THRESHOLD,
        include_confidence=True,
        include_spans=True,
    )
    print(json.dumps(reference, indent=2, default=str)[:4000])

    state = extractor.state_dict()
    head_keys = [
        k
        for k in state
        if k.startswith(
            ("boundary_head.", "record_decoder.", "relation_scorer.", "classifier.")
        )
    ]
    print(f"Head tensors: {len(head_keys)}")
    if len(head_keys) != EXPECTED_HEAD_TENSORS:
        raise SystemExit(
            f"Expected {EXPECTED_HEAD_TENSORS} head tensors, got {len(head_keys)}"
        )

    os.makedirs(os.path.dirname(ref_file) or ".", exist_ok=True)
    with open(ref_file, "w") as f:
        json.dump(
            {"model": model_name, "text": TEXT, "output": reference}, f, default=str
        )

    prepared = prepare_gliner25_model(
        model_name, output_dir=local_model_dir, force=True
    )
    print(f"Prepared model dir: {prepared}")
    print("Phase 1 complete")


def phase_test(
    model_name: str = MODEL,
    local_model_dir: str = LOCAL_MODEL_DIR,
    ref_file: str = REF_FILE,
) -> bool:
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    from vllm.pooling_params import PoolingParams

    from gliner2.inference.runtime import format_results as gliner2_format_results

    from plugins.deberta_gliner2.processor import normalize_gliner2_schema
    from plugins.deberta_gliner25.processor import (
        decode_boundary_output,
        preprocess_boundary,
        schema_format_args,
    )

    print("=" * 60)
    print(f"PHASE 2: vLLM inference + parity ({model_name})")
    print("=" * 60)

    with open(ref_file) as f:
        ref = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    schema = normalize_gliner2_schema(SCHEMA)
    prep = preprocess_boundary(
        tokenizer,
        TEXT,
        schema,
        threshold=THRESHOLD,
        include_confidence=True,
        include_spans=True,
    )
    prompt_ids = prep["input_ids"]
    extra = prep["extra_kwargs"]

    llm = LLM(
        model=local_model_dir,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.78,
    )
    prompt = TokensPrompt(prompt_token_ids=prompt_ids)
    pooling_params = PoolingParams(task="plugin", extra_kwargs=extra)
    _ = llm.encode([prompt], pooling_params=pooling_params, pooling_task="plugin")

    n = 5
    t0 = time.perf_counter()
    for _ in range(n):
        outputs = llm.encode(
            [prompt], pooling_params=pooling_params, pooling_task="plugin"
        )
    latency = (time.perf_counter() - t0) / n * 1000
    raw = outputs[0].outputs.data
    decoded = decode_boundary_output(raw, schema)
    requested_relations, classification_tasks = schema_format_args(schema)
    formatted = gliner2_format_results(
        decoded,
        include_confidence=True,
        requested_relations=requested_relations,
        classification_tasks=classification_tasks,
    )
    print(json.dumps(formatted, indent=2, default=str)[:4000])
    print(f"Latency: {latency:.1f}ms")

    problems = compare_outputs(ref.get("output") or {}, formatted)
    for problem in problems:
        print(f"  MISMATCH {problem}")
    print("PASS" if not problems else f"FAIL ({len(problems)} mismatch(es))")
    return not problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if not args.prepare and not args.test:
        phase_prepare()
        ok = phase_test()
        raise SystemExit(0 if ok else 1)
    if args.prepare:
        phase_prepare()
    if args.test:
        ok = phase_test()
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
