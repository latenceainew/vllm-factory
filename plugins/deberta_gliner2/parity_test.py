"""
GLiNER2 Parity Test — covers both supported checkpoint variants.

Two-phase design:
    Phase 1 (--prepare): GLiNER2 reference + model dir preparation
    Phase 2 (--test):    vLLM inference + parity comparison

The ``--model`` flag selects the HF checkpoint and picks the ``counting_layer``
variant exercised by the pooler:

    large  →  fastino/gliner2-large-v1   (counting_layer == "count_lstm")
    base   →  fastino/gliner2-base-v1    (counting_layer == "count_lstm_v2")

With no flags the script runs both phases for *both* variants in sequence so
any regression in the non-``count_lstm`` path surfaces immediately (see the
long-lived silent-drop bug fixed by the PR that introduced this.)

Usage:
    python plugins/deberta_gliner2/parity_test.py --prepare --model base
    python plugins/deberta_gliner2/parity_test.py --test    --model base
    python plugins/deberta_gliner2/parity_test.py                           # all variants
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Short-name → (HF repo, local model dir, reference-JSON path).
# The two variants correspond to the two ``counting_layer`` values used by
# every currently-published GLiNER2 checkpoint.
MODELS: dict[str, tuple[str, str, str]] = {
    "large": (
        "fastino/gliner2-large-v1",
        "/tmp/gliner2-large-vllm",
        "/tmp/gliner2-large-reference.json",
    ),
    "base": (
        "fastino/gliner2-base-v1",
        "/tmp/gliner2-base-vllm",
        "/tmp/gliner2-base-reference.json",
    ),
}

# Default / fallback — kept for backwards compatibility with any external
# invocation that imports these module-level names.
MODEL, LOCAL_MODEL_DIR, REF_FILE = MODELS["large"]

TEXT = (
    "John Smith works at NVIDIA Corporation in Santa Clara, California. "
    "His email is john.smith@nvidia.com and phone number is 555-123-4567. "
    "He is the VP of AI Research and reports to Jensen Huang."
)

ENTITY_LABELS = ["person", "organization", "location", "email", "phone_number"]
THRESHOLD = 0.5


# ======================================================================
# Phase 1: Generate references + model dir
# ======================================================================


def phase_prepare(
    model_name: str = MODEL, local_model_dir: str = LOCAL_MODEL_DIR, ref_file: str = REF_FILE
):
    import safetensors.torch
    from gliner2 import GLiNER2

    print("=" * 60)
    print(f"PHASE 1: GLiNER2 Reference + Model Directory ({model_name})")
    print("=" * 60)

    model = GLiNER2.from_pretrained(model_name)
    model.eval()

    # Generate entity reference
    entities = model.extract_entities(
        TEXT,
        ENTITY_LABELS,
        threshold=THRESHOLD,
        include_confidence=True,
        include_spans=True,
    )

    # Generate classification reference
    classification = model.classify_text(
        TEXT,
        {
            "topic": {
                "labels": ["technology", "finance", "sports", "healthcare"],
                "multi_label": False,
            }
        },
        include_confidence=True,
    )

    # Generate relation reference
    relations = model.extract_relations(
        TEXT,
        ["works_at", "reports_to"],
        threshold=0.3,
        include_confidence=True,
    )

    # Generate JSON structure reference
    json_result = model.extract_json(
        TEXT,
        {"employee": ["name::str", "title::str", "company::str", "location::str", "email::str"]},
        threshold=0.3,
        include_confidence=True,
    )

    print("\n--- Entity Results ---")
    print(json.dumps(entities, indent=2, default=str))
    print("\n--- Classification Results ---")
    print(json.dumps(classification, indent=2, default=str))
    print("\n--- Relation Results ---")
    print(json.dumps(relations, indent=2, default=str))
    print("\n--- JSON Results ---")
    print(json.dumps(json_result, indent=2, default=str))

    # Per-entity threshold reference — person at 0.9 should return fewer results
    # than the default 0.5 threshold
    from gliner2 import Schema

    per_threshold_schema = Schema().entities({"person": {"threshold": 0.9}, "email": {}})
    per_threshold_result = model.batch_extract(
        [TEXT],
        per_threshold_schema,
        threshold=THRESHOLD,
        include_confidence=True,
        include_spans=True,
    )
    per_threshold_entities = (
        per_threshold_result[0].get("entities", {}) if per_threshold_result else {}
    )

    print("\n--- Per-Entity Threshold Results ---")
    print(json.dumps(per_threshold_entities, indent=2, default=str))

    # Save references
    refs = {
        "model": model_name,
        "text": TEXT,
        "entities": entities,
        "classification": classification,
        "relations": relations,
        "json_structure": json_result,
        "per_threshold_entities": per_threshold_entities,
    }
    with open(ref_file, "w") as f:
        json.dump(refs, f, indent=2, default=str)
    print(f"\nSaved references to {ref_file}")

    # Prepare vLLM model dir
    ec = model.encoder.config
    vllm_config = {
        "model_type": "gliner2",
        "architectures": ["GLiNER2VLLMModel"],
        "num_hidden_layers": 0,
        "num_attention_heads": 1,
        "hidden_size": ec.hidden_size,
        "encoder_model_name": model.config.model_name,
        "vocab_size": len(model.processor.tokenizer),
        "encoder_hidden_size": ec.hidden_size,
        "encoder_num_hidden_layers": ec.num_hidden_layers,
        "encoder_num_attention_heads": ec.num_attention_heads,
        "encoder_intermediate_size": ec.intermediate_size,
        "encoder_hidden_act": ec.hidden_act,
        "encoder_hidden_dropout_prob": 0.0,
        "encoder_attention_probs_dropout_prob": 0.0,
        "encoder_max_position_embeddings": ec.max_position_embeddings,
        "encoder_type_vocab_size": ec.type_vocab_size,
        "encoder_layer_norm_eps": ec.layer_norm_eps,
        "encoder_relative_attention": ec.relative_attention,
        "encoder_max_relative_positions": ec.max_relative_positions,
        "encoder_position_buckets": ec.position_buckets,
        "encoder_pos_att_type": ec.pos_att_type,
        "encoder_share_att_key": ec.share_att_key,
        "encoder_norm_rel_ebd": ec.norm_rel_ebd,
        "encoder_position_biased_input": ec.position_biased_input,
        "encoder_pad_token_id": ec.pad_token_id,
        "max_width": model.config.max_width,
        "counting_layer": model.config.counting_layer,
        "token_pooling": model.config.token_pooling,
    }

    os.makedirs(local_model_dir, exist_ok=True)
    with open(os.path.join(local_model_dir, "config.json"), "w") as f:
        json.dump(vllm_config, f, indent=2)

    # Save weights
    state = model.state_dict()
    deduped = {}
    seen = set()
    for k, v in state.items():
        ptr = v.data_ptr()
        if ptr not in seen:
            deduped[k] = v.contiguous().cpu()
            seen.add(ptr)
        else:
            deduped[k] = v.clone().contiguous().cpu()
    safetensors.torch.save_file(deduped, os.path.join(local_model_dir, "model.safetensors"))

    # Save tokenizer
    model.processor.tokenizer.save_pretrained(local_model_dir)
    print(
        f"Model dir: {local_model_dir} ({len(deduped)} weights, "
        f"counting_layer={model.config.counting_layer})"
    )
    print("✅ Phase 1 complete\n")


# ======================================================================
# Phase 2: vLLM inference + parity comparison
# ======================================================================


def phase_test(
    model_name: str = MODEL, local_model_dir: str = LOCAL_MODEL_DIR, ref_file: str = REF_FILE
) -> bool:
    from transformers import AutoTokenizer
    from vllm import LLM, PoolingParams
    from vllm.inputs import TokensPrompt

    from plugins.deberta_gliner2.processor import (
        decode_output,
        format_results,
        normalize_gliner2_schema,
        preprocess,
    )

    print("=" * 60)
    print(f"PHASE 2: vLLM Inference + Parity ({model_name})")
    print("=" * 60)

    # Load references
    with open(ref_file) as f:
        ref = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

    # ---- TEST 1: Entity Extraction ----
    print("\n--- TEST 1: Entity Extraction ---")
    schema = normalize_gliner2_schema(
        {
            "entities": {
                "person": "",
                "organization": "",
                "location": "",
                "email": "",
                "phone_number": "",
            }
        }
    )
    prep = preprocess(tokenizer, TEXT, schema)
    prompt_ids = prep["input_ids"]
    prep = {k: v for k, v in prep.items() if k != "input_ids"}

    vllm_model = LLM(
        model=local_model_dir,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.78,
    )

    prompt = TokensPrompt(prompt_token_ids=prompt_ids)
    pooling_params = PoolingParams(task="plugin", extra_kwargs=prep)

    # Warmup
    _ = vllm_model.encode(
        [prompt],
        pooling_params=pooling_params,
        pooling_task="plugin",
    )

    # Timed run
    N = 5
    t0 = time.perf_counter()
    for _ in range(N):
        outputs = vllm_model.encode(
            [prompt],
            pooling_params=pooling_params,
            pooling_task="plugin",
        )
    latency = (time.perf_counter() - t0) / N * 1000

    raw = outputs[0].outputs.data
    result = decode_output(raw, schema)
    formatted = format_results(result, include_confidence=True)
    print(f"vLLM output: {json.dumps(formatted, indent=2, default=str)}")
    print(f"Latency: {latency:.1f}ms")

    # Compare entities
    ref_entities = ref.get("entities", {}).get("entities", {})
    vllm_entities = formatted.get("entities", {})

    ref_set = set()
    for etype, elist in ref_entities.items():
        for e in elist if isinstance(elist, list) else [elist]:
            if isinstance(e, dict):
                ref_set.add((e.get("text", e), etype))
            else:
                ref_set.add((e, etype))

    vllm_set = set()
    for etype, elist in vllm_entities.items():
        for e in elist if isinstance(elist, list) else [elist]:
            if isinstance(e, dict):
                vllm_set.add((e.get("text", e), etype))
            else:
                vllm_set.add((e, etype))

    matched = ref_set & vllm_set
    if ref_set:
        recall = len(matched) / len(ref_set)
    else:
        recall = 1.0
    if vllm_set:
        precision = len(matched) / len(vllm_set)
    else:
        precision = 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nEntity Parity: P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
    print(f"Reference: {len(ref_set)}, vLLM: {len(vllm_set)}, Matched: {len(matched)}")
    if ref_set - vllm_set:
        print(f"  Missing: {ref_set - vllm_set}")
    if vllm_set - ref_set:
        print(f"  Extra:   {vllm_set - ref_set}")

    # ---- TEST 2: Classification ----
    print("\n--- TEST 2: Classification ---")
    cls_schema = normalize_gliner2_schema(
        {
            "classifications": [
                {
                    "task": "topic",
                    "labels": ["technology", "finance", "sports", "healthcare"],
                }
            ]
        }
    )
    cls_prep = preprocess(tokenizer, TEXT, cls_schema)
    cls_prompt = TokensPrompt(prompt_token_ids=cls_prep["input_ids"])
    cls_prep = {k: v for k, v in cls_prep.items() if k != "input_ids"}
    cls_pp = PoolingParams(task="plugin", extra_kwargs=cls_prep)
    cls_out = vllm_model.encode(
        [cls_prompt],
        pooling_params=cls_pp,
        pooling_task="plugin",
    )
    cls_result = decode_output(cls_out[0].outputs.data, cls_schema)
    cls_formatted = format_results(cls_result, include_confidence=True)
    print(f"vLLM: {json.dumps(cls_formatted, indent=2, default=str)}")

    ref_cls = ref.get("classification", {})
    ref_label = (
        ref_cls.get("topic", {}).get("label", "")
        if isinstance(ref_cls.get("topic"), dict)
        else ref_cls.get("topic", "")
    )
    vllm_label = (
        cls_formatted.get("topic", {}).get("label", "")
        if isinstance(cls_formatted.get("topic"), dict)
        else cls_formatted.get("topic", "")
    )
    cls_match = ref_label == vllm_label
    print(f"Classification Parity: ref={ref_label}, vllm={vllm_label}, match={cls_match}")

    # ---- TEST 3: Relations ----
    print("\n--- TEST 3: Relation Extraction ---")
    rel_schema = normalize_gliner2_schema(
        {
            "relations": {
                "works_at": "Employment relationship",
                "reports_to": "Reporting relationship",
            }
        }
    )
    rel_prep = preprocess(tokenizer, TEXT, rel_schema)
    rel_prompt = TokensPrompt(prompt_token_ids=rel_prep["input_ids"])
    rel_prep = {k: v for k, v in rel_prep.items() if k != "input_ids"}
    rel_pp = PoolingParams(task="plugin", extra_kwargs=rel_prep)
    rel_out = vllm_model.encode(
        [rel_prompt],
        pooling_params=rel_pp,
        pooling_task="plugin",
    )
    rel_result = decode_output(rel_out[0].outputs.data, rel_schema)
    rel_formatted = format_results(rel_result, include_confidence=True)
    print(f"vLLM: {json.dumps(rel_formatted, indent=2, default=str)}")

    # ---- TEST 4: JSON Structure ----
    print("\n--- TEST 4: JSON Structure ---")
    json_schema = normalize_gliner2_schema(
        {
            "structures": {
                "employee": {
                    "fields": [
                        {"name": "name", "dtype": "str"},
                        {"name": "title", "dtype": "str"},
                        {"name": "company", "dtype": "str"},
                        {"name": "location", "dtype": "str"},
                        {"name": "email", "dtype": "str"},
                    ]
                }
            }
        }
    )
    json_prep = preprocess(tokenizer, TEXT, json_schema)
    json_prompt = TokensPrompt(prompt_token_ids=json_prep["input_ids"])
    json_prep = {k: v for k, v in json_prep.items() if k != "input_ids"}
    json_pp = PoolingParams(task="plugin", extra_kwargs=json_prep)
    json_out = vllm_model.encode(
        [json_prompt],
        pooling_params=json_pp,
        pooling_task="plugin",
    )
    json_result = decode_output(json_out[0].outputs.data, json_schema)
    json_formatted = format_results(json_result, include_confidence=True)
    print(f"vLLM: {json.dumps(json_formatted, indent=2, default=str)}")

    # ---- TEST 5: Mixed Canonical Schema ----
    print("\n--- TEST 5: Mixed Canonical Schema ---")
    mixed_schema = normalize_gliner2_schema(
        {
            "entities": {
                "person": "Person names",
                "organization": "Organization names",
            },
            "classifications": [
                {
                    "task": "topic",
                    "labels": ["technology", "finance", "sports", "healthcare"],
                }
            ],
            "relations": {
                "works_at": "Employment relationship",
            },
            "structures": {
                "employee": {
                    "fields": [
                        {"name": "name", "dtype": "str"},
                        {"name": "company", "dtype": "str"},
                    ]
                }
            },
        }
    )
    mixed_prep = preprocess(tokenizer, TEXT, mixed_schema)
    mixed_prompt = TokensPrompt(prompt_token_ids=mixed_prep["input_ids"])
    mixed_pp = PoolingParams(
        task="plugin",
        extra_kwargs={k: v for k, v in mixed_prep.items() if k != "input_ids"},
    )
    mixed_out = vllm_model.encode(
        [mixed_prompt],
        pooling_params=mixed_pp,
        pooling_task="plugin",
    )
    mixed_result = decode_output(mixed_out[0].outputs.data, mixed_schema)
    mixed_formatted = format_results(
        mixed_result,
        include_confidence=True,
        include_spans=True,
    )
    print(f"vLLM: {json.dumps(mixed_formatted, indent=2, default=str)}")

    # ──────────────────────────────────────────────────────────────
    # Per-entity threshold parity
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Per-entity threshold parity")
    print("=" * 60)

    ref_per_threshold = ref.get("per_threshold_entities", {})
    if ref_per_threshold:
        per_t_schema = normalize_gliner2_schema(
            {
                "entities": {
                    "person": {"threshold": 0.9},
                    "email": {},
                }
            }
        )
        per_t_pp = PoolingParams(
            additional_data=json.dumps(
                {
                    "schema": {
                        "entities": {
                            "person": {"threshold": 0.9},
                            "email": {},
                        }
                    },
                    "threshold": THRESHOLD,
                    "include_confidence": True,
                    "include_spans": True,
                }
            )
        )
        per_t_out = vllm_model.encode(
            TokensPrompt(
                prompt_token_ids=preprocess(
                    tokenizer,
                    TEXT,
                    per_t_schema,
                )["input_ids"]
            ),
            pooling_params=per_t_pp,
            pooling_task="plugin",
        )
        per_t_result = decode_output(per_t_out[0].outputs.data, per_t_schema)
        per_t_formatted = format_results(
            per_t_result,
            threshold=THRESHOLD,
            include_confidence=True,
            include_spans=True,
        )

        per_t_entities = per_t_formatted.get("entities", {})
        print(f"Reference (native):  {json.dumps(ref_per_threshold, indent=2, default=str)}")
        print(f"vLLM per-threshold:  {json.dumps(per_t_entities, indent=2, default=str)}")

        # Verify that high threshold for person reduces results vs default
        default_person_count = len(ref.get("entities", {}).get("person", []))
        per_t_person_count = len(per_t_entities.get("person", []))
        threshold_effective = per_t_person_count <= default_person_count
        print(
            f"Person count default={default_person_count} vs per-threshold={per_t_person_count}: "
            f"{'✅' if threshold_effective else '⚠️'}"
        )
    else:
        print("⚠️  No per-threshold reference found — skipping")
        threshold_effective = True

    # Final summary
    print("\n" + "=" * 60)
    print(f"SUMMARY — {model_name}")
    print("=" * 60)
    print(f"Entity F1:         {f1:.4f}")
    print(f"Classification:    {'✅' if cls_match else '❌'}")
    print(f"Per-threshold:     {'✅' if threshold_effective else '⚠️'}")
    print(f"Latency:           {latency:.1f}ms")
    passed = f1 >= 0.8 and cls_match and threshold_effective
    status = "✅ PARITY OK" if passed else "⚠️  NEEDS WORK"
    print(status)
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLiNER2 Parity Test")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--model",
        choices=sorted(MODELS.keys()),
        default=None,
        help="Which GLiNER2 variant to exercise (default: all, in sequence).",
    )
    args = parser.parse_args()

    selected = [args.model] if args.model else list(MODELS.keys())

    if args.prepare:
        for key in selected:
            hf, ldir, rfile = MODELS[key]
            phase_prepare(hf, ldir, rfile)
    elif args.test:
        all_passed = True
        for key in selected:
            hf, ldir, rfile = MODELS[key]
            if not phase_test(hf, ldir, rfile):
                all_passed = False
        sys.exit(0 if all_passed else 1)
    else:
        print(f"Running both phases in separate processes for variants={selected}...\n")
        for key in selected:
            r1 = subprocess.run(
                [sys.executable, __file__, "--prepare", "--model", key],
                cwd=os.getcwd(),
            )
            if r1.returncode != 0:
                sys.exit(r1.returncode)
            r2 = subprocess.run(
                [sys.executable, __file__, "--test", "--model", key],
                cwd=os.getcwd(),
            )
            if r2.returncode != 0:
                sys.exit(r2.returncode)
