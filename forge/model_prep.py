"""
Prepare GLiNER / GLiNER2 / GLiNER-Linker models for vLLM.

These models require a local directory with a vLLM-compatible config.json
because the HuggingFace config format differs from what vLLM expects.

Usage:
    from forge.model_prep import prepare_gliner_model
    model_dir = prepare_gliner_model("VAGOsolutions/SauerkrautLM-GLiNER", plugin="mmbert_gliner")
    llm = LLM(model=model_dir, trust_remote_code=True)
"""

import json
import logging
import os
import shutil

from huggingface_hub import hf_hub_download, list_repo_files

logger = logging.getLogger(__name__)

# Plugin → (model_type, architecture, config transform)
PLUGIN_REGISTRY = {
    "mmbert_gliner": {
        "model_type": "gliner_mmbert",
        "architectures": ["GLiNERModernBertModel"],
        "extra_fields": {
            "local_attention": 128,
            "global_attn_every_n_layers": 3,
            "global_rope_theta": 160000.0,
            "local_rope_theta": 10000.0,
            "hidden_activation": "gelu",
        },
    },
    "mmbert_gliner2": {
        "model_type": "mmbert_gliner2",
        "architectures": ["GLiNER2ModernBertModel"],
        "extra_fields": {},
    },
    "deberta_gliner": {
        "model_type": "gliner_deberta_v2",
        "architectures": ["GLiNERDebertaV2Model"],
        "extra_fields": {},
    },
    "mt5_gliner": {
        "model_type": "gliner_mt5",
        "architectures": ["GLiNERMT5Model"],
        "extra_fields": {},
        "hidden_size_key": "d_model",
    },
    "deberta_gliner2": {
        "model_type": "gliner2",
        "architectures": ["GLiNER2VLLMModel"],
        "extra_fields": {},
    },
    "deberta_gliner_linker": {
        "model_type": "gliner_linker",
        "architectures": ["GLiNERLinkerModel"],
        "extra_fields": {},
    },
}


def infer_gliner_plugin_from_model_name(
    model_name: str,
    encoder_model_type: str | None = None,
    model_ref: str | None = None,
) -> str | None:
    """Infer the vLLM Factory plugin from GLiNER metadata."""
    name = (model_name or "").lower()
    encoder = (encoder_model_type or "").lower()
    ref = (model_ref or "").lower()
    combined = " ".join([name, encoder, ref])

    if "rerank" in combined:
        return "modernbert_gliner_rerank"
    if "linker" in combined:
        return "deberta_gliner_linker"
    if "mt5" in combined:
        return "mt5_gliner"
    if "modernbert" in combined or "mmb" in name or "ettin" in name:
        return "mmbert_gliner"
    if "deberta" in combined:
        return "deberta_gliner"
    return None


def get_gliner_base_model_name(model_ref: str) -> str | None:
    """Return GLiNER base model name from gliner_config.json, if available."""
    if os.path.exists(model_ref) or "/" not in model_ref:
        return None
    try:
        repo_files = list_repo_files(model_ref)
    except Exception:
        return None
    if "gliner_config.json" not in repo_files:
        return None
    gliner_config_path = _download_file(model_ref, "gliner_config.json")
    if not gliner_config_path:
        return None
    gliner_cfg = _read_json(gliner_config_path)
    model_name = gliner_cfg.get("model_name")
    return model_name if isinstance(model_name, str) else None


def prepare_model_for_vllm_if_needed(
    model_ref: str,
    plugin: str | None = None,
    output_dir: str | None = None,
    force: bool = False,
) -> str:
    """Translate HF GLiNER repos into local vLLM-compatible model dirs.

    Returns the original model reference if no GLiNER translation is needed.
    """
    if os.path.exists(model_ref):
        return model_ref

    # Likely not an HF repo ID; leave untouched.
    if "/" not in model_ref:
        return model_ref

    try:
        repo_files = list_repo_files(model_ref)
    except Exception:
        return model_ref

    config_json_path = (
        _download_file(model_ref, "config.json") if "config.json" in repo_files else None
    )
    hf_config = _read_json(config_json_path) if config_json_path else {}

    if hf_config.get("model_type") == "extractor" and "encoder_config/config.json" in repo_files:
        enc_cfg_path = _download_file(model_ref, "encoder_config/config.json")
        enc_cfg = _read_json(enc_cfg_path) if enc_cfg_path else {}
        enc_model_type = enc_cfg.get("model_type", "")

        if plugin == "mmbert_gliner2" or enc_model_type == "modernbert":
            return prepare_mmbert_gliner2_model(
                hf_model_id=model_ref,
                output_dir=output_dir,
                force=force,
            )
        if plugin == "deberta_gliner2":
            return prepare_gliner2_model(
                hf_model_id=model_ref,
                output_dir=output_dir,
                force=force,
            )

    if "gliner_config.json" not in repo_files:
        return model_ref

    inferred_plugin = plugin
    if inferred_plugin is None:
        model_name = ""
        encoder_model_type = None
        gliner_config_path = _download_file(model_ref, "gliner_config.json")
        if gliner_config_path:
            gliner_cfg = _read_json(gliner_config_path)
            model_name = gliner_cfg.get("model_name", "")
            encoder_cfg = gliner_cfg.get("encoder_config", {})
            if isinstance(encoder_cfg, dict):
                encoder_model_type = encoder_cfg.get("model_type")
        inferred_plugin = infer_gliner_plugin_from_model_name(
            model_name=model_name,
            encoder_model_type=encoder_model_type,
            model_ref=model_ref,
        )

    if inferred_plugin is None:
        raise RuntimeError(
            f"Could not infer GLiNER plugin for '{model_ref}'. "
            "Pass an explicit plugin (e.g. deberta_gliner, mmbert_gliner, mt5_gliner)."
        )

    return prepare_gliner_model(
        hf_model_id=model_ref,
        plugin=inferred_plugin,
        output_dir=output_dir,
        force=force,
    )


def prepare_gliner2_model(
    hf_model_id: str,
    output_dir: str | None = None,
    force: bool = False,
) -> str:
    """Prepare a GLiNER2 extractor repo for the DeBERTa GLiNER2 vLLM plugin."""
    from transformers import AutoTokenizer

    if output_dir is None:
        slug = hf_model_id.replace("/", "--")
        output_dir = f"/tmp/{slug}-vllm"

    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path) and not force:
        cached = _read_json(config_path)
        if cached.get("model_type") == "gliner2":
            logger.info("Model already prepared at %s", output_dir)
            return output_dir
        shutil.rmtree(output_dir, ignore_errors=True)

    logger.info("Preparing %s for vLLM (deberta_gliner2)...", hf_model_id)

    repo_files = list_repo_files(hf_model_id)
    extractor_cfg = _read_json(
        _require_download(hf_model_id, "config.json", "GLiNER2 extractor config")
    )
    encoder_cfg = _read_json(
        _require_download(hf_model_id, "encoder_config/config.json", "GLiNER2 encoder config")
    )

    os.makedirs(output_dir, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load tokenizer for '{hf_model_id}'") from exc
    tokenizer.save_pretrained(output_dir)

    vllm_config = {
        "model_type": "gliner2",
        "architectures": ["GLiNER2VLLMModel"],
        "num_hidden_layers": 0,
        "num_attention_heads": 1,
        "hidden_size": encoder_cfg.get("hidden_size", 1024),
        "encoder_model_name": extractor_cfg.get("model_name", "microsoft/deberta-v3-large"),
        "vocab_size": len(tokenizer),
        "encoder_hidden_size": encoder_cfg.get("hidden_size", 1024),
        "encoder_num_hidden_layers": encoder_cfg.get("num_hidden_layers", 24),
        "encoder_num_attention_heads": encoder_cfg.get("num_attention_heads", 16),
        "encoder_intermediate_size": encoder_cfg.get("intermediate_size", 4096),
        "encoder_hidden_act": encoder_cfg.get("hidden_act", "gelu"),
        "encoder_hidden_dropout_prob": 0.0,
        "encoder_attention_probs_dropout_prob": 0.0,
        "encoder_max_position_embeddings": encoder_cfg.get("max_position_embeddings", 512),
        "encoder_type_vocab_size": encoder_cfg.get("type_vocab_size", 0),
        "encoder_layer_norm_eps": encoder_cfg.get("layer_norm_eps", 1e-7),
        "encoder_relative_attention": encoder_cfg.get("relative_attention", True),
        "encoder_max_relative_positions": encoder_cfg.get("max_relative_positions", -1),
        "encoder_position_buckets": encoder_cfg.get("position_buckets", 256),
        "encoder_pos_att_type": encoder_cfg.get("pos_att_type", ["p2c", "c2p"]),
        "encoder_share_att_key": encoder_cfg.get("share_att_key", True),
        "encoder_norm_rel_ebd": encoder_cfg.get("norm_rel_ebd", "layer_norm"),
        "encoder_position_biased_input": encoder_cfg.get("position_biased_input", False),
        "encoder_pad_token_id": encoder_cfg.get("pad_token_id", 0),
        "max_width": extractor_cfg.get("max_width", 8),
        "counting_layer": extractor_cfg.get("counting_layer", "count_lstm"),
        "token_pooling": extractor_cfg.get("token_pooling", "first"),
    }

    with open(config_path, "w") as f:
        json.dump(vllm_config, f, indent=2)

    weight_files = [f for f in repo_files if f.endswith((".safetensors", ".bin", ".pt"))]
    if not weight_files:
        raise RuntimeError(f"No model weights found for '{hf_model_id}'")
    for wf in weight_files:
        src = _require_download(hf_model_id, wf, "GLiNER2 weight file")
        dst = os.path.join(output_dir, wf)
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except OSError:
                logger.info("Symlink failed for %s; copying instead", wf)
                shutil.copy2(src, dst)

    logger.info("Model prepared at %s", output_dir)
    logger.info("Config: gliner2, GLiNER2VLLMModel")
    logger.info(
        "hidden=%s, encoder_layers=%s",
        vllm_config["encoder_hidden_size"],
        vllm_config["encoder_num_hidden_layers"],
    )
    return output_dir


_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "merges.txt",
    "added_tokens.json",
    "tokenizer.model",
]


def _copy_tokenizer_files(
    hf_model_id: str,
    repo_files: list[str],
    output_dir: str,
) -> None:
    """Download tokenizer files from HF repo and patch out unresolvable tokenizer_class."""
    copied = 0
    for fname in _TOKENIZER_FILES:
        if fname not in repo_files:
            continue
        src = _download_file(hf_model_id, fname)
        if src:
            dst = os.path.join(output_dir, fname)
            shutil.copy2(src, dst)
            copied += 1

    if copied == 0:
        raise RuntimeError(f"No tokenizer files found in {hf_model_id}")

    tok_cfg_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path):
        tok_cfg = _read_json(tok_cfg_path)
        tc = tok_cfg.get("tokenizer_class", "")
        known_classes = {
            "BertTokenizer",
            "BertTokenizerFast",
            "PreTrainedTokenizer",
            "PreTrainedTokenizerFast",
            "DebertaV2Tokenizer",
            "DebertaV2TokenizerFast",
            "T5Tokenizer",
            "T5TokenizerFast",
        }
        dirty = False
        if tc and tc not in known_classes:
            logger.info(
                "Replacing unresolvable tokenizer_class '%s' with 'PreTrainedTokenizerFast'", tc
            )
            tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            dirty = True

        est = tok_cfg.get("extra_special_tokens")
        if isinstance(est, list):
            tok_cfg["extra_special_tokens"] = {t: t for t in est}
            dirty = True

        if dirty:
            with open(tok_cfg_path, "w") as f:
                json.dump(tok_cfg, f, indent=2)


def prepare_mmbert_gliner2_model(
    hf_model_id: str,
    output_dir: str | None = None,
    force: bool = False,
) -> str:
    """Prepare a ModernBERT + GLiNER2 extractor repo for the mmbert_gliner2 plugin."""
    if output_dir is None:
        slug = hf_model_id.replace("/", "--")
        output_dir = f"/tmp/{slug}-vllm"

    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path) and not force:
        cached = _read_json(config_path)
        if cached.get("model_type") == "mmbert_gliner2":
            logger.info("Model already prepared at %s", output_dir)
            return output_dir
        shutil.rmtree(output_dir, ignore_errors=True)

    logger.info("Preparing %s for vLLM (mmbert_gliner2)...", hf_model_id)

    repo_files = list_repo_files(hf_model_id)
    extractor_cfg = _read_json(
        _require_download(hf_model_id, "config.json", "GLiNER2 extractor config")
    )
    encoder_cfg = _read_json(
        _require_download(hf_model_id, "encoder_config/config.json", "GLiNER2 encoder config")
    )

    os.makedirs(output_dir, exist_ok=True)

    _copy_tokenizer_files(hf_model_id, repo_files, output_dir)

    rope_params = encoder_cfg.get("rope_parameters", {})
    full_attn_rope = rope_params.get("full_attention", {})
    sliding_attn_rope = rope_params.get("sliding_attention", {})
    encoder_num_layers = encoder_cfg.get("num_hidden_layers", 22)
    global_attn_every_n_layers = encoder_cfg.get("global_attn_every_n_layers", 3)

    vllm_config = {
        "model_type": "mmbert_gliner2",
        "architectures": ["GLiNER2ModernBertModel"],
        "num_hidden_layers": 0,
        "num_attention_heads": 1,
        "hidden_size": encoder_cfg.get("hidden_size", 384),
        "encoder_num_layers": encoder_num_layers,
        "encoder_num_attention_heads": encoder_cfg.get("num_attention_heads", 6),
        "intermediate_size": encoder_cfg.get("intermediate_size", 1152),
        "vocab_size": encoder_cfg.get("vocab_size", 50368),
        "max_position_embeddings": encoder_cfg.get("max_position_embeddings", 8192),
        "hidden_activation": encoder_cfg.get("hidden_activation", "gelu"),
        "norm_eps": encoder_cfg.get("layer_norm_eps", encoder_cfg.get("norm_eps", 1e-5)),
        "pad_token_id": encoder_cfg.get("pad_token_id", 0),
        "local_attention": encoder_cfg.get("local_attention", 128),
        "global_attn_every_n_layers": global_attn_every_n_layers,
        "global_rope_theta": full_attn_rope.get("rope_theta", 160000.0),
        "local_rope_theta": sliding_attn_rope.get("rope_theta", 160000.0),
        "rope_parameters": {
            "full_attention": {
                "rope_theta": full_attn_rope.get("rope_theta", 160000.0),
                "rope_type": full_attn_rope.get("rope_type", "default"),
            },
            "sliding_attention": {
                "rope_theta": sliding_attn_rope.get("rope_theta", 160000.0),
                "rope_type": sliding_attn_rope.get("rope_type", "default"),
            },
        },
        "layer_types": encoder_cfg.get("layer_types")
        or [
            "full_attention" if i % global_attn_every_n_layers == 0 else "sliding_attention"
            for i in range(encoder_num_layers)
        ],
        "max_width": extractor_cfg.get("max_width", 12),
        "counting_layer": extractor_cfg.get("counting_layer", "count_lstm_v2"),
        "token_pooling": extractor_cfg.get("token_pooling", "first"),
    }

    with open(config_path, "w") as f:
        json.dump(vllm_config, f, indent=2)

    weight_files = [
        f for f in repo_files if f.endswith((".safetensors", ".bin", ".pt")) and "/" not in f
    ]
    if not weight_files:
        raise RuntimeError(f"No model weights found for '{hf_model_id}'")
    for wf in weight_files:
        src = _require_download(hf_model_id, wf, "GLiNER2 weight file")
        dst = os.path.join(output_dir, wf)
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except OSError:
                logger.info("Symlink failed for %s; copying instead", wf)
                shutil.copy2(src, dst)

    logger.info("Model prepared at %s", output_dir)
    logger.info("Config: mmbert_gliner2, GLiNER2ModernBertModel")
    logger.info(
        "hidden=%s, encoder_layers=%s",
        vllm_config["hidden_size"],
        vllm_config["encoder_num_layers"],
    )
    return output_dir


def _download_file(repo_id: str, filename: str) -> str | None:
    """Download a single file from HF repo, return local path or None."""
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception:
        return None


def _require_download(repo_id: str, filename: str, description: str) -> str:
    path = _download_file(repo_id, filename)
    if path is None:
        raise RuntimeError(f"Missing {description} '{filename}' for '{repo_id}'")
    return path


def _read_json(path: str) -> dict:
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)


def _build_gliner_tokenizer(hf_model_id: str, merged: dict, ent_token: str, sep_token: str):
    """Load the base tokenizer and add GLiNER special tokens (<<ENT>>, <<SEP>>)."""
    from transformers import AutoTokenizer

    tokenizer_source = (
        merged.get("model_name") if isinstance(merged.get("model_name"), str) else None
    )

    for source in [hf_model_id, tokenizer_source]:
        if source is None:
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(source)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"Could not load tokenizer from {hf_model_id} or {tokenizer_source}")

    tokens_to_add = []
    for tok in [ent_token, sep_token]:
        if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id:
            tokens_to_add.append(tok)

    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add)
        print(f"   Added special tokens: {tokens_to_add} (vocab {len(tokenizer)})")

    return tokenizer


def prepare_gliner_model(
    hf_model_id: str,
    plugin: str,
    output_dir: str | None = None,
    force: bool = False,
) -> str:
    """Prepare a GLiNER-family model for vLLM by creating a local directory
    with a vLLM-compatible config.json.

    Args:
        hf_model_id: HuggingFace model ID (e.g., "VAGOsolutions/SauerkrautLM-GLiNER")
        plugin: Plugin name (e.g., "mmbert_gliner")
        output_dir: Where to create the local model dir. Defaults to /tmp/<model-slug>-vllm
        force: If True, recreate even if output_dir already exists

    Returns:
        Path to the prepared model directory, ready for LLM(model=...)
    """
    if plugin not in PLUGIN_REGISTRY:
        raise ValueError(f"Unknown plugin '{plugin}'. Available: {list(PLUGIN_REGISTRY.keys())}")

    reg = PLUGIN_REGISTRY[plugin]

    # Default output dir
    if output_dir is None:
        slug = hf_model_id.replace("/", "--")
        output_dir = f"/tmp/{slug}-vllm"

    # Skip if already prepared with the correct plugin
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path) and not force:
        cached = _read_json(config_path)
        if cached.get("model_type") == reg["model_type"]:
            print(f"✅ Model already prepared at {output_dir}")
            return output_dir
        print(
            f"⚠️  Cached model_type '{cached.get('model_type')}' != '{reg['model_type']}', regenerating..."
        )
        shutil.rmtree(output_dir, ignore_errors=True)

    print(f"📦 Preparing {hf_model_id} for vLLM ({plugin})...")

    # List available files in the HF repo
    try:
        repo_files = list_repo_files(hf_model_id)
    except Exception:
        repo_files = []

    # Download config files — GLiNER models use various config formats
    hf_config = {}
    gliner_config = {}

    config_json_path = _download_file(hf_model_id, "config.json")
    if config_json_path:
        hf_config = _read_json(config_json_path)

    gliner_config_path = _download_file(hf_model_id, "gliner_config.json")
    if gliner_config_path:
        gliner_config = _read_json(gliner_config_path)

    # Merge: gliner_config has precedence, but hf_config has encoder details
    merged = {**hf_config, **gliner_config}

    # Extract encoder config (can be nested or top-level)
    encoder_config = merged
    if "encoder_config" in merged and isinstance(merged["encoder_config"], dict):
        encoder_config = merged["encoder_config"]
    elif "model_config" in merged and isinstance(merged["model_config"], dict):
        encoder_config = merged["model_config"]

    os.makedirs(output_dir, exist_ok=True)

    # GLiNER models add special tokens (<<ENT>>, <<SEP>>) at load time.
    # We must save the augmented tokenizer so vLLM can resolve them.
    ent_token = merged.get("ent_token", "<<ENT>>")
    sep_token = merged.get("sep_token", "<<SEP>>")

    tokenizer = _build_gliner_tokenizer(hf_model_id, merged, ent_token, sep_token)
    tokenizer.save_pretrained(output_dir)

    ent_token_id = tokenizer.convert_tokens_to_ids(ent_token)
    sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
    if ent_token_id == tokenizer.unk_token_id:
        ent_token_id = 0
    if sep_token_id == tokenizer.unk_token_id:
        sep_token_id = 0

    # GLiNER models have two distinct hidden sizes:
    #   encoder hidden_size  – the backbone transformer dimension (e.g. 1024 for ModernBERT-large)
    #   gliner  hidden_size  – the span/label projection dimension (top-level in gliner_config.json)
    # The pooler LSTM/span layers use gliner_hidden_size; the encoder uses hidden_size.
    gliner_span_hidden = gliner_config.get("hidden_size") if gliner_config else None

    # Build vLLM-compatible config
    vllm_config = {
        "model_type": reg["model_type"],
        "architectures": reg["architectures"],
        "num_hidden_layers": 0,  # vLLM pooling trick
        "num_attention_heads": 1,
        # Real encoder config
        "encoder_num_layers": encoder_config.get(
            "num_hidden_layers", encoder_config.get("num_layers", 24)
        ),
        "encoder_num_attention_heads": encoder_config.get(
            "num_attention_heads", encoder_config.get("num_heads", 12)
        ),
        "hidden_size": encoder_config.get(reg.get("hidden_size_key", "hidden_size"), 768),
        "intermediate_size": encoder_config.get("intermediate_size", 3072),
        "vocab_size": len(tokenizer),
        "max_position_embeddings": encoder_config.get("max_position_embeddings", 8192),
        "norm_eps": encoder_config.get("norm_eps", encoder_config.get("layer_norm_eps", 1e-5)),
        "pad_token_id": encoder_config.get(
            "pad_token_id", tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ),
        # GLiNER params
        "gliner_dropout": merged.get("dropout", 0.3),
        "max_width": merged.get("max_width", 12),
        "class_token_index": ent_token_id,
        "sep_token_index": sep_token_id,
        "ent_token": ent_token,
        "sep_token": sep_token,
        "has_rnn": merged.get("has_rnn", True),
        "embed_ent_token": merged.get("embed_ent_token", True),
        "max_len": merged.get("max_len", 2048),
        "span_mode": merged.get("span_mode", "markerV0"),
    }

    if gliner_span_hidden is not None:
        vllm_config["gliner_hidden_size"] = gliner_span_hidden

    # Merge plugin-specific extra fields as defaults; encoder_config values take precedence
    for k, v in reg["extra_fields"].items():
        vllm_config[k] = encoder_config.get(k, v)

    # For MT5/T5 models, add encoder-specific fields the config class needs
    if reg["model_type"] == "gliner_mt5":
        t5_fields = {
            "d_model": encoder_config.get("d_model", 1024),
            "d_kv": encoder_config.get("d_kv", 64),
            "d_ff": encoder_config.get("d_ff", 2816),
            "num_layers": encoder_config.get("num_layers", 24),
            "num_heads": encoder_config.get("num_heads", 16),
            "relative_attention_num_buckets": encoder_config.get(
                "relative_attention_num_buckets", 32
            ),
            "relative_attention_max_distance": encoder_config.get(
                "relative_attention_max_distance", 128
            ),
            "dropout_rate": encoder_config.get("dropout_rate", 0.1),
            "layer_norm_epsilon": encoder_config.get("layer_norm_epsilon", 1e-6),
            "feed_forward_proj": encoder_config.get("feed_forward_proj", "gated-gelu"),
            "gliner_hidden_size": encoder_config.get("d_model", 1024),
        }
        for k, v in t5_fields.items():
            vllm_config.setdefault(k, v)

    # Save config
    with open(config_path, "w") as f:
        json.dump(vllm_config, f, indent=2)

    # Preserve the original GLiNER config for IO processors that need the
    # native data processor/decoder stack (notably token-level GLiNER).
    if gliner_config_path:
        dst = os.path.join(output_dir, "gliner_config.json")
        if not os.path.exists(dst):
            os.symlink(gliner_config_path, dst)

    # Symlink model weight files
    weight_files = [f for f in repo_files if f.endswith((".safetensors", ".bin", ".pt"))]
    for wf in weight_files:
        src = _download_file(hf_model_id, wf)
        if src:
            dst = os.path.join(output_dir, wf)
            if not os.path.exists(dst):
                os.symlink(src, dst)

    print(f"✅ Model prepared at {output_dir}")
    print(f"   Config: {reg['model_type']}, {reg['architectures'][0]}")
    print(
        f"   hidden={vllm_config['hidden_size']}, encoder_layers={vllm_config['encoder_num_layers']}"
    )
    return output_dir
