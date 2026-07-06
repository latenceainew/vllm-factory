"""Detection tests for prepare_model_for_vllm_if_needed (issue #20).

GLiNER2 extractor repos (config.json model_type == "extractor" with an
encoder_config/ subdir) must route to the right prepare_* function from
metadata alone — no --plugin flag. Hub access is stubbed; the prepare_*
functions themselves are stubbed so no downloads or weights are involved.
"""

import json

import pytest

from forge import model_prep


def _fake_hub(monkeypatch, tmp_path, repo_files, configs):
    """Stub list_repo_files/hf_hub_download inside forge.model_prep.

    ``configs`` maps repo filename → dict written as JSON to a temp file.
    """

    def fake_list_repo_files(repo_id):
        return list(repo_files)

    def fake_hf_hub_download(repo_id=None, filename=None, **kwargs):
        if filename not in configs:
            raise FileNotFoundError(filename)
        path = tmp_path / filename.replace("/", "__")
        path.write_text(json.dumps(configs[filename]))
        return str(path)

    monkeypatch.setattr(model_prep, "list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(model_prep, "hf_hub_download", fake_hf_hub_download)


@pytest.fixture
def extractor_repo():
    return {
        "repo_files": [
            "config.json",
            "encoder_config/config.json",
            "model.safetensors",
            "tokenizer.json",
        ],
        "configs": {
            "config.json": {
                "model_type": "extractor",
                "model_name": "microsoft/deberta-v3-base",
                "max_width": 8,
            },
        },
    }


def _route_called(monkeypatch):
    """Replace both prepare functions with recorders; return the record."""
    calls = {}

    def _recorder(name, result):
        def _record(**kw):
            calls[name] = kw
            return result

        return _record

    monkeypatch.setattr(
        model_prep, "prepare_gliner2_model", _recorder("deberta_gliner2", "/prepared/deberta")
    )
    monkeypatch.setattr(
        model_prep,
        "prepare_mmbert_gliner2_model",
        _recorder("mmbert_gliner2", "/prepared/mmbert"),
    )
    return calls


def test_deberta_v2_extractor_autodetects_gliner2(monkeypatch, tmp_path, extractor_repo):
    """fastino/gliner2-base-v1 shape: extractor + deberta-v2 encoder, no --plugin."""
    extractor_repo["configs"]["encoder_config/config.json"] = {"model_type": "deberta-v2"}
    _fake_hub(monkeypatch, tmp_path, extractor_repo["repo_files"], extractor_repo["configs"])
    calls = _route_called(monkeypatch)

    result = model_prep.prepare_model_for_vllm_if_needed("fastino/gliner2-base-v1")

    assert result == "/prepared/deberta"
    assert "deberta_gliner2" in calls


def test_modernbert_extractor_autodetects_mmbert_gliner2(monkeypatch, tmp_path, extractor_repo):
    extractor_repo["configs"]["encoder_config/config.json"] = {"model_type": "modernbert"}
    _fake_hub(monkeypatch, tmp_path, extractor_repo["repo_files"], extractor_repo["configs"])
    calls = _route_called(monkeypatch)

    result = model_prep.prepare_model_for_vllm_if_needed("acme/gliner2-modernbert")

    assert result == "/prepared/mmbert"
    assert "mmbert_gliner2" in calls


def test_unknown_extractor_encoder_raises_instead_of_silent_passthrough(
    monkeypatch, tmp_path, extractor_repo
):
    """An extractor repo must never fall through to 'not a GLiNER model'."""
    extractor_repo["configs"]["encoder_config/config.json"] = {"model_type": "llama"}
    _fake_hub(monkeypatch, tmp_path, extractor_repo["repo_files"], extractor_repo["configs"])
    _route_called(monkeypatch)

    with pytest.raises(RuntimeError, match="no registered plugin"):
        model_prep.prepare_model_for_vllm_if_needed("acme/gliner2-exotic")


def test_non_gliner_repo_passes_through(monkeypatch, tmp_path):
    """Plain models keep returning unchanged (the intended no-op path)."""
    _fake_hub(
        monkeypatch,
        tmp_path,
        ["config.json", "model.safetensors"],
        {"config.json": {"model_type": "bert"}},
    )

    result = model_prep.prepare_model_for_vllm_if_needed("bert-base-uncased/x")

    assert result == "bert-base-uncased/x"
