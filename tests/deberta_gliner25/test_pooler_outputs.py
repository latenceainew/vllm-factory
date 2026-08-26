"""The pooler's output list must stay paired with the scheduled batch.

A coalesced batch holds several callers, so returning the wrong number of
payloads hands one caller another's extraction.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType

import torch


def _decode(payload: torch.Tensor) -> dict:
    """Unpack a payload the way the IO processor does."""
    data = payload.tolist()
    length = int(data[0])
    return json.loads(bytes(int(b) for b in data[1 : length + 1]).decode("utf-8"))


class _Ctx:
    """Minimal PoolerContext stand-in."""

    def __init__(self, seq_lengths: list[int], extra_kwargs: list[dict]) -> None:
        self.seq_lengths = seq_lengths
        self.extra_kwargs = extra_kwargs
        self.prompt_token_ids: list[list[int]] = []


def test_loading_the_pooler_leaves_no_stub_behind(pooler_mod: ModuleType):
    """A leaked stub shadows the real adapter for every later test."""
    adapter = sys.modules.get("vllm_factory.pooling.vllm_adapter")
    assert adapter is None or hasattr(adapter, "VllmPoolerAdapter")
    assert pooler_mod.GLiNER25BoundaryPooler is not None


def test_empty_payload_decodes_as_an_empty_result(pooler_mod: ModuleType):
    assert _decode(pooler_mod._pack_json({}, torch.device("cpu"))) == {}


def test_empty_payloads_are_one_per_sequence(pooler_mod: ModuleType):
    payloads = pooler_mod._empty_payloads(3, torch.device("cpu"))

    assert len(payloads) == 3
    assert all(_decode(payload) == {} for payload in payloads)


def test_split_failure_still_answers_every_sequence(pooler_mod: ModuleType, monkeypatch):
    def _boom(hidden_states, seq_lengths):
        raise RuntimeError("bad lengths")

    monkeypatch.setattr(pooler_mod, "split_hidden_states", _boom)
    pooler = object.__new__(pooler_mod.GLiNER25BoundaryPooler)
    ctx = _Ctx([5, 7, 9], [{"a": 1}, {"b": 2}, {"c": 3}])

    outputs = pooler_mod.GLiNER25BoundaryPooler.forward(pooler, torch.zeros(21, 4), ctx)

    assert len(outputs) == 3
    assert all(_decode(payload) == {} for payload in outputs)


def test_missing_extras_do_not_shift_the_other_results(
    pooler_mod: ModuleType, monkeypatch
):
    monkeypatch.setattr(
        pooler_mod,
        "split_hidden_states",
        lambda hidden_states, seq_lengths: [torch.zeros(n, 4) for n in seq_lengths],
    )
    monkeypatch.setattr(
        pooler_mod.GLiNER25BoundaryPooler,
        "_process_one",
        lambda self, token_embs, extra: pooler_mod._pack_json(extra, token_embs.device),
    )
    monkeypatch.setattr(pooler_mod, "_can_batch_compact", lambda extras: False)
    pooler = object.__new__(pooler_mod.GLiNER25BoundaryPooler)
    # vLLM scheduled three sequences but only two carry extras.
    ctx = _Ctx([2, 3, 4], [{"first": True}, {}])

    outputs = pooler_mod.GLiNER25BoundaryPooler.forward(pooler, torch.zeros(9, 4), ctx)

    assert [_decode(payload) for payload in outputs] == [{"first": True}, {}, {}]
