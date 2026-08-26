"""Overflow handling in preprocess_boundary: truncate only when asked."""

from types import SimpleNamespace

import pytest
import torch

from plugins.deberta_gliner25.processor import (
    _FIT_ATTEMPTS,
    collate_word_cap,
    preprocess_boundary,
)

_SCHEMA_TOKENS = 10
_SCHEMA = {"entities": ["person"]}


class _FakeHost:
    """Collate stub whose token count follows the word cap it is given."""

    def __init__(self, words: int, tokens_per_word: int = 2, fixed_tokens: int | None = None):
        self.words = words
        self.tokens_per_word = tokens_per_word
        self.fixed_tokens = fixed_tokens
        self.caps: list[int | None] = []

    def collate_fn_inference(self, batch, *, architecture, error_policy, max_len=None):
        assert architecture == "boundary"
        assert error_policy == "raise"
        self.caps.append(max_len)
        kept = self.words if max_len is None else min(self.words, max_len)
        tokens = self.fixed_tokens or (kept * self.tokens_per_word + _SCHEMA_TOKENS)
        return SimpleNamespace(
            input_ids=[torch.arange(tokens)],
            text_tokens=[["w"] * kept],
        )


def _preprocess(host: _FakeHost, **kwargs):
    return preprocess_boundary(None, "some text", _SCHEMA, transformer=host, **kwargs)


def test_word_cap_leaves_room_for_schema_tokens():
    assert collate_word_cap(4096) == 3968
    assert collate_word_cap(None) is None
    assert collate_word_cap(0) is None


def test_prompt_within_limit_is_untouched():
    host = _FakeHost(words=100)
    result = _preprocess(host, word_cap=900, max_model_len=1024)

    assert len(result["input_ids"]) == 210
    assert host.caps == [None]
    assert "max_len" not in result["extra_kwargs"]


def test_overflow_without_consent_is_rejected():
    host = _FakeHost(words=800)

    with pytest.raises(ValueError, match="above the 1024-token limit"):
        _preprocess(host, word_cap=900, max_model_len=1024)

    assert host.caps == [None], "text must not be silently dropped"


def test_overflow_with_consent_is_truncated_to_fit():
    host = _FakeHost(words=800)
    result = _preprocess(
        host, truncate_overflow_text=True, word_cap=900, max_model_len=1024
    )

    assert len(result["input_ids"]) <= 1024
    # First try uses the caller's cap, then it tightens to the token budget.
    assert host.caps[0] == 900
    assert host.caps[-1] < 900
    assert result["extra_kwargs"]["max_len"] == host.caps[-1]


def test_truncation_gives_up_rather_than_looping():
    host = _FakeHost(words=800, fixed_tokens=99_999)

    with pytest.raises(ValueError, match=f"after {_FIT_ATTEMPTS} tries"):
        _preprocess(host, truncate_overflow_text=True, word_cap=900, max_model_len=1024)

    assert len(host.caps) == _FIT_ATTEMPTS


def test_no_engine_limit_skips_the_check_entirely():
    host = _FakeHost(words=10_000)
    result = _preprocess(host)

    assert len(result["input_ids"]) == 20_010
    assert host.caps == [None]
