"""Splitting vLLM's packed batch back into per-sequence rows."""

import torch

from plugins.deberta_gliner25.packing import pad_batch, sequence_lengths


def _packed(lengths: list[int], pad_slots: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay out ids and positions the way the vLLM runner does."""
    ids: list[int] = []
    positions: list[int] = []
    token = 100
    for length in lengths:
        ids.extend(range(token, token + length))
        positions.extend(range(length))
        token += 1000
    ids.extend([0] * pad_slots)
    positions.extend([0] * pad_slots)
    return torch.tensor(ids), torch.tensor(positions)


def test_single_sequence():
    ids, positions = _packed([7])
    assert sequence_lengths(ids, positions) == [7]


def test_splits_on_position_restart():
    ids, positions = _packed([5, 3, 9])
    assert sequence_lengths(ids, positions) == [5, 3, 9]


def test_trailing_runner_padding_is_not_a_sequence():
    ids, positions = _packed([6], pad_slots=18)
    assert sequence_lengths(ids, positions) == [6]


def test_padding_after_several_sequences():
    ids, positions = _packed([4, 4], pad_slots=8)
    assert sequence_lengths(ids, positions) == [4, 4]


def test_no_positions_falls_back_to_one_sequence():
    ids, _ = _packed([5, 5])
    assert sequence_lengths(ids, None) == [10]


def test_mismatched_positions_falls_back_to_one_sequence():
    ids, positions = _packed([5, 5])
    assert sequence_lengths(ids, positions[:-1]) == [10]


def test_empty_batch():
    empty = torch.tensor([], dtype=torch.long)
    assert sequence_lengths(empty, empty) == [0]


def test_pad_batch_rows_and_mask():
    ids, _ = _packed([3, 5, 2])
    padded, mask = pad_batch(ids, [3, 5, 2], pad_id=7)

    assert padded.shape == (3, 5)
    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]
    assert padded[0].tolist() == [100, 101, 102, 7, 7]
    assert padded[2].tolist() == [2100, 2101, 7, 7, 7]


def test_pad_batch_round_trips_through_the_mask():
    lengths = [3, 5, 2]
    ids, _ = _packed(lengths)
    padded, _ = pad_batch(ids, lengths, pad_id=0)

    repacked = torch.cat([padded[row, :length] for row, length in enumerate(lengths)])
    assert torch.equal(repacked, ids)
