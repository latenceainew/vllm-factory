from __future__ import annotations

import torch


def sequence_lengths(flat_ids: torch.Tensor, positions: torch.Tensor | None) -> list[int]:
    """Recover per-sequence token counts from a packed batch.

    Sequence starts are the offsets where ``positions`` restarts at 0, which is
    how vLLM lays out a batch for a model without a KV cache. The runner pads
    the token count up to a tile size and those pad slots also carry position
    0, so the all-zero tail is trimmed first — a real sequence ends at position
    ``length - 1``, so the last nonzero position marks the last real token.
    Sequences of a single token are indistinguishable from padding and would be
    dropped; boundary prompts always carry schema markers, so they never are.

    Args:
        flat_ids: Flat token ids for the whole batch, shape (total_tokens,).
        positions: Per-sequence position ids, or None when unavailable.

    Returns:
        Token count per real sequence, in batch order. A single entry means the
        batch could not be split and is treated as one sequence.
    """
    total = int(flat_ids.numel())
    if positions is None or total == 0:
        return [total]

    pos = positions.view(-1)
    if pos.numel() != total:
        return [total]

    nonzero = torch.nonzero(pos != 0, as_tuple=False).flatten()
    if nonzero.numel() == 0:
        return [total]
    real = int(nonzero[-1].item()) + 1

    starts = torch.nonzero(pos[:real] == 0, as_tuple=False).flatten().tolist()
    if not starts or starts[0] != 0:
        return [total]

    bounds = [*starts, real]
    return [bounds[i + 1] - bounds[i] for i in range(len(starts))]


def pad_batch(
    flat_ids: torch.Tensor, lengths: list[int], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter a packed token tensor into one padded row per sequence.

    Args:
        flat_ids: Flat token ids for the whole batch, shape (total_tokens,).
        lengths: Token count per sequence, laid out consecutively from index 0.
        pad_id: Token id to write into padding positions.

    Returns:
        A ``(input_ids, attention_mask)`` pair, both shaped
        (len(lengths), max(lengths)), the mask holding 1 on real tokens.
    """
    rows = len(lengths)
    width = max(lengths)
    ids = flat_ids.new_full((rows, width), pad_id)
    mask = torch.zeros((rows, width), dtype=torch.long, device=flat_ids.device)

    offset = 0
    for row, length in enumerate(lengths):
        ids[row, :length] = flat_ids[offset : offset + length]
        mask[row, :length] = 1
        offset += length
    return ids, mask
