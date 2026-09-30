"""Span-level attribution: one gradient row per run of token positions.

A span is a contiguous run of an example's token positions. ``data.span_column``
holds the first token position of each span, so the spans partition the
example: they must be strictly increasing and start at 0.

Row ``t`` of a per-token store is token position ``t`` (the ``gradient``
token influence) or the loss on token ``t + 1`` (the ``output`` one), so a
span owns a different set of rows in each convention. ``shift`` carries that
difference: row ``t`` belongs to the span holding token ``t + shift``.
"""

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

TOKEN_INFLUENCE_SHIFT = {"gradient": 0, "output": 1}
"""Token position row ``t`` is attributed to, minus ``t``, per token influence."""


def check_span_starts(starts: NDArray, num_tokens: int, index: int = 0) -> None:
    """Raise if ``starts`` does not partition a ``num_tokens``-token example."""
    if len(starts) == 0:
        raise ValueError(f"Example {index} has no spans.")
    if starts[0] != 0:
        raise ValueError(
            f"Example {index}'s spans start at token {starts[0]}, not 0, so its "
            "first tokens belong to no span."
        )
    if np.any(np.diff(starts) <= 0):
        raise ValueError(f"Example {index}'s span starts are not increasing: {starts}.")
    if starts[-1] >= max(num_tokens, 1):
        raise ValueError(
            f"Example {index}'s last span starts at token {starts[-1]}, past its "
            f"{num_tokens} tokens."
        )


def span_row_bounds(starts: NDArray, num_rows: int, shift: int = 0) -> NDArray:
    """``[num_spans, 2]`` half-open row ranges, one per span.

    A span whose tokens carry no gradient row -- the last span under
    ``shift`` 1, or any span past ``num_rows`` -- gets an empty range, and so
    a zero gradient, rather than being dropped: the store's shape then depends
    only on the dataset, not on the token influence it was scored with.
    """
    if len(starts) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    lo = np.clip(starts.astype(np.int64) - shift, 0, num_rows)
    hi = np.empty_like(lo)
    hi[:-1], hi[-1] = lo[1:], num_rows
    return np.stack([lo, np.maximum(hi, lo)], axis=1)


def span_gather(
    batch: dict,
    span_column: str,
    seq_len: int,
    device: torch.device | str = "cpu",
    shift: int = 0,
) -> tuple[Tensor, Tensor]:
    """``(index, valid)``, both ``[num_spans, longest_span]``, gathering each
    span's rows out of a batch padded to ``seq_len`` and flattened over its
    first two dimensions. ``valid`` is False where a span is padded out to the
    longest one, so a batch of uneven spans gathers more rows than it has
    positions -- by the ratio of the longest span to the mean one.
    """
    bounds = [
        span_row_bounds(np.asarray(row, dtype=np.int64), max(len(ids) - 1, 0), shift)
        + i * seq_len
        for i, (row, ids) in enumerate(zip(batch[span_column], batch["input_ids"]))
    ]
    lo = np.concatenate([b[:, 0] for b in bounds])
    sizes = np.concatenate([np.diff(b, axis=1).ravel() for b in bounds])

    positions = np.arange(max(int(sizes.max()), 1) if len(sizes) else 1)
    valid = positions < sizes[:, None]
    index = np.where(valid, lo[:, None] + positions, 0)
    return torch.from_numpy(index).to(device), torch.from_numpy(valid).to(device)
