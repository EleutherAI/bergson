"""Subsampled randomized Hadamard transforms (SRHT) of module gradients.

A module's flattened gradient ``x`` of length ``n`` is zero-padded to ``N``, the
next power of two, and sketched as ``sqrt(N / m) S H D x``: ``D`` multiplies each of
the ``N`` entries by a seeded random sign, ``H`` is the orthonormal Walsh-Hadamard
transform, and ``S`` keeps ``m`` of the ``N`` transformed entries. The sketch's inner
products are unbiased estimates of the gradients'. The kept entries are the first
``m`` of a seeded sequence of random orderings of all ``N``, so every entry is kept
once before any repeats, and the first ``k`` entries of an ``m``-entry sketch, scaled
by ``sqrt(m / k)``, are the ``k``-entry sketch.

Signs and kept entries are determined by the module's identifier, so the train and
query sides of an index use the same sketch on any device.
"""

import math
from dataclasses import replace

import numpy as np
import torch
from torch import Tensor

from bergson.collector.projection_matrix import _keys, random_matrix
from bergson.gradients import OuterProductGradients

_CHUNK_NUMEL = 1 << 26
"""Entries of padded gradients held at once."""

_sketches: dict[tuple, tuple[Tensor, Tensor, float]] = {}


def fwht(x: Tensor, dim: int = -1) -> Tensor:
    """Orthonormal fast Walsh-Hadamard transform of ``x`` along ``dim``, whose size
    must be a power of two."""
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"The Hadamard transform needs a power of two, got {n}.")
    lead = x.shape[:-1]
    h = 1
    while h < n:
        y = x.reshape(*lead, n // (2 * h), 2, h)
        x = torch.stack((y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]), -2)
        h *= 2
    return (x.reshape(*lead, n) / math.sqrt(n)).movedim(-1, dim)


def _sketch(
    identifier: str, m: int, n: int, device: torch.device
) -> tuple[Tensor, Tensor, float]:
    """The ``[N]`` signs, the ``m`` kept entries of the transform and the scale for
    module ``identifier``."""
    key = (identifier, m, n, device)
    if key not in _sketches:
        size = 1 << (n - 1).bit_length()
        signs = random_matrix(f"{identifier}/signs", 1, size, device, "rademacher")[0]
        k1, k2 = _keys(f"{identifier}/entries")
        rng = np.random.Generator(
            np.random.Philox(key=(k1 & 0xFFFFFFFF) << 32 | (k2 & 0xFFFFFFFF))
        )
        orderings = [rng.permutation(size) for _ in range(-(-m // size))]
        entries = np.concatenate(orderings)[:m]
        _sketches[key] = (
            signs,
            torch.from_numpy(entries).to(device),
            math.sqrt(size / m),
        )
    return _sketches[key]


def hadamard_project(
    identifier: str, P: Tensor | OuterProductGradients, m: int
) -> Tensor:
    """Sketch each row of ``P``, a module's ``[N, ...]`` gradients or the vectors
    they are formed from, to ``m`` entries in float32."""
    if isinstance(P, OuterProductGradients):
        rows, n, device = P.g.shape[0], P.g.shape[-1] * P.a.shape[-1], P.g.device
    else:
        rows, n, device = P.shape[0], math.prod(P.shape[1:]), P.device
    signs, entries, scale = _sketch(identifier, m, n, device)

    out = torch.empty(rows, m, dtype=torch.float32, device=device)
    step = max(1, _CHUNK_NUMEL // len(signs))
    for i in range(0, rows, step):
        if isinstance(P, OuterProductGradients):
            x = _rows(P, i, i + step).materialize()
        else:
            x = P[i : i + step]
        x = torch.nn.functional.pad(x.flatten(1).float(), (0, len(signs) - n))
        out[i : i + step] = fwht(x * signs)[:, entries] * scale
    return out


def _rows(P: OuterProductGradients, start: int, stop: int) -> OuterProductGradients:
    """Rows ``start:stop`` of ``P``."""
    return replace(
        P,
        g=P.g[start:stop],
        a=P.a[start:stop],
        bias=P.bias[start:stop] if P.bias is not None else None,
    )
