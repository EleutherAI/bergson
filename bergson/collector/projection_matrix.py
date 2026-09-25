"""Random projection matrices that are the same on every device.

Entries come from Philox4x32-10 (Salmon et al., "Parallel random numbers: as
easy as 1, 2, 3", 2011), a counter-based generator: each output is a function
of a counter and a key only. The key comes from the matrix's identifier and the
counter from each entry's row and column, so the values don't depend on the
device, on how the work is split across threads, or on the matrix's shape:
``A[:k, :l]`` is the ``[k, l]`` matrix with the same identifier. Rademacher
matrices are bit-identical everywhere; normal matrices can differ in the last
bits because ``log``, ``sin`` and ``cos`` round differently across devices.
"""

import hashlib
import math
from typing import Callable, Literal

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

from bergson.utils.logger import get_logger

logger = get_logger("projection_matrix", level="INFO")

# NumPy is single-threaded, so the CPU path isn't slowed down by torch's thread
# pool when the machine has fewer free cores than it reports.
_CPU_CHUNK = 1 << 16
"""Entries per CPU chunk, small enough to keep its temporaries in cache."""

_GPU_SLAB = 1 << 25
"""Entries generated in fp32 at a time when the output dtype is narrower."""

_PHILOX_M = (0xD2511F53, 0xCD9E8D57)
_PHILOX_W = (0x9E3779B9, 0xBB67AE85)

# Philox4x32-10 on the counter (w, row, 0, 0), giving x0..x3. The kernels are
# written as one function each because the jiterator only accepts a single
# template function.
_PHILOX = """
    unsigned int x0 = w, x1 = static_cast<unsigned int>(row), x2 = 0u, x3 = 0u;
    unsigned int key0 = static_cast<unsigned int>(k1);
    unsigned int key1 = static_cast<unsigned int>(k2);
    for (int round = 0; round < 10; ++round) {
        if (round > 0) { key0 += 0x9e3779b9u; key1 += 0xbb67ae85u; }
        unsigned long long p0 = 0xd2511f53ull * x0;
        unsigned long long p1 = 0xcd9e8d57ull * x2;
        x0 = static_cast<unsigned int>(p1 >> 32) ^ x1 ^ key0;
        x1 = static_cast<unsigned int>(p1);
        x2 = static_cast<unsigned int>(p0 >> 32) ^ x3 ^ key1;
        x3 = static_cast<unsigned int>(p0);
    }
"""
# Each Philox output gives two Box-Muller pairs, (x0, x1) and (x2, x3). Pair q
# gives r and t, and its two entries are r * cos(t) and r * sin(t).
_BOX_MULLER = """
    unsigned int a = q ? x2 : x0, b = q ? x3 : x1;
    float u1 = (__uint2float_rn(a) + 0.5f) * 2.3283064365386963e-10f;
    float u2 = __uint2float_rn(b) * 2.3283064365386963e-10f;
    float r = sqrtf(-2.0f * logf(u1));
    float t = 6.2831853071795864769f * u2;
"""
# Bit b of the 128 bits x0..x3.
_BIT = """
    auto bit = [&](unsigned int b) {
        unsigned int x = b < 64 ? (b < 32 ? x0 : x1) : (b < 96 ? x2 : x3);
        return (x >> (b & 31u)) & 1u;
    };
"""


def _header(name: str, index: str) -> str:
    return f"template <typename T> T {name}(T row, T {index}, T k1, T k2) {{"


# Rounds to nearest even, like torch's float -> bfloat16 conversion.
_BF16 = """
    auto bf16 = [](float z) {
        unsigned int b = __float_as_uint(z);
        return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
    };
"""
_KERNELS = {
    # Column c is entry c % 4 of Philox output c / 4.
    "normal": _header("philox_normal", "col")
    + """
    unsigned int c = static_cast<unsigned int>(col);
    unsigned int w = c >> 2, q = (c >> 1) & 1u;"""
    + _PHILOX
    + _BOX_MULLER
    + """
    float z = (c & 1u) ? r * sinf(t) : r * cosf(t);
    return static_cast<T>(__float_as_int(z));
}
""",
    # Column c is bit c % 128 of Philox output c / 128.
    "rademacher": _header("philox_rademacher", "col")
    + """
    unsigned int c = static_cast<unsigned int>(col);
    unsigned int w = c >> 7;"""
    + _PHILOX
    + _BIT
    + """
    float z = bit(c & 127u) ? 1.0f : -1.0f;
    return static_cast<T>(__float_as_int(z));
}
""",
    # Columns 2p and 2p + 1 as bfloat16, packed into one 32-bit output.
    "normal_bf16": _header("philox_normal_bf16", "pair")
    + _BF16
    + """
    unsigned int p = static_cast<unsigned int>(pair);
    unsigned int w = p >> 1, q = p & 1u;"""
    + _PHILOX
    + _BOX_MULLER
    + """
    return static_cast<T>(bf16(r * cosf(t)) | (bf16(r * sinf(t)) << 16));
}
""",
    "rademacher_bf16": _header("philox_rademacher_bf16", "pair")
    + """
    unsigned int c = 2 * static_cast<unsigned int>(pair);
    unsigned int w = c >> 7;"""
    + _PHILOX
    + _BIT
    + """
    unsigned int lo = bit(c & 127u) ? 0x3f80u : 0xbf80u;
    unsigned int hi = bit((c & 127u) + 1) ? 0x3f80u : 0xbf80u;
    return static_cast<T>(lo | (hi << 16));
}
""",
}
_jit_fns: dict[str, Callable] = {}
_kernels_failed = False


def random_matrix(
    identifier: str,
    m: int,
    n: int,
    device: torch.device | str,
    projection_type: Literal["normal", "rademacher"],
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """An ``[m, n]`` matrix of standard normal or ±1 entries determined by
    ``identifier``."""
    if projection_type not in ("normal", "rademacher"):
        raise ValueError(f"Unknown projection type: {projection_type}")
    if max(m, n) >= 1 << 31:
        raise ValueError(
            f"Projection matrices are limited to 2^31 rows and columns, got [{m}, {n}]."
        )

    device = torch.device(device)
    k1, k2 = _keys(identifier)
    if device.type == "cuda" and not _kernels_failed:
        try:
            return _random_matrix_cuda(k1, k2, m, n, device, projection_type, dtype)
        except torch.OutOfMemoryError:
            raise
        except Exception as e:
            _disable_kernels(e)

    # One torch op at the end: each torch op on the CPU may wait for torch's
    # whole thread pool.
    out = np.empty((m, n), dtype=np.float32)
    generate = _normal_cpu if projection_type == "normal" else _rademacher_cpu
    rows_per = max(1, _CPU_CHUNK // n)
    # Chunks start at a multiple of 128 so each Philox output lands in one chunk.
    cols_per = min(n, _CPU_CHUNK)
    for r0 in range(0, m, rows_per):
        r1 = min(m, r0 + rows_per)
        rows = np.arange(r0, r1, dtype=np.uint32)[:, None]
        for c0 in range(0, n, cols_per):
            c1 = min(n, c0 + cols_per)
            out[r0:r1, c0:c1] = generate(k1, k2, rows, c0, c1)
    return torch.from_numpy(out).to(device, dtype)


def _keys(identifier: str) -> tuple[int, int]:
    digest = hashlib.md5(identifier.encode()).digest()
    return (
        int.from_bytes(digest[:4], "little", signed=True),
        int.from_bytes(digest[4:8], "little", signed=True),
    )


def _random_matrix_cuda(
    k1: int,
    k2: int,
    m: int,
    n: int,
    device: torch.device,
    projection_type: str,
    dtype: torch.dtype,
) -> Tensor:
    rows = torch.arange(m, dtype=torch.int32, device=device)[:, None]
    if dtype == torch.bfloat16:
        kernel = _kernel(f"{projection_type}_bf16")
        pairs = torch.arange((n + 1) // 2, dtype=torch.int32, device=device)[None, :]
        A = kernel(rows, pairs, k1=k1, k2=k2).view(torch.bfloat16)
        return A if n % 2 == 0 else A[:, :n].contiguous()

    kernel = _kernel(projection_type)

    def columns(c0: int, c1: int) -> Tensor:
        cols = torch.arange(c0, c1, dtype=torch.int32, device=device)[None, :]
        return kernel(rows, cols, k1=k1, k2=k2).view(torch.float32)

    if dtype == torch.float32 or m * n <= _GPU_SLAB:
        return columns(0, n).to(dtype)

    # Generate fp32 in slabs so the fp32 copy of the whole matrix never exists.
    out = torch.empty(m, n, dtype=dtype, device=device)
    cols_per = max(1, _GPU_SLAB // m)
    for c0 in range(0, n, cols_per):
        c1 = min(n, c0 + cols_per)
        out[:, c0:c1] = columns(c0, c1)
    return out


def _kernel(name: str) -> Callable:
    """A fused CUDA/ROCm kernel, compiled for each device on first use."""
    if name not in _jit_fns:
        from torch.cuda.jiterator import _create_jit_fn

        _jit_fns[name] = _create_jit_fn(_KERNELS[name], k1=0, k2=0)
    return _jit_fns[name]


def _disable_kernels(e: Exception) -> None:
    # The CPU path computes the same values, so falling back only costs speed.
    global _kernels_failed
    _kernels_failed = True
    logger.warning(
        f"Couldn't run the projection matrix kernel ({e}); generating projection "
        "matrices on the CPU instead."
    )


def philox4x32(
    counter: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    key: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Philox4x32-10 on uint32 counter words, which broadcast together."""
    x0, x1, x2, x3 = (np.asarray(c, dtype=np.uint64) for c in counter)
    k0, k1 = (np.uint64(k & 0xFFFFFFFF) for k in key)
    mask = np.uint64(0xFFFFFFFF)
    for round in range(10):
        if round > 0:
            k0 = (k0 + np.uint64(_PHILOX_W[0])) & mask
            k1 = (k1 + np.uint64(_PHILOX_W[1])) & mask
        p0 = np.uint64(_PHILOX_M[0]) * x0
        p1 = np.uint64(_PHILOX_M[1]) * x2
        x0, x1, x2, x3 = (
            (p1 >> np.uint64(32)) ^ x1 ^ k0,
            p1 & mask,
            (p0 >> np.uint64(32)) ^ x3 ^ k1,
            p0 & mask,
        )
    return tuple(x.astype(np.uint32) for x in (x0, x1, x2, x3))  # type: ignore


def _philox_block(
    k1: int, k2: int, rows: np.ndarray, w0: int, w1: int
) -> tuple[np.ndarray, ...]:
    """Philox outputs for counters ``(w, row, 0, 0)`` with ``w`` in ``[w0, w1)``."""
    w = np.arange(w0, w1, dtype=np.uint32)
    zero = np.uint32(0)
    return philox4x32((w, rows, zero, zero), (k1, k2))


def _normal_cpu(k1: int, k2: int, rows: np.ndarray, c0: int, c1: int) -> np.ndarray:
    x0, x1, x2, x3 = _philox_block(k1, k2, rows, c0 // 4, (c1 + 3) // 4)
    z = []
    for a, b in [(x0, x1), (x2, x3)]:
        # Rounded to fp32 like __uint2float_rn.
        u1 = (a.astype(np.float32) + np.float32(0.5)) * np.float32(2.0**-32)
        u2 = b.astype(np.float32) * np.float32(2.0**-32)
        r = np.sqrt(np.float32(-2.0) * np.log(u1))
        t = np.float32(2 * math.pi) * u2
        z += [r * np.cos(t), r * np.sin(t)]
    z = np.stack(z, axis=-1).reshape(len(rows), -1)
    start = c0 - 4 * (c0 // 4)
    return z[:, start : start + c1 - c0]


_SIGNS = np.array([-1.0, 1.0], dtype=np.float32)


def _rademacher_cpu(k1: int, k2: int, rows: np.ndarray, c0: int, c1: int) -> np.ndarray:
    words = np.stack(_philox_block(k1, k2, rows, c0 // 128, (c1 + 127) // 128), -1)
    bits = np.unpackbits(words.view(np.uint8), axis=-1, bitorder="little")
    bits = bits.reshape(len(rows), -1)
    start = c0 - 128 * (c0 // 128)
    return _SIGNS[bits[:, start : start + c1 - c0]]
