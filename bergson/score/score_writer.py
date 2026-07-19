import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import ml_dtypes  # noqa: F401  # register bfloat16 dtype with numpy
import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset

from bergson.data import compute_num_token_grads
from bergson.utils.utils import convert_dtype_to_np, tensor_to_numpy


def _score_struct_dtype(num_scores: int, np_dtype: np.dtype) -> tuple[dict, dict]:
    """Interleaved ``(score_i, written_i)`` structured dtype shared by every
    score writer -- per-document or per-token -- so every stored score
    carries a per-cell written flag for resumability. Each pair is aligned
    to the next power of 2 for efficiency.

    Returns ``(numpy_dtype_spec, json_safe_dtype_spec)``.
    """
    score_size = np_dtype.itemsize
    bool_size = np.dtype("bool").itemsize
    pair_size = score_size + bool_size
    aligned_pair_size = 1 << (pair_size - 1).bit_length()

    names = []
    formats = []
    offsets = []
    for i in range(num_scores):
        names.append(f"score_{i}")
        formats.append(np_dtype)
        offsets.append(i * aligned_pair_size)

        names.append(f"written_{i}")
        formats.append("bool")
        offsets.append(i * aligned_pair_size + score_size)

    total_bytes = num_scores * aligned_pair_size
    # Round up to the nearest 8 bytes
    itemsize = ((total_bytes + 7) // 8) * 8

    struct_dtype = {
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": itemsize,
    }
    # For JSON serialization, convert numpy dtype to string
    struct_dtype_json = {
        "names": names,
        "formats": [str(f) if isinstance(f, np.dtype) else f for f in formats],
        "offsets": offsets,
        "itemsize": itemsize,
    }
    return struct_dtype, struct_dtype_json


class ScoreWriter(ABC):
    """
    Base class for score writers.
    """

    scores: Any

    @abstractmethod
    def __call__(
        self,
        indices: list[int],
        scores: torch.Tensor,
    ):
        """
        Write the scores to the score writer.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def flush(self):
        """
        Flush the score writer.
        """
        raise NotImplementedError("Subclasses must implement this method")


class InMemoryTokenScoreWriter(ScoreWriter):
    """Stores scores in memory as a torch tensor."""

    def __init__(
        self,
        data: Dataset,
        num_scores: int,
        dtype: torch.dtype = torch.float32,
    ):
        num_token_grads = compute_num_token_grads(data)
        self.num_token_grads = num_token_grads
        self.offsets = np.zeros(len(num_token_grads) + 1, dtype=np.int64)

        np.cumsum(num_token_grads, out=self.offsets[1:])

        self.scores = [
            torch.zeros((num_grads, num_scores), device="cpu", dtype=dtype)
            for num_grads in num_token_grads
        ]
        self.dtype = dtype

    def __call__(self, indices: list[int], scores: torch.Tensor):
        # scores: [total_valid_in_batch, num_scores]
        row = 0
        for idx in indices:
            sl = int(self.num_token_grads[idx])
            self.scores[idx] = scores[row : row + sl].to(dtype=self.dtype).cpu()
            row += sl

    def flush(self):
        # No-op for in-memory storage
        pass


class InMemorySequenceScoreWriter(ScoreWriter):
    """Stores scores in memory as a torch tensor."""

    def __init__(
        self, num_items: int, num_scores: int, dtype: torch.dtype = torch.float32
    ):
        self.scores = torch.zeros((num_items, num_scores), device="cpu", dtype=dtype)

    def __call__(self, indices: list[int], scores: torch.Tensor):
        self.scores[indices] = scores.to(dtype=self.scores.dtype).cpu()

    def flush(self):
        # No-op for in-memory storage
        pass


class MemmapTokenScoreWriter(ScoreWriter):
    """Writes per-token scores to a flat memory-mapped file, the same
    ``(score_i, written_i)`` structured layout as
    :class:`MemmapSequenceScoreWriter` -- so a per-token store is a plain
    per-document store with ``num_rows = total_tokens`` instead of
    ``num_items``, plus an ``offsets.npy`` saying which rows belong to which
    document. Example *i*'s scores live at rows ``offsets[i]:offsets[i+1]``.
    """

    def __init__(
        self,
        path: Path,
        data: Dataset,
        num_scores: int,
        *,
        dtype: torch.dtype = torch.float32,
        flush_interval: int = 64,
    ):
        self.path = path
        self.num_scores = num_scores
        self.dtype = dtype
        self.flush_interval = flush_interval
        self.num_batches_since_flush = 0

        num_token_grads = compute_num_token_grads(data)
        num_items = len(data)
        self.num_token_grads = num_token_grads
        self.offsets = np.zeros(len(num_token_grads) + 1, dtype=np.int64)
        np.cumsum(num_token_grads, out=self.offsets[1:])
        total_tokens = int(self.offsets[-1])

        self.path.mkdir(parents=True, exist_ok=True)
        scores_file_path = self.path / "scores.bin"
        np_dtype = convert_dtype_to_np(dtype)
        struct_dtype, struct_dtype_json = _score_struct_dtype(num_scores, np_dtype)

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not scores_file_path.exists():
            print(f"Creating new token scores file: {scores_file_path}")

            self.scores = np.memmap(
                str(scores_file_path),
                dtype=np.dtype(struct_dtype),  # type: ignore
                mode="w+",
                shape=(total_tokens,),
            )

            with (path / "info.json").open("w") as f:
                json.dump(
                    {
                        "attribute_tokens": True,
                        "num_items": num_items,
                        "num_rows": total_tokens,
                        "num_scores": num_scores,
                        "dtype": struct_dtype_json,
                    },
                    f,
                    indent=2,
                )

            np.save(path / "offsets.npy", self.offsets)

        if dist.is_initialized():
            dist.barrier()

        self.scores = np.memmap(
            str(scores_file_path),
            dtype=np.dtype(struct_dtype),  # type: ignore
            mode="r+",
            shape=(total_tokens,),
        )

    def __call__(self, indices: list[int], scores: torch.Tensor):
        # scores: [total_valid_in_batch, num_scores]
        scores = scores.to(dtype=self.dtype)

        row = 0
        for idx in indices:
            sl = int(self.num_token_grads[idx])
            buf_start = int(self.offsets[idx])
            buf_end = int(self.offsets[idx + 1])
            for i in range(self.num_scores):
                col = tensor_to_numpy(scores[row : row + sl, i].cpu())
                self.scores[f"score_{i}"][buf_start:buf_end] = col
                self.scores[f"written_{i}"][buf_start:buf_end] = True
            row += sl

        self.num_batches_since_flush += 1
        if self.num_batches_since_flush >= self.flush_interval:
            self.flush()

    def flush(self):
        self.scores.flush()
        self.num_batches_since_flush = 0


def save_token_scores(
    path: Path,
    scores: np.ndarray,
    offsets: np.ndarray,
    *,
    dtype: torch.dtype = torch.float32,
) -> None:
    """One-shot equivalent of :class:`MemmapTokenScoreWriter`'s on-disk
    layout, for callers that already hold the full flat ``(total_tokens,
    num_scores)`` array in memory (e.g. summed across upstream per-token
    score dirs sharing the same ``offsets``), rather than streaming batches
    through ``__call__``.
    """
    if scores.ndim == 1:
        scores = scores[:, None]
    total_tokens, num_scores = scores.shape
    num_items = len(offsets) - 1

    path.mkdir(parents=True, exist_ok=True)
    np_dtype = convert_dtype_to_np(dtype)
    struct_dtype, struct_dtype_json = _score_struct_dtype(num_scores, np_dtype)

    mmap = np.memmap(
        path / "scores.bin",
        dtype=np.dtype(struct_dtype),  # type: ignore
        mode="w+",
        shape=(total_tokens,),
    )
    scores_np = tensor_to_numpy(torch.from_numpy(scores).to(dtype))
    for i in range(num_scores):
        mmap[f"score_{i}"] = scores_np[:, i]
        mmap[f"written_{i}"] = True
    mmap.flush()

    np.save(path / "offsets.npy", offsets)

    with (path / "info.json").open("w") as f:
        json.dump(
            {
                "attribute_tokens": True,
                "num_items": num_items,
                "num_rows": total_tokens,
                "num_scores": num_scores,
                "dtype": struct_dtype_json,
            },
            f,
            indent=2,
        )


class MemmapSequenceScoreWriter(ScoreWriter):
    """
    Writes scores to a memory-mapped file on disk.

    Supports bfloat16 via ml_dtypes.
    """

    def __init__(
        self,
        path: Path,
        num_items: int,
        num_scores: int,
        *,
        dtype: torch.dtype = torch.float32,
        flush_interval: int = 64,
    ):
        self.path = path
        self.num_scores = num_scores
        self.dtype = dtype
        self.flush_interval = flush_interval
        self.num_batches_since_flush = 0

        self.path.mkdir(parents=True, exist_ok=True)
        scores_file_path = self.path / "scores.bin"

        # Convert torch dtype to numpy dtype (handles bfloat16 via ml_dtypes)
        np_dtype = convert_dtype_to_np(dtype)
        struct_dtype, struct_dtype_json = _score_struct_dtype(num_scores, np_dtype)

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not scores_file_path.exists():
            print(f"Creating new scores file: {scores_file_path}")

            # w+ mode creates a zero-filled file.
            self.scores = np.memmap(
                str(scores_file_path),
                dtype=np.dtype(struct_dtype),  # type: ignore
                mode="w+",
                shape=(num_items,),
            )

            # Persist metadata for future runs
            with (path / "info.json").open("w") as f:
                json.dump(
                    {
                        "attribute_tokens": False,
                        "num_items": num_items,
                        "num_rows": num_items,
                        "num_scores": num_scores,
                        "dtype": struct_dtype_json,
                    },
                    f,
                    indent=2,
                )

        if dist.is_initialized():
            dist.barrier()

        self.scores = np.memmap(
            str(scores_file_path),
            dtype=np.dtype(struct_dtype),  # type: ignore
            mode="r+",
            shape=(num_items,),
        )

    def __call__(self, indices: list[int], scores: torch.Tensor):
        # scores: [num_indices, num_scores]
        scores = scores.to(dtype=self.dtype)
        for i in range(self.num_scores):
            score_col = tensor_to_numpy(scores[:, i].cpu()).flatten()
            self.scores[f"score_{i}"][indices] = score_col
            self.scores[f"written_{i}"][indices] = True

        self.num_batches_since_flush += 1
        if self.num_batches_since_flush >= self.flush_interval:
            self.flush()

    def flush(self):
        self.scores.flush()
        self.num_batches_since_flush = 0


def save_sequence_scores(
    path: Path,
    scores: np.ndarray,
    *,
    dtype: torch.dtype = torch.float32,
) -> None:
    """One-shot equivalent of :class:`MemmapSequenceScoreWriter`, for callers
    that already hold the full ``[num_items, num_scores]`` matrix in memory
    (e.g. summed across upstream per-segment score dirs) rather than
    streaming batches through ``__call__``.
    """
    if scores.ndim == 1:
        scores = scores[:, None]
    num_items, num_scores = scores.shape

    writer = MemmapSequenceScoreWriter(path, num_items, num_scores, dtype=dtype)
    writer(list(range(num_items)), torch.from_numpy(np.ascontiguousarray(scores)))
    writer.flush()
