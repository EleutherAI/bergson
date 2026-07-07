import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import ml_dtypes  # noqa: F401  # register bfloat16 dtype with numpy
import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset

from bergson.data import (
    compute_num_token_grads,
    leading_written_count,
    open_shared_memmap,
)
from bergson.utils.utils import convert_dtype_to_np, tensor_to_numpy


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

    def leading_written_count(self, batches: list[list[int]]) -> int:
        """Number of leading batches already written, to skip on resume.
        Default 0: in-memory writers don't persist progress."""
        return 0


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
    """Writes per-token scores to a flat memory-mapped file.

    The flat buffer has shape ``(total_tokens, num_scores)`` where
    ``total_tokens = sum(num_token_grads)``.  Example *i*'s scores live at
    rows ``offsets[i]:offsets[i+1]``.
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

        self.num_items = num_items
        self.path.mkdir(parents=True, exist_ok=True)
        scores_file_path = self.path / "token_scores.bin"
        np_dtype = convert_dtype_to_np(dtype)

        rank = dist.get_rank() if dist.is_initialized() else 0
        fresh = rank == 0 and not scores_file_path.exists()

        self.scores = open_shared_memmap(
            scores_file_path, np_dtype, (total_tokens, num_scores)
        )
        # Per-row written flags for resume (token scores have no natural one:
        # a zero score is ambiguous).
        self.written = open_shared_memmap(
            self.path / "written.bin", np.bool_, (num_items,)
        )

        if fresh:
            print(f"Creating new token scores file: {scores_file_path}")
            with (path / "info.json").open("w") as f:
                json.dump(
                    {
                        "attribute_tokens": True,
                        "total_tokens": total_tokens,
                        "num_items": num_items,
                        "num_scores": num_scores,
                        "dtype": np_dtype.name,
                    },
                    f,
                    indent=2,
                )
            np.save(path / "num_token_grads.npy", num_token_grads)
            np.save(path / "offsets.npy", self.offsets)

    def __call__(self, indices: list[int], scores: torch.Tensor):
        # scores: [total_valid_in_batch, num_scores]
        scores_np = tensor_to_numpy(scores.to(dtype=self.dtype).cpu())

        row = 0
        for idx in indices:
            sl = int(self.num_token_grads[idx])
            buf_start = int(self.offsets[idx])
            buf_end = int(self.offsets[idx + 1])
            self.scores[buf_start:buf_end] = scores_np[row : row + sl]
            row += sl
        self.written[indices] = True

        self.num_batches_since_flush += 1
        if self.num_batches_since_flush >= self.flush_interval:
            self.flush()

    def flush(self):
        # Scores before flags: a persisted "written" must imply scores on disk.
        self.scores.flush()
        self.written.flush()
        self.num_batches_since_flush = 0

    def leading_written_count(self, batches: list[list[int]]) -> int:
        return leading_written_count(self.written, batches)


class MemmapSequenceScoreWriter(ScoreWriter):
    """Writes per-sequence scores to a memory-mapped ``(num_items, num_scores)``
    array on disk, with per-row completion flags in a sibling ``written.bin``.

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
        np_dtype = convert_dtype_to_np(dtype)

        rank = dist.get_rank() if dist.is_initialized() else 0
        fresh = rank == 0 and not scores_file_path.exists()

        self.scores = open_shared_memmap(
            scores_file_path, np_dtype, (num_items, num_scores)
        )
        self.written = open_shared_memmap(
            self.path / "written.bin", np.bool_, (num_items,)
        )

        if fresh:
            print(f"Creating new scores file: {scores_file_path}")
            with (path / "info.json").open("w") as f:
                json.dump(
                    {
                        "num_items": num_items,
                        "num_scores": num_scores,
                        "dtype": np_dtype.name,
                        "format": "flat",
                    },
                    f,
                    indent=2,
                )

    def __call__(self, indices: list[int], scores: torch.Tensor):
        # scores: [num_indices, num_scores]
        self.scores[indices] = tensor_to_numpy(scores.to(dtype=self.dtype).cpu())
        self.written[indices] = True

        self.num_batches_since_flush += 1
        if self.num_batches_since_flush >= self.flush_interval:
            self.flush()

    def flush(self):
        # Scores before flags: a persisted "written" must imply scores on disk.
        self.scores.flush()
        self.written.flush()
        self.num_batches_since_flush = 0

    def leading_written_count(self, batches: list[list[int]]) -> int:
        return leading_written_count(self.written, batches)
