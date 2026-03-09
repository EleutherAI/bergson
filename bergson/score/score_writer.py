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

        self.path.mkdir(parents=True, exist_ok=True)
        scores_file_path = self.path / "token_scores.bin"
        np_dtype = convert_dtype_to_np(dtype)

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not scores_file_path.exists():
            print(f"Creating new token scores file: {scores_file_path}")

            self.scores = np.memmap(
                str(scores_file_path),
                dtype=np_dtype,
                mode="w+",
                shape=(total_tokens, num_scores),
            )
            self.scores[:] = 0
            self.flush()

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

        if dist.is_initialized():
            dist.barrier()

        self.scores = np.memmap(
            str(scores_file_path),
            dtype=np_dtype,
            mode="r+",
            shape=(total_tokens, num_scores),
        )

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

        self.num_batches_since_flush += 1
        if self.num_batches_since_flush >= self.flush_interval:
            self.flush()

    def flush(self):
        self.scores.flush()
        self.num_batches_since_flush = 0


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
        score_size = np_dtype.itemsize
        bool_size = np.dtype("bool").itemsize

        # Build a structured dtype with (score, written) pairs per query
        # Align each pair to the next power of 2 for efficiency
        pair_size = score_size + bool_size
        aligned_pair_size = 1 << (pair_size - 1).bit_length()  # Next power of 2

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

        # For JSON serialization, convert numpy dtype to string
        format_strs = [str(f) if isinstance(f, np.dtype) else f for f in formats]
        struct_dtype_json = {
            "names": names,
            "formats": format_strs,
            "offsets": offsets,
            "itemsize": itemsize,
        }

        struct_dtype = {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": itemsize,
        }

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
                        "num_items": num_items,
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
