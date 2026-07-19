import heapq
import json
import math
import os
import random
import re
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Iterator, Sequence

import ml_dtypes  # noqa: F401  # registers bfloat16 dtype with numpy
import numpy as np
import pyarrow as pa
import torch
import torch.distributed as dist
from datasets import (
    Dataset,
    DatasetDict,
    IterableDatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.typing import DTypeLike, NDArray
from transformers import PreTrainedTokenizerFast, logging

from .config.config import DataConfig
from .utils.utils import (
    assert_type,
    simple_parse_kwargs_string,
)


def compute_num_token_grads(data: Dataset) -> np.ndarray:
    """Number of per-token gradient rows stored per example.

    Position ``t``'s row is ``g_t (x) a_t``. ``g_t`` is generally nonzero even
    at prompt / masked positions, because the completion-token losses backprop
    through them via causal attention -- masking is applied to the *loss*, not
    to the gradient. Storing every position therefore makes the per-token rows
    sum to the per-document gradient. Only the final position (which predicts
    nothing after ``logits[:, :-1]``) and right-padding are excluded, so
    ``num_token_grads = length - 1`` regardless of the label mask.

    Returns
    -------
    np.ndarray of shape ``(len(data),)`` with dtype int64.
    """
    if "length" in data.column_names:
        lengths = np.asarray(data["length"], dtype=np.int64)
    elif "input_ids" in data.column_names:
        lengths = np.asarray([len(x) for x in data["input_ids"]], dtype=np.int64)
    else:
        lengths = np.asarray([len(x) for x in data["labels"]], dtype=np.int64)
    return lengths - 1


def create_token_index(
    root: Path,
    num_token_grads: np.ndarray,
    grad_sizes: dict[str, int],
    dtype: DTypeLike,
) -> tuple[np.memmap, np.ndarray]:
    """Allocate a flat memory-mapped file for ragged per-token gradients.

    Parameters
    ----------
    root : Path
        Directory in which ``token_gradients.bin``, ``num_token_grads.npy``,
        ``offsets.npy`` and ``info.json`` will be created.
    num_token_grads : np.ndarray
        Number of valid gradient rows per example, shape ``(num_items,)``.
    grad_sizes : dict[str, int]
        Per-module gradient dimensions (same as :func:`create_index`).
    dtype : DTypeLike
        Element dtype for the gradient array.

    Returns
    -------
    (memmap, offsets) where *memmap* has shape ``(total_tokens, total_grad_dim)``
    and *offsets* is ``cumsum([0] + num_token_grads)`` of length
    ``num_items + 1``.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    total_grad_dim = sum(grad_sizes.values())
    offsets = np.zeros(len(num_token_grads) + 1, dtype=np.int64)
    np.cumsum(num_token_grads, out=offsets[1:])
    total_tokens = int(offsets[-1])

    np_dtype = np.dtype(dtype)
    grad_path = root / "token_gradients.bin"

    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        nbytes = np_dtype.itemsize * total_tokens * total_grad_dim
        with open(grad_path, "wb") as f:
            f.truncate(nbytes)
            os.fsync(f.fileno())

        np.save(root / "num_token_grads.npy", num_token_grads)
        np.save(root / "offsets.npy", offsets)

        with (root / "info.json").open("w") as f:
            json.dump(
                {
                    "attribute_tokens": True,
                    "total_tokens": total_tokens,
                    "total_grad_dim": total_grad_dim,
                    "num_items": len(num_token_grads),
                    "grad_sizes": grad_sizes,
                    "base_dtype": np_dtype.name,
                },
                f,
                indent=2,
            )

    if dist.is_initialized():
        dist.barrier()

    mmap = np.memmap(
        grad_path,
        dtype=np_dtype,
        mode="r+",
        shape=(total_tokens, total_grad_dim),
    )
    return mmap, offsets


def load_token_gradients(
    root_dir: Path | str,
) -> tuple[np.memmap, np.ndarray, np.ndarray]:
    """Load per-token gradients stored by :func:`create_token_index`.

    Returns
    -------
    (mmap, num_token_grads, offsets)
        *mmap* has shape ``(total_tokens, total_grad_dim)``.
        Example *i*'s gradients are ``mmap[offsets[i]:offsets[i+1]]`` with
        shape ``(num_token_grads[i], total_grad_dim)``.
    """
    root_dir = Path(root_dir)
    with (root_dir / "info.json").open("r") as f:
        info = json.load(f)

    total_tokens = info["total_tokens"]
    total_grad_dim = info["total_grad_dim"]
    base_dtype = info["base_dtype"]

    mmap = np.memmap(
        root_dir / "token_gradients.bin",
        dtype=np.dtype(base_dtype),
        mode="r",
        shape=(total_tokens, total_grad_dim),
    )
    num_token_grads = np.load(root_dir / "num_token_grads.npy")
    offsets = np.load(root_dir / "offsets.npy")
    return mmap, num_token_grads, offsets


class TokenGradients:
    """Convenience wrapper around the flat per-token gradient memmap.

    Provides ``__getitem__`` to retrieve a single example's gradients as
    a contiguous array of shape ``(num_token_grads[i], grad_dim)``.

    Parameters
    ----------
    root_dir : Path | str
        Directory produced by :func:`create_token_index`.
    """

    def __init__(self, root_dir: Path | str):
        self.mmap, self._num_token_grads, self._offsets = load_token_gradients(root_dir)

    @property
    def num_token_grads(self) -> np.ndarray:
        return self._num_token_grads

    def __len__(self) -> int:
        return len(self._num_token_grads)

    def __getitem__(self, i: int) -> np.ndarray:
        return np.asarray(self.mmap[self._offsets[i] : self._offsets[i + 1]])


def ceildiv(a: int, b: int) -> int:
    """Ceiling division of two integers."""
    return -(-a // b)  # Equivalent to math.ceil(a / b) but faster for integers


def allocate_batches(
    doc_lengths: list[int],
    N: int,
    seed: int = 42,
    max_batch_size: int | None = None,
) -> list[list[int]]:
    """
    Allocate documents into batches that are then distributed evenly across
    a fixed number of workers.

    Parameters
    ----------
    doc_lengths : Sequence[int]
        Length (in tokens) of each document.  The *i-th* document is referred to
        internally by its index ``i``.
    N : int
        Hard memory budget per *batch*, expressed as
        ``max(length in batch) * (# docs in batch) ≤ N``.
    seed : int
        Random seed for shuffling batches within each worker's allocation.
    Returns
    -------
    list[list[int]]
        ``allocation[w][b]`` is the list of document indices that belong to the
        *b-th* batch assigned to worker ``w``.  Every worker receives the same
        number of (non-empty) batches.

    Raises
    ------
    AllocationError
        If the three hard constraints cannot be satisfied.

    Notes
    -----
    1.  **Per-batch cost constraint**:  Each batch is padded to the maximum
        sequence length *inside that batch*, so its cost in “token × examples”
        units is ``max_len_in_batch * batch_size``.  This must stay ≤ ``N``.
    2.  **Bin-packing strategy**:  We use a simple greedy bin-packing algorithm
        that sorts the documents by length and tries to fit them into batches
        without exceeding the cost constraint.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    (batches,) = _allocate_batches_world(
        doc_lengths, N, world_size, seed, ranks=[rank], max_batch_size=max_batch_size
    )
    return batches


def _allocate_batches_world(
    doc_lengths: list[int],
    N: int,
    world_size: int,
    seed: int = 42,
    ranks: list[int] | None = None,
    max_batch_size: int | None = None,
) -> list[list[list[int]]]:
    """Lower-level version of allocate_batches that returns batches for specified ranks.

    If ranks is None, returns batches for all ranks.
    """
    if ranks is None:
        ranks = list(range(world_size))
    # Skip documents shorter than 2 tokens: they cannot produce a valid
    # next-token label so they contribute no gradient.  Keeping them would
    # also be problematic for length-0 documents which create [N, 0] tensors
    # that hang the model forward pass.  These documents are simply omitted
    # from batches; their gradient index entries stay at the pre-initialized
    # zero value.
    docs_sorted = sorted(
        ((i, l) for i, l in enumerate(doc_lengths) if l >= 2),
        key=lambda x: x[1],
        reverse=True,
    )
    n_skipped = len(doc_lengths) - len(docs_sorted)
    if n_skipped > 0 and (not dist.is_initialized() or dist.get_rank() == 0):
        print(
            f"Skipping {n_skipped} documents with fewer than 2 tokens "
            f"(no valid next-token labels)."
        )
    if len(docs_sorted) < world_size:
        raise RuntimeError(
            "Not enough documents (with 2+ tokens) to distribute across workers."
        )
    if docs_sorted[0][1] > N:  # a single document would overflow any batch
        raise RuntimeError(
            f"At least one document is too long for the token batch size {N}."
        )

    # ---------------------------------------------------------------------
    # 1) Bin packing under the cost function
    #    cost(batch) = max_len_in_batch * len(batch)
    # ---------------------------------------------------------------------
    batches: list[list[int]] = []  # holds document *indices*
    cur_batch: list[int] = []  # holds document *indices* in the current batch

    for idx, length in docs_sorted:
        if not cur_batch:
            # Start a new batch with the current document
            cur_batch.append(idx)
        else:
            # Check if adding this document would exceed the budget
            new_cost = max(length, doc_lengths[cur_batch[0]]) * (len(cur_batch) + 1)
            within_count = (
                max_batch_size is None or len(cur_batch) + 1 <= max_batch_size
            )
            if new_cost <= N and within_count:
                # It fits, so add it to the current batch
                cur_batch.append(idx)
            else:
                # It doesn't fit, finalize the current batch and start a new one
                batches.append(cur_batch)
                cur_batch = [idx]

    # Finalize the last batch if it's not empty
    if cur_batch:
        batches.append(cur_batch)

    # ---------------------------------------------------------------------
    # 2) Ensure every worker gets ≥ 1 batch
    # ---------------------------------------------------------------------
    if len(batches) < world_size:
        # Heapify
        counter = 0
        heap: list[tuple[int, int, list[int]]] = []
        for batch in batches:
            heapq.heappush(heap, (-len(batch), counter, batch))
            counter += 1

        # Split large batches until we have >= workers batches.
        # Use an inverted min heap rather than a max heap for old
        # python version support.
        while len(heap) < world_size:
            _, _, big = heapq.heappop(heap)
            if len(big) == 1:
                # Every remaining bin in the heap is a singleton,
                # so the constraint is unsatisfiable.
                raise RuntimeError(
                    "Not enough documents to give each worker at least one batch."
                )
            singleton = [big.pop()]
            heapq.heappush(heap, (-len(big), counter, big))
            counter += 1
            heapq.heappush(heap, (-1, counter, singleton))
            counter += 1
        batches = [b for _, _, b in heap]

    # ---------------------------------------------------------------------
    # 3) Pad the number of batches to a multiple of `workers`
    # ---------------------------------------------------------------------
    k = -(-len(batches) // world_size)  # ceiling division
    target_batches = world_size * k  # == k batches per worker

    # Split arbitrary (non-singleton) batches until we reach the target
    i = 0
    while len(batches) < target_batches and i < len(batches):
        batch = batches[i % len(batches)]
        if len(batch) == 1:
            i += 1  # try another batch
            continue
        batches.append([batch.pop()])  # split off a singleton
        i += 1

    assert len(batches) == target_batches, (
        "Could not construct a number of batches divisible by the world size."
        " If variability of item lengths in your dataset is low "
        "consider using a different dataset size or token batch size."
    )
    assert all(
        max(doc_lengths[i] for i in batch) * len(batch) <= N for batch in batches
    )

    # ---------------------------------------------------------------------
    # 4) Round-robin assignment to workers
    # ---------------------------------------------------------------------
    allocation: list[list[list[int]]] = [[] for _ in range(world_size)]
    for b_idx, batch in enumerate(batches):
        allocation[b_idx % world_size].append(batch)

    # Sanity: equal # of batches per worker
    assert len({len(b) for b in allocation}) == 1

    # Break any systematic ordering of batches (shuffle only requested ranks)
    for rank in ranks:
        random.seed(seed)
        random.shuffle(allocation[rank])

    return [allocation[rank] for rank in ranks]


def column_offsets(grad_sizes: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Map each module name to its ``(start, end)`` column range in a flat
    ``(num_grads, total_grad_dim)`` gradient array laid out in `grad_sizes`
    order (the on-disk layout of :func:`create_index`)."""
    offsets = {}
    start = 0
    for name, size in grad_sizes.items():
        offsets[name] = (start, start + size)
        start += size
    return offsets


def create_index(
    root: Path,
    num_grads: int,
    grad_sizes: dict[str, int],
    dtype: DTypeLike,
) -> np.memmap:
    """Create a memory-mapped ``(num_grads, total_grad_dim)`` gradient file
    (modules laid out in `grad_sizes` order) and persist metadata."""
    grad_path = root / "gradients.bin"
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Legacy structured dtype, still written for external readers of older stores
    struct_dtype = {
        "names": [name for name in grad_sizes.keys()],
        "formats": [f"({size},){np.dtype(dtype).str}" for size in grad_sizes.values()],
        "itemsize": np.dtype(dtype).itemsize * sum(grad_sizes.values()),
    }

    # ── 1. Rank-0 creates file & metadata exactly once ─────────────────────────
    if rank == 0:
        # Ensure the directory exists
        root.mkdir(parents=True, exist_ok=True)

        # Allocate (extends file to right size without writing zeros byte-by-byte)
        nbytes = struct_dtype["itemsize"] * num_grads
        with open(grad_path, "wb") as f:
            f.truncate(nbytes)

            # Force the directory entry + data to disk *before* other ranks continue
            os.fsync(f.fileno())

        # Persist metadata for future runs
        with (root / "info.json").open("w") as f:
            json.dump(
                {
                    "num_grads": num_grads,
                    "dtype": struct_dtype,
                    "grad_sizes": grad_sizes,
                    "base_dtype": np.dtype(dtype).name,
                },
                f,
                indent=2,
            )

    # ── 2. Everyone blocks until the file is definitely there & sized ─────────────
    if dist.is_initialized():
        dist.barrier()

    return np.memmap(
        grad_path,
        dtype=np.dtype(dtype),
        mode="r+",
        shape=(num_grads, sum(grad_sizes.values())),
    )


def load_data_string(
    data_str: str,
    split: str = "train",
    subset: str | None = None,
    data_kwargs: str = "",
) -> Dataset:
    """Load a dataset from a string identifier or path."""
    if data_str.endswith(".csv"):
        ds = load_dataset("csv", data_files=data_str, split=split)
    elif data_str.endswith(".json") or data_str.endswith(".jsonl"):
        ds = load_dataset("json", data_files=data_str, split=split)
    elif Path(data_str).is_dir() and (
        (Path(data_str) / "dataset_info.json").exists()
        or (Path(data_str) / "dataset_dict.json").exists()
    ):
        ds = load_from_disk(data_str, keep_in_memory=False)
    else:
        try:
            kwargs = simple_parse_kwargs_string(data_kwargs)
            ds = load_dataset(data_str, subset, split=split, **kwargs)

            if isinstance(ds, DatasetDict) or isinstance(ds, IterableDatasetDict):
                raise NotImplementedError(
                    "DatasetDicts and IterableDatasetDicts are not supported."
                )
        except ValueError as e:
            # Automatically use load_from_disk if appropriate
            if "load_from_disk" in str(e):
                ds = load_from_disk(data_str, keep_in_memory=False)
            else:
                raise e

    if isinstance(ds, DatasetDict):
        if "[" in split or "+" in split:
            raise NotImplementedError(
                f"Split slicing/concatenation expressions are not supported "
                f"for load_from_disk paths: {split!r}"
            )
        ds = ds[split]

    ds = assert_type(Dataset, ds)
    return ds


def load_gradients(root_dir: Path | str) -> np.memmap:
    """Map the gradients stored in `root_dir` into memory as a flat
    ``(num_grads, total_grad_dim)`` array, with modules laid out in
    ``grad_sizes`` order (see :func:`column_offsets` for per-module slicing).

    Dispatches on ``info.json["attribute_tokens"]``: a per-token index (as
    written by :func:`create_token_index`) maps ``token_gradients.bin`` with
    ``num_grads = total_tokens``; a plain index (as written by
    :func:`create_index`) maps ``gradients.bin``.
    """
    root_dir = Path(root_dir)
    with (root_dir / "info.json").open("r") as f:
        info = json.load(f)

    if "base_dtype" in info:
        dtype = np.dtype(info["base_dtype"])
    else:
        # Old stores lack base_dtype; recover the scalar dtype from a field format
        dtype = np.dtype(info["dtype"]["formats"][0]).base

    if info.get("attribute_tokens"):
        return np.memmap(
            root_dir / "token_gradients.bin",
            dtype=dtype,
            mode="r",
            shape=(info["total_tokens"], info["total_grad_dim"]),
        )

    return np.memmap(
        root_dir / "gradients.bin",
        dtype=dtype,
        mode="r",
        shape=(info["num_grads"], sum(info["grad_sizes"].values())),
    )


class ModuleGradients:
    """Dict-like, module-name-keyed view over a flat memmapped gradient store:
    ``grads[name]`` returns the ``(num_grads, size)`` column slice for that
    module."""

    def __init__(self, mmap: np.memmap, grad_sizes: dict[str, int]):
        self.mmap = mmap
        self.offsets = column_offsets(grad_sizes)

    def __getitem__(self, name: str) -> np.ndarray:
        lo, hi = self.offsets[name]
        return self.mmap[:, lo:hi]

    def __len__(self) -> int:
        return len(self.mmap)

    def __iter__(self) -> Iterator[str]:
        return iter(self.offsets)

    def __contains__(self, name: str) -> bool:
        return name in self.offsets

    def keys(self):
        return self.offsets.keys()


def load_module_gradients(root_dir: Path | str) -> ModuleGradients:
    """Load the gradients stored in `root_dir` keyed by module name."""
    root_dir = Path(root_dir)
    with (root_dir / "info.json").open("r") as f:
        grad_sizes = json.load(f)["grad_sizes"]

    return ModuleGradients(load_gradients(root_dir), grad_sizes)


def load_gradient_dataset(root_dir: Path) -> Dataset:
    """Load a dataset of gradients from `root_dir`."""

    def load_shard(dir: Path) -> Dataset:
        ds = Dataset.load_from_disk(str(dir / "data.hf"))

        # Add gradients to HF dataset.
        mmap = load_gradients(dir)

        def _to_arrow(arr: np.ndarray) -> pa.Array:
            """Convert numpy array to PyArrow, casting bfloat16 to float32."""
            if arr.dtype == np.dtype("bfloat16"):
                arr = arr.astype(np.float32)
            return pa.array(arr)

        flat = _to_arrow(mmap.reshape(-1).copy())
        col_arrow = pa.FixedSizeListArray.from_arrays(flat, mmap.shape[1])
        ds = ds.add_column("gradients", col_arrow, new_fingerprint="gradients")

        return ds

    if (root_dir / "data.hf").exists():
        return load_shard(root_dir)

    # Flatten indices to avoid CPU OOM
    return concatenate_datasets(
        [load_shard(path) for path in sorted(root_dir.iterdir()) if path.is_dir()]
    ).flatten_indices()


class Scores:
    def __init__(self, mmap: np.memmap, info: dict[str, Any]):
        self.mmap = mmap
        self.info = info
        self.num_scores = info["num_scores"]

        self._score_fields = [f"score_{i}" for i in range(self.num_scores)]

    def __len__(self) -> int:
        return len(self.mmap)

    def __getitem__(self, key: Any) -> NDArray:
        items = self.mmap[key]
        return structured_to_unstructured(items[self._score_fields])

    def get(self, key: Any, score_idx: int = 0) -> NDArray:
        """Get scores for a specific score index."""
        return self.mmap[key][f"score_{score_idx}"]

    def is_written(self) -> bool:
        """Check whether all scores in the structured mmap have
        been written to (i.e. are not still zeros)"""
        return all(np.all(self.mmap[f"written_{i}"]) for i in range(self.num_scores))


class ArrayScores:
    """Dense ``[num_items, num_scores]`` score matrix with the same interface
    as :class:`Scores`, for scores saved as a plain ``.npy`` (e.g. an ad hoc
    array from outside the standard scoring pipeline)."""

    def __init__(self, arr: np.ndarray):
        if arr.ndim == 1:
            arr = arr[:, None]
        self.arr = arr
        self.num_scores = arr.shape[1]

    def __len__(self) -> int:
        return self.arr.shape[0]

    def __getitem__(self, key: Any) -> NDArray:
        return self.arr[key]

    def get(self, key: Any, score_idx: int = 0) -> NDArray:
        return self.arr[key, score_idx]

    def is_written(self) -> bool:
        return True


class TokenScores:
    """Flat ``(total_tokens, num_scores)`` per-token score store, as written
    by :class:`bergson.score.score_writer.MemmapTokenScoreWriter`. Document
    ``i``'s per-token scores live at rows ``offsets[i]:offsets[i+1]``."""

    def __init__(self, mmap: np.memmap, info: dict[str, Any], offsets: NDArray):
        self.mmap = mmap
        self.info = info
        self.offsets = offsets
        self.num_scores = info["num_scores"]

    def __len__(self) -> int:
        return self.info["num_items"]

    def __getitem__(self, key: Any) -> NDArray:
        return np.asarray(self.mmap[key])

    def get(self, key: Any, score_idx: int = 0) -> NDArray:
        return np.asarray(self.mmap[key, score_idx])

    def is_written(self) -> bool:
        return True

    def to_grid(self) -> torch.Tensor:
        """Unpack into a zero-padded ``[docs, seq_len]`` grid (``[docs,
        seq_len, num_scores]`` when multi-query), so token ``t`` of doc ``d``
        lands at ``grid[d, t]`` -- the per-token training weight layout."""
        num_docs = len(self)
        tokens_per_doc = np.diff(self.offsets).astype(np.int64)
        seq_len = int(tokens_per_doc.max()) + 1

        grid = np.zeros((num_docs, seq_len, self.num_scores), dtype=np.float32)
        for doc in range(num_docs):
            grid[doc, : tokens_per_doc[doc]] = self.mmap[
                self.offsets[doc] : self.offsets[doc + 1]
            ]

        scores = torch.from_numpy(grid)
        return scores[..., 0] if self.num_scores == 1 else scores


def load_scores(path: Path) -> Scores | ArrayScores | TokenScores:
    """Load a score store written by the standard scoring pipeline.

    Dispatches on ``path``: a plain ``.npy`` array loads as
    :class:`ArrayScores`; a directory whose ``info.json`` has
    ``attribute_tokens`` set loads its ``token_scores.bin`` as
    :class:`TokenScores`; any other score directory loads its ``scores.bin``
    as :class:`Scores`.
    """
    if path.suffix == ".npy":
        return ArrayScores(np.load(path))

    info_path = path / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)

    if info.get("attribute_tokens"):
        offsets = np.load(path / "offsets.npy")
        mmap = np.memmap(
            path / "token_scores.bin",
            dtype=info["dtype"],
            mode="r",
            shape=(info["total_tokens"], info["num_scores"]),
        )
        return TokenScores(mmap, info, offsets)

    mmap = np.memmap(
        path / "scores.bin",
        dtype=info["dtype"],
        mode="r",
        shape=(info["num_items"],),
    )

    return Scores(mmap, info)


def sorted_checkpoints(folder: str) -> list[tuple[int, str]]:
    """
    Return a list of (step, filepath) sorted by step
    for files named like: step_<index>.ckpt
    """
    pattern = re.compile(r"step_(\d+)\.ckpt$")

    checkpoints = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        match = pattern.match(name)
        if match:
            step = int(match.group(1))
            checkpoints.append((step, path))

    return sorted(checkpoints, key=lambda x: x[0])


def pad_and_tensor(
    sequences: list[list[int]],
    labels: list[list[int]] | None = None,
    *,
    padding_value: int = 0,
    dtype: torch.dtype | None = torch.long,
    device: torch.device | None = None,
    sync_max_len: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad a list of sequences to the same length and convert them to tensors.
    Returns a tuple of (padded sequences, padded labels, shift_loss_masks,
    collection_masks). The labels are the same as the sequences, but with -100
    for the padding positions, which is useful for ignoring padding in loss
    calculations.

    Both masks are aligned with input positions. ``shift_loss_masks[i]`` is
    True iff ``labels[i+1] != -100``, i.e. it marks the input positions whose
    next-token prediction is supervised. ``collection_masks`` marks every
    non-padding position but each sequence's last, independent of label
    masking (the gradient-carrying positions).

    When ``sync_max_len`` is True (default) and a process group is
    initialized, the padding length is reduced to the global max across
    ranks via an ``all_reduce``. This is required by callers that need
    matching activation shapes across ranks (MAGIC, see PR #227). The
    build-time gradient collector passes ``sync_max_len=False`` because
    its bin-packer enforces a per-rank token budget
    (``max_len × batch_size ≤ N``) that the global all-reduce silently
    violates when ranks are assigned batches with very different
    max_lens — a single long-example batch on one rank forces every
    short-example batch on its peers to pad up to the long length,
    blowing the budget by orders of magnitude.
    """
    if labels is None:
        labels = sequences

    # find max length
    max_len = max(len(seq) for seq in sequences)
    if sync_max_len and dist.is_initialized():
        max_len_tensor = torch.tensor(max_len, device=device)
        dist.all_reduce(max_len_tensor, op=dist.ReduceOp.MAX)
        max_len = int(max_len_tensor)

    # pad each sequence
    padded = [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]
    labels = [label + [-100] * (max_len - len(label)) for label in labels]

    # convert to tensor
    padded_tokens = torch.tensor(padded, dtype=dtype, device=device)
    padded_labels = torch.tensor(labels, dtype=dtype, device=device)

    # shift_loss_masks[i] = labels[i+1] != -100
    N, S = padded_tokens.shape
    shift_loss_masks = torch.zeros(N, S, dtype=torch.bool, device=device)
    shift_loss_masks[:, :-1] = padded_labels[:, 1:] != -100

    # Compute collection_masks: every non-padding position but each
    # sequence's last, regardless of label masking.
    lengths = torch.tensor([len(seq) for seq in sequences], device=device)
    positions = torch.arange(S, device=device)
    collection_masks = (positions.unsqueeze(0) + 1) < lengths.unsqueeze(1)

    return padded_tokens, padded_labels, shift_loss_masks, collection_masks


def tokenize(
    batch: dict,
    *,
    args: DataConfig,
    tokenizer,
    max_length: int | None = None,
):
    """Tokenize a batch of data with `tokenizer` according to `args`."""
    kwargs: dict[str, Any] = dict(
        return_attention_mask=False,
        return_length=True,
        truncation=args.truncation,
    )
    if args.truncation and max_length is not None:
        kwargs["max_length"] = max_length
    if args.completion_column:
        # We're dealing with a prompt-completion dataset
        convos = [
            [
                {"role": "user", "content": assert_type(str, prompt)},
                {"role": "assistant", "content": assert_type(str, resp)},
            ]
            for prompt, resp in zip(
                batch[args.prompt_column], batch[args.completion_column]
            )
        ]
    elif args.conversation_column:
        # We're dealing with a conversation dataset
        convos = assert_type(list, batch[args.conversation_column])
    else:
        # We're dealing with vanilla next-token prediction
        return tokenizer(batch[args.prompt_column], **kwargs)

    # Make sure we only compute loss on the assistant's responses
    # Chat templates already include special tokens (BOS/EOS) in the rendered
    # string, so we must disable add_special_tokens to avoid duplicates (e.g.
    # Llama 3 renders <|begin_of_text|> and would otherwise add a second BOS).
    strings = tokenizer.apply_chat_template(convos, tokenize=False)
    encodings = tokenizer(strings, add_special_tokens=False, **kwargs)
    labels_list: list[list[int]] = []

    for i, convo in enumerate(convos):
        # Find the spans (start, end) of the assistant's responses in the tokens
        spans: list[tuple[int, int]] = []

        # We use rfind to find the last match in the string so if the answer is
        # duplicated in the user prompt we don't select that.

        # We work backwards through the messages and restrict the search to not
        # use previously matching substrings so if two assistant messages
        # have the same content they won't both match the right-most substring.
        search_end = len(strings[i])

        for msg in reversed(convo):
            if msg["role"] != "assistant":
                continue

            ans = msg["content"]
            start = strings[i].rfind(ans, 0, search_end)
            if start < 0:
                print("String under test: ", strings[i])
                print("Substring search: ", ans)
                raise RuntimeError(
                    "Failed to find completion in the chat-formatted conversation. "
                    "Make sure the chat template does not alter the completion, e.g. "
                    "by removing leading whitespace."
                )

            # move past this match
            end = start + len(ans)

            start_token_idx = encodings.char_to_token(i, start)
            end_token_idx = encodings.char_to_token(i, end)
            # When end falls on the last char, char_to_token returns None;
            # look up the token for the previous char and advance by one.
            if end_token_idx is None and end > 0:
                prev = encodings.char_to_token(i, end - 1)
                if prev is not None:
                    end_token_idx = prev + 1
            spans.append((start_token_idx, end_token_idx))

            # Update the boundary; the next message must be found before this one
            search_end = start

        spans = list(reversed(spans))

        # Labels are -100 everywhere except where the assistant's response is
        tokens = encodings["input_ids"][i]
        labels = [-100] * len(tokens)
        for start, end in spans:
            if start is None:
                # Entire span is beyond the truncation boundary, so we skip.
                continue
            if end is None:
                # Span starts in-bounds but extends past truncation, so we
                # label up to the end of the truncated sequence
                end = len(tokens)
            labels[start:end] = tokens[start:end]

        labels_list.append(labels)

    return dict(**encodings, labels=labels_list)


def tokenize_and_chunk(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    chunk_size: int,
    text_column: str = "text",
    *,
    num_proc: int | None = None,
) -> Dataset:
    """
    Tokenizes a text dataset and chunks tokens into uniform-length sequences.

    Uses a streaming generator to carry remainder tokens across document
    boundaries — no tokens are ever dropped, no padding is used, and memory
    usage stays flat regardless of dataset size.

    Args:
        dataset:     HuggingFace Dataset with a text column.
        tokenizer:   A HuggingFace tokenizer (must have eos_token_id set).
        chunk_size:  Number of tokens per output chunk.
        text_column: Name of the column containing raw text.
        num_proc:    Number of processes for the tokenization .map() pass.
                     Defaults to half the CPUs, capped so each process gets
                     at least ~256 documents.

    Returns:
        A Dataset where every row has a single 'input_ids' list of length chunk_size.
        The dataset has a doc_ids column of the same size as input_ids, that
        maps chunk tokens to the corresponding doc ID in the input dataset.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer does not have an eos_token_id set.")

    # Suppress long sequence errors
    original_verbosity = logging.get_verbosity()
    logging.set_verbosity_error()

    # Add original index as column
    n_before = len(dataset)
    dataset = dataset.map(
        lambda _, idx: {"_orig_idx": idx},
        with_indices=True,
        desc="Tagging original indices",
    )

    # Drop and log empty rows.
    dataset = dataset.filter(
        lambda row: isinstance(row[text_column], str)
        and row[text_column].strip() != "",
        desc="Filtering empty documents",
    )
    n_dropped = n_before - len(dataset)
    if n_dropped > 0:
        pct = n_dropped / n_before * 100
        print(f"Warning: {n_dropped}/{n_before} empty documents " f"({pct:.1f}%).")

    if num_proc is None:
        num_proc = min(cpu_count() // 2, max(1, len(dataset) // 256))

    # ── Step 1: tokenize each document in parallel ───────────────────────────
    def tokenize_batch(batch):
        out = tokenizer(
            batch[text_column],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        out["_orig_idx"] = batch["_orig_idx"]
        return out

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        desc="Tokenizing",
        num_proc=num_proc,
        remove_columns=[c for c in dataset.column_names if c != "_orig_idx"],
    )

    # ── Step 2: concatenate the token stream and slice into fixed-size chunks ──
    def chunk_batch(batch):
        # Flatten all token lists in this batch into one stream
        doc_stream = []
        token_stream = []
        for doc_idx, ids in zip(batch["_orig_idx"], batch["input_ids"]):
            # Mark these tokens as belonging to the current document
            doc_stream.extend([doc_idx] * (len(ids) + 1))  # +1 for EOS token

            token_stream.extend(ids)
            token_stream.append(eos_id)  # ensure document boundary tokens

        # Slice into complete chunks; keep the remainder for the next batch
        # NOTE: .map() does NOT carry state between batches, so any tokens that
        # don't fill a complete chunk at the end of a batch are dropped.
        # To avoid this, set batch_size large enough (or process as a single batch)
        # so that tail loss is negligible, OR use the single-process approach below.
        n_chunks = len(token_stream) // chunk_size
        doc_chunks = [
            doc_stream[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)
        ]
        token_chunks = [
            token_stream[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)
        ]
        return {
            "input_ids": token_chunks,
            "doc_ids": doc_chunks,
            "length": [chunk_size] * n_chunks,
        }

    # Cap parallelism so each worker gets enough documents to fill many chunks,
    # since tail tokens that don't fill a chunk are dropped per-worker.
    min_docs_per_worker = chunk_size * 4
    num_proc = min(num_proc, max(len(tokenized) // min_docs_per_worker, 1))
    bs = min(10_000, len(tokenized) // num_proc)
    chunked = tokenized.map(
        chunk_batch,
        batched=True,
        batch_size=bs,
        num_proc=num_proc,
        desc="Chunking",
        remove_columns=tokenized.column_names,
    )
    # Make sure HuggingFace didn't change the dataset type!
    assert isinstance(chunked, type(dataset))

    # Warn if chunking dropped a significant number of documents
    n_docs = len(dataset)
    if n_docs > 0 and "doc_ids" in chunked.column_names:
        represented = set()
        for doc_ids in chunked["doc_ids"]:
            represented.update(doc_ids)
        n_dropped = n_docs - len(represented)
        if n_dropped > 0:
            pct = n_dropped / n_docs * 100
            print(
                f"Warning: chunking dropped {n_dropped}/{n_docs} short documents"
                f" ({pct:.1f}%) that didn't fill a {chunk_size}-token chunk "
                f"when using a document batch size of {bs}."
            )

    # Restore original logging level
    logging.set_verbosity(original_verbosity)
    return chunked


def unflatten(x: torch.Tensor, shapes: dict[str, Sequence[int]], dim: int = -1):
    """Unflatten a tensor `x` into a dictionary of tensors with specified shapes."""
    numels = [math.prod(shape) for shape in shapes.values()]
    return {
        name: x.unflatten(dim, shape)
        for (name, shape), x in zip(shapes.items(), x.split(numels, dim=dim))
    }
