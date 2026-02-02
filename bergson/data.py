import json
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence, cast, overload

import ml_dtypes  # noqa: F401  # registers bfloat16 dtype with numpy
import numpy as np
import pyarrow as pa
import torch
import torch.distributed as dist
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from numpy.typing import DTypeLike

from .config import DataConfig, ReduceConfig
from .utils.utils import (
    assert_type,
    convert_dtype_to_np,
    simple_parse_args_string,
    tensor_to_numpy,
)


def compute_num_seq_grads(data: Dataset) -> np.ndarray:
    """Compute the number of valid gradient positions per example.

    A token at position t produces a gradient iff the *next* token's label
    is not -100 (the ignore index).  For vanilla NTP (no explicit ``labels``
    column) every position except the last is valid, so
    ``seq_length = length - 1``.

    Returns
    -------
    np.ndarray of shape ``(len(data),)`` with dtype int64.
    """
    if "labels" in data.column_names:
        # Count positions where labels[t+1] != -100 for t in 0..len-2
        seq_lengths = []
        for labels in data["labels"]:
            labels_arr = np.asarray(labels)
            seq_lengths.append(int(np.sum(labels_arr[1:] != -100)))
        return np.array(seq_lengths, dtype=np.int64)
    else:
        lengths = np.array(data["length"], dtype=np.int64)
        return lengths - 1


def create_token_index(
    root: Path,
    seq_lengths: np.ndarray,
    grad_sizes: dict[str, int],
    dtype: DTypeLike,
) -> tuple[np.memmap, np.ndarray]:
    """Allocate a flat memory-mapped file for ragged per-token gradients.

    Parameters
    ----------
    root : Path
        Directory in which ``token_gradients.bin``, ``seq_lengths.npy``,
        ``offsets.npy`` and ``info.json`` will be created.
    seq_lengths : np.ndarray
        Number of valid gradient rows per example, shape ``(num_items,)``.
    grad_sizes : dict[str, int]
        Per-module gradient dimensions (same as :func:`create_index`).
    dtype : DTypeLike
        Element dtype for the gradient array.

    Returns
    -------
    (memmap, offsets) where *memmap* has shape ``(total_tokens, total_grad_dim)``
    and *offsets* is ``cumsum([0] + seq_lengths)`` of length ``num_items + 1``.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    total_grad_dim = sum(grad_sizes.values())
    offsets = np.zeros(len(seq_lengths) + 1, dtype=np.int64)
    np.cumsum(seq_lengths, out=offsets[1:])
    total_tokens = int(offsets[-1])

    np_dtype = np.dtype(dtype)
    grad_path = root / "token_gradients.bin"

    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        nbytes = np_dtype.itemsize * total_tokens * total_grad_dim
        with open(grad_path, "wb") as f:
            f.truncate(nbytes)
            os.fsync(f.fileno())

        np.save(root / "seq_lengths.npy", seq_lengths)
        np.save(root / "offsets.npy", offsets)

        with (root / "info.json").open("w") as f:
            json.dump(
                {
                    "token_attributions": True,
                    "total_tokens": total_tokens,
                    "total_grad_dim": total_grad_dim,
                    "num_items": len(seq_lengths),
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
    (mmap, seq_lengths, offsets)
        *mmap* has shape ``(total_tokens, total_grad_dim)``.
        Example *i*'s gradients are ``mmap[offsets[i]:offsets[i+1]]`` with
        shape ``(seq_lengths[i], total_grad_dim)``.
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
    seq_lengths = np.load(root_dir / "seq_lengths.npy")
    offsets = np.load(root_dir / "offsets.npy")
    return mmap, seq_lengths, offsets


class TokenGradients:
    """Convenience wrapper around the flat per-token gradient memmap.

    Provides ``__getitem__`` to retrieve a single example's gradients as
    a contiguous array of shape ``(seq_lengths[i], grad_dim)``.

    Parameters
    ----------
    root_dir : Path | str
        Directory produced by :func:`create_token_index`.
    """

    def __init__(self, root_dir: Path | str):
        self.mmap, self._seq_lengths, self._offsets = load_token_gradients(root_dir)

    @property
    def seq_lengths(self) -> np.ndarray:
        return self._seq_lengths

    def __len__(self) -> int:
        return len(self._seq_lengths)

    def __getitem__(self, i: int) -> np.ndarray:
        return np.asarray(self.mmap[self._offsets[i] : self._offsets[i + 1]])


class TokenBuilder:
    """Creates and writes per-token gradients to disk.

    Parameters
    ----------
    path : Path
        Root directory for the index artifacts.
    data : Dataset
        The dataset being indexed (used only for length).
    grad_sizes : dict[str, int]
        Per-module gradient dimensions.
    dtype : torch.dtype
        Torch dtype for the gradients (converted to numpy internally).
    seq_lengths : np.ndarray
        Number of valid positions per example.
    """

    def __init__(
        self,
        path: Path,
        data: Dataset,
        grad_sizes: dict[str, int],
        dtype: torch.dtype,
        seq_lengths: np.ndarray,
    ):
        self.grad_sizes = grad_sizes
        self.num_items = len(data)
        np_dtype = convert_dtype_to_np(dtype)
        self.grad_buffer, self.offsets = create_token_index(
            path, seq_lengths, grad_sizes, np_dtype,
        )
        self.seq_lengths = seq_lengths

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ):
        """Write a batch of per-token gradients to the flat buffer.

        ``mod_grads`` values have shape ``[total_valid_in_batch, grad_dim_mod]``
        (already filtered to valid positions).  Batch indices may be
        non-contiguous, so each example's chunk is written individually.
        """
        torch.cuda.synchronize()

        per_example_lengths = self.seq_lengths[indices]

        col_offset = 0
        for module_name in self.grad_sizes.keys():
            g_np = tensor_to_numpy(mod_grads[module_name])
            dim = g_np.shape[1]
            row = 0
            for idx, sl in zip(indices, per_example_lengths):
                buf_start = int(self.offsets[idx])
                buf_end = int(self.offsets[idx + 1])
                self.grad_buffer[
                    buf_start:buf_end, col_offset : col_offset + dim
                ] = g_np[row : row + sl]
                row += sl
            col_offset += dim

    def flush(self):
        self.grad_buffer.flush()

    def dist_reduce(self):
        # Token-level gradients don't support reduction.
        pass


def ceildiv(a: int, b: int) -> int:
    """Ceiling division of two integers."""
    return -(-a // b)  # Equivalent to math.ceil(a / b) but faster for integers


def allocate_batches(
    doc_lengths: list[int],
    N: int,
    seed: int = 42,
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
    if len(doc_lengths) < world_size:
        raise RuntimeError("Not enough documents to distribute across workers.")

    docs_sorted = sorted(enumerate(doc_lengths), key=lambda x: x[1], reverse=True)
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
            if new_cost <= N:
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
        # split the largest batches (by size) until we have ≥ workers batches
        batches.sort(key=len, reverse=True)
        while len(batches) < world_size:
            big = batches.pop(0)  # take the current largest
            if len(big) == 1:  # cannot split a singleton
                raise RuntimeError(
                    "Not enough documents to give each worker at least one batch."
                )
            batches.append([big.pop()])  # move one doc into new batch
            batches.append(big)  # put the remainder back
            # preserve cost constraint automatically

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

    # Break any systematic ordering of batches
    random.seed(seed)
    random.shuffle(allocation[rank])

    return allocation[rank]


def create_index(
    root: Path,
    num_grads: int,
    grad_sizes: dict[str, int],
    dtype: DTypeLike,
    with_structure: bool = True,
) -> np.memmap:
    """Create a memory-mapped file for storing structured gradients
    and persist metadata."""
    grad_path = root / "gradients.bin"
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Build a json-serializable structured dtype
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

    if with_structure:
        dtype = np.dtype(struct_dtype)  # type: ignore
        shape = (num_grads,)
    else:
        dtype = np.dtype(dtype)
        shape = (num_grads, sum(grad_sizes.values()))

    return np.memmap(
        grad_path,
        dtype=dtype,
        mode="r+",
        shape=shape,
    )


def load_data_string(
    data_str: str,
    split: str = "train",
    subset: str | None = None,
    data_args: str = "",
) -> Dataset | IterableDataset:
    """Load a dataset from a string identifier or path."""
    if data_str.endswith(".csv"):
        ds = assert_type(Dataset, Dataset.from_csv(data_str))
    elif data_str.endswith(".json") or data_str.endswith(".jsonl"):
        ds = assert_type(Dataset, Dataset.from_json(data_str))
    elif Path(data_str).is_dir() and (Path(data_str) / "dataset_info.json").exists():
        ds = load_from_disk(data_str, keep_in_memory=False)
        if isinstance(ds, DatasetDict):
            ds = ds[split]
    else:
        try:
            kwargs = simple_parse_args_string(data_args)
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

    return ds


def load_gradients(root_dir: Path | str, structured: bool = True) -> np.memmap:
    """Map the structured gradients stored in `root_dir` into memory."""
    root_dir = Path(root_dir)
    with (root_dir / "info.json").open("r") as f:
        info = json.load(f)

    num_grads = info["num_grads"]

    if structured:
        dtype = info["dtype"]
        shape = (num_grads,)
    else:
        dtype = info["base_dtype"]
        grad_sizes = info["grad_sizes"]
        shape = (num_grads, sum(grad_sizes.values()))

    return np.memmap(
        root_dir / "gradients.bin",
        dtype=dtype,
        mode="r",
        shape=shape,
    )


def load_gradient_dataset(root_dir: Path, structured: bool = True) -> Dataset:
    """Load a dataset of gradients from `root_dir`."""

    def load_shard(dir: Path) -> Dataset:
        ds = Dataset.load_from_disk(str(dir / "data.hf"))

        # Add gradients to HF dataset.
        mmap = load_gradients(dir, structured=structured)

        if structured:
            assert mmap.dtype.names is not None
            for field_name in mmap.dtype.names:
                flat = pa.array(mmap[field_name].reshape(-1).copy())
                col = pa.FixedSizeListArray.from_arrays(flat, mmap[field_name].shape[1])
                ds = ds.add_column(field_name, col, new_fingerprint=field_name)
        else:
            flat = pa.array(mmap.reshape(-1).copy())
            col_arrow = pa.FixedSizeListArray.from_arrays(flat, mmap.shape[1])
            ds = ds.add_column("gradients", col_arrow, new_fingerprint="gradients")

        return ds

    if (root_dir / "data.hf").exists():
        return load_shard(root_dir)

    # Flatten indices to avoid CPU OOM
    return concatenate_datasets(
        [load_shard(path) for path in sorted(root_dir.iterdir()) if path.is_dir()]
    ).flatten_indices()


class Scores(np.memmap):
    @overload
    def __getitem__(self, key: str) -> np.ndarray[Any, Any]: ...

    @overload
    def __getitem__(self, key: int | slice) -> Any: ...

    def __getitem__(self, key: Any) -> Any:  # type: ignore
        return super().__getitem__(key)


def load_scores(
    path: Path,
) -> Scores:
    bin_path = path / "scores.bin"
    info_path = path / "info.json"

    with open(info_path, "r") as f:
        info = json.load(f)

    mmap = np.memmap(
        bin_path,
        dtype=info["dtype"],
        mode="r",
        shape=(info["num_items"],),
    )

    return cast(Scores, mmap)


class Builder:
    """Creates and writes gradients to disk, with optional distributed reduction.
    Scores are always saved as float32."""

    num_items: int

    grad_buffer: np.memmap

    reduce_cfg: ReduceConfig | None

    def __init__(
        self,
        path: Path,
        data: Dataset,
        grad_sizes: dict[str, int],
        dtype: torch.dtype,
        reduce_cfg: ReduceConfig | None = None,
    ):
        self.grad_sizes = grad_sizes
        self.num_items = len(data)
        self.reduce_cfg = reduce_cfg
        self.eps = torch.finfo(torch.float32).eps
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if reduce_cfg is not None:
            num_grads = 1
            np_dtype = np.float32
            self.in_memory_grad_buffer = torch.zeros(
                (num_grads, sum(self.grad_sizes.values())),
                dtype=torch.float32,
                device=f"cuda:{self.rank}",
            )
        else:
            num_grads = self.num_items
            np_dtype = convert_dtype_to_np(dtype)
            self.in_memory_grad_buffer = None

        self.grad_buffer = create_index(
            path,
            num_grads=num_grads,
            grad_sizes=self.grad_sizes,
            dtype=np_dtype,
            with_structure=False,
        )

    def reduce(self, indices: list[int], mod_grads: dict[str, torch.Tensor]):
        assert self.reduce_cfg is not None and self.in_memory_grad_buffer is not None
        device = next(iter(mod_grads.values())).device

        if self.reduce_cfg.unit_normalize:
            ssqs = torch.zeros(len(indices), device=device)
            for mod_grad in mod_grads.values():
                ssqs += mod_grad.pow(2).sum(dim=-1)
            norms = ssqs.sqrt()
        else:
            norms = torch.ones(len(indices), device=device)

        offset = 0
        for module_name in self.grad_sizes.keys():
            grads = mod_grads[module_name]
            if self.reduce_cfg.unit_normalize:
                grads = grads / (norms.unsqueeze(1) + self.eps)

            grads = grads.sum(dim=0).to(torch.float32)

            self.in_memory_grad_buffer[0, offset : offset + grads.shape[0]] += grads
            offset += grads.shape[0]

    def __call__(self, indices: list[int], mod_grads: dict[str, torch.Tensor]):
        torch.cuda.synchronize()

        if self.reduce_cfg is not None:
            self.reduce(indices, mod_grads)
        else:
            # It turns out that it's very important for efficiency to write the
            # gradients sequentially instead of first concatenating them, then
            # writing to one vector
            offset = 0
            for module_name in self.grad_sizes.keys():
                self.grad_buffer[
                    indices, offset : offset + mod_grads[module_name].shape[1]
                ] = tensor_to_numpy(mod_grads[module_name])
                offset += mod_grads[module_name].shape[1]

    def flush(self):
        self.grad_buffer.flush()

    def dist_reduce(self):
        if self.reduce_cfg is None:
            return

        assert self.in_memory_grad_buffer is not None

        self.in_memory_grad_buffer = self.in_memory_grad_buffer.cuda()

        if dist.is_initialized():
            dist.reduce(self.in_memory_grad_buffer, dst=0, op=dist.ReduceOp.SUM)

        if self.reduce_cfg.method == "mean":
            self.in_memory_grad_buffer /= self.num_items

        self.in_memory_grad_buffer = self.in_memory_grad_buffer.cpu()

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            self.grad_buffer[:] = tensor_to_numpy(self.in_memory_grad_buffer).astype(
                self.grad_buffer.dtype
            )

        self.in_memory_grad_buffer = self.in_memory_grad_buffer.cpu()


def pad_and_tensor(
    sequences: list[list[int]],
    labels: list[list[int]] | None = None,
    *,
    padding_value: int = 0,
    dtype: torch.dtype | None = torch.long,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of sequences to the same length and convert them to tensors.
    Returns a tuple of padded sequences and labels. The labels are the same as the
    sequences, but with -100 for the padding positions, which is useful for ignoring
    padding in loss calculations.
    """
    if labels is None:
        labels = sequences

    # find max length
    max_len = max(len(seq) for seq in sequences)
    # pad each sequence
    padded = [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]
    labels = [label + [-100] * (max_len - len(label)) for label in labels]

    # convert to tensor
    padded_tokens = torch.tensor(padded, dtype=dtype, device=device)
    padded_labels = torch.tensor(labels, dtype=dtype, device=device)
    return padded_tokens, padded_labels


def tokenize(batch: dict, *, args: DataConfig, tokenizer):
    """Tokenize a batch of data with `tokenizer` according to `args`."""
    kwargs = dict(
        return_attention_mask=False,
        return_length=True,
        truncation=args.truncation,
    )
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
    strings = tokenizer.apply_chat_template(convos, tokenize=False)
    encodings = tokenizer(strings, **kwargs)
    labels_list: list[list[int]] = []

    for i, convo in enumerate(convos):
        # Find the spans of the assistant's responses in the tokenized output
        pos = 0
        spans: list[tuple[int, int]] = []

        for msg in convo:
            if msg["role"] != "assistant":
                continue

            ans = msg["content"]
            start = strings[i].rfind(ans, pos)
            if start < 0:
                raise RuntimeError(
                    "Failed to find completion in the chat-formatted conversation. "
                    "Make sure the chat template does not alter the completion, e.g. "
                    "by removing leading whitespace."
                )

            # move past this match
            pos = start + len(ans)

            start_token = encodings.char_to_token(i, start)
            end_token = encodings.char_to_token(i, pos)
            spans.append((start_token, end_token))

        # Labels are -100 everywhere except where the assistant's response is
        tokens = encodings["input_ids"][i]
        labels = [-100] * len(tokens)
        for start, end in spans:
            if start is not None and end is not None:
                labels[start:end] = tokens[start:end]

        labels_list.append(labels)

    return dict(**encodings, labels=labels_list)


def unflatten(x: torch.Tensor, shapes: dict[str, Sequence[int]], dim: int = -1):
    """Unflatten a tensor `x` into a dictionary of tensors with specified shapes."""
    numels = [math.prod(shape) for shape in shapes.values()]
    return {
        name: x.unflatten(dim, shape)
        for (name, shape), x in zip(shapes.items(), x.split(numels, dim=dim))
    }
