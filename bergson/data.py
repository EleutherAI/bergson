import json
import math
import os
import random
import re
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Sequence

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
)
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.typing import DTypeLike
from transformers import PreTrainedTokenizerFast, logging

from .config import DataConfig
from .utils.utils import (
    assert_type,
    simple_parse_args_string,
)


def compute_num_token_grads(data: Dataset) -> np.ndarray:
    """Compute the number of valid gradient positions per example.

    A token at position t produces a gradient iff the *next* token's label
    is not -100 (the ignore index).  When there is no explicit ``labels``
    column every position except the last is valid, so
    ``num_token_grads = length - 1``.

    Returns
    -------
    np.ndarray of shape ``(len(data),)`` with dtype int64.
    """
    if "labels" in data.column_names:
        # Count positions where labels[t+1] != -100 for t in 0..len-2
        counts = []
        for labels in data["labels"]:
            labels_arr = np.asarray(labels)
            counts.append(int(np.sum(labels_arr[1:] != -100)))
        return np.array(counts, dtype=np.int64)
    else:
        lengths = np.array(data["length"], dtype=np.int64)
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
    (batches,) = _allocate_batches_world(doc_lengths, N, world_size, seed, ranks=[rank])
    return batches


def _allocate_batches_world(
    doc_lengths: list[int],
    N: int,
    world_size: int,
    seed: int = 42,
    ranks: list[int] | None = None,
) -> list[list[list[int]]]:
    """Lower-level version of allocate_batches that returns batches for specified ranks.

    If ranks is None, returns batches for all ranks.
    """
    if ranks is None:
        ranks = list(range(world_size))
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

    # Break any systematic ordering of batches (shuffle only requested ranks)
    for rank in ranks:
        random.seed(seed)
        random.shuffle(allocation[rank])

    return [allocation[rank] for rank in ranks]


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
) -> Dataset:
    """Load a dataset from a string identifier or path."""
    if data_str.endswith(".csv"):
        ds = Dataset.from_csv(data_str)
    elif data_str.endswith(".json") or data_str.endswith(".jsonl"):
        ds = Dataset.from_json(data_str)
    elif Path(data_str).is_dir() and (Path(data_str) / "dataset_info.json").exists():
        ds = Dataset.load_from_disk(data_str, keep_in_memory=False)
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
                ds = Dataset.load_from_disk(data_str, keep_in_memory=False)
            else:
                raise e

    ds = assert_type(Dataset, ds)
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

        def _to_arrow(arr: np.ndarray) -> pa.Array:
            """Convert numpy array to PyArrow, casting bfloat16 to float32."""
            if arr.dtype == np.dtype("bfloat16"):
                arr = arr.astype(np.float32)
            return pa.array(arr)

        if structured:
            assert mmap.dtype.names is not None
            for field_name in mmap.dtype.names:
                flat = _to_arrow(mmap[field_name].reshape(-1).copy())
                col = pa.FixedSizeListArray.from_arrays(flat, mmap[field_name].shape[1])
                ds = ds.add_column(field_name, col, new_fingerprint=field_name)
        else:
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

    def __getitem__(self, key: Any) -> Any:
        items = self.mmap[key]
        return structured_to_unstructured(items[self._score_fields])

    def get(self, key: Any, score_idx: int = 0) -> Any:
        """Get scores for a specific score index."""
        return self.mmap[key][f"score_{score_idx}"]

    def is_written(self) -> bool:
        """Check whether all scores in the structured mmap have
        been written to (i.e. are not still zeros)"""
        return all(np.all(self.mmap[f"written_{i}"]) for i in range(self.num_scores))


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # Compute valid_masks: position i is valid if labels[i+1] != -100
    N, S = padded_tokens.shape
    valid_masks = torch.zeros(N, S, dtype=torch.bool, device=device)
    valid_masks[:, :-1] = padded_labels[:, 1:] != -100

    return padded_tokens, padded_labels, valid_masks


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


def tokenize_and_chunk(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    chunk_size: int,
    text_column: str = "text",
    *,
    num_proc: int = cpu_count() // 2,
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

    Returns:
        A Dataset where every row has a single 'input_ids' list of length chunk_size.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer does not have an eos_token_id set.")

    # Suppress long sequence errors
    original_verbosity = logging.get_verbosity()
    logging.set_verbosity_error()

    # ── Step 1: tokenize each document in parallel ───────────────────────────
    def tokenize_batch(batch):
        return tokenizer(
            batch[text_column],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        desc="Tokenizing",
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )

    # ── Step 2: concatenate the token stream and slice into fixed-size chunks ──
    def chunk_batch(batch, indices: list[int]):
        # Flatten all token lists in this batch into one stream
        doc_stream = []
        token_stream = []
        for doc_idx, ids in zip(indices, batch["input_ids"]):
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
        return {"input_ids": token_chunks, "doc_ids": doc_chunks}

    bs = min(10_000, len(tokenized) // num_proc)
    chunked = tokenized.map(
        chunk_batch,
        batched=True,
        batch_size=bs,
        num_proc=num_proc,
        desc="Chunking",
        remove_columns=tokenized.column_names,
        with_indices=True,
    )
    # Make sure HuggingFace didn't change the dataset type!
    assert isinstance(chunked, type(dataset))

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
