import json
import math
import os
from dataclasses import dataclass
from typing import Literal, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from numpy.typing import DTypeLike
from simple_parsing import field

from .utils import assert_type


@dataclass
class DataConfig:
    dataset: str = "EleutherAI/SmolLM2-135M-10B"
    """Dataset identifier to build the index from."""

    prompt_column: str = "text"
    """Column in the dataset that contains the prompts."""

    completion_column: str = ""
    """Optional column in the dataset that contains the completions."""

    conversation_column: str = ""
    """Optional column in the dataset that contains the conversation."""


@dataclass
class IndexConfig:
    """Config for building the index and running the model/dataset pipeline."""

    run_path: str = field(positional=True)
    """Name of the run. Used to create a directory for the index."""

    data: DataConfig = field(default_factory=DataConfig)
    """Specification of the data on which to build the index."""

    model: str = "HuggingFaceTB/SmolLM2-135M"
    """Name of the model to load."""

    fsdp: bool = False
    """Whether to use Fully Sharded Data Parallel (FSDP) for collecing gradients."""

    precision: Literal["bf16", "fp16", "fp32", "int4", "int8"] = "bf16"
    """Precision to use for the model parameters."""

    projection_dim: int = 16
    """Dimension of the random projection for the index, or 0 to disable it."""

    reshape_to_square: bool = False
    """Whether to reshape the gradients to a square matrix."""

    token_batch_size: int = 8192
    """Batch size in tokens for building the index."""

    normalizer: Literal["adafactor", "adam", "none"] = "adafactor"
    """Type of normalizer to use for the gradients."""

    fisher_fourth_root: bool = False
    """Whether to use the fourth root of the Fisher information matrix."""

    processor_path: str = ""
    """Path to a precomputed processor."""

    skip_preconditioners: bool = False
    """Whether to skip computing preconditioners for the gradients."""

    stats_sample_size: int | None = 10_000
    """Number of examples to use for estimating processor statistics."""

    drop_columns: bool = False
    """Only return the new dataset columns."""

    streaming: bool = False
    """Whether to use streaming mode for the dataset."""

    streaming_chunk_size: int = 100_000
    """Chunk size for streaming the dataset into Dataset objects."""

    revision: str | None = None
    """Revision of the model."""


def ceildiv(a: int, b: int) -> int:
    """Ceiling division of two integers."""
    return -(-a // b)  # Equivalent to math.ceil(a / b) but faster for integers


def allocate_batches(doc_lengths: list[int], N: int) -> list[list[int]]:
    """
    Allocate documents into batches that are then distributed evenly across
    a fixed number of workers.

    Parameters
    ----------
    doc_lengths : Sequence[int]
        Length (in tokens) of each document.  The *i-th* document is referred to
        internally by its index ``i``.
    workers : int
        Number of parallel workers ( 1 ≤ workers ≤ 8).
    N : int
        Hard memory budget per *batch*, expressed as
        ``max(length in batch) * (# docs in batch) ≤ N``.

    Returns
    -------
    list[list[list[int]]]
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
    2.  **Bin-packing strategy**:  We use *first-fit decreasing* (FFD) to obtain
        an initial near-minimal set of batches, then split some of the larger
        batches (never increases cost) until

            * every worker has at least one batch,
            * the total number of batches is a multiple of ``workers``.

        Because each split only lowers the cost of the two resulting batches,
        the constraint in (1) remains satisfied throughout.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if not doc_lengths:
        raise RuntimeError("Empty document list.")
    if max(doc_lengths) > N:  # a single document would overflow any batch
        raise RuntimeError("At least one document is too long for the budget N.")

    # ---------------------------------------------------------------------
    # 1) First-fit decreasing (FFD) bin packing under the cost function
    #    cost(batch) = max_len_in_batch * len(batch)
    # ---------------------------------------------------------------------
    docs_sorted = sorted(enumerate(doc_lengths), key=lambda x: x[1], reverse=True)
    batches: list[list[int]] = []  # holds document *indices*
    batch_meta = []  # (max_len, size) for each batch

    for idx, length in docs_sorted:
        placed = False
        for j, (mx, sz) in enumerate(batch_meta):
            new_mx = max(mx, length)
            new_sz = sz + 1
            if new_mx * new_sz <= N:  # still fits
                batches[j].append(idx)
                batch_meta[j] = (new_mx, new_sz)
                placed = True
                break

        if not placed:  # open a new batch
            batches.append([idx])
            batch_meta.append((length, 1))

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
    while len(batches) < target_batches:
        batch = batches[i % len(batches)]
        if len(batch) == 1:
            i += 1  # try another batch
            continue
        batches.append([batch.pop()])  # split off a singleton
        i += 1

    assert len(batches) == target_batches
    assert all(
        max(doc_lengths[i] for i in batch) * len(batch) <= N for batch in batches
    )

    # ---------------------------------------------------------------------
    # 4) Round-robin assignment to workers
    # ---------------------------------------------------------------------
    allocation: list[list[list[int]]] = [[] for _ in range(world_size)]
    for b_idx, batch in enumerate(batches):
        allocation[b_idx % world_size].append(batch)

    # sanity: equal # of batches per worker
    assert len({len(b) for b in allocation}) == 1
    return allocation[rank]


def create_index(
    root: str, num_grads: int, grad_sizes: dict[str, int], dtype: DTypeLike
) -> np.memmap:
    """Create a memory-mapped file for storing structured gradients, and persist metadata."""
    grad_path = os.path.join(root, "gradients.bin")
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Build a json-serializable structured dtype
    dtype_itemsize = np.dtype(dtype).itemsize
    structured_dtype_itemsize = dtype_itemsize * sum(grad_sizes.values())

    struct_dtype = {
        "names": [name for name in grad_sizes.keys()],
        # "formats": [np.dtype(dtype).name] * len(grad_sizes),
        "formats": [f"({size},){np.dtype(dtype).str}" for size in grad_sizes.values()],
        "itemsize": dtype_itemsize * sum(grad_sizes.values())
    }

    # ── 1. Rank-0 creates file & metadata exactly once ─────────────────────────
    if rank == 0:
        # Ensure the directory exists
        os.makedirs(root, exist_ok=True)

        # Allocate (extends file to right size without writing zeros byte-by-byte)
        nbytes = np.dtype(struct_dtype).itemsize * num_grads # type: ignore
        with open(grad_path, "wb") as f:
            f.truncate(nbytes)

            # Force the directory entry + data to disk *before* other ranks continue
            os.fsync(f.fileno())

        # Persist metadata for future runs
        with open(root + "/info.json", "w") as f:
            json.dump({"num_grads": num_grads, "dtype": struct_dtype}, f, indent=2)

    # ── 2. Everyone blocks until the file is definitely there & sized ─────────────
    if dist.is_initialized():
        dist.barrier()

    return np.memmap(
        grad_path,
        dtype=np.dtype(struct_dtype),  # type: ignore
        mode="r+",
        shape=(num_grads,),
    )


def load_unstructured_gradients(root_dir: str) -> np.memmap:
    """Map the gradients stored in `root_dir` into memory."""
    with open(os.path.join(root_dir, "info.json")) as f:
        info = json.load(f)
        grad_size = info["grad_size"]
        num_grads = info["num_grads"]

    mmap = np.memmap(
        root_dir + "/gradients.bin",
        dtype=np.float16,
        mode="r",
        shape=(num_grads, grad_size),
    )
    return mmap


def load_gradients(root_dir: str) -> np.memmap:
    """Map the structured gradients stored in `root_dir` into memory."""
    with open(os.path.join(root_dir, "info.json")) as f:
        info = json.load(f)

    # Handle legacy format
    if "grad_size" in info:
        return load_unstructured_gradients(root_dir)

    dtype = info["dtype"]
    num_grads = info["num_grads"]

    return np.memmap(
        os.path.join(root_dir, "gradients.bin"),
        dtype=dtype,
        mode="r",
        shape=(num_grads,),
    )


def load_gradient_dataset(root_dir: str, concatenate_gradients: bool = True) -> Dataset:
    """Load a dataset of gradients from `root_dir`."""
    def load_shard(dir: str) -> Dataset:
        mmap = load_gradients(dir)

        arrays = {}
        for field_name in mmap.dtype.names:
            flat = pa.array(mmap[field_name].reshape(-1))
            col = pa.FixedSizeListArray.from_arrays(flat, mmap[field_name].shape[1])
            arrays[field_name] = col
        
        ds = Dataset.load_from_disk(dir + "/data.hf")

        # concatenate the extracted module gradients into a single column
        if concatenate_gradients:
            col = np.concatenate([mmap[field_name] for field_name in mmap.dtype.names], axis=1)
            flat = pa.array(col.reshape(-1))
            col_arrow = pa.FixedSizeListArray.from_arrays(flat, col.shape[1])
            ds = ds.add_column("gradients", col_arrow, new_fingerprint="gradients")       
        # Add a column for each module's gradient vectors
        else:
            for field_name in mmap.dtype.names:
                ds = ds.add_column(field_name, arrays[field_name], new_fingerprint=field_name)
        return ds

    root = Path(root_dir)

    if (root / "data.hf").exists():
        return load_shard(root_dir)

    # Flatten indices to avoid CPU OOM
    return concatenate_datasets(
        [load_shard(str(path)) for path in sorted(root.iterdir()) if path.is_dir()]
    ).flatten_indices()


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
        return tokenizer(batch[args.prompt_column], truncation=True, **kwargs)

    # Make sure we only compute loss on the assistant's responses
    strings = tokenizer.apply_chat_template(convos, tokenize=False)
    encodings = tokenizer(strings, truncation=True, **kwargs)
    labels_list: list[list[int]] = []

    for i, convo in enumerate(convos):
        # Find the spans of the assistant's responses in the tokenized output
        pos = 0
        spans: list[tuple[int, int]] = []

        for msg in convo:
            if msg["role"] != "assistant":
                continue

            ans = msg["content"]
            start = strings[i].find(ans, pos)
            pos = start + len(ans)  # move past this match

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
