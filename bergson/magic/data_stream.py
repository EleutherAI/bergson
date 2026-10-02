from dataclasses import dataclass

import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from torch import Tensor

from ..data import pad_and_tensor


def mask_padded_rows(batch: dict) -> tuple[dict, int]:
    """Remove ``example_weight`` and mask out the rows it zeroes, so all ranks
    keep the same batch size. Returns the batch and its supervised token count.
    """
    weight = batch.pop("example_weight", None)
    if weight is not None:
        live = (weight != 0).reshape(len(weight), -1).any(dim=1)
        batch["labels"] = batch["labels"].masked_fill(~live[:, None], -100)
        if "shift_loss_mask" in batch:
            batch["shift_loss_mask"] = batch["shift_loss_mask"] & live[:, None]
    return batch, int((batch["labels"][:, 1:] != -100).sum())


@dataclass(frozen=True)
class Padding:
    """How much of the padded dataset is padding, and how to keep padding from
    affecting influence scores.

    A run weights either documents, as a vector with one entry per document,
    or tokens, as a grid with one row per example. ``num_docs`` counts the
    vector entries that are pure padding and ``num_examples`` the grid rows.
    Each is one if the pad rows share a synthetic id, ``doc_ids`` for the
    first and ``example_ids`` for the second, and ``num_rows`` if they do not;
    the two are decided separately. Pass the weights to :meth:`zero_weights`
    and the scores to :meth:`trim`, which use whichever count matches.
    """

    num_rows: int = 0
    num_docs: int = 0
    num_examples: int = 0

    def __bool__(self) -> bool:
        return bool(self.num_rows)

    def _trailing_pad_len(self, t: Tensor) -> int:
        return self.num_docs if t.ndim == 1 else self.num_examples

    def zero_weights(self, weights: Tensor) -> None:
        """Zero the pad rows' weights in place."""
        if n := self._trailing_pad_len(weights):
            weights.data[-n:] = 0.0

    def trim(self, scores: Tensor) -> Tensor:
        """Return ``scores`` without the entries the pad rows produced."""
        n = self._trailing_pad_len(scores)
        return scores[:-n] if n else scores


def pad_dataset_to_batch_size(
    dataset: Dataset,
    batch_size: int,
    num_docs: int,
    label: str,
    global_rank: int,
) -> tuple[Dataset, int, Padding]:
    """Pad dataset to be divisible by batch_size by repeating the last example.

    Repeating a row copies its ``doc_ids``, which would add the pad rows'
    scores to the last document; they are given a separate document id
    instead, so one zeroed weight entry covers all of them. An ``example_ids``
    column, which per-token weights index by, is re-pointed the same way.
    """
    remainder = len(dataset) % batch_size
    if not remainder:
        return dataset, num_docs, Padding()

    pad_count = batch_size - remainder
    total = len(dataset)
    last = dataset[total - 1]

    if "example_ids" in dataset.column_names:
        last = {**last, "example_ids": max(dataset["example_ids"]) + 1}
        num_examples = 1
    else:
        num_examples = pad_count

    if "doc_ids" in dataset.column_names:
        last = {**last, "doc_ids": [num_docs] * len(last["doc_ids"])}
        num_docs += 1
        padding = Padding(pad_count, num_docs=1, num_examples=num_examples)
    else:
        num_docs = total + pad_count
        padding = Padding(pad_count, num_docs=pad_count, num_examples=num_examples)

    # Built in memory: the pad rows are a handful of copies of the last row.
    pad_rows = Dataset.from_dict(
        {k: [v] * pad_count for k, v in last.items()}, features=dataset.features
    )
    dataset = concatenate_datasets([dataset, pad_rows])
    if global_rank == 0:
        print(
            f"{label}: padded {pad_count}/{total + pad_count} examples "
            f"(weight=0) to fill last batch"
        )
    return dataset, num_docs, padding


class DataStream:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        input_key: str = "text",
        weight_shape: tuple[int, ...] | None = None,
    ):
        self.batch_size = batch_size
        self.dataset = dataset
        self.device = torch.device(device)
        self.input_key = input_key
        self.n = len(dataset)
        self.num_batches = self.n // batch_size

        # If a shape isn't provided, assume that each sequence contains one document
        if weight_shape is None:
            weight_shape = (self.n,)

        # Rows are split across data-parallel peers only. Under tensor
        # parallelism the ranks inside a tp group jointly hold one model and
        # must therefore process the SAME rows; splitting by global rank would
        # give each of them different data and silently compute a different
        # model's gradient. data_parallel_group() is None until a tp mesh is
        # configured, and None is torch's own spelling of the default group, so
        # this is the previous behaviour verbatim in the flat case.
        from .tensor_parallel import data_parallel_group

        group = data_parallel_group()
        self.rank = dist.get_rank(group=group) if dist.is_initialized() else 0
        self.world_size = (
            dist.get_world_size(group=group) if dist.is_initialized() else 1
        )
        self.weights = torch.nn.Parameter(torch.ones(*weight_shape, device=device))

    @property
    def requires_grad(self) -> bool:
        return self.weights.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self.weights.requires_grad = value

    def batch_rows(self, i: int) -> list[int]:
        """The current rank's dataset row indices for batch ``i``."""
        rng = range(
            i * self.batch_size,
            min((i + 1) * self.batch_size, len(self.dataset)),
        )
        return list(rng)[self.rank :: self.world_size]

    def __getitem__(self, i: int) -> dict:
        if i < 0 or i >= len(self):
            raise IndexError("DataStream index out of range")

        indices = self.batch_rows(i)
        batch = self.dataset[indices]
        x, y, shift_loss_mask, _ = pad_and_tensor(
            batch["input_ids"],
            labels=batch.get("labels"),
            device=self.device,
        )
        # If the weights are 1D, we assume they correspond to documents and look for
        # "doc_ids" in the batch to index them. If they're 2D, they correspond to tokens
        if self.weights.ndim == 2:
            # One weight row per example: a shuffled multi-epoch stream reaches
            # its rows through "example_ids". Truncate to the max sequence
            # length in the batch to avoid indexing errors.
            rows = (
                torch.tensor(batch["example_ids"], device=self.device)
                if "example_ids" in batch
                else indices
            )
            indices = (rows, slice(None, x.shape[1]))
        elif "doc_ids" in batch:
            indices = torch.tensor(batch["doc_ids"], device=self.device)
            # doc_ids may be longer than the per-batch padded seq_len (unpacked
            # path stores doc_ids at dataset-wide max_len); truncate to match.
            if indices.ndim == 2:
                indices = indices[:, : x.shape[1]]

        return {
            "input_ids": x,
            "labels": y,
            "example_weight": self.weights[indices],
            "shift_loss_mask": shift_loss_mask,
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.num_batches

    def __reversed__(self):
        for i in reversed(range(len(self))):
            yield self[i]
