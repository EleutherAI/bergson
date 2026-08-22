import torch
import torch.distributed as dist
from datasets import Dataset

from ..data import pad_and_tensor


def mask_padded_rows(batch: dict) -> tuple[dict, int]:
    """Remove ``example_weight`` and mask out the rows it zeroes, so all ranks
    keep the same batch width. Returns the batch and its supervised token count.
    """
    weight = batch.pop("example_weight", None)
    if weight is not None:
        live = (weight != 0).reshape(len(weight), -1).any(dim=1)
        batch["labels"] = batch["labels"].masked_fill(~live[:, None], -100)
        if "shift_loss_mask" in batch:
            batch["shift_loss_mask"] = batch["shift_loss_mask"] & live[:, None]
    return batch, int((batch["labels"][:, 1:] != -100).sum())


def pad_dataset_to_batch_size(
    dataset: Dataset,
    batch_size: int,
    num_docs: int,
    label: str,
    global_rank: int,
) -> tuple[Dataset, int, int, int]:
    """Pad dataset to be divisible by batch_size by repeating the last example.

    Returns (padded_dataset, num_docs, pad_count, weight_pad_count).

    `pad_count` is the number of rows appended to the dataset (0 if unchanged).
    `weight_pad_count` is the number of trailing entries of a *1D* per-doc
    weight tensor that should be zeroed to silence the pad rows' training
    contribution.

    - If the dataset has a "doc_ids" column, `.select(total - 1, ...)` copies
      the last doc's doc_ids into every pad row. Zeroing the last `pad_count`
      entries of a weights-indexed-by-doc_id tensor would silence real docs,
      so we instead route pad rows to a fresh synthetic doc id (=num_docs),
      bump num_docs by 1, and set `weight_pad_count = 1`.
    - Otherwise rows are self-identified docs: num_docs becomes the padded
      length and `weight_pad_count = pad_count` zeros the pad rows directly.

    In per-token (2D) mode callers should zero `weights[-pad_count:]` instead
    — `weight_pad_count` applies only to 1D per-doc weights.
    """
    remainder = len(dataset) % batch_size
    if not remainder:
        return dataset, num_docs, 0, 0

    pad_count = batch_size - remainder
    total = len(dataset)
    pad_indices = list(range(total)) + [total - 1] * pad_count
    dataset = dataset.select(pad_indices)

    if "doc_ids" in dataset.column_names:
        synthetic_doc_id = num_docs
        new_doc_ids = [
            row if i < total else [synthetic_doc_id] * len(row)
            for i, row in enumerate(dataset["doc_ids"])
        ]
        dataset = dataset.remove_columns("doc_ids").add_column("doc_ids", new_doc_ids)
        num_docs += 1
        weight_pad_count = 1
    else:
        num_docs = len(dataset)
        weight_pad_count = pad_count

    if global_rank == 0:
        print(
            f"{label}: padded {pad_count}/{total + pad_count} examples "
            f"(weight=0) to fill last batch"
        )
    return dataset, num_docs, pad_count, weight_pad_count


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

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
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
            # Truncate to the max sequence length in the batch to avoid indexing errors
            indices = (indices, slice(None, x.shape[1]))
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
