import torch
import torch.distributed as dist
from datasets import Dataset

from ..data import pad_and_tensor


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


def doc_rows(dataset: Dataset, num_docs: int) -> list[list[int]]:
    """Row indices holding each document's tokens.

    A chunked dataset (``chunk_length > 0``) packs several documents into a row
    and splits documents across rows, so document ``i`` is not row ``i``; its
    per-token ``doc_ids`` column says which is which. Pad rows carry a
    synthetic id past the real documents, and a document that chunking dropped
    (the tail that doesn't fill a chunk) gets no rows.
    """
    if "doc_ids" not in dataset.column_names:
        return [[i] for i in range(num_docs)]

    rows: list[list[int]] = [[] for _ in range(num_docs)]
    for r, ids in enumerate(dataset["doc_ids"]):
        for doc_id in set(ids):
            if doc_id < num_docs:
                rows[doc_id].append(r)
    return rows


class DataStream:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        input_key: str = "text",
        weight_shape: tuple[int, ...] | None = None,
        rows: list[int] | None = None,
        doc_id: int | None = None,
    ):
        """``rows`` restricts the stream to those dataset rows, cycling them to
        fill whole batches; ``doc_id`` restricts the loss to that document's
        tokens, which is how a single document is scored out of rows that pack
        several (see :func:`doc_rows`)."""
        self.batch_size = batch_size
        self.dataset = dataset
        self.device = torch.device(device)
        self.input_key = input_key
        self.n = len(dataset)
        self.doc_id = doc_id

        self.rows = list(range(self.n)) if rows is None else list(rows)
        if rows is not None:
            # Cycle rather than append dead pad rows: a pad row is a real term
            # in the loss wherever the data weights are discarded, and an
            # all-pad batch on some rank would scale that loss down.
            n = len(self.rows) + (-len(self.rows)) % batch_size
            self.rows = [self.rows[i % len(self.rows)] for i in range(n)]
        self.num_batches = len(self.rows) // batch_size

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
        rows = self.rows[i * self.batch_size : (i + 1) * self.batch_size]
        return rows[self.rank :: self.world_size]

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
        doc_ids = batch.get("doc_ids")
        if doc_ids is not None:
            doc_ids = torch.tensor(doc_ids, device=self.device)
            # doc_ids may be longer than the per-batch padded seq_len (unpacked
            # path stores doc_ids at dataset-wide max_len); truncate to match.
            if doc_ids.ndim == 2:
                doc_ids = doc_ids[:, : x.shape[1]]

        # If the weights are 1D, we assume they correspond to documents and look for
        # "doc_ids" in the batch to index them. If they're 2D, they correspond to
        # tokens. A doc_id-restricted stream weights by row, since its rows repeat.
        if self.weights.ndim == 2:
            # Truncate to the max sequence length in the batch to avoid indexing errors
            indices = (indices, slice(None, x.shape[1]))
        elif doc_ids is not None and self.doc_id is None:
            indices = doc_ids

        # Drop the other documents sharing these rows from the loss, so it is
        # this document's own mean cross-entropy. The shift mask is the loss
        # denominator, so it has to follow the labels.
        if self.doc_id is not None and doc_ids is not None and doc_ids.ndim == 2:
            y = y.where(doc_ids == self.doc_id, -100)
            shift_loss_mask = torch.zeros_like(y, dtype=torch.bool)
            shift_loss_mask[:, :-1] = y[:, 1:] != -100

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
