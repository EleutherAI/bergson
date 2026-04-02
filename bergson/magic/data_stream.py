import torch
import torch.distributed as dist
from datasets import Dataset

from ..data import pad_and_tensor


class DataStream:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        input_key: str = "text",
        num_docs: int | None = None,
    ):
        self.batch_size = batch_size
        self.dataset = dataset
        self.device = torch.device(device)
        self.input_key = input_key
        self.n = len(dataset)
        self.num_batches = self.n // batch_size

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # If num_docs isn't provided, assume that each sequence contains one document
        num_docs = num_docs or self.n
        self.weights = torch.nn.Parameter(torch.ones(num_docs, device=self.device))

    @property
    def requires_grad(self) -> bool:
        return self.weights.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self.weights.requires_grad = value

    def __getitem__(self, i: int) -> dict:
        if i < 0 or i >= len(self):
            raise IndexError("DataStream index out of range")

        rng = range(
            i * self.batch_size,
            min((i + 1) * self.batch_size, len(self.dataset)),
        )
        indices = list(rng)[self.rank :: self.world_size]

        batch = self.dataset[indices]
        x, y, valid_mask = pad_and_tensor(
            batch["input_ids"],
            labels=batch.get("labels"),
            device=self.device,
        )
        if "doc_ids" in batch:
            indices = torch.tensor(batch["doc_ids"], device=self.device)

        return {
            "input_ids": x,
            "labels": y,
            "example_weight": self.weights[indices],
            "valid_mask": valid_mask,
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.num_batches

    def __reversed__(self):
        for i in reversed(range(len(self))):
            yield self[i]
