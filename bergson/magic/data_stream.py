import torch
import torch.distributed as dist
from datasets import Dataset

from ..data import pad_and_tensor


class DataStream:
    def __init__(
        self,
        dataset: Dataset,
        batches: list[list[int]],
        *,
        device: torch.device | str = "cpu",
        input_key: str = "text",
    ):
        self.dataset = dataset
        self.batches = batches
        self.input_key = input_key
        self.device = torch.device(device)

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        n = sum(len(batch) for batch in batches)
        self.weights = torch.nn.Parameter(torch.ones(n, device=self.device))

    @property
    def requires_grad(self) -> bool:
        return self.weights.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self.weights.requires_grad = value

    def __getitem__(self, i: int) -> dict:
        if i < 0 or i >= len(self.batches):
            raise IndexError("DataStream index out of range")

        indices = self.batches[i]
        batch = self.dataset[indices]
        x, y, _ = pad_and_tensor(
            batch["input_ids"],
            labels=batch.get("labels"),
            device=self.device,
        )
        return {
            "input_ids": x,
            "labels": y,
            "example_weight": self.weights[indices],
        }

    def __iter__(self):
        for i in range(len(self.batches)):
            yield self[i]

    def __len__(self):
        return len(self.batches)

    def __reversed__(self):
        for i in reversed(range(len(self.batches))):
            yield self[i]
