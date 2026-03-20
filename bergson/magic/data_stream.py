import torch
import torch.distributed as dist
from datasets import Dataset


class DataStream:
    def __init__(
        self,
        dataset: Dataset,
        processor,
        batch_size: int,
        num_batches: int = 0,
        *,
        device: torch.device | str = "cpu",
        input_key: str = "text",
        max_length: int = 256,
    ):
        self.dataset = dataset
        self.processor = processor
        self.input_key = input_key
        self.max_length = max_length

        self.batch_size = batch_size
        self.device = device
        self.num_batches = num_batches or len(self.dataset) // self.batch_size

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        if self.batch_size % self.world_size != 0:
            raise ValueError(
                f"Batch size {self.batch_size} must be divisible by world size "
                f"{self.world_size}"
            )

        needed = self.batch_size * self.num_batches
        assert len(self.dataset) >= needed, (
            f"Dataset has {len(self.dataset)} examples but {self.num_batches} "
            f"batches of size {self.batch_size} require {needed}. "
            f"Pass a larger split or reduce --num_batches."
        )

        n = self.batch_size * self.num_batches
        self.weights = torch.nn.Parameter(torch.ones(n, device=device))

    @property
    def requires_grad(self) -> bool:
        return self.weights.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self.weights.requires_grad = value

    def __getitem__(self, i: int) -> dict:
        if i < 0 or i >= self.num_batches:
            raise IndexError("DataStream index out of range")

        # Fetch only this rank's interleaved slice of the batch directly from the
        # dataset, avoiding tokenizing examples that will just be discarded.
        indices = list(
            range(
                i * self.batch_size + self.rank,
                (i + 1) * self.batch_size,
                self.world_size,
            )
        )
        raw = self.dataset[indices]

        # padding="max_length" ensures uniform shape across ranks without needing
        # to process all ranks' examples together.
        x = self.processor(
            raw[self.input_key],
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        x["labels"] = x["input_ids"]
        x["example_weight"] = self.weights[
            i * self.batch_size
            + self.rank : (i + 1) * self.batch_size : self.world_size
        ]
        return {k: v.to(self.device) for k, v in x.items()}

    def __iter__(self):
        for i in range(self.num_batches):
            yield self[i]

    def __len__(self):
        return self.num_batches

    def __reversed__(self):
        for i in reversed(range(self.num_batches)):
            yield self[i]
