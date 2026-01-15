"""
Toy language model for EKFAC testing.

Provides a minimal transformers-compatible model and dataset generation
utilities for testing EkfacComputer without loading real models.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from datasets import Dataset
from torch import Tensor
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class ToyLMConfig(PretrainedConfig):
    """Configuration for ToyLM - a minimal language model for testing."""

    model_type = "toy_lm"

    def __init__(
        self,
        vocab_size: int = 8,
        hidden_size: int = 4,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        super().__init__(**kwargs)


class ToyLMModule(nn.Module):
    """The base model (what hooks attach to)."""

    def __init__(self, config: ToyLMConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden = self.embed(input_ids)  # [B, S] -> [B, S, H]
        return self.linear(hidden)  # [B, S, H] -> [B, S, V]


class ToyLM(PreTrainedModel):
    """Toy language model compatible with EkfacComputer."""

    config_class = ToyLMConfig
    base_model_prefix = "model"

    def __init__(
        self,
        config: ToyLMConfig,
        *,
        training_data=None,
        training_batches: list[list[int]] | None = None,
        device: torch.device | None = None,
        num_steps: int = 5000,
    ):
        super().__init__(config)
        self.model = ToyLMModule(config)

        if training_data is not None and training_batches is not None:
            self._train(training_data, training_batches, device, num_steps)

    def _train(
        self,
        dataset,
        batches: list[list[int]],
        device: torch.device | None,
        num_steps: int,
        lr: float = 0.1,
    ) -> None:
        """Train the model to make logits more peaked (like a real LLM)."""
        import torch.nn.functional as F

        if device is not None:
            nn.Module.to(self, device)

        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        step = 0
        while step < num_steps:
            for batch_indices in batches:
                for idx in batch_indices:
                    input_ids = torch.tensor(
                        dataset[idx]["input_ids"], device=device
                    ).unsqueeze(0)
                    labels = torch.tensor(
                        dataset[idx]["labels"], device=device
                    ).unsqueeze(0)

                    logits = self(input_ids).logits
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        labels[:, 1:].reshape(-1),
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    step += 1
                    if step >= num_steps:
                        return

    @property
    def base_model(self) -> nn.Module:
        return self.model

    def forward(self, input_ids: Tensor, **kwargs) -> CausalLMOutput:
        logits = self.model(input_ids)
        return CausalLMOutput(logits=logits)


@dataclass
class ToyDataConfig:
    """Configuration for toy data generation."""

    vocab_size: int = 8
    hidden_size: int = 4
    seq_lengths: tuple[int, ...] = (2,)
    num_batches: int = 2000

    @property
    def max_seq_len(self) -> int:
        return max(self.seq_lengths)

    @property
    def batch_size(self) -> int:
        return len(self.seq_lengths)


def generate_dataset(config: ToyDataConfig) -> Dataset:
    """Generate a HuggingFace Dataset for use with EkfacComputer."""
    data = {"input_ids": [], "labels": []}

    for _ in range(config.num_batches):
        for seq_len in config.seq_lengths:
            input_ids = torch.randint(0, config.vocab_size, (seq_len,)).tolist()
            data["input_ids"].append(input_ids)
            data["labels"].append(input_ids)

    return Dataset.from_dict(data)


def generate_batches(config: ToyDataConfig) -> list[list[int]]:
    """Generate batch indices for EkfacComputer."""
    batch_size = len(config.seq_lengths)
    return [
        list(range(i * batch_size, (i + 1) * batch_size))
        for i in range(config.num_batches)
    ]
