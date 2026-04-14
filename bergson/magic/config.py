from dataclasses import dataclass, field
from typing import Literal

from ..config import AttributionConfig, DataConfig, TrainingConfig

MagicQueryMethod = Literal["mean", "sum"]
MagicSaveMode = Literal["all", "sqrt", "log"]


@dataclass
class MagicConfig(AttributionConfig, TrainingConfig):
    """Special config for MAGIC attribution."""

    query: DataConfig = field(
        default_factory=lambda: DataConfig(split="train"),
    )
    """Query/eval dataset for computing attribution target gradients.
    If not specified, defaults to the training dataset."""

    query_method: MagicQueryMethod = "mean"
    """Method for reducing query gradients across batches."""

    save_mode: MagicSaveMode = "sqrt"
    """Checkpoint saving mode.

    'all' saves every checkpoint.
    'log' saves at a log-spaced interval.
    'sqrt' saves at a linearly-spaced interval, every sqrt(N) steps.
    """

    subset_jitter_std: float = 0.0
    """Standard deviation of Gaussian noise added to scores for subset selection."""

    num_subsets: int = 100
    """Number of leave-k-out subsets for Spearman correlation."""

    seed: int = 42
    """Random seed for subset permutation."""

    wandb_project: str = ""
    """Weights & Biases project name. If set, logs training loss to W&B."""

    resume: bool = False
    """Resume a previously interrupted run from the last checkpoint."""

    backward_save_every: int = 0
    """How often (in steps) to save backward state for resume."""

    per_token: bool = False
    """Whether to compute attribution scores per token (instead of per sequence)."""

    def __post_init__(self):
        assert not self.fsdp, "PyTorch FSDP is not currently supported for MAGIC."
