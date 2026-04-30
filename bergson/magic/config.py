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

    - 'all' saves every checkpoint. This method uses O(N) space and O(N) time.
    - 'log' saves at a log-spaced interval, more frequently near the end of a training
      segment. Training is recursively divided into segments. This method uses O(log N)
      space and O(N log N) time.
    - 'sqrt' saves at a linearly-spaced interval, every sqrt(N) steps. This method uses
      O(sqrt N) space and O(N) time.

    The original MAGIC paper used 'log', but 'sqrt' is often a better choice when disk
    space is not a concern.
    """

    subset_strategy: Literal["random", "sorted"] = "sorted"
    """Strategy for selecting leave-k-out subsets for validation."""

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
