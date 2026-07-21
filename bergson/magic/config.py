from dataclasses import dataclass
from typing import Literal

from ..config.config import ValidationConfig

MagicSaveMode = Literal["all", "sqrt", "log", "interval"]


@dataclass
class MagicConfig(ValidationConfig):
    """Special config for MAGIC attribution."""

    save_mode: MagicSaveMode = "sqrt"
    """Checkpoint saving mode.

    - 'all' saves every checkpoint. This method uses O(N) space and O(N) time.
    - 'log' saves at a log-spaced interval, more frequently near the end of a training
      segment. Training is recursively divided into segments. This method uses O(log N)
      space and O(N log N) time.
    - 'sqrt' saves at a linearly-spaced interval, every sqrt(N) steps. This method uses
      O(sqrt N) space and O(N) time.

    - 'interval' saves every `save_interval` steps (plus the final state when the
      cadence lands on it). Use for SOURCE-style runs that need a few evenly
      spaced checkpoints rather than backward-replay coverage.

    The original MAGIC paper used 'log', but 'sqrt' is often a better choice when disk
    space is not a concern.
    """

    save_interval: int = 0
    """Snapshot spacing in steps for `save_mode: interval`."""

    backward_save_every: int = 0
    """How often (in steps) to save backward state for resume."""

    cleanup_ckpts: bool = True
    """Whether to delete all but the last checkpoint during the backward pass."""

    per_token: bool = False
    """Whether to compute attribution scores per token (instead of per sequence)."""

    skip_validation: bool = False
    """Stop after computing and saving attribution scores, before the
    leave-k-out retraining loop. Useful for score-only MAGIC runs."""
