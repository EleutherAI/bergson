from dataclasses import dataclass

from ..config.config import ValidationConfig


@dataclass
class MagicConfig(ValidationConfig):
    """Special config for MAGIC attribution."""

    backward_save_every: int = 0
    """How often (in steps) to save backward state for resume."""

    cleanup_ckpts: bool = True
    """Whether to delete all but the last checkpoint during the backward pass."""

    per_token: bool = False
    """Whether to compute attribution scores per token (instead of per sequence)."""

    skip_validation: bool = False
    """Stop after computing and saving attribution scores, before the
    leave-k-out retraining loop. Useful for score-only MAGIC runs."""
