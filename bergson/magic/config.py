from dataclasses import dataclass

from ..config.config import ValidationConfig


@dataclass
class MagicConfig(ValidationConfig):
    """Special config for MAGIC attribution."""

    backward_save_every: int = 0
    """How often (in steps) to save backward state for resume."""

    cleanup_ckpts: bool = True
    """Whether to delete all but the last checkpoint during the backward pass."""

    attribute_tokens: bool = False
    """Whether to compute attribution scores per token (instead of per sequence);
    the same toggle as ``IndexConfig.attribute_tokens``."""

    skip_validation: bool = True
    """Set to False to run a leave-k-out retraining validation loop in the
    same job."""

    use_modula: bool = False
    """Normalize weight updates in the modular norm (modula package).
    Improves metasmoothness of the training trajectory for better attribution."""

    # TODO(Lucia Quirke, December 2026): remove per_token backward compatibility.
    per_token: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.per_token:
            self.attribute_tokens = True
        if self.query.aggregation == "none" and self.query.contrast is not None:
            raise ValueError(
                "query.contrast subtracts one aggregated query from another; "
                "use query.aggregation 'mean' or 'sum', not 'none'."
            )
        # Per-query MAGIC needs one document per row.
        if self.query.aggregation == "none" and self.query.data.chunk_length > 0:
            raise ValueError(
                "query.data.chunk_length must be 0 for per-query MAGIC "
                "(query.aggregation='none'); use query.data.truncation for "
                "long documents."
            )
