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

    # TODO(Lucia Quirke, December 2026): remove per_token backward compatibility.
    per_token: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.per_token:
            self.attribute_tokens = True
        # Per-query MAGIC needs one document per row.
        if self.query_method == "none" and self.query.chunk_length > 0:
            raise ValueError(
                "query.chunk_length must be 0 for per-query MAGIC "
                "(query_method='none'); use query.truncation for long documents."
            )
