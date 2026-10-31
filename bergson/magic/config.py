from dataclasses import dataclass
from typing import Literal

from simple_parsing import Serializable, field

from ..config.config import ValidationConfig


@dataclass
class GaussNewtonConfig(Serializable):
    """Settings for ``curvature: gauss_newton``."""

    query_batch_size: int = 8
    """Number of queries each rank scores per pass back through the run."""

    micro_batch_size: int = 4
    """Batch size when computing Gauss-Newton products on each rank."""

    tf32: bool = True
    """Whether to use TF32 matmuls in the Gauss-Newton products."""

    bf16: bool = False
    """Whether to autocast the Gauss-Newton products to bf16."""


@dataclass
class MagicConfig(ValidationConfig):
    """Special config for MAGIC attribution."""

    backward_save_every: int = 0
    """How often (in steps) to save backward state for resume."""

    cleanup_ckpts: bool = True
    """Whether to delete all but the last checkpoint during the backward pass."""

    curvature: Literal["exact", "gauss_newton"] = "exact"
    """Curvature of each training step in the backward pass. ``exact``
    differentiates through the step (MAGIC). ``gauss_newton`` replaces the
    step's loss Hessian with its Gauss-Newton matrix and backpropagates on the
    weights of the linear layers only, replaying the AdamW run from its
    checkpoints (unrolled Gauss-Newton)."""

    gauss_newton: GaussNewtonConfig = field(default_factory=GaussNewtonConfig)

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
        if self.data.span_column:
            raise ValueError(
                "MAGIC does not support data.span_column; attribute_tokens and sum "
                "each span's rows, which gives the same scores."
            )
        if self.per_token:
            self.attribute_tokens = True
        if self.curvature == "gauss_newton":
            if self.fsdp or self.optimizer != "adamw" or self.max_grad_norm:
                raise ValueError(
                    "curvature='gauss_newton' supports AdamW without fsdp or clipping."
                )
            if self.attribute_tokens:
                raise ValueError(
                    "curvature='gauss_newton' does not support attribute_tokens."
                )
        if self.query.aggregation == "none" and self.query.contrast is not None:
            raise ValueError(
                "query.contrast subtracts one aggregated query from another; "
                "use query.aggregation 'mean' or 'sum', not 'none'."
            )
        # Per-query MAGIC needs one document per row.
        if self.query.aggregation == "none" and self.query.chunk_length > 0:
            raise ValueError(
                "query.chunk_length must be 0 for per-query MAGIC "
                "(query.aggregation='none'); use query.truncation for "
                "long documents."
            )
