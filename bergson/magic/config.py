from dataclasses import dataclass
from typing import Literal

from simple_parsing import Serializable, field

from ..config.config import ValidationConfig


@dataclass
class UnrollConfig(Serializable):
    """Unrolled Gauss-Newton attribution: the AdamW training run differentiated
    backward with each step's loss Hessian replaced by its Gauss-Newton matrix
    on the batch, keeping only the weights of the linear layers."""

    enabled: bool = False

    query_batch_size: int = 8
    """Queries carried backward together; each group replays the trajectory once."""

    micro_batch_size: int = 4
    """Documents per Gauss-Newton forward on each rank."""

    tf32: bool = True
    """TF32 matmuls in the Gauss-Newton products."""

    bf16: bool = False
    """Autocast the curvature products to bf16."""

    curvature: Literal["gauss_newton", "gradient_outer"] = "gauss_newton"
    """Each step's curvature: the Gauss-Newton matrix of the batch loss or the
    mean outer product of the documents' loss gradients."""

    parameters: Literal["linear", "all"] = "linear"
    """Parameters carried through the backward: the linear-layer weights, or
    every trainable parameter."""


@dataclass
class MagicConfig(ValidationConfig):
    """Special config for MAGIC attribution."""

    backward_save_every: int = 0
    """How often (in steps) to save backward state for resume."""

    cleanup_ckpts: bool = True
    """Whether to delete all but the last checkpoint during the backward pass."""

    unroll: UnrollConfig = field(default_factory=UnrollConfig)

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
        if self.unroll.enabled:
            if self.fsdp or self.optimizer != "adamw" or self.max_grad_norm:
                raise ValueError("unroll supports AdamW without fsdp or clipping.")
            if self.attribute_tokens:
                raise ValueError("unroll scores whole documents.")
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
