from dataclasses import dataclass

from simple_parsing import Serializable, field

from ..config.config import ValidationConfig


@dataclass
class UnrollConfig(Serializable):
    """Config for unrolled Gauss-Newton attribution, which backpropagates through
    the AdamW training run on the weights of the linear layers with each step's
    loss Hessian replaced by its Gauss-Newton matrix."""

    enabled: bool = False
    """Whether to compute unrolled Gauss-Newton scores instead of MAGIC scores."""

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

    unroll: UnrollConfig = field(default_factory=UnrollConfig)
    """Configuration for unrolled Gauss-Newton attribution."""

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
                raise ValueError("unroll does not support attribute_tokens.")
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
