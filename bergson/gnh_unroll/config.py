from dataclasses import dataclass, field

from ..config.config import QuerySetConfig, TrainingConfig


@dataclass
class GnhUnrollConfig(TrainingConfig):
    """Config for unrolled Gauss-Newton attribution."""

    query: QuerySetConfig = field(default_factory=QuerySetConfig)
    """Query dataset for the attribution target and how its gradients are
    combined; ``none`` scores each query separately."""

    ckpt_avg_k: int = 1
    """Average the query gradient over the last ``k`` saved checkpoints."""

    checkpoints: str = ""
    """Checkpoint directory of the AdamW run to attribute, as ``bergson train``
    and ``bergson magic`` write them. The training options must match that
    run. Empty trains the run under ``<run_path>/checkpoints`` first."""

    query_batch_size: int = 8
    """Number of queries each rank scores per pass back through the run."""

    double_backward_batch_size: int | None = None
    """Max sequences per Gauss-Newton product. Each product holds a forward
    graph over its sequences, so this bounds the backward's peak memory.
    Exact under any split."""

    def __post_init__(self):
        super().__post_init__()
        if self.fsdp or self.optimizer != "adamw" or self.max_grad_norm:
            raise ValueError(
                "gnh-unroll supports AdamW without fsdp or gradient clipping."
            )
        if self.data.span_column:
            raise ValueError("gnh-unroll does not support data.span_column.")
        if self.query.aggregation == "none" and self.query.contrast is not None:
            raise ValueError(
                "query.contrast subtracts one aggregated query from another; "
                "use query.aggregation 'mean' or 'sum', not 'none'."
            )
        # Per-query scoring needs one document per row.
        if self.query.aggregation == "none" and self.query.chunk_length > 0:
            raise ValueError(
                "query.chunk_length must be 0 for per-query scores "
                "(query.aggregation='none'); use query.truncation for "
                "long documents."
            )
