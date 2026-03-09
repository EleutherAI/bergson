import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from simple_parsing import field


@dataclass
class DataConfig:
    dataset: str = "NeelNanda/pile-10k"
    """Dataset identifier to build the index from."""

    split: str = "train"
    """Split of the dataset to use for building the index."""

    subset: str | None = None
    """Subset of the dataset to use for building the index."""

    prompt_column: str = "text"
    """Column in the dataset that contains the prompts."""

    completion_column: str = ""
    """Optional column in the dataset that contains the completions."""

    conversation_column: str = ""
    """Optional column in the dataset that contains the conversation."""

    reward_column: str = ""
    """Optional column in the dataset that contains the rewards.
    When specified, gradients are calculated using the policy
    gradient loss from Dr. GRPO. https://arxiv.org/abs/2503.20783"""

    skip_nan_rewards: bool = False
    """Whether to skip examples with NaN rewards."""

    truncation: bool = False
    """Whether to truncate long documents to fit the token budget."""

    data_args: str = ""
    """Arguments to pass to the dataset constructor in the format
    arg1=val1,arg2=val2."""


@dataclass
class AttentionConfig:
    """Config for splitting an attention module into head matrices."""

    num_heads: int = 0
    """Number of attention heads."""

    head_size: int = 0
    """Size of each attention head."""

    head_dim: int = 0
    """Axis index for `num_heads` in the weight matrix."""


@dataclass
class DistributedConfig:
    """Configuration for multi-node preconditioner computation."""

    nnode: int = 1
    """The number of nodes to use for preconditioner computation."""

    nproc_per_node: int = field(default_factory=lambda: torch.cuda.device_count())
    """The number of processes per node."""

    node_rank: int | None = None
    """The rank of the current node. If not set Bergson will attempt to infer
    it from environment variables."""

    @property
    def _node_rank(self) -> int:
        """Get the rank of the node from config or environment variables."""
        if self.node_rank is not None:
            return self.node_rank

        if self.nnode == 1:
            return 0

        for var in ("SLURM_NODEID", "GROUP_RANK", "NODE_RANK"):
            if var in os.environ:
                return int(os.environ[var])

        raise ValueError("Node rank not found. Set it with --node_rank.")

    @property
    def world_size(self) -> int:
        """Total number of processes across all nodes."""
        return self.nnode * self.nproc_per_node

    @property
    def start_rank(self) -> int:
        """Starting rank for processes on this node."""
        return self._node_rank * self.nproc_per_node

    @property
    def local_rank(self) -> int:
        """Local rank of the current process."""
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def rank(self) -> int:
        """Rank of the current process."""
        return self.start_rank + self.local_rank


@dataclass
class IndexConfig:
    """Config for building the index and running the model/dataset pipeline."""

    run_path: str = field(positional=True)
    """Name of the run. Used to create a directory for run artifacts."""

    data: DataConfig = field(default_factory=DataConfig)
    """Specification of the data on which to build the index."""

    model: str = "EleutherAI/pythia-160m"
    """Name of the model to load."""

    tokenizer: str = ""
    """Name of the tokenizer to use. If not set the model tokenizer is used."""

    fsdp: bool = False
    """Whether to use Fully Sharded Data Parallel (FSDP) for collecting gradients."""

    precision: Literal["auto", "bf16", "fp16", "fp32", "int4", "int8"] = "fp32"
    """Precision (dtype) to use for the model parameters."""

    projection_dim: int = 16
    """Dimension of the random projection for the index, or 0 to disable it."""

    include_bias: bool = False
    """Whether to include linear layers' bias gradients."""

    reshape_to_square: bool = False
    """Whether to reshape the gradients to a square matrix."""

    projection_type: Literal["normal", "rademacher"] = "rademacher"
    """Type of random projections to use for the gradients."""

    token_batch_size: int = 2048
    """Batch size in tokens for building the index."""

    auto_batch_size: bool = False
    """Whether to automatically determine the optimal token batch size.
    Experimental feature only enabled for `build`."""

    processor_path: str = ""
    """Path to a precomputed processor."""

    normalizer: Literal["adafactor", "adam", "none"] = "none"
    """Type of normalizer to use for the gradients."""

    skip_preconditioners: bool = False
    """Whether to skip estimating preconditioner statistics"""

    skip_index: bool = False
    """Whether to skip building the gradient index."""

    stats_sample_size: int | None = 10_000
    """Number of examples to use for estimating normalizer statistics."""

    drop_columns: bool = True
    """Only save the new dataset columns. If false, the original dataset
    columns will be saved as well."""

    loss_fn: Literal["ce", "kl"] = "ce"
    """Loss function to use."""

    loss_reduction: Literal["mean", "sum"] = "mean"
    """Reduction method for the loss function."""

    label_smoothing: float = 0.0
    """Label smoothing coefficient for cross-entropy loss. When > 0, prevents
    near-zero gradients for high-confidence predictions that can cause numerical
    instability. Recommended value: 0.005-0.01."""

    stream_shard_size: int = 400_000
    """Shard size for streaming the dataset into Dataset objects."""

    revision: str | None = None
    """Revision of the model."""

    split_attention_modules: list[str] = field(default_factory=list)
    """Modules to split into head matrices."""

    attention: AttentionConfig = field(default_factory=AttentionConfig)
    """Configuration for each attention module to be split into head matrices.
    Used for attention modules specified in `split_attention_modules`."""

    profile: bool = False
    """Whether to enable profiling during gradient collection.
    If true, by default the first 4 steps will be profiled."""

    debug: bool = False
    """Whether to enable debug mode with additional logging."""

    filter_modules: str | None = None
    """If provided, a glob pattern to filter out modules from gradient collection.
    For example, "transformer.h.*.mlp.*" will exclude all MLP layers in a
    standard transformer architecture."""

    overwrite: bool = False
    """Whether to overwrite any existing index in the run path."""

    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    """Configuration for multi-node distributed preconditioner computation."""

    max_tokens: int | None = None
    """Max tokens to process. If None, all tokens processed. Dataset only.
    This experimental feature may be removed in the future."""

    attribute_tokens: bool = False
    """Whether to compute per-token gradients instead of per-example.
    Incompatible with reduce mode."""

    modules: list[str] = field(default_factory=list)
    """Modules to use for the query. If empty, all modules will be used."""

    @property
    def partial_run_path(self) -> Path:
        """Temporary path to use while writing build artifacts."""
        return Path(self.run_path + ".part")

    def __post_init__(self):
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)

        if isinstance(self.attention, dict):
            self.attention = AttentionConfig(**self.attention)

        if isinstance(self.distributed, dict):
            self.distributed = DistributedConfig(**self.distributed)


@dataclass
class QueryConfig:
    """Config for querying an existing gradient index."""

    index: str = ""
    """Path to the existing index."""

    model: str = ""
    """Model to use for the query. When not provided the model used to build the
    index is used."""

    text_field: str = "text"
    """Field to use for the query."""

    unit_norm: bool = True
    """Whether to unit normalize the query."""

    device_map_auto: bool = False
    """Load the model onto multiple devices if necessary."""

    faiss: bool = False
    """Whether to use FAISS for the query."""

    top_k: int = 5
    """Number of top (and bottom) results to return per query."""

    record: str = ""
    """Path to a CSV file for recording query results. Each query appends
    its top and bottom results as rows with columns:
    query, direction, result, result_index, score."""


@dataclass
class PreprocessConfig:
    """Config for gradient preprocessing, shared across build, reduce, and score."""

    unit_normalize: bool = False
    """Whether to unit normalize the gradients."""

    preconditioner_path: str | None = None
    """Path to a precomputed preconditioner."""

    aggregation: Literal["mean", "sum", "none"] = "none"
    """Method for aggregating the gradients. In score, only query
    gradients will be aggregated."""

    normalize_aggregated_grad: bool = False
    """Whether to unit normalize the aggregated gradient. This has
    no effect on future relative score rankings but does affect score
    magnitudes."""


@dataclass
class ScoreConfig:
    """Config for querying an index on the fly."""

    query_path: str = ""
    """Path to the existing query index."""

    score: Literal["nearest", "individual"] = "individual"
    """Method for scoring the gradients with the query.
        `nearest`: compute each gradient's similarity to the most
            similar query gradient (the maximum score).
        `individual`: compute a separate score for each query gradient."""

    batch_size: int = 1024
    """Batch size for processing the query dataset."""

    precision: Literal["auto", "bf16", "fp16", "fp32"] = "fp32"
    """Precision (dtype) to convert the query and index gradients to before
    computing the scores. If "auto", the model's gradient dtype is used."""

    modules: list[str] = field(default_factory=list)
    """Modules to use for the query. If empty, all modules will be used."""


@dataclass
class HessianConfig:
    """Config for reducing the gradients."""

    method: Literal["kfac", "tkfac", "shampoo"] = "kfac"
    """Method for approximating the Hessian."""

    ev_correction: bool = False
    """Whether to additionally compute eigenvalue correction."""

    hessian_dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    """Precision (dtype) to use for the Hessian approximation."""

    use_dataset_labels: bool = False
    """Whether to use dataset labels for Hessian (empirical Fisher) approximation.
    If false, the model predictions will be used."""


@dataclass
class FaissConfig:
    """Configuration for FAISS index."""

    index_factory: str = "Flat"
    """
    The [FAISS index factory string](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

    Common FAISS factory strings:
        - "IVF1,SQfp16": exact nearest neighbors with brute force search and fp16.
            Valid for CPU or memmapped indices.
        - "IVF1024,SQfp16": approximate nearest neighbors with 1024 cluster centers
            and fp16. Fast approximate queries are produced at the cost of a slower
            initial index build.
        - "PQ6720": nearest neighbors with vector product quantization to 6720 elements.
            Reduces memory usage at the cost of accuracy.
    """

    mmap_index: bool = False
    """Whether to query the gradients on-disk."""

    max_train_examples: int | None = None
    """The maximum number of examples to train the index on.
        If `None`, all examples will be used."""

    batch_size: int = 1024
    """The batch size for pre-processing gradients."""

    num_shards: int = 1
    """The number of shards to build for an index.
        Using more shards reduces peak RAM usage."""

    nprobe: int = 10
    """The number of FAISS vector clusters to search if using ANN."""


@dataclass
class TrackstarConfig:
    """Config for the trackstar pipeline query dataset."""

    query: DataConfig = field(default_factory=DataConfig)
    """Query dataset specification."""

    target_downweight_components: int = 1000
    """Number of gradient components to downweight via automatic lambda
    selection (§A.1.3 of Chang et al., 2024). The mixing coefficient is
    computed so that the sorted singular-value curves of the query and
    index preconditioners intersect at this component. Typical value is
    ~1000 out of ~65K total components."""

    num_stats_sample_preconditioner: bool = True
    """Whether to use num_stats_sample items or the full dataset to
    compute preconditioners."""
