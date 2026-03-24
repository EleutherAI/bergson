import os
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from simple_parsing import Serializable, field


@dataclass
class DataConfig(Serializable):
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

    format_template: str = ""
    """Path to a YAML containing a Jinja2 template specifying how to
    format dataset rows into text. The YAML must contain `doc_to_text`
    and optionally `doc_to_target` and `doc_to_choice`. MCQA YAML
    available at `bergson/templates/mcqa.yaml`."""

    data_args: str = ""
    """Arguments to pass to the dataset constructor in the format
    arg1=val1,arg2=val2."""

    chunk_length: int = 0
    """When non-zero, concatenate and chunk the documents into fixed-length token
    sequences of length `chunk_length`."""


@dataclass
class DistributedConfig(Serializable):
    """Configuration for multi-node computation."""

    nnode: int = 1
    """The number of nodes to use for computation."""

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
class ModelConfig(ABC):
    """Base config for model loading."""

    run_path: str = field(positional=True)
    """Directory to save results."""

    model: str = "EleutherAI/pythia-160m"
    """Name of the model to load."""

    precision: Literal["auto", "bf16", "fp16", "fp32", "int4", "int8"] = "fp32"
    """Precision (dtype) to use for the model parameters."""

    revision: str | None = None
    """Revision of the model."""

    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    """Configuration for multi-node distributed computation."""

    fsdp: bool = False
    """Whether to use PyTorch Fully Sharded Data Parallel (FSDP)"""


@dataclass
class TrainingConfig(ModelConfig, Serializable):
    """Configuration for the MAGIC trainer."""

    lr: float = 1e-5
    """Base learning rate after warmup."""

    warmup_steps: int = 10
    """Number of warmup steps before applying base lr."""

    batch_size: int = 16
    """Batch size for both training and query streams. Adjust based on GPU memory."""

    beta1: float = 0.95
    """Beta1 for AdamW optimizer."""

    beta2: float = 0.975
    """Beta2 for AdamW optimizer."""

    eps_root: float = 1e-8
    """Epsilon root for AdamW optimizer. Use 1e-2 for better stability
    with small models."""

    grad_checkpointing: bool = False
    """Whether to use gradient checkpointing during the forward pass."""


@dataclass
class AttributionConfig(ModelConfig, ABC):
    """Base config for attribution methods."""

    data: DataConfig = field(default_factory=DataConfig)
    """Specification of the data on which to build the index."""

    tokenizer: str = ""
    """Name of the tokenizer to use. If not set the model tokenizer is used."""

    drop_columns: bool = True
    """Only save the new dataset columns. If false, the original dataset
    columns will be saved as well."""

    max_tokens: int | None = None
    """Max tokens to process. If None, all tokens processed. Dataset only.
    This experimental feature may be removed in the future."""

    overwrite: bool = False
    """Whether to overwrite any existing index in the run path."""


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
class IndexConfig(AttributionConfig, Serializable):
    """Config for building the index and running the model/dataset pipeline."""

    use_tf32: bool = False
    """Enable TF32 matmuls. Recommended for large FP32 runs."""

    set_float32_matmul_precision_high: bool = False
    """Set matmul precision to 'high'."""

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

    normalizer: Literal["none"] = "none"  # "adafactor", "adam",
    """Type of normalizer to use for the gradients. We are disabling
    optimizers due to lack of empirical validation - contact Eleuther
    if you'd like to use them."""

    skip_preconditioners: bool = False
    """Whether to skip estimating preconditioner statistics"""

    skip_index: bool = False
    """Whether to skip building the gradient index."""

    stats_sample_size: int | None = 10_000
    """Number of examples to use for estimating normalizer statistics."""

    loss_fn: Literal["ce", "kl"] = "ce"
    """Loss function to use."""

    loss_reduction: Literal["mean", "sum"] = "mean"
    """Reduction method for the loss function."""

    label_smoothing: float = 0.0
    """Label smoothing coefficient for cross-entropy loss. When > 0, prevents
    near-zero gradients for high-confidence predictions that can cause numerical
    instability."""

    stream_shard_size: int = 400_000
    """Shard size for streaming the dataset into Dataset objects."""

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

        if self.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.set_float32_matmul_precision_high:
            torch.set_float32_matmul_precision("high")


@dataclass
class QueryConfig(Serializable):
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
class PreprocessConfig(Serializable):
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
class ScoreConfig(Serializable):
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
class HessianConfig(Serializable):
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
class HessianPipelineConfig:
    """Config for the Hessian-preconditioned influence pipeline."""

    query: DataConfig = field(default_factory=DataConfig)
    """Query dataset specification."""

    lambda_damp_factor: float = 0.1
    """Damping factor for EKFAC eigenvalue correction."""

    resume: bool = False
    """Skip pipeline steps whose output directory already exists."""


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

    resume: bool = False
    """Skip pipeline steps whose output directory already exists."""
