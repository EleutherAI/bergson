from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from simple_parsing import field


@dataclass
class DataConfig:
    dataset: str = "NeelNanda/pile-10k"
    """Dataset identifier to build the index from."""

    split: str = "train"
    """Split of the dataset to use for building the index."""

    subset: str | None = None
    """Subset of the dataset to use for building the index."""

    streaming: bool = False
    """Whether to use streaming mode for the dataset."""

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

    truncation: bool = False
    """Whether to truncate long documents to fit the token budget."""


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
class IndexConfig:
    """Config for building the index and running the model/dataset pipeline."""

    run_path: str = field(positional=True)
    """Name of the run. Used to create a directory for run artifacts."""

    data: DataConfig = field(default_factory=DataConfig)
    """Specification of the data on which to build the index."""

    model: str = "EleutherAI/pythia-160m"
    """Name of the model to load."""

    tokenizer: str | None = None
    """Name of the tokenizer to use. If not provided, the model tokenizer
    is used."""

    fsdp: bool = False
    """Whether to use Fully Sharded Data Parallel (FSDP) for collecting gradients."""

    precision: Literal["auto", "bf16", "fp16", "fp32", "int4", "int8"] = "auto"
    """Precision (dtype) to use for the model parameters."""

    projection_dim: int = 16
    """Dimension of the random projection for the index, or 0 to disable it."""

    include_bias: bool = False
    """Whether to include linear layers' bias gradients."""

    reshape_to_square: bool = False
    """Whether to reshape the gradients to a square matrix."""

    projection_type: Literal["normal", "rademacher"] = "rademacher"
    """Type of random projections to use for the gradients."""

    token_batch_size: int = 8192
    """Batch size in tokens for building the index."""

    processor_path: str = ""
    """Path to a precomputed processor."""

    skip_preconditioners: bool = False
    """Whether to skip computing preconditioners for the gradients."""

    skip_index: bool = False
    """Whether to skip building the gradient index."""

    stats_sample_size: int | None = 10_000
    """Number of examples to use for estimating processor statistics."""

    drop_columns: bool = True
    """Only save the new dataset columns. If false, the original dataset
    columns will be saved as well."""

    loss_fn: Literal["ce", "kl"] = "ce"
    """Loss function to use."""

    loss_reduction: Literal["mean", "sum"] = "mean"
    """Reduction method for the loss function."""

    stream_shard_size: int = 400_000
    """Shard size for streaming the dataset into Dataset objects."""

    revision: str | None = None
    """Revision of the model."""

    split_attention_modules: list[str] = field(default_factory=list)
    """Modules to split into head matrices."""

    attention: AttentionConfig = field(default_factory=AttentionConfig)
    """Configuration for each attention module to be split into head matrices.
    Used for attention modules specified in `split_attention_modules`."""

    @property
    def partial_run_path(self) -> Path:
        """Temporary path to use while writing build artifacts."""
        return Path(self.run_path + ".part")


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

    unit_norm: bool = False
    """Whether to unit normalize the query."""

    faiss: bool = False
    """Whether to use FAISS for the query."""


@dataclass
class ScoreConfig:
    """Config for querying an index on the fly."""

    query_path: str = ""
    """Path to the existing query index."""

    score: Literal["mean", "nearest", "individual"] = "mean"
    """Method for scoring the gradients with the query.
        `mean`: compute each gradient's similarity to the mean
            query gradient.
        `nearest`: compute each gradient's similarity to the most
            similar query gradient (the maximum score).
        `individual`: compute a separate score for each query gradient."""

    query_preconditioner_path: str | None = None
    """Path to a precomputed preconditioner to be applied to
    the query dataset gradients."""

    index_preconditioner_path: str | None = None
    """Path to a precomputed preconditioner to be applied to
    the query dataset gradients. This does not affect the
    ability to compute a new preconditioner during the query."""

    mixing_coefficient: float = 0.99
    """Coefficient to weight the application of the query preconditioner
    and the pre-computed index preconditioner. 0.0 means only use the
    index preconditioner and 1.0 means only use the query preconditioner."""

    modules: list[str] = field(default_factory=list)
    """Modules to use for the query. If empty, all modules will be used."""

    unit_normalize: bool = False
    """Whether to unit normalize the gradients before computing the scores."""

    batch_size: int = 1024
    """Batch size for processing the query dataset."""


@dataclass
class ReduceConfig:
    """Config for reducing the gradients."""

    method: Literal["mean", "sum"] = "mean"
    """Method for reducing the gradients."""

    unit_normalize: bool = False
    """Whether to unit normalize the gradients before reducing them."""


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
