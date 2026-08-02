import math
import os
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from simple_parsing import Serializable, field

from ..hessians.inversion import Inversion


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

    data_kwargs: str = ""
    """Arguments to pass to the dataset constructor in the format
    arg1=val1,arg2=val2."""

    chunk_length: int = 0
    """When positive, concatenate and chunk the documents into fixed-length token
    sequences of this length. Incompatible with truncation and format_template."""

    def __post_init__(self):
        if self.chunk_length > 0:
            if self.truncation:
                raise ValueError("chunk_length and truncation cannot both be True")
            if self.format_template:
                raise ValueError(
                    "chunk_length and format_template cannot both be specified"
                )


@dataclass
class InversionConfig(Serializable):
    inversion: Inversion = "damped_inverse"
    """Eigenvalue function used to invert a Hessian with regularization.
    Setting ``c = damping_factor`` and ``λ`` for an eigenvalue:

    - "damped_inverse" (default): ``1 / (λ + c·mean(λ))`` — uniform Tikhonov
      damping.
    - "factored_tikhonov": damped inverse with the damping split across the
      activation (A) and gradient (G) Kronecker factors via the Martens &
      Grosse trace ratio ``π = sqrt(mean(λ_A) / mean(λ_G))``. Factored EKFAC
      only; falls back to ``damped_inverse`` for the dense autocorrelation
      Hessian, which has no Kronecker structure.
    - "pseudoinverse": ``1/λ`` where ``λ > c·mean(λ)``, else ``0`` (truncated
      Moore-Penrose pseudoinverse).
    - "pseudoinverse_rank": ``1/λ`` where ``λ > clamp(rank_rtol·max(λ),
      min=rank_atol)``, else ``0`` — the ``torch.linalg.matrix_rank`` cutoff
      used by Meta distributed_shampoo's ``PseudoInverseConfig``, applied to
      the joint spectrum. Ignores ``damping_factor``.
    - "pseudoinverse_factored": the same cutoff computed separately per
      Kronecker factor; a grid cell ``λ_G·λ_A`` survives only if both factor
      eigenvalues pass their own factor's threshold — matching
      distributed_shampoo's per-side truncated roots ``L^{†p} G R^{†p}``.
      Factored EKFAC only (falls back to "pseudoinverse_rank" for the dense
      autocorrelation Hessian); incompatible with ``ev_correction``.
    - "tikhonov_filtered": ``λ / (λ² + α²)`` with ``α = c·mean(λ)`` — the
      Tikhonov filter factor / ridge solution ``(H² + α²I)⁻¹H``."""

    damping_factor: float = 0.1
    """Damping / truncation strength, relative to the mean eigenvalue. Ignored
    by the "pseudoinverse_rank" / "pseudoinverse_factored" modes, which use
    ``rank_rtol`` / ``rank_atol`` instead."""

    rank_rtol: float | None = None
    """Relative truncation tolerance for the "pseudoinverse_rank" /
    "pseudoinverse_factored" modes: eigenvalues at or below
    ``rank_rtol·max(λ)`` are zeroed. ``None`` uses the
    ``torch.linalg.matrix_rank`` / Meta distributed_shampoo default,
    ``numel(λ)·machine_eps``. Being relative, it is invariant to the trace
    normalization and sample-count scaling of the stored factors."""

    rank_atol: float = 0.0
    """Absolute floor for the truncation threshold of the rank-based
    pseudoinverse modes. Compared in units of the stored eigenvalues, which
    for factored Hessians are trace-normalized and scaled by
    ``sqrt(total_processed)`` per factor."""


@dataclass
class DistributedConfig(Serializable):
    """Configuration for multi-node computation."""

    nnode: int = 1
    """The number of nodes to use for computation."""

    nproc_per_node: int = field(default_factory=torch.cuda.device_count)
    """The number of processes per node."""

    node_rank: int | None = None
    """The rank of the current node. If not set Bergson will attempt to infer
    it from environment variables."""

    @property
    def _node_rank(self) -> int:
        """Get the rank of the node from config or environment variables."""
        if self.node_rank is not None:
            return self.node_rank

        for var in ("SLURM_NODEID", "GROUP_RANK", "NODE_RANK"):
            if var in os.environ:
                return int(os.environ[var])

        if self.nnode == 1:
            return 0

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

    overwrite: bool = False
    """Whether to overwrite any existing index in the run path."""

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

    peft_init_kwargs: str = ""
    """peft.LoraConfig arguments for initializing a PEFT adapter on the
    base model in the format 'arg1=val1,arg2=val2'.
    Use | to separate list values, e.g. target_modules=q_proj|k_proj|v_proj."""

    model_kwargs: str = ""
    """HF Model kwargs for in the format 'arg1=val1,arg2=val2'."""


@dataclass
class LRScheduleConfig(Serializable):
    """Learning rate schedule configuration."""

    lr: float = 1e-5
    """The peak learning rate."""

    lr_scheduler_type: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = "linear"
    """The learning rate scheduler type."""

    lr_start: float = 0.0
    """Initial learning rate at the beginning of warmup."""

    lr_end: float = 0.0
    """Final learning rate after decay (only available for polynomial)."""

    warmup_steps: float = 0
    """Number of warmup steps before applying base lr.
    A value >= 1 is an exact step count; a value in [0, 1)
    is interpreted as a fraction of total training steps."""

    num_cycles: float = 0.5
    """Number of cosine cycles (used by cosine and cosine_with_restarts).
    Default 0.5 gives a single half-cosine decay."""

    power: float = 1.0
    """Exponent for polynomial decay."""

    def get_schedule(self, num_steps: int):
        """Return a learning rate schedule function: step → lr.

        Supports HF-compatible scheduler types and an optional non-zero warmup
        start (``lr_start``).
        """
        if self.warmup_steps >= 1:
            warmup_steps = int(self.warmup_steps)
        else:
            warmup_steps = math.ceil(num_steps * self.warmup_steps)

        lr = self.lr
        lr_start = self.lr_start
        decay_steps = max(num_steps - warmup_steps, 1)

        def _warmup(step):
            """Linear warmup from lr_start to lr."""
            progress = step / max(warmup_steps, 1)
            return lr_start + (lr - lr_start) * progress

        match self.lr_scheduler_type:
            case "constant":
                return lambda step: lr
            case "constant_with_warmup":
                return lambda step: _warmup(step) if step < warmup_steps else lr
            case "linear":

                def lin_schedule(step):
                    if step < warmup_steps:
                        return _warmup(step)
                    progress = (step - warmup_steps) / decay_steps
                    return lr * (1 - progress)

                return lin_schedule
            case "cosine":

                def cos_schedule(step):
                    if step < warmup_steps:
                        return _warmup(step)
                    progress = (step - warmup_steps) / decay_steps
                    omega = math.pi * self.num_cycles * 2.0 * progress
                    return lr * 0.5 * (1 + math.cos(omega))

                return cos_schedule
            case "cosine_with_restarts":

                def cos_restart_schedule(step):
                    if step < warmup_steps:
                        return _warmup(step)
                    progress = (step - warmup_steps) / decay_steps
                    omega = math.pi * 2.0 * ((self.num_cycles * progress) % 1.0)
                    return lr * 0.5 * (1 + math.cos(omega))

                return cos_restart_schedule
            case "polynomial":

                def poly_schedule(step):
                    if step < warmup_steps:
                        return _warmup(step)
                    progress = (step - warmup_steps) / decay_steps
                    return (
                        self.lr_end + (lr - self.lr_end) * (1 - progress) ** self.power
                    )

                return poly_schedule
            case other:
                raise ValueError(f"Unknown lr_scheduler_type: {other!r}")


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

    use_tf32_matmuls: bool = False
    """Set matmul precision to 'high'."""

    debug: bool = False
    """Whether to enable debug mode with additional logging."""

    def __post_init__(self):
        if self.use_tf32_matmuls:
            torch.set_float32_matmul_precision("high")


@dataclass
class TrainingConfig(AttributionConfig, Serializable):
    """Configuration for the MAGIC trainer."""

    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)
    """Learning rate schedule configuration."""

    batch_size: int = 16
    """Batch size for both training and query streams.
    Adjust based on GPU memory."""

    num_epochs: int = 1
    """Number of full passes over the training data."""

    seed: int = 42
    """Random seed for dataset shuffling."""

    adam_beta1: float = 0.95
    """Beta1 for AdamW optimizer."""

    adam_beta2: float = 0.975
    """Beta2 for AdamW optimizer."""

    eps_root: float = 1e-8
    """Epsilon root for AdamW optimizer.

    Note for TrackStar attribution: Adam normalization with a non-zero
    eps_root is untested. We recommend setting it to zero."""

    adam_eps: float = 1e-8
    """Epsilon (outside the square root) for AdamW. MAGIC
    uses 1e-6."""

    loss_reduction: Literal["mean", "sum_of_means"] = "mean"
    """How the per-token training losses are reduced to a scalar.
    "mean" (default): mean over all valid tokens in the batch (standard).
    "sum_of_means": mean over each sample's tokens,
    then summed over the batch with no batch-size division — sensitive
    MAGIC/metagradients design choice (arXiv 2503.13751 App. D)."""

    optimizer: Literal["adamw", "muon", "sgd"] = "adamw"
    """Optimizer to use for the training steps. Muon is an efficient
    optimizer that can reduce memory usage and speed up training."""

    save_optimizer_state: bool = False
    """After training, export the optimizer's second moments to
    ``<run_path>/optimizer.pt`` for use as an attribution normalizer.
    AdamW only."""

    weight_decay: float = 0.01
    """Weight decay coefficient for AdamW and Muon."""

    max_grad_norm: float | None = None
    """Clip gradients to this global norm before each optimizer step, matching
    HuggingFace Trainer's ``max_grad_norm``. ``None`` or ``0`` disables clipping."""

    grad_checkpointing: bool = False
    """Whether to use gradient checkpointing during the forward pass."""

    resume: bool = False
    """Resume a previously interrupted run from the last checkpoint."""

    wandb_project: str = ""
    """Weights & Biases project name. If set, logs training loss to W&B."""


@dataclass
class MetasmoothnessConfig(TrainingConfig):
    """Config for empirical metasmoothness (arXiv 2503.13751, Def. 2).

    Trains three models with data weights ``1``, ``1 + h*v`` and ``1 + 2h*v``
    (``v ~ N(0, I)`` over docs) and scores the movement-weighted sign
    agreement of consecutive finite-difference derivatives in parameter
    space. Costs three trainings; no retraining bank. The metagradients
    authors select eps_root and lr by maximizing this metric."""

    h: float = 0.1
    """Finite-difference step size for the data-weight perturbation."""

    direction_seed: int = 0
    """Seed for the random perturbation direction ``v``."""


@dataclass
class ValidationConfig(TrainingConfig, ABC):
    """Config for leave-k-out validation of attribution scores."""

    query: DataConfig = field(
        default_factory=lambda: DataConfig(split="train"),
    )
    """Query/eval dataset for computing attribution target gradients.
    If not specified, defaults to the training dataset."""

    query_method: Literal["mean", "sum"] = "mean"
    """Method for reducing query gradients across batches."""

    num_subsets: int = 100
    """Number of leave-k-out subsets for Spearman correlation."""

    subset_strategy: Literal["random"] = "random"
    """Strategy for selecting leave-k-out subsets for validation."""

    exclude_zero_scores: bool = False
    """When True, drop doc_ids with score == 0 from the validation
    permutation. These scores may be produced by items with fewer than
    2 tokens."""

    save_retrained_models: bool = False
    """When True, save each leave-k-out retrained model (HF format, weights +
    tokenizer) to ``<run_path>/retrained/subset_<i>/`` so it can be reused for
    later attribution queries without retraining. ~0.5 GB per subset for GPT-2."""

    subset_fraction: float = 0.0
    """When > 0, each of the ``num_subsets`` leave-k-out subsets is an
    independent draw (without replacement within a subset, overlapping across
    subsets) of ``round(subset_fraction * pool)`` docs from the validation
    pool — e.g. 0.05 drops 5 percent of the data per subset."""


@dataclass
class RecallDataConfig(Serializable):
    """Identity of the cached synthetic-facts datasets used by ``recall``.

    Datasets are generated once per ``(num_people, seed, single_paraphrase)``
    and cached under ``data_dir`` with size-keyed names, e.g.
    ``statements_1000p_seed0.hf`` / ``questions_1000p_seed0.hf``, so runs are
    reproducible and datasets of different sizes coexist."""

    num_people: int = 1000
    """Number of synthetic people to generate facts for. Capped at the size
    of the first-name list (~9333 with the stock ``names/`` files)."""

    seed: int = 0
    """Seed for profile generation. Part of the dataset identity."""

    single_paraphrase: bool = False
    """Keep only one statement per person and field, so each question has
    exactly one gold document. By default every paraphrase template is kept
    and retrieving any of them counts as a hit."""

    data_dir: str = "data"
    """Directory containing the ``names/`` and ``templates/`` source lists,
    and where the generated ``*.hf`` datasets are cached."""


@dataclass
class RecallConfig(Serializable):
    """Config for evaluating attribution scores by synthetic factual recall."""

    run_path: str = field(positional=True)
    """Directory to save results."""

    scores: str = ""
    """Path to a directory written by ``score`` containing ``scores.bin``
    and ``info.json`` with one score column per question."""

    data: RecallDataConfig = field(default_factory=RecallDataConfig)
    """Identity of the synthetic facts datasets that were trained on and
    scored. Statement/question paths are derived from this so they cannot
    drift from what was scored."""

    k: int = 10
    """Cutoff for Recall@k."""

    higher_is_better: bool = True
    """True when a higher score means a stronger proponent of the query
    (e.g. influence functions). False for unrolled differentiation."""


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

    projection_dim: int = 0
    """Dimension of the random projection for the index, or 0 to disable it."""

    include_bias: bool = False
    """Whether to include linear layers' bias gradients."""

    reshape_to_square: bool = False
    """Whether to reshape the gradients to a square matrix."""

    projection_type: Literal["normal", "rademacher"] = "rademacher"
    """Type of random projections to use for the gradients."""

    projection_target: Literal["per_module", "global"] = "per_module"
    """Projection target. ``per_module`` does a double-sided random projection of
    each module gradient. ``global`` projects each module's flattened gradient with
    an independent right-side matrix and sums into one vector per example."""

    token_batch_size: int = 2048
    """Batch size in tokens for building the index."""

    max_batch_size: int | None = None
    """Cap the number of documents per batch."""

    auto_batch_size: bool = False
    """Whether to automatically determine the optimal token batch size.
    Experimental feature only enabled for `build`."""

    optimizer_state: str = ""
    """Source for optimizer second moments used to normalize gradients.
    Either a local path (a checkpoint directory containing ``optimizer.pt``,
    or a path to an optimizer state file directly) or a Hugging Face URI
    ``hf://<repo>[@<revision>][/<path>]``.

    Note: Untested with the AdamW eps_root in the bergson trainer -
    consider setting this to 0 when using optimizer normalization."""

    loss_fn: Literal["ce", "kl"] = "ce"
    """Loss function to use."""

    loss_reduction: Literal["mean", "sum"] = "sum"
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

    filter_modules: str | None = None
    """If provided, a glob pattern to filter out modules from gradient collection.
    For example, "transformer.h.*.mlp.*" will exclude all MLP layers in a
    standard transformer architecture."""

    force_math_sdp: bool = False
    """Disable flash and memory-efficient SDPA backends, forcing the
    math-only kernel. Some models produce inconsistent gradients across
    different padding lengths when using optimized attention backends.
    Run `bergson test_model_configuration` to check whether your model
    needs this."""

    attribute_tokens: bool = False
    """Whether to compute per-token gradients instead of per-example.
    Incompatible with reduce mode."""

    modules: list[str] = field(default_factory=list)
    """Modules to use for the query. If empty, all modules will be used."""

    @property
    def partial_run_path(self) -> Path:
        """Temporary path to use while writing build artifacts."""
        return Path(self.run_path + ".part")


@dataclass
class TrackstarIndexConfig(IndexConfig):
    """IndexConfig for the Trackstar command, which uses random-projection
    compression and so defaults ``projection_dim`` to 16."""

    projection_dim: int = 16


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

    hessian_path: str | None = None
    """Path to a Hessian to precondition the gradients with. Two-sided
    preconditioning (H^(-1/2) applied to query and index) will be used
    if ``unit_norm`` is enabled."""

    ev_correction: bool = False
    """Use the corrected eigenvalues of a factored Hessian at ``hessian_path``
    (requires it to have been fit with ``HessianConfig.ev_correction=True``)."""

    inversion_cfg: InversionConfig = field(default_factory=InversionConfig)
    """How to invert the Hessian at ``hessian_path``."""

    device_map_auto: bool = False
    """Load the model onto multiple devices if necessary."""

    faiss: bool = False
    """Whether to use FAISS for the query."""

    faiss_cfg: FaissConfig = field(default_factory=FaissConfig)
    """FAISS index configuration, used only when ``faiss=True``."""

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

    hessian_path: str | None = None
    """Path to a precomputed gradient processor. Set to apply Hessian approx."""

    inversion_cfg: InversionConfig = field(default_factory=InversionConfig)
    """How to invert the (dense autocorrelation) Hessian at ``hessian_path``."""

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

    higher_is_better: bool = True
    """True when a positive scoring item is a proponent of the query
    capability (e.g. in influence functions). False for unrolled
    differentiation."""


@dataclass
class ApproxUnrollingConfig(Serializable):
    """Config for approximate unrolling of the influence function."""

    checkpoints: list[str] = field(default_factory=list)
    """List of training checkpoint paths (HF model ids or local dirs) to use for
    approximating."""

    model_path: str | None = None
    """Path to the model used for the training checkpoints."""

    segments: int = 3
    """Number of segments to split the training trajectory into for approximation.
    Must divide len(checkpoints)."""

    lr: float = 1e-5
    """Learning rate to use for the approximate unrolling optimization."""

    lr_list: list[float] = field(default_factory=list)
    """Per-segment average LR; length must == segments. If empty,
    inferred from the training run's log_history.json."""

    step_size_list: list[int] = field(default_factory=list)
    """Per-segment optimizer step count; length must == segments. If
    empty, inferred from per-checkpoint step counts."""

    query: DataConfig = field(default_factory=DataConfig)
    """Query dataset spec; gradients computed at the final checkpoint."""

    query_aggregation: Literal["mean", "sum", "none"] = "mean"
    """How to aggregate the query gradients. "none" produces one
    score column per query."""


@dataclass
class HessianConfig(Serializable):
    """Config for reducing the gradients."""

    method: Literal["kfac", "tkfac", "shampoo", "autocorrelation"]
    """Method for approximating the Hessian."""

    apply_power: float = -1.0
    """Inverse power applied to the fitted eigenvalue grid when preconditioning
    the query gradient. -1.0 is the full inverse H^-1."""

    ev_correction: bool = False
    """Whether to additionally compute eigenvalue correction."""

    hessian_dtype: Literal["bf16", "fp16", "fp32"] = "fp32"
    """Precision (dtype) to use for the Hessian approximation."""

    use_dataset_labels: bool = False
    """Whether to use dataset labels for Hessian (empirical Fisher) approximation.
    If false, the model predictions will be used."""


@dataclass
class HessianPipelineConfig:
    """Config for the Hessian-preconditioned influence pipeline."""

    query: DataConfig = field(default_factory=DataConfig)
    """Query dataset specification."""

    query_aggregation: Literal["mean", "sum", "none"] = "mean"
    """How to aggregate the query gradients. "none" produces
    one score column per query."""

    inversion_cfg: InversionConfig = field(default_factory=InversionConfig)
    """How to invert the fitted EKFAC Hessian when applying it to the query."""

    resume: bool = False
    """Skip pipeline steps whose output directory already exists."""


@dataclass
class MixConfig(Serializable):
    """Config for mixing two autocorrelation hessians."""

    query_path: str = ""
    """Directory containing the query autocorrelation hessian
    (a saved GradientProcessor)."""

    index_path: str = ""
    """Directory containing the index autocorrelation hessian
    (a saved GradientProcessor)."""

    output_path: str = ""
    """Directory to write the mixed hessian to."""

    target_downweight_components: int = 1000
    """Number of gradient components to downweight via automatic lambda
    selection (§A.1.3 of Chang et al., 2024). The mixing coefficient is
    computed so that the sorted singular-value curves of the query and
    index hessians intersect at this component. Typical value is
    ~1000 out of ~65K total components."""


@dataclass
class TrackstarConfig:
    """Config for the trackstar pipeline query dataset."""

    query: DataConfig = field(default_factory=DataConfig)
    """Query dataset specification."""

    preprocess_cfg: PreprocessConfig = field(default_factory=PreprocessConfig)

    score_cfg: ScoreConfig = field(default_factory=ScoreConfig)

    target_downweight_components: int = 1000
    """Number of gradient components to downweight via automatic lambda
    selection (§A.1.3 of Chang et al., 2024). The mixing coefficient is
    computed so that the sorted singular-value curves of the query and
    index hessians intersect at this component. Typical value is
    ~1000 out of ~65K total components."""

    stats_sample_size: int | None = 10_000
    """Number of examples to use for estimating the autocorrelation Hessian
    in the trackstar pipeline's hessian-fitting steps. Set to None to use
    the full dataset."""

    resume: bool = False
    """Skip pipeline steps whose output directory already exists."""
