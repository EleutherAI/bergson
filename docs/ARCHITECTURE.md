# Claude-Generated Bergson Architecture Overview

This document provides a comprehensive overview of the Bergson architecture, including code structure, key components, design patterns, and data flow.

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Code Structure](#code-structure)
3. [Core Abstractions](#core-abstractions)
4. [Data Flow](#data-flow)
5. [Gradient Collection](#gradient-collection)
6. [Indexing and Querying](#indexing-and-querying)
7. [FAISS Integration](#faiss-integration)
8. [Distributed Training](#distributed-training)
9. [Design Patterns](#design-patterns)
10. [File Formats](#file-formats)

---

## High-Level Overview

Bergson is a library for **gradient-based data attribution** using the TrackStar algorithm. It enables tracing which training examples most influenced a model's predictions through efficient per-sample gradient computation and similarity search.

### Key Capabilities

- **Efficient Gradient Collection**: Per-sample gradients computed via PyTorch hooks
- **Memory-Efficient Compression**: Random projection and factored normalization
- **Scalable Indexing**: Memory-mapped storage and FAISS approximate search
- **Distributed Training**: Multi-GPU/multi-node support with FSDP
- **Flexible Querying**: Interactive CLI and programmatic API

### Core Workflow

```
Training Data → Gradient Collection → Index Building → Query → Attribution
                      ↓                      ↓            ↑
              Compression/Normalization   Storage    FAISS/Search
```

---

## Code Structure

The codebase is organized into focused modules with clear responsibilities:

```
bergson/
├── __init__.py              # Public API exports
├── __main__.py              # CLI entry point with command routing
├── config.py                # Dataclass configurations for all components
│
├── build.py                 # Index building orchestration
├── collection.py            # High-level gradient collection API
├── reduce.py                # Gradient reduction (mean/sum)
│
├── data.py                  # Data loading, batching, and storage utilities
├── gradients.py             # GradientProcessor and normalizer abstractions
├── distributed.py           # Multi-GPU/multi-node orchestration
├── process_preconditioners.py  # Preconditioner computation and eigendecomposition
│
├── collector/               # Hook-based gradient collection
│   ├── collector.py         # Base classes for hook collectors
│   ├── gradient_collectors.py  # GradientCollector, TraceCollector
│   └── in_memory_collector.py  # In-memory gradient collection
│
├── query/                   # Index querying and attribution
│   ├── attributor.py        # Main attribution interface
│   ├── faiss_index.py       # FAISS integration for ANN search
│   └── query_index.py       # Interactive query CLI
│
├── score/                   # On-the-fly scoring
│   ├── score.py             # Score dataset against query index
│   ├── scorer.py            # Scorer class for computing similarities
│   └── score_writer.py      # Memory-mapped score storage
│
├── normalizer/              # Gradient normalization
│   └── fit_normalizers.py   # Estimate Adam/Adafactor normalizers
│
├── utils/                   # Utility functions
│   ├── worker_utils.py      # Model/data setup for distributed workers
│   ├── logger.py            # Logging utilities
│   ├── peft.py              # PEFT adapter detection
│   ├── math.py              # Mathematical utilities
│   └── auto_batch_size.py   # Automatic batch size tuning
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| **collector/** | How to collect gradients (hooks, computation) |
| **query/** | How to search gradients (attribution, FAISS) |
| **score/** | How to compute similarities on-the-fly |
| **data.py** | Where to store gradients (memory-mapped files) |
| **gradients.py** | How to transform gradients (normalization, compression) |
| **distributed.py** | How to coordinate workers (multi-GPU/node) |

---

## Core Abstractions

### GradientProcessor

**Location**: `gradients.py`

The central configuration object for gradient processing.

```python
@dataclass
class GradientProcessor:
    normalizers: Dict[str, Normalizer]  # Per-module normalizers
    preconditioners: Dict[str, Tensor]  # Preconditioner matrices
    preconditioners_eigen: Dict[str, Tuple[Tensor, Tensor]]  # Eigendecompositions
    projection_dim: int  # Target dimension for compression
    projection_type: str  # "normal" or "rademacher"
    include_bias: bool  # Whether to include bias gradients
```

**Responsibilities**:
- Stores normalization state (Adam/Adafactor)
- Stores preconditioners (Hessian approximations)
- Configures compression (projection dimension)
- Provides serialization (`save()` / `load()`)

### Normalizer

**Location**: `gradients.py`

Abstract base class for gradient normalization strategies.

```python
class Normalizer(ABC):
    @abstractmethod
    def normalize_(self, grad: Tensor) -> Tensor:
        """Normalize gradient in-place"""
```

**Implementations**:

1. **AdafactorNormalizer**: Factored second moments
   - Memory: O(O + I) for layer with shape [O, I]
   - Stores `row` (size O) and `col` (size I) statistics
   - Approximates full second moment matrix

2. **AdamNormalizer**: Full second moment
   - Memory: O(O × I) for layer with shape [O, I]
   - Stores complete second moment matrix
   - More accurate but less scalable

**Usage**:
```python
normalizer.normalize_(grad)  # In-place normalization
```

### HookCollectorBase

**Location**: `collector/collector.py`

Abstract base class for all gradient collectors using PyTorch hooks.

```python
class HookCollectorBase(ABC):
    def __enter__(self):
        """Register hooks on model"""

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup hooks and state"""

    @abstractmethod
    def forward_hook(self, module, input, output):
        """Cache activations during forward pass"""

    @abstractmethod
    def backward_hook(self, module, grad_input, grad_output):
        """Compute per-sample gradients during backward"""

    @abstractmethod
    def process_batch(self):
        """Process collected gradients after batch"""

    @abstractmethod
    def teardown(self):
        """Final processing and save"""
```

**Lifecycle**:
```python
with collector:  # __enter__: registers hooks
    model(input).loss.backward()  # Hooks execute
    collector.process_batch()     # Process gradients
# __exit__: cleanup hooks
collector.teardown()  # Final processing
```

### GradientCollector

**Location**: `collector/gradient_collectors.py`

Main collector for building gradient indexes.

**Key Features**:
- Per-sample gradient computation via hooks
- Random projection compression
- Adafactor/Adam normalization
- Preconditioner accumulation
- Distributed gradient aggregation

**Example**:
```python
collector = GradientCollector(
    model=model,
    processor=processor,
    builder=builder,  # Writes gradients to disk
    attention_config=attention_cfg
)

with collector:
    for batch in batches:
        loss = model(**batch).loss
        loss.backward()
        collector.process_batch()

collector.teardown()  # Process preconditioners, save
```

### Attributor

**Location**: `query/attributor.py`

High-level interface for querying gradient indexes.

```python
class Attributor:
    def __init__(
        self,
        index_path: Path,
        model: nn.Module,
        unit_norm: bool = True,
        faiss_config: Optional[FaissConfig] = None
    ):
        """Load index and prepare for querying"""

    def trace(self, model: nn.Module, k: int = 5):
        """Context manager for computing attribution"""
```

**Usage**:
```python
attributor = Attributor(index_path="runs/my_index", model=model)

with attributor.trace(model, k=5) as trace_result:
    loss = model(query_input).loss
    loss.backward()

# trace_result contains top-k training example indices and scores
```

### Scorer

**Location**: `score/scorer.py`

Computes similarity scores on-the-fly without saving gradients.

**Scoring Modes**:
- `mean`: Score against mean query gradient
- `nearest`: Score against most similar query gradient
- `individual`: Separate score for each query gradient

**Example**:
```python
scorer = Scorer(
    query_path="runs/query_vector",
    score_method="mean",
    writer=score_writer
)

with scorer:
    for batch in dataset:
        loss = model(**batch).loss
        loss.backward()
        scorer.process_batch()

scorer.teardown()
```

### Builder

**Location**: `data.py`

Handles writing gradients to memory-mapped files.

```python
class Builder:
    def __init__(
        self,
        modules: List[str],
        grad_sizes: Dict[str, int],
        num_grads: int,
        output_path: Path
    ):
        """Create memory-mapped gradient storage"""

    def write(self, indices: List[int], gradients: Dict[str, Tensor]):
        """Write gradients at specified indices"""
```

**Storage Format**:
- Structured numpy array with one field per module
- Memory-mapped for efficient out-of-core processing
- Supports concurrent writes in distributed setting

---

## Data Flow

### Build Command Flow

```
CLI Input
    ↓
bergson build <path> --model <model> --dataset <dataset>
    ↓
__main__.py: Parse args into Build dataclass
    ↓
build.py: build(index_cfg)
    ├─ Setup data pipeline
    ├─ Validate paths
    └─ Launch distributed run
        ↓
    distributed.py: launch_distributed_run()
        ↓
    build.py: build_worker() [on each GPU/node]
        ├─ Initialize process group
        ├─ Setup model
        ├─ Create GradientProcessor
        │   └─ Fit normalizers if needed
        ├─ Create GradientCollector
        └─ Run collection
            ↓
        collection.py: collect_gradients()
            ├─ Create CollectorComputer
            └─ Run with collector hooks
                ↓
            collector/collector.py: run_with_collector_hooks()
                ├─ For each batch:
                │   ├─ Enter collector (register hooks)
                │   ├─ Forward pass → forward_hook caches activations
                │   ├─ Backward pass → backward_hook computes gradients
                │   ├─ process_batch() writes to disk
                │   └─ Exit collector (cleanup hooks)
                └─ teardown() processes preconditioners
                    ↓
                data.py: Builder writes gradients to memory-mapped file
                    ↓
                process_preconditioners.py: Aggregate and eigendecompose
                    ↓
                Save: gradients.bin, processor, dataset, metadata
```

### Query Command Flow

```
CLI Input
    ↓
bergson query --index <path>
    ↓
query/query_index.py: query()
    ├─ Load index config and dataset
    ├─ Create Attributor (with optional FAISS)
    └─ Interactive loop:
        ├─ Get query text from user
        ├─ Tokenize query
        └─ attributor.trace(model, k=5)
            ├─ Enter context: Create TraceCollector
            ├─ Forward pass with query
            ├─ Backward pass → collect query gradients
            ├─ Search index for top-k matches
            │   ├─ If FAISS: faiss_index.search()
            │   └─ Else: In-memory search via matmul
            └─ Return TraceResult(indices, scores)
                ↓
        Display top-k training examples from dataset
```

### Score Command Flow

```
CLI Input
    ↓
bergson score <path> --query_path <query> --score mean
    ↓
score/score.py: score()
    ├─ Load query index
    ├─ Setup data pipeline
    ├─ Create Scorer with ScoreWriter
    └─ Launch distributed run
        ↓
    build_worker()
        ├─ Initialize process group
        ├─ Setup model
        ├─ Create Scorer
        └─ For each batch:
            ├─ Forward pass
            ├─ Backward pass → collect gradients
            ├─ Compute similarity to query
            ├─ Write scores to memory-mapped file
            └─ Continue
                ↓
        Save: scores.bin with similarity values
```

---

## Gradient Collection

Bergson implements the **TrackStar algorithm** for scalable gradient-based attribution.

### Per-Sample Gradient Computation

The core innovation is computing per-sample gradients efficiently without storing full batch gradients.

#### Forward Hook: Cache Preprocessed Activations

```python
def forward_hook(self, module, input, output):
    """Cache activations with optional preprocessing"""
    a = input[0]  # [N, S, I] - batch, sequence, input_dim

    # Apply Adafactor column normalization
    if self.adafactor_normalizer:
        col_norm = self.adafactor_normalizer.col.rsqrt()
        a = a * col_norm

    # Apply random projection
    if self.projection_dim:
        proj = self._get_projection(module)  # Cached [I, p]
        a = a @ proj  # [N, S, p]

    # Cache preprocessed activations
    module._cached_inputs = a
```

#### Backward Hook: Compute Per-Sample Gradients

```python
def backward_hook(self, module, grad_input, grad_output):
    """Compute per-sample gradients via outer product"""
    a = module._cached_inputs  # [N, S, I] or [N, S, p]
    g = grad_output[0]  # [N, S, O]

    # Apply Adafactor row normalization
    if self.adafactor_normalizer:
        row_norm = self.adafactor_normalizer.row
        g = g * (row_norm.mean().sqrt() * row_norm.rsqrt())

    # Apply gradient projection
    if self.projection_dim:
        g_proj = self._get_grad_projection(module)  # [O, p]
        g = g @ g_proj.T  # [N, S, p]

    # Compute per-sample gradient as outer product
    # P[i] = g[i].T @ a[i] for each sample i
    P = g.mT @ a  # [N, O/p, I/p]
    P = P.flatten(1)  # [N, (O/p)*(I/p)]

    # Accumulate preconditioner (gradient covariance)
    if self.compute_preconditioners:
        self.preconditioner += P.mT @ P

    # Write gradients to disk
    self.builder.write(batch_indices, {module_name: P})
```

### Memory Efficiency Techniques

#### 1. Random Projections

Compress gradients from `[O, I]` to `[p, p]` where `p << min(O, I)`.

**Properties**:
- Preserves inner products approximately (Johnson-Lindenstrauss)
- Rademacher matrices: `{-1, +1}` entries (fast, no random generation)
- Gaussian matrices: N(0, 1) entries (better theoretical guarantees)

**Memory Savings**:
- Original: O × I parameters
- Projected: p × p parameters
- Typical: p=16, O=4096, I=4096 → 99.99% compression

**Example**:
```python
# Without projection: 4096 × 4096 = 16M parameters
# With projection: 16 × 16 = 256 parameters
compression_ratio = (O * I) / (p * p)  # 65,536x
```

#### 2. Adafactor Normalization

Factored representation of second moment matrix.

**Standard Adam**:
- Second moment: [O, I] matrix
- Memory: O(O × I)

**Adafactor**:
- Row factors: [O] vector
- Column factors: [I] vector
- Memory: O(O + I)

**Normalization**:
```python
# Full second moment (conceptual):
M = row[:, None] * col[None, :]  # [O, I]

# Applied factorized:
a_normalized = a * col.rsqrt()  # Apply to activations
g_normalized = g * row.rsqrt()  # Apply to gradients
```

**Memory Savings**:
- For O=4096, I=4096:
  - Adam: 16M parameters
  - Adafactor: 8K parameters (2000× reduction)

#### 3. Lazy Materialization

Gradients never fully materialized in memory:

1. **Forward pass**: Cache preprocessed activations
2. **Backward pass**: Compute gradient on-the-fly
3. **Immediate write**: Write to disk via memory-mapped file
4. **Discard**: Clear cache for next batch

**Benefits**:
- Constant memory usage per batch
- Supports datasets larger than RAM
- Enables distributed gradient aggregation

### TrackStar-Specific Components

#### Preconditioners (Hessian Approximation)

Accumulated during gradient collection:

```python
# Gradient covariance matrix
preconditioner = sum(g_i @ g_i.T for g_i in gradients)
preconditioner /= num_examples

# Eigendecomposition for efficient inversion
eigval, eigvec = torch.linalg.eigh(preconditioner)

# Inverse square root (for influence computation)
inv_sqrt = eigvec @ diag(eigval ** -0.5) @ eigvec.T
```

**Usage in Attribution**:
```python
# Apply preconditioning to query gradient
q_preconditioned = inv_sqrt @ q

# Compute influence scores
influences = q_preconditioned @ gradients.T
```

#### Distributed Preconditioner Aggregation

```python
# Each worker computes local preconditioner
local_prec = local_gradients.T @ local_gradients / local_count

# Reduce to rank 0 (on CPU to save GPU memory)
dist.reduce(local_prec, dst=0, op=dist.ReduceOp.SUM)

# Rank 0 computes eigendecomposition
if rank == 0:
    global_prec = local_prec / world_size
    eigval, eigvec = torch.linalg.eigh(global_prec)
```

---

## Indexing and Querying

### Index Structure

Gradients stored in **structured memory-mapped numpy arrays**:

```python
# Create structured dtype with one field per module
dtype = {
    'names': ['gpt_neox.layers.0.attention.dense',
              'gpt_neox.layers.0.mlp.dense_h_to_4h', ...],
    'formats': ['(256,)float16', '(512,)float16', ...]
}

# Create memory-mapped array
gradients = np.memmap(
    'gradients.bin',
    dtype=dtype,
    mode='w+',
    shape=(num_examples,)
)

# Access gradients by module
layer_grads = gradients['gpt_neox.layers.0.attention.dense']  # [num_examples, 256]
```

**Benefits**:
- Efficient out-of-core processing
- Named field access
- Supports partial loading (select modules)
- Works with datasets larger than RAM

### Metadata Format

**info.json**:
```json
{
    "num_grads": 100000,
    "dtype": {
        "names": ["layer1", "layer2"],
        "formats": ["(256,)float16", "(512,)float16"]
    },
    "grad_sizes": {
        "layer1": 256,
        "layer2": 512
    },
    "base_dtype": "float16"
}
```

### Query Methods

#### In-Memory Search

Fast exact search when index fits in GPU memory:

```python
# Load gradients into GPU
grads = {}
for name in module_names:
    grads[name] = torch.tensor(mmap[name], device='cuda')

# Compute scores via batch matrix multiplication
scores = sum(
    query_grad[name] @ grads[name].T
    for name in module_names
)  # [num_examples]

# Get top-k
topk_values, topk_indices = torch.topk(scores, k)
```

**Complexity**:
- Time: O(num_examples × grad_dim)
- Space: O(num_examples × grad_dim) GPU memory

#### FAISS Search

Approximate nearest neighbor for large-scale indices:

```python
# Build index
index = faiss.index_factory(
    grad_dim,
    "IVF1024,SQfp16",
    faiss.METRIC_INNER_PRODUCT
)
index.train(gradients[:train_size])
index.add(gradients)

# Search
distances, indices = index.search(query, k)
```

**Complexity**:
- Time: O(log(num_examples) × grad_dim) with IVF
- Space: Compressed on disk, partial loading

---

## FAISS Integration

### Index Creation Workflow

```python
def create_faiss_index(
    gradient_path: Path,
    factory_string: str = "IVF1024,SQfp16",
    num_shards: int = 1,
    mmap_index: bool = False
):
    """Create FAISS index from gradients"""

    # 1. Load gradients from memory-mapped files
    gradients = load_gradients(gradient_path)

    # 2. Normalize if needed
    if unit_norm:
        gradients = normalize_on_gpu(gradients)

    # 3. Create sharded indexes
    shard_size = len(gradients) // num_shards

    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = start + shard_size
        shard_grads = gradients[start:end]

        # 4. Build FAISS index
        index = faiss.index_factory(
            grad_dim,
            factory_string,
            faiss.METRIC_INNER_PRODUCT
        )

        # 5. Train (for IVF indexes)
        if "IVF" in factory_string:
            train_size = min(len(shard_grads), 1_000_000)
            index.train(shard_grads[:train_size])

        # 6. Add vectors
        index.add(shard_grads)

        # 7. Save to disk
        faiss.write_index(index, f"{shard_id}.faiss")
```

### Multi-Shard Search

Enables querying indices larger than memory:

```python
def search_sharded(query: Tensor, k: int):
    """Search across multiple FAISS shards"""
    all_distances = []
    all_indices = []

    # Search each shard independently
    for shard_id, shard in enumerate(shards):
        # Load shard (optionally mmap)
        if mmap_index:
            index = faiss.read_index(f"{shard_id}.faiss", faiss.IO_FLAG_MMAP)
        else:
            index = faiss.read_index(f"{shard_id}.faiss")

        # Search this shard
        distances, indices = index.search(query, k)

        # Offset indices by shard position
        offset = shard_id * shard_size
        indices += offset

        all_distances.append(distances)
        all_indices.append(indices)

    # Concatenate results from all shards
    combined_distances = np.concatenate(all_distances, axis=1)
    combined_indices = np.concatenate(all_indices, axis=1)

    # Rerank to get global top-k
    topk_positions = np.argsort(-combined_distances, axis=1)[:, :k]
    topk_distances = np.take_along_axis(combined_distances, topk_positions, axis=1)
    topk_indices = np.take_along_axis(combined_indices, topk_positions, axis=1)

    return topk_distances, topk_indices
```

### FAISS Factory Strings

Common configurations:

| Factory String | Description | Speed | Memory | Accuracy |
|---------------|-------------|-------|--------|----------|
| `"Flat"` | Exact search (brute force) | Slow | High | 100% |
| `"IVF1,SQfp16"` | Exact with fp16 quantization | Medium | Medium | 100% |
| `"IVF1024,SQfp16"` | ANN with 1024 clusters, fp16 | Fast | Low | ~95% |
| `"IVF4096,PQ32"` | ANN with product quantization | Very Fast | Very Low | ~90% |
| `"HNSW32"` | Hierarchical graph search | Fast | Medium | ~98% |

**Parameters**:
- `nprobe`: Number of clusters to search (IVF)
  - Higher → more accurate, slower
  - Default: 1024
- `mmap_index`: Query on disk vs load into memory
  - `True` → lower memory, slower
  - `False` → higher memory, faster

---

## Distributed Training

### Multi-GPU/Multi-Node Setup

Bergson supports distributed gradient collection across multiple GPUs and nodes.

#### Configuration

```python
@dataclass
class DistributedConfig:
    nnode: int = 1                    # Number of nodes
    nproc_per_node: int = 1           # GPUs per node
    node_rank: int = 0                # Current node rank
    master_addr: str = "localhost"    # Master node address
    master_port: str = "29500"        # Master node port
```

#### Launch Distributed Run

```python
from bergson.distributed import launch_distributed_run

launch_distributed_run(
    name="build",
    worker_fn=build_worker,
    const_worker_args=[index_cfg, dataset],
    dist_config=cfg.distributed
)
```

**What it does**:
1. Spawns `nproc_per_node` processes
2. Each process gets a unique `local_rank` (0 to nproc_per_node-1)
3. Sets environment variables for distributed training
4. Calls `worker_fn` on each process

#### Worker Initialization

```python
def build_worker(local_rank, index_cfg, dataset):
    # Set CUDA device
    torch.cuda.set_device(local_rank)

    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=global_rank
    )

    # Compute work assignment
    batches_per_worker = len(batches) // world_size
    start_idx = global_rank * batches_per_worker
    end_idx = start_idx + batches_per_worker
    my_batches = batches[start_idx:end_idx]

    # Run gradient collection
    collect_gradients(model, my_batches, ...)
```

### Data Distribution

#### Batch Allocation

Sophisticated bin-packing algorithm ensures equal work per worker:

```python
def allocate_batches(
    dataset,
    token_batch_size: int,
    world_size: int
):
    """Distribute batches across workers"""

    # Constraint: max_len * batch_size ≤ token_batch_size
    # Goal: Equal batches per worker

    # 1. Sort documents by length
    sorted_docs = sorted(dataset, key=lambda x: len(x['input_ids']))

    # 2. Greedy bin packing
    batches = []
    current_batch = []
    current_max_len = 0

    for doc in sorted_docs:
        doc_len = len(doc['input_ids'])

        # Check if adding doc exceeds token budget
        new_max_len = max(current_max_len, doc_len)
        new_size = len(current_batch) + 1

        if new_max_len * new_size > token_batch_size:
            # Start new batch
            batches.append(current_batch)
            current_batch = [doc]
            current_max_len = doc_len
        else:
            current_batch.append(doc)
            current_max_len = new_max_len

    # 3. Round to multiple of world_size
    total_batches = len(batches)
    batches_per_worker = total_batches // world_size

    # 4. Return worker assignments
    return [
        batches[i * batches_per_worker : (i+1) * batches_per_worker]
        for i in range(world_size)
    ]
```

### Gradient Aggregation

#### Preconditioner Reduction

```python
def process_preconditioners(
    local_preconditioners: Dict[str, Tensor],
    num_local_examples: int
):
    """Aggregate preconditioners across workers"""

    # Normalize by local dataset size
    for name, prec in local_preconditioners.items():
        prec /= num_local_examples

    # Reduce to rank 0 (on CPU to save GPU memory)
    for name, prec in local_preconditioners.items():
        prec_cpu = prec.cpu()
        dist.reduce(prec_cpu, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            # Average across workers
            prec_cpu /= world_size

            # Eigendecomposition
            eigval, eigvec = torch.linalg.eigh(prec_cpu)

            # Save
            save_preconditioner(name, prec_cpu, eigval, eigvec)
```

#### Loss Aggregation

```python
# Each worker tracks local per-document losses
local_losses = torch.zeros(len(dataset), device='cpu')
local_losses[local_indices] = computed_losses

# Reduce to rank 0
dist.reduce(local_losses, dst=0, op=dist.ReduceOp.SUM)

# Rank 0 saves complete loss vector
if rank == 0:
    save_losses(local_losses)
```

### FSDP Support

Optional Fully Sharded Data Parallel for models that don't fit on single GPU:

```python
from torch.distributed.fsdp import fully_shard

if cfg.fsdp:
    # Shard each transformer layer
    for layer in model.layers:
        fully_shard(layer)

    # Shard root module
    fully_shard(model)
```

**Benefits**:
- Shard model parameters across GPUs
- Shard gradients and optimizer states
- Enables training models larger than single GPU memory
- Automatic all-gather/reduce-scatter communication

---

## Design Patterns

### 1. Hook-Based Architecture

**Pattern**: Template Method + Strategy

```python
class HookCollectorBase(ABC):
    """Template for hook lifecycle"""

    def __enter__(self):
        # Template: register hooks
        for module in self.modules:
            module.register_forward_hook(self.forward_hook)
            module.register_full_backward_hook(self.backward_hook)

    @abstractmethod
    def forward_hook(self, module, input, output):
        """Strategy: how to cache activations"""

    @abstractmethod
    def backward_hook(self, module, grad_input, grad_output):
        """Strategy: how to compute gradients"""
```

**Benefits**:
- Non-intrusive: works with any PyTorch model
- Flexible: different strategies via subclasses
- Efficient: intercepts gradients at computation time

### 2. Context Manager Protocol

All collectors use context managers for resource management:

```python
with GradientCollector(...) as collector:
    loss.backward()
    collector.process_batch()
# Automatic cleanup: hooks removed, memory freed
```

**Benefits**:
- Automatic resource cleanup
- Exception-safe
- Clear API boundaries
- Prevents resource leaks

### 3. Lazy Evaluation + Streaming

Gradients never fully materialized in memory:

```
Forward → Cache activations → Backward → Compute gradients → Write to disk → Discard
   ↓                                                                              ↑
   └──────────────────────────── Constant memory ────────────────────────────────┘
```

**Benefits**:
- O(1) memory per batch
- Supports datasets larger than RAM
- Enables distributed processing

### 4. Composition Over Inheritance

**GradientProcessor** composes strategies:
```python
processor = GradientProcessor(
    normalizers={"layer1": AdafactorNormalizer(...)},  # Strategy
    preconditioners={"layer1": torch.tensor(...)},     # Data
    projection_dim=16                                   # Config
)
```

**CollectorComputer** composes components:
```python
computer = CollectorComputer(
    model=model,              # Component
    dataset=dataset,          # Component
    collector=collector,      # Strategy
    batching=batching_fn      # Strategy
)
```

### 5. Dataclass-Based Configuration

All configs use `@dataclass` for type safety and serialization:

```python
@dataclass
class IndexConfig:
    model: str = "EleutherAI/pythia-160m"
    dataset: str = "NeelNanda/pile-10k"
    projection_dim: int = 16
    normalizer: str = "adafactor"

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(dataclasses.asdict(self), f)

    @classmethod
    def load(cls, path: Path):
        with open(path) as f:
            return cls(**json.load(f))
```

**Benefits**:
- Type checking
- Default values
- Easy serialization
- CLI parsing via simple_parsing

### 6. Memory-Mapped Storage

Uses `numpy.memmap` for out-of-core processing:

```python
# Create memory-mapped array
gradients = np.memmap(
    'gradients.bin',
    dtype=dtype,
    mode='w+',
    shape=(num_examples,)
)

# Write gradients (appends to file)
gradients[indices] = new_gradients

# Read gradients (loads from disk on access)
batch = gradients[start:end]
```

**Benefits**:
- Supports datasets larger than RAM
- Efficient random access
- Concurrent read/write
- OS-level caching

### 7. Separation of Concerns

Clear module boundaries:

| Concern | Module |
|---------|--------|
| Gradient computation | `collector/` |
| Gradient storage | `data.py` |
| Gradient transformation | `gradients.py` |
| Similarity search | `query/` |
| Distributed coordination | `distributed.py` |
| Configuration | `config.py` |

**Benefits**:
- Testable components
- Reusable abstractions
- Clear dependencies
- Easy to extend

---

## File Formats

### Gradient Index Directory

```
runs/my_index/
├── index_config.json          # IndexConfig serialized
├── data.hf/                   # HuggingFace Dataset
│   ├── dataset_info.json      # Dataset metadata
│   ├── state.json             # Dataset state
│   └── data-00000-of-00001.arrow  # Arrow format data
├── gradients.bin              # Memory-mapped gradients
├── info.json                  # Gradient metadata
├── processor_config.json      # GradientProcessor config
├── normalizers.pth            # Normalizer state dicts (PyTorch)
├── preconditioners.pth        # Preconditioner matrices (PyTorch)
└── preconditioners_eigen.pth  # Eigendecompositions (PyTorch)
```

### Gradient Binary Format

**Structured numpy array** with one field per module:

```python
dtype = {
    'names': ['layer1', 'layer2', ...],
    'formats': ['(256,)float16', '(512,)float16', ...]
}

# Shape: (num_examples,)
# Size: num_examples * sum(grad_dims) * dtype_bytes
```

**Example**:
```python
# 10,000 examples
# 2 layers: 256-dim and 512-dim
# float16 (2 bytes)
total_size = 10_000 * (256 + 512) * 2 = 15.36 MB
```

### info.json Format

```json
{
    "num_grads": 10000,
    "dtype": {
        "names": ["layer1", "layer2"],
        "formats": ["(256,)float16", "(512,)float16"]
    },
    "grad_sizes": {
        "layer1": 256,
        "layer2": 512
    },
    "base_dtype": "float16"
}
```

### FAISS Index Directory

```
runs/my_index/faiss_IVF1024_SQfp16_cosine/
├── config.json       # FaissConfig + metadata
├── 0.faiss          # Shard 0
├── 1.faiss          # Shard 1
├── 2.faiss          # Shard 2
└── ...
```

**config.json**:
```json
{
    "factory_string": "IVF1024,SQfp16",
    "metric": "cosine",
    "num_shards": 4,
    "shard_size": 250000,
    "total_vectors": 1000000,
    "dim": 768
}
```

### Score Storage

```
runs/scores/
├── info.json        # Metadata
└── scores.bin       # Memory-mapped scores
```

**Structured array format**:
```python
# For mean/nearest scoring
dtype = [
    ('score_0', 'float32'),     # Score value
    ('written_0', 'bool')       # Whether score has been written
]

# For individual scoring (multiple queries)
dtype = [
    ('score_0', 'float32'),
    ('score_1', 'float32'),
    ...
    ('written_0', 'bool')
]
```

### Training Gradients

```
runs/training/
├── train/
│   ├── gradients.bin          # Accumulated gradients
│   ├── info.json
│   └── processor_config.json
├── train/epoch_0/             # If not accumulating across epochs
│   └── gradients.bin
├── train/epoch_1/
│   └── gradients.bin
└── order.hf/                  # If track_order=True
    └── data.arrow             # Training order tracking
```

---

## Summary

Bergson's architecture demonstrates several key principles:

1. **Modularity**: Clear separation of concerns with well-defined interfaces
2. **Scalability**: Distributed training, memory-mapped storage, FAISS integration
3. **Efficiency**: Lazy evaluation, streaming, random projections, factored normalizers
4. **Flexibility**: Multiple normalizers, collectors, scoring methods
5. **Usability**: Simple CLI, context managers, sensible defaults
6. **Extensibility**: Hook-based design, composition patterns, strategy pattern

The codebase is well-structured for both research experimentation and production deployment of gradient-based data attribution at scale.

### Key Innovations

- **Hook-based gradient collection**: Non-intrusive per-sample gradients
- **Factored normalization**: 1000× memory reduction for second moments
- **Random projections**: 10,000× compression with preserved similarity
- **Memory-mapped storage**: Process datasets larger than RAM
- **Distributed preconditioners**: Scalable Hessian approximation
- **Sharded FAISS**: Query billion-scale indices

These architectural choices enable Bergson to scale from small models (14M parameters) to large models (billions of parameters) and from small datasets (10K examples) to massive datasets (millions of examples).
