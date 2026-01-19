# Claude-Generated Bergson CLI Usage Guide

Bergson is a library for tracing the memory of deep neural networks using gradient-based data attribution. This guide covers practical usage of the Bergson CLI with hands-on examples.

## Quick Start

The fastest way to get started is to build a gradient index and query it:

```bash
# Build an index from a small dataset
bergson build runs/quickstart \
    --model EleutherAI/pythia-14m \
    --dataset NeelNanda/pile-10k \
    --truncation \
    --token_batch_size 4096

# Query the index interactively
bergson query --index runs/quickstart
```

When prompted, enter any text and Bergson will show you the top 5 most influential training examples.

## CLI Commands Overview

Bergson provides 4 main commands:

| Command | Purpose |
|---------|---------|
| `build` | Build a gradient index from training data |
| `query` | Interactively query a pre-built index |
| `reduce` | Aggregate dataset gradients to a query vector |
| `score` | Score a dataset against a query vector |

---

## 1. Building a Gradient Index

The `build` command collects per-example gradients from your training data and optionally compresses them.

### Basic Usage

```bash
bergson build <output_path> \
    --model <model_name> \
    --dataset <dataset_name>
```

### Example: Small Model on Pile

```bash
bergson build runs/pile_index \
    --model EleutherAI/pythia-160m \
    --dataset NeelNanda/pile-10k \
    --token_batch_size 2048 \
    --truncation
```

### Example: Larger Model with Compression

```bash
# Compress gradients to dimension 32 using random projection
bergson build runs/compressed_index \
    --model EleutherAI/pythia-410m \
    --dataset NeelNanda/pile-10k \
    --token_batch_size 4096 \
    --projection_dim 32 \
    --normalizer adafactor
```

### Example: No Compression (Full Gradients)

```bash
# Set projection_dim to 0 to disable compression
bergson build runs/full_gradients \
    --model EleutherAI/pythia-70m \
    --dataset NeelNanda/pile-10k \
    --projection_dim 0 \
    --token_batch_size 8192
```

### Example: Custom Dataset Format

```bash
# For datasets with specific column names
bergson build runs/custom_data \
    --model EleutherAI/pythia-160m \
    --dataset my_org/my_dataset \
    --prompt_column "input" \
    --completion_column "output" \
    --token_batch_size 4096
```

### Example: Distributed Multi-GPU Build

```bash
# Using FSDP (Fully Sharded Data Parallel)
bergson build runs/distributed_index \
    --model EleutherAI/pythia-1b \
    --dataset NeelNanda/pile-10k \
    --token_batch_size 16384 \
    --fsdp \
    --precision bf16
```

### Key Parameters

- `--token_batch_size`: Token budget per batch (controls memory usage)
- `--projection_dim`: Compression dimension (default: 16, set to 0 to disable)
- `--normalizer`: Gradient normalization method (`adafactor`, `adam`, `none`)
- `--truncation`: Truncate long documents to fit token budget
- `--precision`: Model dtype (`auto`, `bf16`, `fp16`, `fp32`, `int4`, `int8`)
- `--fsdp`: Enable Fully Sharded Data Parallel for multi-GPU
- `--skip_index`: Only compute preconditioners (don't build index)

---

## 2. Querying an Index

The `query` command launches an interactive session where you can enter text and find the most influential training examples.

### Basic Usage

```bash
bergson query --index <index_path>
```

### Example: Basic Query

```bash
bergson query --index runs/pile_index

# Interactive prompt appears:
> Enter your query: The quick brown fox jumps over
# Returns top 5 most similar training examples
```

### Example: Query with Model Override

```bash
# Use a different model than the one that built the index
bergson query \
    --index runs/pile_index \
    --model EleutherAI/pythia-70m
```

### Example: FAISS Approximate Search

```bash
# Use FAISS for faster approximate nearest neighbor search
bergson query \
    --index runs/large_index \
    --faiss
```

### Example: Show Least Influential Examples

```bash
# Reverse the ranking to show lowest influences
bergson query \
    --index runs/pile_index \
    --reverse
```

### Example: Custom Text Field

```bash
# Display a specific column from the dataset
bergson query \
    --index runs/custom_index \
    --text_field "content"
```

### Key Parameters

- `--index`: Path to the pre-built gradient index
- `--model`: Model to use (defaults to the model that built the index)
- `--faiss`: Use FAISS for approximate nearest neighbor search
- `--reverse`: Show lowest influences instead of highest
- `--unit_norm`: Unit normalize query gradient (default: True)
- `--text_field`: Dataset column to display (default: "text")

---

## 3. Reducing a Dataset to a Query Vector

The `reduce` command aggregates all examples in a dataset into a single gradient vector, useful for creating query vectors.

### Basic Usage

```bash
bergson reduce <output_path> \
    --model <model_name> \
    --dataset <dataset_name> \
    --method mean
```

### Example: Mean Query Vector

```bash
# Create a mean gradient from WikiText
bergson reduce runs/wikitext_query \
    --model EleutherAI/pythia-160m \
    --dataset wikitext \
    --method mean \
    --unit_normalize
```

### Example: Sum Aggregation

```bash
bergson reduce runs/sum_query \
    --model EleutherAI/pythia-160m \
    --dataset NeelNanda/pile-10k \
    --method sum
```

### Key Parameters

- `--method`: Reduction method (`mean` or `sum`)
- `--unit_normalize`: Unit normalize gradients before reduction
- All standard `IndexConfig` parameters (`model`, `dataset`, `token_batch_size`, etc.)

---

## 4. Scoring a Dataset

The `score` command computes attribution scores for a dataset against a pre-built query vector, without storing full gradients.

### Basic Usage

```bash
bergson score <output_path> \
    --model <model_name> \
    --dataset <dataset_name> \
    --query_path <query_vector_path> \
    --score mean
```

### Example: Score Against Mean Query

```bash
# Score training data against WikiText query vector
bergson score runs/pile_scores \
    --model EleutherAI/pythia-160m \
    --dataset NeelNanda/pile-10k \
    --query_path runs/wikitext_query \
    --score mean \
    --unit_normalize
```

### Example: Nearest Neighbor Scoring

```bash
# For each example, find max similarity across all query examples
bergson score runs/nearest_scores \
    --model EleutherAI/pythia-160m \
    --dataset NeelNanda/pile-10k \
    --query_path runs/query_index \
    --score nearest
```

### Example: Individual Query Scores

```bash
# Get per-query-example scores (returns a matrix)
bergson score runs/individual_scores \
    --model EleutherAI/pythia-160m \
    --dataset NeelNanda/pile-10k \
    --query_path runs/query_index \
    --score individual
```

### Example: With Preconditioner Mixing

```bash
bergson score runs/mixed_scores \
    --model EleutherAI/pythia-160m \
    --dataset NeelNanda/pile-10k \
    --query_path runs/query_vector \
    --query_preconditioner_path runs/query_vector/preconditioner.safetensors \
    --score mean \
    --mixing_coefficient 0.5
```

### Key Parameters

- `--query_path`: Path to query index (from `reduce` or `build`)
- `--score`: Scoring method (`mean`, `nearest`, `individual`)
- `--unit_normalize`: Unit normalize before scoring
- `--batch_size`: Processing batch size (default: 1024)
- `--query_preconditioner_path`: Path to precomputed query preconditioner
- `--mixing_coefficient`: Weight between query/index preconditioners (0-1)

---

## Common Workflows

### Workflow 1: Build and Query

The simplest workflow for data attribution:

```bash
# Step 1: Build index
bergson build runs/my_index \
    --model EleutherAI/pythia-70m \
    --dataset NeelNanda/pile-10k \
    --token_batch_size 4096

# Step 2: Query interactively
bergson query --index runs/my_index
```

### Workflow 2: Build → Reduce → Score

For on-the-fly scoring without storing full gradients:

```bash
# Step 1: Build index from training data
bergson build runs/training_index \
    --model EleutherAI/pythia-160m \
    --dataset NeelNanda/pile-10k \
    --token_batch_size 4096

# Step 2: Reduce evaluation dataset to query vector
bergson reduce runs/eval_query \
    --model EleutherAI/pythia-160m \
    --dataset wikitext \
    --method mean \
    --unit_normalize

# Step 3: Score training data against evaluation query
bergson score runs/attribution_scores \
    --model EleutherAI/pythia-160m \
    --dataset NeelNanda/pile-10k \
    --query_path runs/eval_query \
    --score mean
```

### Workflow 3: Distributed Build

For large-scale distributed training:

```bash
# Distributed training with appropriate batch size
bergson build runs/large_exp/index \
    --model EleutherAI/pythia-1b \
    --dataset NeelNanda/pile-10k \
    --token_batch_size 8192 \
    --fsdp \
    --precision bf16
```

### Workflow 4: Data Filtering with Attribution

Filter your dataset based on attribution scores:

```bash
# Step 1: Build index from high-quality data
bergson build runs/quality_index \
    --model EleutherAI/pythia-160m \
    --dataset high_quality_dataset

# Step 2: Score candidate data against quality index
bergson score runs/candidate_scores \
    --model EleutherAI/pythia-160m \
    --dataset candidate_dataset \
    --query_path runs/quality_index \
    --score mean

# Step 3: Filter data using scores (custom script)
python scripts/filter_data.py \
    --scores runs/candidate_scores/scores.npy \
    --dataset candidate_dataset \
    --threshold 0.5 \
    --output runs/filtered_dataset
```

---

## Advanced Usage

### Attention Head Gradients

Split attention modules into per-head gradients for finer-grained attribution:

```bash
bergson build runs/head_gradients \
    --model EleutherAI/pythia-160m \
    --dataset NeelNanda/pile-10k \
    --split_attention_modules \
    --token_batch_size 4096
```

### Module Filtering

Exclude specific layers using glob patterns:

```bash
bergson build runs/filtered_modules \
    --model EleutherAI/pythia-160m \
    --dataset NeelNanda/pile-10k \
    --filter_modules "*.embed*" "*.ln_*" \
    --token_batch_size 4096
```

### Custom Precision and Memory Optimization

```bash
# Use INT8 quantization for memory-constrained environments
bergson build runs/int8_index \
    --model EleutherAI/pythia-410m \
    --dataset NeelNanda/pile-10k \
    --precision int8 \
    --token_batch_size 8192

# Use BF16 for better numerical stability on modern GPUs
bergson build runs/bf16_index \
    --model EleutherAI/pythia-410m \
    --dataset NeelNanda/pile-10k \
    --precision bf16 \
    --token_batch_size 16384
```

### GRPO (Policy Gradient) Support

For RL/preference data with reward columns:

```bash
bergson build runs/grpo_index \
    --model EleutherAI/pythia-160m \
    --dataset rlhf_dataset \
    --reward_column "reward" \
    --token_batch_size 4096
```

### Skip Index Building (Preconditioners Only)

Build only the preconditioners without creating the full index:

```bash
bergson build runs/preconditioners_only \
    --model EleutherAI/pythia-160m \
    --dataset NeelNanda/pile-10k \
    --skip_index
```

---

## Tips and Best Practices

1. **Start Small**: Test with small models (pythia-14m, pythia-70m) and small datasets before scaling up.

2. **Use Auto Batch Size**: Always run `autobatchsize` first to avoid OOM errors and maximize GPU utilization.

3. **Enable Truncation**: Use `--truncation` to handle variable-length documents and prevent memory issues.

4. **Choose Compression Wisely**:
   - Use `--projection_dim 16-32` for large-scale builds
   - Use `--projection_dim 0` for maximum accuracy on small datasets

5. **Normalize for Stability**: Use `--normalizer adafactor` or `--normalizer adam` for better gradient stability.

6. **Use BF16 on Modern GPUs**: `--precision bf16` provides better numerical stability than FP16 on A100/H100 GPUs.

7. **Multi-GPU Strategy**:
   - Use `--fsdp` for models that don't fit on a single GPU
   - Run `autobatchsize` once before launching distributed jobs

8. **FAISS for Large Indices**: Use `--faiss` in `query` command for faster searches on large indices.

---

## Troubleshooting

### Out of Memory Errors

```bash
# Reduce token_batch_size
bergson build runs/my_index \
    --model EleutherAI/pythia-160m \
    --dataset NeelNanda/pile-10k \
    --token_batch_size 1024  # Lower value

# Use a lower token_batch_size value
```

### Slow Query Performance

```bash
# Use FAISS for approximate nearest neighbor search
bergson query --index runs/large_index --faiss
```

### Dataset Column Name Issues

```bash
# Specify custom column names
bergson build runs/my_index \
    --model EleutherAI/pythia-160m \
    --dataset my_dataset \
    --prompt_column "input_text" \
    --completion_column "output_text"
```

### Distributed Training Issues

```bash
# Ensure you set an appropriate token_batch_size for multi-GPU training
bergson build runs/index --fsdp --token_batch_size 8192
```

---

## Further Reading

- For implementation details, see the main codebase in `bergson/`
- For advanced scripting examples, check `scripts/` directory
- For API usage beyond CLI, explore the library imports in your Python code
