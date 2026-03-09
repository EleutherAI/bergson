# Bergson
This library enables you to trace the memory of deep neural nets with gradient-based data attribution techniques. We currently focus on TrackStar, as described in [Scalable Influence and Fact Tracing for Large Language Model Pretraining](https://arxiv.org/abs/2410.17413v3) by Chang et al. (2024), and also include support for several alternative influence functions. We plan to add support for [Magic](https://arxiv.org/abs/2504.16430) soon.

We view attribution as a counterfactual question: **_If we "unlearned" this training sample, how would the model's behavior change?_** This formulation ties attribution to some notion of what it means to "unlearn" a training sample. Here we focus on a very simple notion of unlearning: taking a gradient _ascent_ step on the loss with respect to the training sample.

## Core features

- Gradient store for serial queries. We provide collection-time gradient compression for efficient storage, and integrate with FAISS for fast KNN search over large stores.
- On-the-fly queries. Query gradients without disk I/O overhead via a single pass over a dataset with a set of precomputed query gradients.
  - Experiment with multiple query strategies based on [LESS](https://arxiv.org/pdf/2402.04333).
  - Ideal for compression-free gradients.
- Per-token scores.
- Train‑time gradient collection. Capture gradients produced during training with a ~17% performance overhead.
- Scalable. We use [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html), BitsAndBytes, and other performance optimizations to support large models, datasets, and clusters.
- Integrated with HuggingFace Transformers and Datasets. We also support on-disk datasets in a variety of formats.
- Structured gradient views and per-attention head gradient collection. Bergson enables mechanistic interpretability via easy access to per‑module or per-attention head gradients.

# Announcements

**February 2026**
- Support per-token gradients

**January 2026**
- Support EK-FAC
- [Experimental] Support distributing preconditioners across nodes and devices for VRAM-efficient computation through the GradientCollectorWithDistributedPreconditioners. If you would like this functionality exposed via the CLI please get in touch! https://github.com/EleutherAI/bergson/pull/100

# Installation

```bash
pip install bergson
```

# Quickstart

To construct an index of randomly projected gradients:

```
bergson build runs/index --model EleutherAI/pythia-14m --dataset NeelNanda/pile-10k --truncation --token_batch_size 4096
```

To collect Trackstar attribution scores:

```
bergson trackstar runs/trackstar --model EleutherAI/pythia-14m --query.dataset NeelNanda/pile-10k --data.dataset NeelNanda/pile-10k --data.truncation --token_batch_size 4096 --query.truncation --query.split "train[:20]"
```

# Usage

There are two ways to use Bergson. The first is to write an index of dataset gradients to disk using `build` then query it programmatically or using the `Attributor` or `query` CLI. The second is to specify your query upfront, then map over the dataset and collect and process gradients on the fly. When using this second strategy only influence scores will be saved.

You can build an index of gradients for each training sample from the command line, using `bergson` as a CLI tool:

```bash
bergson build <output_path> --model <model_name> --dataset <dataset_name>
```

This will create a directory at `<output_path>` containing the gradients for each training sample in the specified dataset. The `--model` and `--dataset` arguments should be compatible with the Hugging Face `transformers` library. By default it assumes that the dataset has a `text` column, but you can specify other columns using `--prompt_column` and optionally `--completion_column`. The `--help` flag will show you all available options.

You can also use the library programmatically to build an index. The `collect_gradients` function is just a bit lower level the CLI tool, and allows you to specify the model and dataset directly as arguments. The result is a HuggingFace dataset which contains a handful of new columns, including `gradients`, which contains the gradients for each training sample. You can then use this dataset to compute attributions.

At the lowest level of abstraction, the `GradientCollector` context manager allows you to efficiently collect gradients for _each individual example_ in a batch during a backward pass, simultaneously randomly projecting the gradients to a lower-dimensional space to save memory. If you use Adafactor normalization we will do this in a very compute-efficient way which avoids computing the full gradient for each example before projecting it to the lower dimension. There are two main ways you can use `GradientCollector`:

1. Using a `closure` argument, which enables you to make use of the per-example gradients immediately after they are computed, during the backward pass. If you're computing summary statistics or other per-example metrics, this is the most efficient way to do it.
2. Without a `closure` argument, in which case the gradients are collected and returned as a dictionary mapping module names to batches of gradients. This is the simplest and most flexible approach but is a bit more memory-intensive.

## On-the-fly Query

You can score a large dataset against a previously built query index without saving its gradients to disk:

```bash
bergson score <output_path> --model <model_name> --dataset <dataset_name> --query_path <existing_index_path> --score individual
```

We provide a utility to reduce a dataset into its mean or sum query gradient, for use as a query index:

```bash
bergson reduce <output_path> --model <model_name> --dataset <dataset_name> --aggregation mean --unit_normalize
```

## Index Query

We provide a query Attributor which supports unit normalized gradients and KNN search out of the box. Access it via CLI with

```bash
bergson query --index  <index_path> --model <model_name> --unit_norm
```

or programmatically with

```python
from bergson import Attributor, FaissConfig

attr = Attributor(args.index, device="cuda")

...
query_tokens = tokenizer(query, return_tensors="pt").to("cuda:0")["input_ids"]

# Query the index
with attr.trace(model.base_model, 5) as result:
    model(query_tokens, labels=query_tokens).loss.backward()
    model.zero_grad()
```

To efficiently query on-disk indexes, perform ANN searches, and explore many other scalability features add a FAISS config:

```python
attr = Attributor(args.index, device="cuda", faiss_cfg=FaissConfig("IVF1,SQfp16", mmap_index=True))

with attr.trace(model.base_model, 5) as result:
    model(query_tokens, labels=query_tokens).loss.backward()
    model.zero_grad()
```

## Training Gradients

Gradient collection during training is supported via an integration with HuggingFace's Trainer and SFTTrainer classes. Training gradients are saved in the original order corresponding to their dataset items, and when the `track_order` flag is set the training steps associated with each training item are separately saved.

```python
from bergson import GradientCollectorCallback, prepare_for_gradient_collection

callback = GradientCollectorCallback(
    path="runs/example",
    track_order=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    callbacks=[callback],
)
trainer = prepare_for_gradient_collection(trainer)
trainer.train()
```

## Attention Head Gradients

By default Bergson collects gradients for named parameter matrices, but per-attention head gradients may be collected by configuring an AttentionConfig for each module of interest.

```python
from bergson import AttentionConfig, IndexConfig, DataConfig
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("RonenEldan/TinyStories-1M", trust_remote_code=True, use_safetensors=True)

collect_gradients(
    model=model,
    data=data,
    processor=processor,
    path="runs/split_attention",
    attention_cfgs={
        # Head configuration for the TinyStories-1M transformer
        "h.0.attn.attention.out_proj": AttentionConfig(num_heads=16, head_size=4, head_dim=2),
    },
)
```

## GRPO

Where a reward signal is available we compute gradients using a weighted advantage estimate based on Dr. GRPO:

```bash
bergson build <output_path> --model <model_name> --dataset <dataset_name> --reward_column <reward_column_name>
```

# CLI Reference

All subcommands use [simple_parsing](https://github.com/lebrice/SimpleParsing) dataclasses for configuration. Nested configs use dot notation (e.g. `--data.dataset`, `--query.split`). Boolean flags can be passed without a value (e.g. `--truncation` is equivalent to `--truncation true`).

## Subcommands

| Command | Description |
|---|---|
| `bergson build` | Build a gradient index from a dataset |
| `bergson score` | Score a dataset against an existing query gradient index on the fly |
| `bergson query` | Interactively query an existing gradient index |
| `bergson reduce` | Reduce a gradient index to an aggregated (mean/sum) gradient |
| `bergson preconditioners` | Compute normalizers and preconditioners without gradient collection |
| `bergson hessian` | Approximate Hessian matrices using KFAC or EKFAC |
| `bergson trackstar` | Run the full TrackStar pipeline (preconditioners, build, and score) in one command |

## Handling Length-Based Differences in Text

When working with datasets containing texts of varying lengths, attribution scores can be biased by sequence length. Three flags interact to control this:

**`--loss_reduction`** (default: `mean`, choices: `mean`, `sum`)

Controls how per-token losses are reduced across a sequence before backpropagation. Neither option is length-neutral on its own.

- `--loss_reduction mean` **(default)**: Divides the summed per-token loss by the number of tokens. This produces an inverse length dependence: shorter sequences yield larger per-example loss values, larger gradient magnitudes, and therefore higher attribution scores. **This biases toward shorter texts.**
- `--loss_reduction sum`: Uses the raw sum of per-token losses. Longer sequences accumulate more loss, producing larger gradients. **This biases toward longer texts.**

**`--unit_normalize`** (default: `true`)

Normalizes all gradient vectors to unit length so that scoring becomes cosine similarity (purely directional) rather than a raw dot product. This removes gradient magnitude as a factor, mitigating length bias regardless of which `--loss_reduction` is used. **Enabled by default.** To disable it and use raw dot-product scoring (where gradient magnitude matters), pass `--unit_normalize false`.

```bash
# Opt out of unit normalization (raw dot-product scoring)
bergson score runs/scores --model <model_name> --dataset <dataset_name> \
    --query_path runs/query --unit_normalize false
```

**`--aggregation`** (default: `none`, choices: `mean`, `sum`, `none`)

Controls how multiple query gradients are combined when aggregating them into a single query vector (used in `reduce` and `score`).

- `--aggregation mean`: Averages multiple query gradients, producing a length-independent aggregate.
- `--aggregation sum`: Sums multiple query gradients. The aggregate magnitude scales with the number of examples.
- `--aggregation none`: No aggregation; each query gradient is kept separate.

## Shared Flags: Model, Data & Index (`IndexConfig`)

These flags are shared across `build`, `score`, `reduce`, `preconditioners`, `hessian`, and `trackstar`.

**Positional:**

| Flag | Description |
|---|---|
| `run_path` | Name of the run. Creates a directory for run artifacts. |

**Model:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `EleutherAI/pythia-160m` | Name or path of the HuggingFace model to load. |
| `--tokenizer` | `""` | Tokenizer name/path. If empty, the model's tokenizer is used. |
| `--revision` | `None` | Model revision (branch, tag, or commit hash). |
| `--precision` | `fp32` | Model parameter dtype. Choices: `auto`, `bf16`, `fp16`, `fp32`, `int4`, `int8`. |

**Data (prefixed with `--data.`):**

| Flag | Default | Description |
|---|---|---|
| `--data.dataset` | `NeelNanda/pile-10k` | HuggingFace dataset identifier. |
| `--data.split` | `train` | Dataset split. Supports HF slice notation (e.g. `train[:100]`). |
| `--data.subset` | `None` | Dataset subset/configuration. |
| `--data.prompt_column` | `text` | Column containing the prompts. |
| `--data.completion_column` | `""` | Optional column containing completions. |
| `--data.conversation_column` | `""` | Optional column containing conversations. |
| `--data.reward_column` | `""` | Optional column containing rewards (enables GRPO loss). |
| `--data.skip_nan_rewards` | `false` | Skip examples with NaN rewards. |
| `--data.truncation` | `false` | Truncate long documents to fit the token budget. |
| `--data.format_template` | `""` | Path to a YAML with a Jinja2 template for formatting dataset rows. |
| `--data.data_args` | `""` | Extra dataset constructor args as `arg1=val1,arg2=val2`. |

**Gradient Collection:**

| Flag | Default | Description |
|---|---|---|
| `--projection_dim` | `16` | Dimension of the random projection, or `0` to disable. |
| `--projection_type` | `rademacher` | Random projection type. Choices: `normal`, `rademacher`. |
| `--include_bias` | `false` | Include linear layers' bias gradients. |
| `--reshape_to_square` | `false` | Reshape gradients to a square matrix. |
| `--token_batch_size` | `2048` | Batch size in tokens. |
| `--auto_batch_size` | `false` | Automatically determine optimal token batch size (experimental, `build` only). |
| `--attribute_tokens` | `false` | Compute per-token gradients instead of per-example. Incompatible with `reduce`. |

**Loss & Normalization:**

| Flag | Default | Description |
|---|---|---|
| `--loss_fn` | `ce` | Loss function. Choices: `ce` (cross-entropy), `kl` (KL divergence). |
| `--loss_reduction` | `mean` | Loss reduction method. Choices: `mean`, `sum`. See [Handling Length-Based Differences](#handling-length-based-differences-in-text). |
| `--label_smoothing` | `0.0` | Label smoothing coefficient. Prevents near-zero gradients for high-confidence predictions. Recommended: `0.005`–`0.01`. |
| `--normalizer` | `none` | Gradient normalizer type. Choices: `adafactor`, `adam`, `none`. |
| `--stats_sample_size` | `10000` | Number of examples for estimating normalizer statistics. |

**Performance & Scaling:**

| Flag | Default | Description |
|---|---|---|
| `--fsdp` | `false` | Use Fully Sharded Data Parallel (FSDP) for gradient collection. |
| `--processor_path` | `""` | Path to a precomputed processor. |
| `--stream_shard_size` | `400000` | Shard size for streaming datasets into Dataset objects. |

**Index Control:**

| Flag | Default | Description |
|---|---|---|
| `--skip_preconditioners` | `false` | Skip estimating preconditioner statistics. |
| `--skip_index` | `false` | Skip building the gradient index. |
| `--overwrite` | `false` | Overwrite any existing index in the run path. |
| `--drop_columns` | `true` | Only save new dataset columns; drop originals. |

**Module Selection:**

| Flag | Default | Description |
|---|---|---|
| `--modules` | `[]` | Restrict gradient collection to specific modules. If empty, all modules are used. |
| `--filter_modules` | `None` | Glob pattern to exclude modules (e.g. `transformer.h.*.mlp.*`). |
| `--split_attention_modules` | `[]` | Modules to split into per-head matrices. |

**Attention Config (prefixed with `--attention.`):**

| Flag | Default | Description |
|---|---|---|
| `--attention.num_heads` | `0` | Number of attention heads. |
| `--attention.head_size` | `0` | Size of each attention head. |
| `--attention.head_dim` | `0` | Axis index for `num_heads` in the weight matrix. |

**Distributed Config (prefixed with `--distributed.`):**

| Flag | Default | Description |
|---|---|---|
| `--distributed.nnode` | `1` | Number of nodes for preconditioner computation. |
| `--distributed.nproc_per_node` | GPU count | Number of processes per node. |
| `--distributed.node_rank` | `None` | Rank of the current node. Inferred from `SLURM_NODEID`, `GROUP_RANK`, or `NODE_RANK` env vars if not set. |

**Debug & Profiling:**

| Flag | Default | Description |
|---|---|---|
| `--profile` | `false` | Enable profiling (first 4 steps by default). |
| `--debug` | `false` | Enable debug mode with additional logging. |
| `--max_tokens` | `None` | Max tokens to process. Experimental. |

## Shared Flags: Preprocessing (`PreprocessConfig`)

Used in `build`, `reduce`, `score`, and `trackstar`.

| Flag | Default | Description |
|---|---|---|
| `--unit_normalize` | `true` | Unit normalize the gradients. Mitigates length bias by converting scoring to cosine similarity. |
| `--preconditioner_path` | `None` | Path to a precomputed preconditioner. |
| `--aggregation` | `none` | Gradient aggregation method. Choices: `mean`, `sum`, `none`. In `score`, only query gradients are aggregated. See [Handling Length-Based Differences](#handling-length-based-differences-in-text). |
| `--normalize_aggregated_grad` | `false` | Unit normalize the aggregated gradient. Does not affect relative score rankings but affects score magnitudes. |

## `bergson score` Flags (`ScoreConfig`)

| Flag | Default | Description |
|---|---|---|
| `--query_path` | `""` | Path to the existing query gradient index. Required. |
| `--score` | `individual` | Scoring method. Choices: `nearest` (use the most similar query gradient's score), `individual` (compute a separate score for each query gradient). |
| `--batch_size` | `1024` | Batch size for processing the query dataset. |
| `--precision` | `fp32` | Dtype for score computation. Choices: `auto`, `bf16`, `fp16`, `fp32`. |

## `bergson query` Flags (`QueryConfig`)

| Flag | Default | Description |
|---|---|---|
| `--index` | `""` | Path to the existing gradient index. Required. |
| `--model` | `""` | Model for the query. Falls back to the index's model if empty. |
| `--text_field` | `text` | Field to use for the query text. |
| `--unit_norm` | `true` | Unit normalize the query gradients. |
| `--device_map_auto` | `false` | Load the model onto multiple devices if necessary. |
| `--faiss` | `false` | Use FAISS for the query. |
| `--top_k` | `5` | Number of top (and bottom) results to return per query. |
| `--record` | `""` | Path to a CSV file for recording results. Appends top/bottom results with columns: `query`, `direction`, `result`, `result_index`, `score`. |

## `bergson hessian` Flags (`HessianConfig`)

| Flag | Default | Description |
|---|---|---|
| `--method` | `kfac` | Hessian approximation method. Choices: `kfac`, `tkfac`, `shampoo`. |
| `--ev_correction` | `false` | Additionally compute eigenvalue correction. |
| `--hessian_dtype` | `auto` | Dtype for the Hessian approximation. Choices: `auto`, `bf16`, `fp16`, `fp32`. |
| `--use_dataset_labels` | `false` | Use dataset labels for empirical Fisher approximation. If false, model predictions are used. |

## `bergson trackstar` Flags (`TrackstarConfig`)

TrackStar accepts all shared flags plus score flags, and adds query-specific data flags and pipeline options.

**Query Data (prefixed with `--query.`):**

Accepts the same fields as `--data.*` (see [Data flags](#shared-flags-model-data--index-indexconfig)) but for the query dataset.

| Flag | Default | Description |
|---|---|---|
| `--query.dataset` | `NeelNanda/pile-10k` | Query dataset identifier. |
| `--query.split` | `train` | Query dataset split. |
| `--query.subset` | `None` | Query dataset subset. |
| `--query.prompt_column` | `text` | Query prompt column. |
| `--query.completion_column` | `""` | Query completion column. |
| `--query.conversation_column` | `""` | Query conversation column. |
| `--query.truncation` | `false` | Truncate long query documents. |
| `--query.format_template` | `""` | Query format template path. |

**Pipeline Options:**

| Flag | Default | Description |
|---|---|---|
| `--target_downweight_components` | `1000` | Number of gradient components to downweight via automatic lambda selection (§A.1.3 of Chang et al., 2024). |
| `--num_stats_sample_preconditioner` | `true` | Use `stats_sample_size` items (instead of the full dataset) to compute preconditioners. |

# Benchmarks

![CLI Benchmark](docs/benchmarks/cli_benchmark_NVIDIA_GH200_120GB.png)

See `benchmarks/` for scripts to reproduce and generate benchmarks on your own hardware.

# Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
pyright
```

We use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) for releases.

# Citation

If you found Bergson useful in your research, please cite us:

```bibtex
@software{bergson,
  author       = {Lucia Quirke and Nora Belrose and Louis Jaburi and William Li and David Johnston and Michael Mulet and Guillaume Martres and Goncalo Paulo},
  title        = {Bergson: Mapping out the "memory" of neural nets with data attribution},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18906967}
  url          = {https://doi.org/10.5281/zenodo.18906967}
}
```

# Support

If you have suggestions, questions, or would like to collaborate, please email lucia@eleuther.ai or drop us a line in the #data-attribution channel of the EleutherAI Discord!
