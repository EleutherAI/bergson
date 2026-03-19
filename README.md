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
bergson score <output_path> --model <model_name> --dataset <dataset_name> --query_path <existing_index_path> --score mean
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

## Numerical Stability

Some models produce inconsistent per-example gradients when sequences of different lengths are batched together. This is caused by optimized SDPA attention backends (flash, memory-efficient) computing slightly different results depending on the padding length.

Use the built-in diagnostic to check your model:

```bash
bergson test_numerical_stability --model <model_name>
```

This automatically tests escalating configurations and reports exactly which flags (if any) you need. If your model fails the default test, add the recommended flags to your `build`/`score`/`trackstar` commands:

```bash
bergson build <output_path> --model <model_name> --force_math_sdp
# or if needed:
bergson build <output_path> --model <model_name> --force_math_sdp --precision fp32
```

### Performance impact

The overhead of `--force_math_sdp` varies by model. Benchmarked on A100-80GB with 500 documents from pile-10k:

| Model | Precision | `--force_math_sdp` | Build time | Overhead | Needs it? |
|-------|-----------|-------------------|------------|----------|-----------|
| Pythia-160M | bf16 | no | 30.2s | baseline | |
| Pythia-160M | bf16 | yes | 30.4s | +0.8% | |
| Pythia-160M | fp32 | no | 35.6s | baseline | |
| Pythia-160M | fp32 | yes | 39.6s | +11.5% | Yes (needs both) |
| OLMo-2-1B | bf16 | no | 43.1s | baseline | No |
| OLMo-2-1B | bf16 | yes | 53.6s | +24.5% | |
| OLMo-2-7B | bf16 | no | 105.5s | baseline | |
| OLMo-2-7B | bf16 | yes | 151.1s | +43.2% | Yes (`--force_math_sdp` only) |
| OLMo-2-7B | fp32 | no | 569.2s | baseline | |
| OLMo-2-7B | fp32 | yes | 603.6s | +6.1% | |

Not all models are affected — run `bergson test_numerical_stability` before enabling these flags to avoid unnecessary overhead.

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
  author       = {Lucia Quirke and Nora Belrose and Louis Jaburi and William Li and David Johnston and Michael Mulet and Guillaume Martres and Goncalo Paulo and Stella Biderman},
  title        = {Bergson: Mapping out the "memory" of neural nets with data attribution},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18906967},
  url          = {https://doi.org/10.5281/zenodo.18906967}
}
```

# Support

If you have suggestions, questions, or would like to collaborate, please email lucia@eleuther.ai or drop us a line in the #data-attribution channel of the EleutherAI Discord!
