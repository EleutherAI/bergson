# Bergson
This library enables you to trace the memory of deep neural nets with gradient-based data attribution techniques. We currently focus on TrackStar, as described in [Scalable Influence and Fact Tracing for Large Language Model Pretraining](https://arxiv.org/abs/2410.17413v3) by Chang et al. (2024), and also include support for several alternative influence functions. We plan to add support for [Magic](https://arxiv.org/abs/2504.16430) soon.

We view attribution as a counterfactual question: **_If we "unlearned" this training sample, how would the model's behavior change?_** This formulation ties attribution to some notion of what it means to "unlearn" a training sample. Here we focus on a very simple notion of unlearning: taking a gradient _ascent_ step on the loss with respect to the training sample.

## Core features

- Gradient store for serial queries. We provide collection-time gradient compression for efficient storage, and integrate with FAISS for fast KNN search over large stores.
- On-the-fly queries. Query gradients without disk I/O overhead via a single pass over a dataset with a set of precomputed query gradients.
  - Experiment with multiple query strategies based on [LESS](https://arxiv.org/pdf/2402.04333).
  - Ideal for compression-free gradients.
- Train‑time gradient collection. Capture gradients produced during training with a ~17% performance overhead.
- Scalable. We use [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html), BitsAndBytes, and other performance optimizations to support large models, datasets, and clusters.
- Integrated with HuggingFace Transformers and Datasets. We also support on-disk datasets in a variety of formats.
- Structured gradient views and per-attention head gradient collection. Bergson enables mechanistic interpretability via easy access to per‑module or per-attention head gradients.

# Announcements

**January 2026**
- [Experimental] Support distributing preconditioners across nodes and devices for VRAM-efficient computation through the GradientCollectorWithDistributedPreconditioners. If you would like this functionality exposed via the CLI please get in touch! https://github.com/EleutherAI/bergson/pull/100

**October 2025**
- Support bias parameter gradients in linear modules: https://github.com/EleutherAI/bergson/pull/54
- Support convolution modules: https://github.com/EleutherAI/bergson/pull/50
- Query datasets on-the-fly: https://github.com/EleutherAI/bergson/pull/47

**September 2025**
- Save per-head attention gradients: https://github.com/EleutherAI/bergson/pull/40
- Eigendecompose preconditioners: https://github.com/EleutherAI/bergson/pull/34
- Dr. GRPO-based loss gradients: https://github.com/EleutherAI/bergson/pull/35
- Choose between summing and averaging losses across tokens: https://github.com/EleutherAI/bergson/pull/36
- Save the order training data is seen in while using the gradient collector callback for HF's Trainer/SFTTrainer: https://github.com/EleutherAI/bergson/pull/40
  - Saving training gradients adds a ~17% wall clock overhead
- Improved static index build ETA accuracy: https://github.com/EleutherAI/bergson/pull/41
- Several small quality of life improvements for querying indexes: https://github.com/EleutherAI/bergson/pull/38

# Installation

```bash
pip install bergson
```

# Quickstart

```
bergson build runs/quickstart --model EleutherAI/pythia-14m --dataset NeelNanda/pile-10k --truncation --token_batch_size 4096
```

# Usage

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
bergson reduce <output_path> --model <model_name> --dataset <dataset_name> --method mean --unit_normalize
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

# Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

We use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) for releases.

# Support

If you have suggestions, questions, or would like to collaborate, please email lucia@eleuther.ai or drop us a line in the #data-attribution channel of the EleutherAI Discord!
