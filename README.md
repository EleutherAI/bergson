# Bergson
Bergson is a python library which provides scalable, state-of-the-art data attribution methods for large language models. Data attribution methods estimate the effect on a behavior of interest of removing data points from a model's training corpus. We support [EK-FAC](https://arxiv.org/abs/2308.03296) (2023), [TrackStar](https://arxiv.org/abs/2410.17413v3) (2024), [SOURCE](https://arxiv.org/abs/2405.12186) (2024), [MAGIC](https://arxiv.org/abs/2504.16430) (2025), gradient cosine similarity, and more.

We try to make research and application straightforward. You can reproduce any of your runs with a single line, use a few CLI flags to save and re-use useful intermediate artifacts, do everything using just the CLI or just code, train models here or use existing ones, tune and evaluate your methods, and scale them up to multi-node runs with 70B+ parameters. More information is available in the [Bergson](https://bergson.readthedocs.io) docs.

<!-- To get the best out of our methods, check out our tuning guide. -->

# Installation

Bergson can be installed using pip:

```bash
pip install bergson
```

Or you can clone the repository and install locally:

```bash
git clone https://github.com/EleutherAI/bergson.git
cd bergson
pip install -e .
```

## Leaderboard

Indicative performance of data attribution methods in the finetuning regime - see [leaderboard](LEADERBOARD.md) for more methods.

[Linear datamodeling score](https://arxiv.org/abs/2303.14186) (LDS) is the accuracy of a method for producing global data rankings by influence. The query loss difference (QLD) shows how much model loss can be increased by retraining without the most highly ranked data by influence (here the top 1%), compared to a random removal baseline.

| Method | Proponent QLD [95% CI] | LDS [95% CI] |
|:---|:---:|:---:|
| MAGIC | 0.100 [0.090, 0.112] | 0.931 [0.925, 0.936] |
| Shampoo | 0.071 [0.060, 0.082] | 0.517 [0.491, 0.539] |
| EK-FAC | 0.070 [0.058, 0.082] | 0.454 [0.426, 0.479] |
| KFAC | 0.067 [0.056, 0.080] | 0.420 [0.391, 0.446] |
| BM25 | 0.062 [0.048, 0.076] | 0.220 [0.185, 0.252] |
| [Qwen3-Embedding-8B](https://huggingface.co/spaces/mteb/leaderboard) semantic search | 0.049 [0.038, 0.061] | 0.132 [0.093, 0.169] |
| TrackStar (no optimizer correction, projection 64) | 0.045 [0.036, 0.055] | 0.270 [0.240, 0.295] |
| SOURCE (Adam) | 0.024 [0.018, 0.030] | 0.154 [0.126, 0.181] |
| Gradient cosine similarity | 0.021 [0.016, 0.027] | 0.156 [0.131, 0.181] |
| TRAK (8-model ensemble) | 0.007 [0.005, 0.010] | 0.045 [0.015, 0.073] |
| Activation similarity | 0.000 [-0.000, 0.001] | 0.110 [0.070, 0.149] |

Results for GPT-2 finetuned on 4 epochs of the WikiText corpus. Held-out loss dropped from 3.545 to 3.111 over training.

## Functionality

### Attribute through Training

`bergson magic` runs a powerful attribution method that backpropagates through the training process to compute the gradient of a model behavior loss with respect to a weighting placed on each training item. It is powered by our twice-differentiable trainer, which can also be called directly using `bergson train`.

**Note: unrolled differentiation efficacy is proportional to [metasmoothness](https://bergson.readthedocs.io/en/latest/magic.html#metasmoothness). Untuned training hyperparameters can result in disappointing linear datamodeling scores. Check your run's estimated metasmoothness with `bergson metasmoothness`.**

`bergson approxunrolling` is an approximation of `bergson magic` that uses a handful of training checkpoints to run the multi-step SOURCE attribution pipeline. This is roughly equivalent to an influence function averaged over several checkpoints.

To build a train‑time gradient store, use our HF Trainer callback. This will incur a ~17% performance overhead.

### Attribute Post-Hoc

Bergson supports on-disk gradient stores and on-the-fly queries, and per-token and per-sequence attribution.

`bergson trackstar`, `bergson ekfac`, and `bergson trak` all orchestrate multi-step attribution recipes over a model checkpoint. `bergson trak` also supports [ensembling](https://arxiv.org/abs/2303.14186) over independently trained models.

At a lower level, you can build your own gradient store for efficient serial queries using `bergson build`. Collection-time gradient compression makes the store space-efficient, and a FAISS integration enables fast KNN search over large stores - see `bergson query`, or `Attributor` in the programmatic interface. For small queries and methods that don't use gradient compression (e.g., EK-FAC), you can score a dataset in a single pass using an in-memory query index of precomputed gradients. Dataset items may be scored using max, mean, and individual scoring strategies, enabling [LESS](https://arxiv.org/pdf/2402.04333)-style data filtering. See `bergson score` and `bergson build`.

Per-module and per-attention head gradients can be extracted from the store.

**Note: influence functions can be sensitive to both Hessian approximation inversion hyperparameters (in tiny models) and metasmoothness. Untuned hyperparameters can result in a disappointing linear datamodeling score.**

### Evaluate

Use `bergson validate` and `bergson recall` to compute LDS and recall@k metrics respectively.

# Examples

There are many example YAMLs in the `examples` directory. use `bergson <yaml_path>` to try them. For example, to MAGIC-attribute a GPT-2 WikiText fine-tune:

```bash
bergson examples/magic/gpt2_wikitext_tiny.yaml
```

Or check out a notebook for programmatic usage:

| | Notebook | Description |
|---|----------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EleutherAI/bergson/blob/colab-notebooks-v2/notebooks/poison_detection.ipynb) | **Poison Detection** | Detect poisoned training examples with gradient attribution (T4, ~5 min) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EleutherAI/bergson/blob/colab-notebooks-v2/notebooks/style_ablation.ipynb) | **Style Ablation** | Suppress style to recover semantic matching (A100, ~20 min) |

Construct and query an on-disk index of randomly projected gradients:

```bash
bergson build runs/index --model EleutherAI/pythia-14m --dataset NeelNanda/pile-10k --truncation --token_batch_size 4096 --projection_dim 16
bergson query --index runs/index --unit_norm
```

Collect TrackStar attribution scores for an I.I.D sample query:

```bash
bergson trackstar runs/trackstar --model EleutherAI/pythia-14m --query.dataset NeelNanda/pile-10k --data.dataset NeelNanda/pile-10k --data.truncation --token_batch_size 4096 --query.truncation --query.split "train[:20]"
```

# Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
pyright
```

We use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) for releases.

If you have multiple GPUs, you can run pytest in distributed mode: `pytest -n 8 --dist loadgroup`.

# Citation

If you found Bergson useful in your research, please cite us:

```bibtex
@misc{quirke2026bergsonopensourcelibrary,
      title={Bergson: An Open Source Library for Data Attribution},
      author={Lucia Quirke and Louis Jaburi and David Johnston and William Z. Li and Gonçalo Paulo and Guillaume Martres and Girish Gupta and Stella Biderman and Nora Belrose},
      year={2026},
      eprint={2606.11660},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.11660},
}
```

# Support

If you have suggestions, questions, or would like to collaborate, please email lucia@eleuther.ai or drop us a line in the #data-attribution channel of the EleutherAI Discord!
