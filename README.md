# Bergson
Bergson is a python library which provides scalable, state-of-the-art data attribution methods for large language models. Data attribution methods estimate the effect on a behavior of interest of removing data points from a model's training corpus. We support [EK-FAC](https://arxiv.org/abs/2308.03296) (2023), [TrackStar](https://arxiv.org/abs/2410.17413v3) (2024), [SOURCE](https://arxiv.org/abs/2405.12186) (2024), [MAGIC](https://arxiv.org/abs/2504.16430) (2025), [ASTRA](https://arxiv.org/abs/2507.14740) (2025), gradient cosine similarity, and more.

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

Indicative performance of data attribution methods in the finetuning regime - see the [leaderboard](https://bergson.readthedocs.io/en/latest/leaderboard.html) for more methods.

[Linear datamodeling score](https://arxiv.org/abs/2303.14186) (LDS) is the accuracy of a method for producing global data rankings by influence. The query loss difference (QLD) shows how much model loss for a held-out query can be increased by retraining without the most highly ranked data by influence (here the top 1%), compared to a random removal baseline. The per-token QLD removes the loss terms of the top 1% of training tokens, ranked by the forward-mode equivalent of each method's scores.

| Method | QLD | LDS | Per-token QLD |
|:---|:---:|:---:|:---:|
| MAGIC | 0.100 [0.090, 0.112] | 0.931 [0.925, 0.936] | 1.375 [1.348, 1.403] |
| EK-FAC + ASTRA | 0.074 [0.063, 0.087] | 0.643 [0.624, 0.660] | 1.082 [1.047, 1.116] |
| Eigenvalue-corrected Shampoo | 0.071 [0.060, 0.082] | 0.517 [0.491, 0.539] | 0.922 [0.887, 0.957] |
| SOURCE (Adam, EK-FAC) | 0.071 [0.060, 0.084] | 0.473 [0.446, 0.498] | 0.908 [0.877, 0.942] |
| EK-FAC | 0.070 [0.058, 0.082] | 0.454 [0.426, 0.479] | 0.863 [0.833, 0.894] |
| BM25 | 0.062 [0.048, 0.076] | 0.220 [0.185, 0.252] | 0.677 [0.650, 0.704] |
| Semantic search ([Qwen3 8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)) | 0.049 [0.038, 0.061] | 0.132 [0.093, 0.169] | 0.008 [0.006, 0.010] |
| TrackStar (no optimizer) | 0.045 [0.036, 0.055] | 0.276 [0.252, 0.299] | 0.492 [0.467, 0.517] |
| TRAK (8-model ensemble) | 0.032 [0.024, 0.040] | 0.138 [0.111, 0.165] | 0.119 [0.112, 0.125] |
| Gradient cosine similarity | 0.021 [0.016, 0.027] | 0.156 [0.131, 0.181] | 0.217 [0.206, 0.229] |
| Activation similarity | 0.000 [-0.000, 0.001] | 0.110 [0.070, 0.149] | 0.028 [0.018, 0.039] |

Results with 95% confidence intervals for GPT-2 finetuned on 4 epochs of the WikiText corpus. Held-out loss dropped from 3.545 to 3.111 over training.

## Functionality

### Attribute through Training

`bergson magic` runs a powerful attribution method that backpropagates through the training process to compute the gradient of a model behavior loss with respect to a weighting placed on each training item. It is powered by our twice-differentiable trainer, which can also be called directly using `bergson train`.

**Note: unrolled differentiation efficacy is proportional to [metasmoothness](https://bergson.readthedocs.io/en/latest/magic.html#metasmoothness), which is low in some settings, including early pretraining steps. Check your run's estimated metasmoothness with `bergson metasmoothness`.**

`bergson approxunrolling` is an approximation of `bergson magic` that uses a handful of training checkpoints to run the multi-step SOURCE attribution pipeline. This is roughly equivalent to an influence function averaged over several checkpoints.

To build a train‑time gradient store, use our HF Trainer callback. This will incur a ~17% performance overhead.

### Attribute Post-Hoc

Bergson supports on-disk gradient stores and on-the-fly queries, and per-token and per-sequence attribution.

The CLI commands `bergson trackstar`, `bergson ekfac`, and `bergson trak` all orchestrate multi-step attribution recipes over a model checkpoint. `bergson trak` also supports [ensembling](https://arxiv.org/abs/2303.14186) over independently trained models.

At a lower level, you can build your own gradient store for efficient serial queries using `bergson build`. Collection-time gradient compression makes the store space-efficient, and a FAISS integration enables fast KNN search over large stores - see `bergson query`, or `Attributor` in the programmatic interface. For small queries and methods that don't use gradient compression (e.g., EK-FAC), you can score a dataset in a single pass using an in-memory query index of precomputed gradients. Dataset items may be scored using max, mean, and individual scoring strategies, enabling [LESS](https://arxiv.org/pdf/2402.04333)-style data filtering. See `bergson score` and `bergson build`.

Per-module and per-attention head gradients can be extracted from the store.

**Note: influence functions are sensitive to the Hessian approximation damping hyperparameter in tiny models. Untuned hyperparameters can result in a suboptimal linear datamodeling score.**

### Evaluate

Use `bergson validate` and `bergson recall` to compute LDS and recall@k metrics respectively.

# Getting Started

There are many example YAMLs in the `examples` directory, including various paper experiment replications. Use `bergson <yaml_path>` to run them. For example, to MAGIC-attribute a GPT-2 WikiText fine-tune:

```bash
bergson examples/magic/gpt2_wikitext_tiny.yaml
```

You can use the same fields used to specify experiments in the YAMLs to run experiments directly in the Bergson CLI. For example, to construct and query an on-disk index of randomly projected gradients from the CLI:

```bash
bergson build runs/index --model EleutherAI/pythia-14m --dataset NeelNanda/pile-10k --truncation --token_batch_size 4096 --projection_dim 16
bergson query --index runs/index --unit_norm
```

Or check out a notebook for programmatic usage:

| | Notebook | Description |
|---|----------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EleutherAI/bergson/blob/main/notebooks/poison_detection.ipynb) | **Poison Detection** | Detect poisoned training examples with gradient attribution (T4, ~5 min) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EleutherAI/bergson/blob/main/notebooks/style_ablation.ipynb) | **Style Ablation** | Suppress style to recover semantic matching (A100, ~20 min) |

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
