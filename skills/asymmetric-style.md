# Asymmetric Style Suppression Experiment

**Goal**: Test whether gradient-based data attribution can find semantically matching training examples when the query is in a different style than the training data.

This simulates a realistic scenario: your training data is mostly in one style (e.g., 95% formal/shakespeare), but users query in a different style (e.g., casual/pirate). Without intervention, gradient similarity is dominated by style—queries match training examples with similar style rather than similar content. This experiment evaluates strategies (preconditioners, PCA, gradient summing) to suppress style and recover semantic matching.

## Usage

There is no CLI interface for this experiment. It is run via the Python API.

```python
from examples.semantic.asymmetric import run_asymmetric_experiment, AsymmetricConfig

config = AsymmetricConfig(hf_dataset="EleutherAI/bergson-asymmetric-style")
results = run_asymmetric_experiment(config=config, base_path="runs/asymmetric_style")
```

**Semantic PCA ablation** (PCA from answer-only gradients instead of full gradients):

```bash
python scripts/semantic_pca_ablation.py
```

Tests whether computing the PCA style subspace from semantic (answer-only) gradients gives a cleaner subspace than full gradients. Builds separate semantic gradient indices, then compares results against the full-gradient PCA baseline.

## What this does

1. Creates asymmetric train/eval split:
   - Train: 95% shakespeare (dominant), 5% pirate (minority)
   - Eval: pirate style queries for facts only in shakespeare style in train
2. Tests whether gradient-based attribution can find semantic matches despite style mismatch
3. Compares strategies: baseline, preconditioners (R_between, H_eval, H_train, mixed), PCA projection, semantic-only eval

## Strategies Tested

### Baseline
- **no_precond**: Raw cosine similarity between query and training gradients. Expected to fail because style dominates the gradient representation.

### Preconditioners
Transform gradients by `g' = g @ H^(-1)` before computing similarity, downweighting certain directions.

- **R_between**: Computed from the difference between style means on **training data**: `delta = mean(shakespeare_train) - mean(pirate_train)`, then `R = delta @ delta.T`. This is a rank-1 matrix that captures the "style direction".

  *Dataset*: Training set (95% shakespeare, 5% pirate). The shakespeare mean is computed over ~950 samples, pirate mean over ~50 samples.

  Hypothesis: inverting this downweights the style axis, exposing semantic signal.

- **H_eval** (`eval_second_moment`): Second moment of eval gradients: `H = (1/n) * G_eval.T @ G_eval`. Hypothesis: directions that vary a lot in the eval set (which is all one style) might be style-related, so downweighting high-variance eval directions could help.

- **H_train** (`train_second_moment` / `index`): Second moment of training gradients: `H = (1/n) * G_train.T @ G_train`. This has theoretical grounding from influence functions: `g_eval @ H^{-1} @ g_train.T` approximates the change in eval loss from upweighting a training point (second-order Taylor expansion). So H_train is the "correct" similarity metric for influence-based attribution.

- **train_eval_mixed**: `H = α * H_train + (1-α) * H_eval`. Combines intuitions from both.

### Dimensionality Reduction (PCA)
- **PCA projection**: Uses separate full-gradient style indices (path configurable via `config.style_index_path`, default `runs/precond_comparison/`). Computes pairwise differences between corresponding dominant/minority gradients (same underlying fact, different style), then PCA on those difference vectors. Projects out the top-k components of this "style difference" subspace.

  **Important**: Eval facts are excluded from the PCA computation to prevent data leakage.

  Hypothesis: the difference `g_shakespeare(fact) - g_pirate(fact)` isolates pure style variation (content is held constant). The top PCs of these differences capture the dominant style directions. Projecting them out should remove style signal while preserving semantic content.

  K values are configurable via `config.pca_k_values` (default: 10, 100, 500, 1000). PCA is combined with both no preconditioning and H_train (`index`) preconditioning.

### Semantic-only Eval (Best Performing)
- **semantic_index**, **semantic_no_precond**, etc.: Transform eval data into Q&A format like `"Where does Paul Tilmouth work? Siemens"` and mask all gradients up to the `?`. This isolates the semantic content (answer tokens) from any style in the query. All preconditioners and PCA can be combined with the semantic prefix; these strategies are prefixed with `semantic_`.

### Optional Strategies (not in main table)

These are available via `run_asymmetric_experiment()` parameters but disabled by default:

- **majority_no_precond**: Query in the majority (shakespeare) style—no style mismatch. Control showing what's achievable when styles match. Enable with `include_majority_control=True`.
- **summed_eval**: For each query, compute gradients in both styles (pirate + shakespeare), then sum them. Tests whether style-specific components cancel out. Enable with `include_summed_eval=True`.
- **summed_loss**: Sum gradients from loss on multiple style variants. Enable with `include_summed_loss=True`.

## Instructions

### Run the experiment

```python
from examples.semantic.asymmetric import run_asymmetric_experiment, AsymmetricConfig

results = run_asymmetric_experiment(
    config=AsymmetricConfig(hf_dataset="EleutherAI/bergson-asymmetric-style"),
    base_path="runs/asymmetric_style",
    include_pca=True,
    include_second_moments=True,
    include_semantic_eval=True,
    damping_factor=0.1,
)
```

### Print existing results summary

```python
import json
from pathlib import Path

base_path = Path("runs/asymmetric_style")
with open(base_path / "experiment_results.json") as f:
    results = json.load(f)

sorted_results = sorted(results.items(), key=lambda x: -x[1]["top1_semantic"])
print(f"{'Strategy':<35} {'Top-1 Sem':<12} {'Top-5 Recall':<13} {'Top-1 Leak':<12}")
print("-" * 72)
for name, m in sorted_results:
    print(f"{name:<35} {m['top1_semantic']:<12.2%} {m['top5_semantic_recall']:<13.2%} {m['top1_leak']:<12.2%}")
```

## `run_asymmetric_experiment()` Parameters

```python
def run_asymmetric_experiment(
    config: AsymmetricConfig | None = None,
    base_path: Path | str = "runs/asymmetric_style",
    analysis_model: str | None = None,
    include_pca: bool = True,
    include_summed_loss: bool = True,
    include_second_moments: bool = True,
    include_majority_control: bool = True,
    include_summed_eval: bool = True,
    include_semantic_eval: bool = True,
    damping_factor: float = 0.1,
) -> dict[str, AsymmetricMetrics]
```

PCA k values and the style index path are configured via `AsymmetricConfig`:
- `config.pca_k_values`: Tuple of k values to sweep (default: `(10, 100, 500, 1000)`)
- `config.style_index_path`: Path to style-specific indices (default: `"runs/precond_comparison"`)

Example with custom PCA settings:

```python
config = AsymmetricConfig(
    hf_dataset="EleutherAI/bergson-asymmetric-style",
    pca_k_values=(10, 50, 200),           # custom sweep values
    style_index_path="runs/my_indices",    # custom style index location
)
results = run_asymmetric_experiment(config=config, base_path="runs/asymmetric_style")
```

## Cached Data

The experiment caches intermediate results to avoid recomputation:

```
runs/asymmetric_style/
├── data/
│   ├── train.hf               # Training set (95% shakespeare, 5% pirate)
│   ├── eval.hf                # Eval set (pirate style)
│   ├── eval_majority.hf       # Eval in majority style (control)
│   ├── eval_summed.hf         # Eval with summed gradients
│   └── rewrites/              # Additional style rewrites for ablations
├── index/                     # Training gradients
├── eval_grads/                # Eval gradients (minority style)
├── eval_grads_majority/       # Eval gradients (majority style)
├── preconditioners/           # Various preconditioner matrices
├── pca_subspace/              # Cached PCA style subspace components
├── scores_*/                  # Score matrices for each strategy
└── experiment_results.json    # Cached metrics summary
```

**What each cache level means:**
- `data/` - Dataset creation and Qwen rewording (~10-20 min)
- `index/` - bergson build for training gradients (~2 min)
- `eval_grads*/` - bergson build for eval gradients (~1 min each)
- `preconditioners/` - Preconditioner computation (~30 sec)
- `pca_subspace/` - PCA style subspace computation
- `scores_*/` - Score computation (~10 sec each)
- `experiment_results.json` - Metrics computed from scores

To recompute scores only:
```bash
rm -rf runs/asymmetric_style/scores_* runs/asymmetric_style/experiment_results.json
```

To recompute everything including data:
```bash
rm -rf runs/asymmetric_style/
```

## Key Metrics

- **Top-1 Semantic Accuracy**: Top match has same underlying fact (higher is better)
- **Top-5 Semantic Recall**: Any of top-5 matches has same underlying fact (higher is better)
- **Top-1 Style Leakage**: Top match is minority style (lower is better - means not style matching)

## Datasets & Models

The datasets and fine-tuned model for this experiment are available on Hugging Face:

- **Dataset**: [EleutherAI/bergson-asymmetric-style](https://huggingface.co/datasets/EleutherAI/bergson-asymmetric-style)
  - `train`: 13,500 samples (95% shakespeare, 5% pirate)
  - `eval`: 4,500 samples (pirate style queries)
  - `eval_majority_style`: 4,500 samples (shakespeare style control)
  - `eval_original_style`: 4,500 samples (unstyled)
  - `eval_pirate_style`: 4,500 samples (pirate style variant)

- **Model**: [EleutherAI/bergson-asymmetric-style-qwen3-8b-lora](https://huggingface.co/EleutherAI/bergson-asymmetric-style-qwen3-8b-lora)
  - LoRA adapter for Qwen/Qwen3-8B-Base
  - Used as the `analysis_model` for gradient collection

## Key Findings

| Strategy | Top-1 Semantic | Top-5 Recall | Style Leak | Notes |
|----------|---------------|-------------|------------|-------|
| semantic_pca_k100_index | 12.72% | 28.43% | 56.71% | **Best**: PCA k=100 + H_train + semantic |
| semantic_pca_k10_index | 9.79% | 20.78% | 65.31% | PCA k=10 + H_train + semantic |
| semantic_eval_second_moment | 9.34% | 21.17% | 61.73% | H_eval + semantic |
| semantic_index | 9.00% | 19.33% | 66.00% | H_train + semantic |
| semantic_train_eval_mixed | 8.90% | 20.68% | 64.76% | Mixed H + semantic |
| semantic_pca_projection_k100 | 7.01% | 17.30% | 54.67% | PCA k=100 + semantic (no precond) |
| eval_second_moment | 4.32% | 8.95% | 81.26% | H_eval (no semantic) |
| train_second_moment | 3.58% | 6.46% | 87.38% | H_train (no semantic) |
| r_between | 2.58% | 5.47% | 83.55% | Style-direction rank-1 precond |
| no_precond | 1.74% | 2.73% | 90.76% | Baseline: style dominates |

**Main insights**:
- Semantic masking (Q&A format, answer-only gradients) is the most impactful single intervention
- PCA projection (k=100) combined with H_train preconditioning and semantic masking gives the best results
- Preconditioners alone (without semantic masking) provide only marginal improvement over baseline
- Over-aggressive PCA (k=500, k=1000) removes too much signal and hurts performance
