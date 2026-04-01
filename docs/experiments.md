# Experiment Walkthroughs

This page provides walkthroughs for running bergson experiments. Skill files for reproducing the results are available in `skills/`.

## Asymmetric Style Suppression

This experiment evaluates various influence functions on a fact retrieval task where the query is in a different writing style to the training data.

This is analogous to a common use case where your query differs stylistically from the training corpus you want to query; for example, because the query comes from an evaluation set written in a multi-choice question format while the training data is sampled from a more general distribution. We are often only interested in examining how the "content" was learned, and not the style.

### Requirements

**Using existing HuggingFace artifacts**:
- GPU with ~24GB VRAM for the analysis model (Qwen3-8B with LoRA)
This option will pull [EleutherAI/bergson-asymmetric-style](https://huggingface.co/datasets/EleutherAI/bergson-asymmetric-style) and [EleutherAI/bergson-asymmetric-style-qwen3-8b-lora](https://huggingface.co/EleutherAI/bergson-asymmetric-style-qwen3-8b-lora) from Hugging Face.

**Using a regenerated dataset**:
- Qwen3-8B-Base for style rewording
- Additional disk space for intermediate datasets

### Quickstart

**Reproduce results with an AI agent**: Point Claude Code or another AI agent at `skills/asymmetric-style.md` for detailed instructions and options (`--recompute`, `--sweep-pca`, `--rewrite-ablation`, `--summary`).

**Reproduce results manually**:

```python
from examples.semantic.asymmetric import run_asymmetric_experiment, AsymmetricConfig

config = AsymmetricConfig(
    # Use pre-computed data
    hf_dataset="EleutherAI/bergson-asymmetric-style",
)

results = run_asymmetric_experiment(
    config=config,
    base_path="runs/asymmetric_style",
)

sorted_results = sorted(results.items(), key=lambda x: -x[1]["top1_semantic"])
print(f"{'Strategy':<35} {'Top-1 Sem':<12} {'Top-5 Recall':<13} {'Top-1 Leak':<12}")
print("-" * 72)
for name, m in sorted_results:
    print(f"{name:<35} {m['top1_semantic']:<12.2%} {m['top5_semantic_recall']:<13.2%} {m['top1_leak']:<12.2%}")
```

### Dataset Structure
The experiment creates train/eval splits with disjoint fact-style combinations:

- **Training set**: Each fact appears in exactly one style (95% shakespeare, 5% pirate)
- **Eval set**: Queries use the *opposite* style from training—facts that were trained in shakespeare are queried in pirate

This design means style leakage and semantic accuracy are mutually exclusive: if attribution finds a training example with matching style, it necessarily has the wrong fact (since that fact-style combo doesn't exist in training). The exception is the `majority_no_precond` control, which queries in the majority (shakespeare) style—here style and semantic matches align.

### Pipeline

The experiment (`run_asymmetric_experiment`) runs these steps:

1. **Create dataset** - Downloads from HuggingFace or generates locally with style rewording
2. **Build gradient index** - Collects gradients for all training samples using `bergson build`
3. **Collect eval gradients** - Computes gradients for eval queries
4. **Compute preconditioners** - Builds various preconditioner matrices (R_between, H_train, H_eval, PCA projection)
5. **Score and evaluate** - Computes similarity scores and metrics for each strategy

### Output Structure

```
runs/asymmetric_style/
├── data/
│   ├── train.hf               # Training set (HuggingFace Dataset)
│   ├── eval.hf                # Eval set (HuggingFace Dataset)
│   └── eval_majority.hf       # Eval in majority style (control)
├── index/                     # Training gradients (bergson index format)
├── eval_grads/                # Eval gradients
├── preconditioners/           # Preconditioner matrices (.pt files)
├── scores_*/                  # Score matrices for each strategy
└── experiment_results.json    # Metrics summary
```

### Key Metrics

- **Top-1 Semantic Accuracy**: Top match has same underlying fact (higher is better)
- **Top-5 Semantic Recall**: Any of top-5 matches has same underlying fact (higher is better)
- **Top-1 Style Leakage**: Top match is minority style (lower is better). Due to the disjoint partitioning, high leakage implies low semantic accuracy and vice versa.

### Strategies

The experiment compares multiple strategies for suppressing style and recovering semantic matching.

#### Baseline

- **no_precond**: Bare gradient cosine similarity. Expected to fail because style dominates.

#### Controls (alternative evaluation sets)

- **majority_no_precond**: Query in shakespeare style (the majority/dominant style). No style mismatch, so this is the upper bound—style and semantic matches align.
- **original_style_no_precond**: Eval set uses original (unstyled) facts instead of pirate style.
- **summed_majority_minority**: Eval gradients are the sum of pirate and shakespeare style gradients for each fact. Hypothesis: style-specific components cancel out.

#### Preconditioners

Without preconditioning, similarity is computed as cosine similarity of gradients:

```python
score(q, t) = cos(g_q, g_t)
           = (g_q · g_t) / (||g_q|| ||g_t||)
```

where `g_q` is the eval gradient and `g_t` is a training gradient (row vectors).

With a preconditioner matrix `H`, we transform the eval gradient before computing similarity:

```python
H_inv = (H + λI)^(-1)           # damped inverse
g_eval_precond = g_eval @ H_inv
g_eval_norm = g_eval_precond / ||g_eval_precond||
g_train_norm = g_train / ||g_train||
score(q, t) = g_eval_norm · g_train_norm
```

The unnormalized inner product `g_eval @ H^(-1) @ g_train.T` is the classic influence function formula. Preconditioning downweights directions where `H` has large eigenvalues.

- **index**: `H = G_train.T @ G_train` (training set second moment). This is the classic [influence function](https://arxiv.org/abs/1703.04730) formulation: a second-order Taylor approximation shows that the change in loss from upweighting a training point is proportional to `g_eval @ H^(-1) @ g_train.T`. Intuitively, H^(-1) gives each training point less credit for similarity in "common" directions (where many training points contribute) and more credit in rare/specific directions.

- **eval_second_moment**: `H = G_eval.T @ G_eval`. Since training gradients average to ~0 at convergence, directions where eval gradients deviate from zero will dominate `H_eval`. Preconditioning downweights these directions. If style causes systematic deviation in eval gradients (e.g., pirate queries all shift in a similar direction), this suppresses the style signal.

- **train_eval_mixed**: `H = α * H_train + (1-α) * H_eval`. Combines intuitions from both.

- **r_between**: `H = (μ_pirate - μ_shakespeare)(μ_pirate - μ_shakespeare)^T + λI` (computed on train set). A rank-1 matrix capturing the "style direction" directly. Preconditioning projects out this direction.

#### Dimensionality Reduction

- **pca_k{n}_index**: Compute PCA on the matrix of paired style differences (differences between matched facts, train set only). Project gradients onto the orthogonal complement of the top-n principal components, then precondition with the train set second moment matrix. This removes the dominant style directions while preserving semantic signal. The k values and style index path are configurable via `AsymmetricConfig.pca_k_values` and `AsymmetricConfig.style_index_path`.

#### Semantic-only Eval

- **semantic_index**, **semantic_no_precond**, etc.: Transform eval data into Q&A format like `"Where does Paul Tilmouth work? Siemens"` and mask all gradients up to the `?`. This isolates the semantic content (answer tokens) from any style in the query. Combined with preconditioning (`semantic_index`), this method achieves the best results by a significant margin.

### Running the Experiment

**With an AI agent**: Point Claude Code or another AI agent at `skills/asymmetric-style.md` for detailed instructions.

**Run the experiment**:

```python
from examples.semantic.asymmetric import run_asymmetric_experiment, AsymmetricConfig

config = AsymmetricConfig(
    hf_dataset="EleutherAI/bergson-asymmetric-style",  # Use pre-computed data
)

results = run_asymmetric_experiment(
    config=config,
    base_path="runs/asymmetric_style",
)
```

See `skills/asymmetric-style.md` for full parameter documentation.

### View Results

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
