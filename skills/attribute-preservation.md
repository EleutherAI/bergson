# Attribute Preservation Experiment

**Goal**: Test whether style suppression preconditioners can remove stylistic signal from gradient embeddings while preserving the ability to match on semantic attributes (occupation, employer type, etc.).

The core challenge is that gradient-based data attribution tends to match based on surface-level features like writing style rather than underlying content. This experiment creates synthetic data with correlated attributes (e.g., scientists work at research labs, business people work at banks) and tests whether we can surgically remove style signal without damaging these attribute-based matching capabilities.

## Usage

```
/attribute-preservation [options]
```

Options:
- `--base-path PATH` - Output directory (default: runs/attribute_preservation)
- `--no-h-eval` - Skip H_eval preconditioner comparison
- `--no-majority` - Skip majority style control
- `--no-semantic` - Skip semantic-only eval (Q&A format)
- `--no-pca` - Skip PCA style projection strategies
- `--pca-k K` - Number of principal components for PCA (default: 100)
- `--recompute` - Clear cached results and recompute from scratch

## What this does

1. Creates a synthetic dataset with occupational clusters (scientists, business, creative)
2. Each cluster has correlated attributes (employers, universities, degrees, titles)
3. Styles are assigned by occupation (scientists→shakespeare, business→pirate, creative→shakespeare)
4. Eval set: scientists in "wrong" style (pirate) to test style suppression
5. Compares preconditioner strategies: none, R_between, H_eval
6. Majority control: scientists in matching style (shakespeare) as upper bound
7. Semantic-only eval: Gradients computed only from answer tokens (Q&A format) to isolate semantic content from style
8. PCA style projection: Projects eval gradients orthogonal to style directions learned from the asymmetric experiment

## Strategies Tested

### Baseline
- **no_precond**: Raw cosine similarity between query and training gradients. Expected to mostly match based on style (pirate queries → pirate training examples) rather than occupation.

### Preconditioners
Transform gradients by `g' = g @ H^(-1)` before computing similarity.

- **R_between**: Computed from training data style means: `delta = mean(shakespeare_grads) - mean(pirate_grads)`, then `R = delta @ delta.T`. This rank-1 matrix captures the "style direction" in gradient space.

  *Dataset*: Training set with style-occupation mapping:
  - shakespeare mean = scientists (400) + creatives (400) = 800 samples
  - pirate mean = business (400) = 400 samples

  *Caveat*: Because shakespeare mixes two occupations, the "style direction" is actually `(scientists + creatives)/2 - business`, which conflates style with occupation. This is meant to represent a situation where you can't rewrite scientist data in different styles, and have to work with different styles that already exist in the data. If you can rewrite, majority_no_precond may be the best option.

  Hypothesis: preconditioning with R^(-1) shrinks the style axis, allowing occupation signal to dominate.

- **H_eval**: Second moment of eval gradients: `H = (1/n) * G_eval.T @ G_eval`. Hypothesis: the eval set is all scientists in pirate style, so directions with high variance in eval might capture style-independent scientist features. Downweighting these could paradoxically help by normalizing the representation.

### Controls
- **majority_no_precond**: Scientists queried in shakespeare (their training style)—no style mismatch.

  ⚠️ **Data leak warning**: The eval facts are the same as training facts (100% overlap), just reworded separately. This means majority_no_precond is essentially matching the same semantic content with the same style, making it an inflated upper bound. It's useful for showing that style mismatch is the problem, but shouldn't be interpreted as the achievable accuracy for attribute matching with disjoint facts.

### Semantic-only Eval
- **semantic_index**, **semantic_no_precond**, **semantic_r_between**, **semantic_h_eval**: Instead of computing gradients from the full stylized text, compute gradients only from answer tokens using a Q&A format:
  - Question: "Where does {name} work?" (masked, no gradient)
  - Answer: "Fermilab" (gradient computed only here)

  This isolates semantic content from style by removing style tokens entirely from the gradient computation. Combined with preconditioners, this can significantly improve attribute matching.

  - **semantic_index**: Standard influence function approach (H_train preconditioner + semantic masking)

### PCA Style Projection
- **pca_k100**, **pca_k100_index**, **semantic_pca_k100**, **semantic_pca_k100_index**: Project eval gradients orthogonal to the top-k style directions before computing similarity.

  The PCA style subspace is loaded from the asymmetric style experiment (`runs/asymmetric_style/pca_subspace`), which contains style directions learned from 9000 fact pairs reworded in both pirate and shakespeare styles. Since there's zero overlap between the asymmetric experiment facts and attribute preservation eval facts, this introduces no data leak.

  - **semantic_pca_k100**: Best performing strategy - combines Q&A format with PCA projection

## Why This Experiment Matters

Previous experiments showed preconditioners have minimal effect on fact-level retrieval. But maybe they work for coarser attribute matching? This tests whether style suppression preserves the ability to match "scientists to scientists" even if it can't match "Alice's employer fact to Alice's employer fact".

## Instructions

### Run full experiment (using HuggingFace data)

The easiest way to run the experiment is using pre-generated data from HuggingFace:

```python
from examples.semantic.attribute_preservation import (
    run_attribute_preservation_experiment,
    AttributePreservationConfig,
)

# Use HF dataset - no local generation needed
config = AttributePreservationConfig(
    hf_dataset="EleutherAI/bergson-attribute-preservation",
)

results = run_attribute_preservation_experiment(
    config=config,
    base_path='runs/attribute_preservation',
    # analysis_model defaults to EleutherAI/bergson-asymmetric-style-qwen3-8b-lora
    include_h_eval=True,
    include_majority_control=True,
    include_semantic_eval=True,
    include_pca=True,  # Uses asymmetric experiment's PCA subspace
    pca_top_k=100,
)
```

### Run full experiment (generate locally)

To generate fresh data locally (requires Qwen model for rewording):

```python
from examples.semantic.attribute_preservation import run_attribute_preservation_experiment

results = run_attribute_preservation_experiment(
    base_path='runs/attribute_preservation',
    reword_model='Qwen/Qwen3-8B-Base',
    include_h_eval=True,
    include_majority_control=True,
    include_semantic_eval=True,
    include_pca=True,  # Requires asymmetric experiment to be run first
    pca_top_k=100,
)
```

## Cached Data

The experiment caches intermediate results to avoid recomputation:

```
runs/attribute_preservation/
├── data/
│   ├── base_train.hf          # Raw facts (no style)
│   ├── base_eval.hf           # Raw eval facts
│   ├── train_shakespeare.hf   # Reworded train (shakespeare)
│   ├── train_pirate.hf        # Reworded train (pirate)
│   ├── train.hf               # Combined styled training set
│   ├── eval_pirate.hf         # Eval in minority style
│   ├── eval.hf                # Final eval set
│   ├── eval_majority.hf       # Eval in majority style (control)
│   └── eval_with_qa.hf        # Eval with question/answer columns (for semantic eval)
├── index/                     # Training gradients (bergson build)
├── eval_grads/                # Eval gradients (minority style)
├── eval_grads_semantic/       # Eval gradients (semantic Q&A format)
├── eval_grads_majority/       # Eval gradients (majority style)
├── r_between/                 # R_between preconditioner
├── h_eval/                    # H_eval preconditioner
├── scores_no_precond/         # Score matrix (no preconditioner)
├── scores_r_between/          # Score matrix (R_between)
├── scores_h_eval/             # Score matrix (H_eval)
├── scores_*_question_answer/  # Score matrices for semantic eval
└── scores_majority_no_precond/ # Score matrix (majority control)
```

**What each cache level means:**
- `data/` - Regenerating requires re-running Qwen rewording (~10 min)
- `index/` - Regenerating requires re-running bergson build (~2 min)
- `eval_grads*/` - Regenerating requires re-running bergson build (~1 min each)
- `r_between/`, `h_eval/` - Preconditioner computation (~30 sec each)
- `scores_*/` - Score computation (~10 sec each)

If the user specifies `--recompute`, first delete cached data:
```bash
rm -rf runs/attribute_preservation/index runs/attribute_preservation/eval_grads* runs/attribute_preservation/scores_* runs/attribute_preservation/r_between runs/attribute_preservation/h_eval
```

To recompute everything including data rewording:
```bash
rm -rf runs/attribute_preservation/
```

## Key Metrics

- **Occupation Accuracy**: How often top-1 match has same occupation cluster (higher is better)
- **Style-Only Match**: Style matches but occupation doesn't (lower is better)
- **Trade-off**: Occ Acc - Style Only (higher is better)

## Datasets & Models

The dataset and fine-tuned model for this experiment are available on Hugging Face:

- **Dataset**: [EleutherAI/bergson-attribute-preservation](https://huggingface.co/datasets/EleutherAI/bergson-attribute-preservation)
  - `train`: 1,200 samples (scientists + business + creative occupations with correlated styles)
  - `eval`: 400 samples (scientists in pirate style - "wrong" style)
  - `eval_majority`: 400 samples (scientists in shakespeare style - control)

- **Model**: [EleutherAI/bergson-asymmetric-style-qwen3-8b-lora](https://huggingface.co/EleutherAI/bergson-asymmetric-style-qwen3-8b-lora)
  - LoRA adapter for Qwen/Qwen3-8B-Base
  - Used as the `analysis_model` for gradient collection

## Expected Output

A summary table comparing strategies:
```
Strategy                  Fact Acc     Occ Acc      Style Only   Trade-off
---------------------------------------------------------------------------
no_precond                0.25%        7.75%        89.75%       -82.00%
r_between                 0.50%        12.25%       84.00%       -71.75%
h_eval                    3.25%        16.25%       80.50%       -64.25%
majority_no_precond       6.75%        76.00%       23.25%       +52.75%  (⚠️ data leak)
semantic_index            1.25%        48.00%       47.00%       +1.00%
semantic_no_precond       0.75%        12.00%       86.00%       -74.00%
semantic_r_between        0.25%        30.00%       67.50%       -37.50%
semantic_h_eval           1.75%        23.50%       71.00%       -47.50%
pca_k100                  1.50%        16.50%       79.50%       -63.00%
pca_k100_index            4.25%        17.25%       80.75%       -63.50%
semantic_pca_k100         1.25%        59.50%       33.50%       +26.00%  ← BEST
semantic_pca_k100_index   2.25%        55.50%       40.00%       +15.50%
```

**Key findings**: The `semantic_pca_k100` combination achieves the best legitimate results, with 59.5% occupation accuracy, only 33.5% style leakage, and +26% trade-off. This combines semantic gradients (Q&A format) with PCA style projection using the asymmetric experiment's precomputed style subspace. The PCA subspace learned from a separate dataset (asymmetric experiment) transfers effectively to attribute preservation.

### LaTeX Table

```latex
\begin{table}[h]
\centering
\begin{tabular}{lccc}
\toprule
Strategy & Occ Acc & Style Only & Trade-off \\
\midrule
\multicolumn{4}{l}{\textit{Full stylized gradients}} \\
\midrule
no\_precond & 7.75\% & 89.75\% & -82.00\% \\
r\_between & 12.25\% & 84.00\% & -71.75\% \\
h\_eval & 16.25\% & 80.50\% & -64.25\% \\
pca\_k100 & 16.50\% & 79.50\% & -63.00\% \\
pca\_k100\_index & 17.25\% & 80.75\% & -63.50\% \\
\midrule
\multicolumn{4}{l}{\textit{Semantic-only gradients (Q\&A format)}} \\
\midrule
semantic\_no\_precond & 12.00\% & 86.00\% & -74.00\% \\
semantic\_h\_eval & 23.50\% & 71.00\% & -47.50\% \\
semantic\_r\_between & 30.00\% & 67.50\% & -37.50\% \\
semantic\_index & 48.00\% & 47.00\% & +1.00\% \\
semantic\_pca\_k100\_index & 55.50\% & 40.00\% & +15.50\% \\
semantic\_pca\_k100 & \textbf{59.50\%} & \textbf{33.50\%} & \textbf{+26.00\%} \\
\midrule
\multicolumn{4}{l}{\textit{Control}} \\
\midrule
majority\_no\_precond$^\dagger$ & 76.00\% & 23.25\% & +52.75\% \\
\bottomrule
\end{tabular}
\caption{Attribute preservation under style mismatch. Higher Occ Acc and lower Style Only is better. Trade-off = Occ Acc - Style Only. $^\dagger$Data leak: 100\% of eval facts overlap with training.}
\end{table}
```

**Key observations:**
- Full stylized gradients perform poorly (all negative trade-offs) because style dominates the similarity signal
- Semantic-only gradients (Q&A format) dramatically improve results by removing style from the query
- PCA style projection provides additional gains, with `semantic_pca_k100` achieving the best legitimate result (+26% trade-off)
- The control (`majority_no_precond`) shows inflated numbers due to 100% fact overlap with training
