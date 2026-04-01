"""Asymmetric style distribution experiment for style suppression validation.

This module creates datasets where semantic matches are only available in the dominant
style, forcing attribution to choose between style similarity and semantic similarity.
"""

from dataclasses import dataclass
from pathlib import Path

import ml_dtypes  # noqa: F401  # registers bfloat16 dtype with numpy
import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from bergson.config import IndexConfig
from examples.semantic.data import (
    HF_ANALYSIS_MODEL,
    load_experiment_data,
)


def _load_gradients_as_float(grads: np.memmap, name: str) -> np.ndarray:
    """Load a gradient field and convert from bfloat16 to float32.

    Args:
        grads: Structured gradient memmap.
        name: Field name to access.

    Returns:
        Float32 numpy array.
    """
    g = grads[name]
    # Gradients are stored as bfloat16 (2-byte void)
    if g.dtype == np.dtype("|V2"):
        g = g.view(ml_dtypes.bfloat16).astype(np.float32)
    return g


def _load_hf_dataset(path: Path | str) -> Dataset:
    """Load a HuggingFace dataset, unwrapping DatasetDict if needed."""
    ds = load_from_disk(str(path))
    if isinstance(ds, DatasetDict):
        ds = ds["train"]
    return ds


def _run_bergson_build(
    output_path: Path | str,
    model: str,
    dataset_path: Path | str,
    prompt_column: str = "fact",
    completion_column: str = "reworded",
    projection_dim: int = 16,
    skip_preconditioners: bool = True,
    label: str = "eval",
) -> None:
    """Run bergson build subprocess with standard arguments.

    Args:
        output_path: Where to write the gradient index.
        model: Model name/path for gradient collection.
        dataset_path: Path to the HuggingFace dataset.
        prompt_column: Column name for prompts.
        completion_column: Column name for completions.
        projection_dim: Random projection dimensionality.
        skip_preconditioners: Whether to skip preconditioner computation.
        label: Description for error messages (e.g. "eval", "majority eval").
    """
    import subprocess

    cmd = [
        "bergson",
        "build",
        str(output_path),
        "--model",
        model,
        "--dataset",
        str(dataset_path),
        "--drop_columns",
        "False",
        "--prompt_column",
        prompt_column,
        "--completion_column",
        completion_column,
        "--fsdp",
        "--projection_dim",
        str(projection_dim),
        "--token_batch_size",
        "6000",
    ]
    if skip_preconditioners:
        cmd.append("--skip_preconditioners")

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"bergson build for {label} failed")
    print(result.stdout)


def _compute_metrics_from_scores(
    scores: np.ndarray,
    train_ds: Dataset,
    eval_ds: Dataset,
    minority_style: str,
) -> "AsymmetricMetrics":
    """Compute AsymmetricMetrics from a score matrix.

    Args:
        scores: Score matrix of shape (n_eval, n_train).
        train_ds: Training dataset with style/identifier/field columns.
        eval_ds: Eval dataset with identifier/field columns.
        minority_style: Name of the minority style for leakage computation.

    Returns:
        AsymmetricMetrics dataclass.
    """
    train_styles = train_ds["style"]  # type: ignore[index]
    train_identifiers = train_ds["identifier"]  # type: ignore[index]
    train_fields = train_ds["field"]  # type: ignore[index]
    eval_identifiers = eval_ds["identifier"]  # type: ignore[index]
    eval_fields = eval_ds["field"]  # type: ignore[index]

    n_eval = len(eval_ds)
    top_indices = np.argsort(-scores, axis=1)[:, :10]

    semantic_top1 = 0
    semantic_top5 = 0
    semantic_top10 = 0
    style_leak_top1 = 0
    style_leak_top5 = 0.0
    style_leak_top10 = 0.0
    subject_top1 = 0
    field_top1 = 0

    for i in range(n_eval):
        query_identifier = eval_identifiers[i]
        query_field = eval_fields[i]
        top_k_idx = top_indices[i]

        # Check semantic matching (same identifier AND field)
        for k, idx in enumerate(top_k_idx):
            if (
                train_identifiers[idx] == query_identifier
                and train_fields[idx] == query_field
            ):
                if k == 0:
                    semantic_top1 += 1
                if k < 5:
                    semantic_top5 += 1
                    break
                if k < 10:
                    semantic_top10 += 1
                    break

        # Check style leakage
        if train_styles[top_k_idx[0]] == minority_style:
            style_leak_top1 += 1

        top5_minority = sum(
            1 for idx in top_k_idx[:5] if train_styles[idx] == minority_style
        )
        style_leak_top5 += top5_minority / 5

        top10_minority = sum(
            1 for idx in top_k_idx[:10] if train_styles[idx] == minority_style
        )
        style_leak_top10 += top10_minority / 10

        # Check attribute matching for top-1
        top1_idx = top_k_idx[0]
        if train_identifiers[top1_idx] == query_identifier:
            subject_top1 += 1
        if train_fields[top1_idx] == query_field:
            field_top1 += 1

    return AsymmetricMetrics(
        top1_semantic_accuracy=semantic_top1 / n_eval,
        top5_semantic_recall=semantic_top5 / n_eval,
        top10_semantic_recall=semantic_top10 / n_eval,
        top1_style_leakage=style_leak_top1 / n_eval,
        top5_style_leakage=style_leak_top5 / n_eval,
        top10_style_leakage=style_leak_top10 / n_eval,
        top1_subject_accuracy=subject_top1 / n_eval,
        top1_field_accuracy=field_top1 / n_eval,
    )


@dataclass
class AsymmetricConfig:
    """Configuration for asymmetric style experiment."""

    dominant_style: str = "shakespeare"
    minority_style: str = "pirate"
    dominant_ratio: float = 0.95  # Fraction of training in dominant style
    exclusive_ratio: float = 0.5  # Fraction of facts exclusive to dominant style
    seed: int = 42
    # HuggingFace dataset repo. If set, skips local generation and downloads from HF.
    hf_dataset: str | None = None
    # Template split for train/test segregation (only used for local generation)
    # Train uses templates < cutoff, eval majority uses templates >= cutoff
    train_template_cutoff: int = 5
    # Path to style-specific indices (pirate/shakespeare) for PCA and summed loss
    style_index_path: str = "runs/precond_comparison"
    # PCA k values to sweep. First value is used for the initial PCA computation;
    # all values are swept in the semantic eval section.
    pca_k_values: tuple[int, ...] = (10, 100, 500, 1000)


def create_asymmetric_dataset(
    config: AsymmetricConfig,
    output_dir: Path | str,
) -> tuple[Dataset, Dataset]:
    """Create asymmetric training and evaluation datasets.

    Splits facts into:
    - Exclusive facts: only appear in dominant style (for testing semantic matching)
    - Shared facts: appear in both styles (for style ratio control)

    For train/test segregation:
    - Training uses templates < train_template_cutoff (default: 0-4)
    - Eval majority style control uses templates >= cutoff (default: 5+)
    This ensures no exact text overlap between train and eval majority control.

    Args:
        config: Experiment configuration.
        output_dir: Directory to save datasets.

    Returns:
        (train_dataset, eval_dataset) tuple.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.hf"
    eval_path = output_dir / "eval.hf"

    # Return cached if exists
    if train_path.exists() and eval_path.exists():
        print(f"Loading cached datasets from {output_dir}")
        train_cached = _load_hf_dataset(train_path)
        eval_cached = _load_hf_dataset(eval_path)
        return train_cached, eval_cached

    # Load original facts to get metadata columns
    original = _load_hf_dataset("data/facts_dataset.hf")
    fact_to_meta = {row["fact"]: row for row in original}  # type: ignore[index]

    # Load style-specific datasets (Qwen only for consistency)
    style_datasets = {
        "shakespeare": load_from_disk(
            "data/facts_dataset_shakespeare-Qwen3-8B-Base.hf"
        ),
        "pirate": load_from_disk("data/facts_dataset_pirate-Qwen3-8B-Base.hf"),
    }
    for name in style_datasets:
        if isinstance(style_datasets[name], DatasetDict):
            style_datasets[name] = style_datasets[name]["train"]

        # Add back metadata columns from original
        ds = style_datasets[name]
        for col in original.column_names:
            if col not in ds.column_names:
                restored_col = [fact_to_meta[row["fact"]][col] for row in ds]  # type: ignore[index]
                ds = ds.add_column(col, restored_col)  # type: ignore[union-attr]
        style_datasets[name] = ds

    dominant_ds = style_datasets[config.dominant_style]
    minority_ds = style_datasets[config.minority_style]

    # Get unique (identifier, field) pairs - these represent underlying semantic facts
    # Each pair has multiple templates (different surface forms of the same fact)
    semantic_facts = list({(row["identifier"], row["field"]) for row in original})  # type: ignore[index]
    n_semantic_facts = len(semantic_facts)

    # Split into exclusive (dominant-only) and shared by semantic fact
    rng = np.random.default_rng(config.seed)
    rng.shuffle(semantic_facts)

    n_exclusive = int(n_semantic_facts * config.exclusive_ratio)
    exclusive_semantic_facts = set(semantic_facts[:n_exclusive])
    shared_semantic_facts = set(semantic_facts[n_exclusive:])

    print(f"Total unique semantic facts (identifier, field pairs): {n_semantic_facts}")
    print(f"Exclusive to {config.dominant_style}: {len(exclusive_semantic_facts)}")
    print(f"Shared between styles: {len(shared_semantic_facts)}")
    print(f"Template cutoff for train/eval split: {config.train_template_cutoff}")

    # Build training set with template filtering
    # 1. Dominant style: only templates < cutoff
    # (to reserve rest for eval majority control)
    train_dominant_indices = [
        i
        for i, row in enumerate(dominant_ds)
        if row["template"] < config.train_template_cutoff  # type: ignore[index]
    ]
    train_dominant = dominant_ds.select(train_dominant_indices)  # type: ignore[union-attr]

    # 2. Minority style only for shared facts
    # (any template since minority eval is different)
    minority_shared_indices = [
        i
        for i, row in enumerate(minority_ds)
        if (row["identifier"], row["field"]) in shared_semantic_facts  # type: ignore[index]
        and row["template"] < config.train_template_cutoff  # type: ignore[index]
    ]
    train_minority = minority_ds.select(minority_shared_indices)  # type: ignore[union-attr]

    # Add style column
    train_dominant = train_dominant.add_column(
        "style", [config.dominant_style] * len(train_dominant)
    )
    train_minority = train_minority.add_column(
        "style", [config.minority_style] * len(train_minority)
    )

    # Combine and shuffle
    train_ds = concatenate_datasets([train_dominant, train_minority])
    train_ds = train_ds.shuffle(seed=config.seed)

    print("\nTraining set composition:")
    print(f"  {config.dominant_style}: {len(train_dominant)} samples")
    print(f"  {config.minority_style}: {len(train_minority)} samples")
    print(f"  Total: {len(train_ds)} samples")
    print(f"  Dominant ratio: {len(train_dominant) / len(train_ds):.2%}")

    # Build eval set: query exclusive facts in minority style
    # Use templates >= cutoff to ensure no overlap with train
    # These facts don't exist in minority style in training, so the model
    # must use semantic matching (not style matching) to find them
    eval_minority_indices = [
        i
        for i, row in enumerate(minority_ds)
        if (row["identifier"], row["field"]) in exclusive_semantic_facts  # type: ignore[index]
        and row["template"] >= config.train_template_cutoff  # type: ignore[index]
    ]
    eval_ds = minority_ds.select(eval_minority_indices)  # type: ignore[union-attr]
    eval_ds = eval_ds.add_column("style", [config.minority_style] * len(eval_ds))

    # Add expected_match_style to indicate where the ground truth is
    eval_ds = eval_ds.add_column(
        "expected_match_style", [config.dominant_style] * len(eval_ds)
    )

    print("\nEval set:")
    print(f"  Queries in {config.minority_style} style: {len(eval_ds)}")
    print(f"  Ground truth only in {config.dominant_style} style")
    print(
        f"  Using templates >= {config.train_template_cutoff} (no overlap with train)"
    )

    # Save datasets
    train_ds.save_to_disk(str(train_path))
    eval_ds.save_to_disk(str(eval_path))
    print(f"\nSaved datasets to {output_dir}")

    return train_ds, eval_ds


def create_asymmetric_index(
    config: AsymmetricConfig,
    base_path: Path | str,
    analysis_model: str | None = None,
) -> Path:
    """Create bergson index for asymmetric training set.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        analysis_model: Model to use for gradient collection.
            Defaults to HF_ANALYSIS_MODEL.

    Returns:
        Path to the created index.
    """

    if analysis_model is None:
        analysis_model = HF_ANALYSIS_MODEL

    base_path = Path(base_path)
    data_path = base_path / "data"
    index_path = base_path / "index"

    # Load or create dataset
    if config.hf_dataset:
        # Download from HuggingFace and save locally for bergson
        print(f"Loading dataset from HuggingFace: {config.hf_dataset}")
        dataset_dict = load_experiment_data(hf_repo=config.hf_dataset)
        data_path.mkdir(parents=True, exist_ok=True)
        for split_name, split_ds in dataset_dict.items():
            split_path = data_path / f"{split_name}.hf"
            if not split_path.exists():
                split_ds.save_to_disk(str(split_path))
                print(f"  Saved {split_name} to {split_path}")
    else:
        # Generate locally
        create_asymmetric_dataset(config, data_path)

    if index_path.exists():
        print(f"Index already exists at {index_path}, skipping...")
        return index_path

    _run_bergson_build(
        index_path,
        model=analysis_model,
        dataset_path=data_path / "train.hf",
        skip_preconditioners=False,
        label="index",
    )

    return index_path


def score_asymmetric_eval(
    config: AsymmetricConfig,
    base_path: Path | str,
    preconditioner_name: str | None = None,
    damping_factor: float = 0.1,
    regularizer_name: str | None = None,
    eval_prompt_column: str = "fact",
    eval_completion_column: str = "reworded",
) -> np.ndarray:
    """Score eval queries against training index.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        preconditioner_name: Name of preconditioner subdirectory
            (None for no precond).
        damping_factor: Damping factor for matrix inversion
            (default: 0.1).
        regularizer_name: Name of preconditioner to use as
            regularizer instead of identity. If provided, computes
            inv(H + damping_factor * H_regularizer). Useful for
            regularizing rank-deficient preconditioners like
            r_between with a well-conditioned matrix like
            H_train or H_eval.
        eval_prompt_column: Column to use as prompt for eval
            gradients (default: "fact").
        eval_completion_column: Column to use as completion for
            eval gradients (default: "reworded"). Set to
            "question"/"answer" for semantic-only attribution
            where gradients only come from the answer tokens.

    Returns:
        Score matrix of shape (n_eval, n_train).
    """
    import json

    import torch
    from tqdm import tqdm

    from bergson.data import load_gradients
    from bergson.gradients import GradientProcessor
    from bergson.utils.math import damped_psd_power

    base_path = Path(base_path)
    index_path = base_path / "index"
    data_path = base_path / "data"

    # Determine output path
    # (include damping factor, regularizer, and eval columns in cache key)
    damping_suffix = f"_d{damping_factor:.0e}" if damping_factor != 0.1 else ""
    reg_suffix = f"_reg_{regularizer_name}" if regularizer_name else ""
    # Add eval column suffix if not using default columns
    eval_col_suffix = ""
    if eval_prompt_column != "fact" or eval_completion_column != "reworded":
        eval_col_suffix = f"_{eval_prompt_column}_{eval_completion_column}"
    if preconditioner_name:
        scores_path = (
            base_path / f"scores_{preconditioner_name}"
            f"{damping_suffix}{reg_suffix}{eval_col_suffix}"
        )
        precond_path = base_path / preconditioner_name
    else:
        scores_path = (
            base_path
            / f"scores_no_precond{damping_suffix}{reg_suffix}{eval_col_suffix}"
        )
        precond_path = None

    # Return cached if exists
    if (scores_path / "scores.npy").exists():
        print(f"Loading cached scores from {scores_path}")
        return np.load(scores_path / "scores.npy")

    scores_path.mkdir(parents=True, exist_ok=True)

    # Load train and eval datasets
    train_ds = _load_hf_dataset(data_path / "train.hf")
    eval_ds = _load_hf_dataset(data_path / "eval.hf")

    n_train = len(train_ds)
    n_eval = len(eval_ds)

    print(f"Scoring {n_eval} eval queries against {n_train} train samples")

    # Load train gradients
    print("Loading train gradients...")
    train_grads = load_gradients(index_path, structured=True)

    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Load preconditioner if specified
    h_inv = {}
    if precond_path and (precond_path / "preconditioners.pth").exists():
        # Load regularizer preconditioner if specified
        reg_proc = None
        if regularizer_name:
            reg_path = base_path / regularizer_name
            if (reg_path / "preconditioners.pth").exists():
                print(f"Loading regularizer from {reg_path}")
                reg_proc = GradientProcessor.load(reg_path)
            else:
                print(
                    f"Warning: regularizer {regularizer_name} not found at {reg_path}"
                )

        print(f"Loading preconditioner from {precond_path} (damping={damping_factor})")
        proc = GradientProcessor.load(precond_path)
        device = torch.device("cuda:0")
        for name in tqdm(module_names, desc="Computing H^(-1)"):
            H = proc.preconditioners[name].to(device=device)
            regularizer = None
            if reg_proc is not None and name in reg_proc.preconditioners:
                regularizer = reg_proc.preconditioners[name].to(device=device)
            h_inv[name] = damped_psd_power(
                H, power=-1, damping_factor=damping_factor, regularizer=regularizer
            )

    # Concatenate train gradients
    print("Preparing train gradients...")
    train_grad_list = []
    for name in tqdm(module_names, desc="Loading train grads"):
        g = _load_gradients_as_float(train_grads, name)
        train_grad_list.append(torch.from_numpy(g))
    train_grad_tensor = torch.cat(train_grad_list, dim=1)

    # Unit normalize train grads (as index)
    train_norms = train_grad_tensor.norm(dim=1, keepdim=True)
    train_grad_tensor = train_grad_tensor / (train_norms + 1e-8)
    train_grad_tensor = train_grad_tensor.cuda()

    # For eval, we need to compute gradients on the fly
    # Use bergson to compute eval gradients with same projection
    print("Computing eval gradients...")

    # Use different cache path based on eval columns
    if eval_prompt_column == "fact" and eval_completion_column == "reworded":
        eval_grads_path = base_path / "eval_grads"
    else:
        eval_grads_path = (
            base_path / f"eval_grads_{eval_prompt_column}_{eval_completion_column}"
        )

    if not eval_grads_path.exists():
        index_cfg = IndexConfig.load_yaml(index_path / "index_config.yaml")

        _run_bergson_build(
            eval_grads_path,
            model=index_cfg.model,
            dataset_path=data_path / "eval.hf",
            prompt_column=eval_prompt_column,
            completion_column=eval_completion_column,
            projection_dim=index_cfg.projection_dim or 16,
            label="eval",
        )

    # Load eval gradients
    eval_grads = load_gradients(eval_grads_path, structured=True)
    eval_grad_list = []
    for name in tqdm(module_names, desc="Loading eval grads"):
        g = torch.from_numpy(_load_gradients_as_float(eval_grads, name))
        if h_inv:
            # Apply preconditioning
            g = (g.cuda() @ h_inv[name]).cpu()
        eval_grad_list.append(g)
    eval_grad_tensor = torch.cat(eval_grad_list, dim=1)

    # Unit normalize eval grads (as query)
    eval_norms = eval_grad_tensor.norm(dim=1, keepdim=True)
    eval_grad_tensor = eval_grad_tensor / (eval_norms + 1e-8)
    eval_grad_tensor = eval_grad_tensor.cuda()

    # Compute scores: eval @ train.T gives (n_eval, n_train)
    print("Computing scores...")
    scores = (eval_grad_tensor @ train_grad_tensor.T).cpu().numpy()

    # Save scores
    np.save(scores_path / "scores.npy", scores)
    print(f"Saved scores to {scores_path}")

    return scores


@dataclass
class AsymmetricMetrics:
    """Metrics for asymmetric style suppression experiment."""

    # Semantic accuracy (same subject AND field = same underlying fact)
    top1_semantic_accuracy: float  # Top-1 same subject AND field
    top5_semantic_recall: float  # Any of top-5 same subject AND field
    top10_semantic_recall: float  # Any of top-10 same subject AND field

    # Style leakage (lower is better - means not matching on style)
    top1_style_leakage: float  # Top-1 is same (minority) style
    top5_style_leakage: float  # Fraction of top-5 in same (minority) style
    top10_style_leakage: float  # Fraction of top-10 in same (minority) style

    # Breakdown by attribute
    top1_subject_accuracy: float  # Top-1 same subject
    top1_field_accuracy: float  # Top-1 same field type


def compute_asymmetric_metrics(
    config: AsymmetricConfig,
    base_path: Path | str,
    preconditioner_name: str | None = None,
    damping_factor: float = 0.1,
    regularizer_name: str | None = None,
    eval_prompt_column: str = "fact",
    eval_completion_column: str = "reworded",
) -> AsymmetricMetrics:
    """Compute metrics for asymmetric style suppression.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        preconditioner_name: Name of preconditioner to use.
        damping_factor: Damping factor for matrix inversion
            (default: 0.1).
        regularizer_name: Name of preconditioner to use as
            regularizer instead of identity.
        eval_prompt_column: Column to use as prompt for eval
            gradients (default: "fact").
        eval_completion_column: Column to use as completion for
            eval gradients (default: "reworded"). Set to
            "question"/"answer" for semantic-only attribution.

    Returns:
        AsymmetricMetrics dataclass.
    """
    base_path = Path(base_path)
    data_path = base_path / "data"

    # Load datasets
    train_ds = _load_hf_dataset(data_path / "train.hf")
    eval_ds = _load_hf_dataset(data_path / "eval.hf")

    # Load scores
    scores = score_asymmetric_eval(
        config,
        base_path,
        preconditioner_name,
        damping_factor=damping_factor,
        regularizer_name=regularizer_name,
        eval_prompt_column=eval_prompt_column,
        eval_completion_column=eval_completion_column,
    )

    return _compute_metrics_from_scores(
        scores, train_ds, eval_ds, config.minority_style
    )


def print_metrics(metrics: AsymmetricMetrics, name: str) -> None:
    """Print metrics in a formatted way."""
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {name}")
    print("=" * 60)

    print("\nSemantic Accuracy (higher is better):")
    print(f"  Top-1 accuracy:  {metrics.top1_semantic_accuracy:.2%}")
    print(f"  Top-5 recall:    {metrics.top5_semantic_recall:.2%}")
    print(f"  Top-10 recall:   {metrics.top10_semantic_recall:.2%}")

    print("\nStyle Leakage (lower is better):")
    print(f"  Top-1 leakage:   {metrics.top1_style_leakage:.2%}")
    print(f"  Top-5 leakage:   {metrics.top5_style_leakage:.2%}")
    print(f"  Top-10 leakage:  {metrics.top10_style_leakage:.2%}")

    print("\nAttribute Breakdown (Top-1):")
    print(f"  Same subject:    {metrics.top1_subject_accuracy:.2%}")
    print(f"  Same field:      {metrics.top1_field_accuracy:.2%}")


def compute_style_preconditioner(
    base_path: Path | str,
    config: AsymmetricConfig,
) -> Path:
    """Compute R_between preconditioner that isolates the style direction.

    This creates a rank-1 preconditioner from the difference in style means.
    When used for scoring, this should downweight the style direction.

    Args:
        base_path: Base path for experiment outputs.
        config: Experiment configuration.

    Returns:
        Path to the preconditioner.
    """
    import json

    import torch
    from tqdm import tqdm

    from bergson.data import load_gradients
    from bergson.gradients import GradientProcessor

    base_path = Path(base_path)
    index_path = base_path / "index"
    data_path = base_path / "data"
    output_path = base_path / "r_between"

    if (output_path / "preconditioners.pth").exists():
        print(f"Loading cached R_between from {output_path}")
        return output_path

    print("Computing R_between preconditioner from style means...")

    # Load training data and gradients
    train_ds = _load_hf_dataset(data_path / "train.hf")

    train_styles = train_ds["style"]  # type: ignore[index]
    train_grads = load_gradients(index_path, structured=True)

    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Separate indices by style
    dominant_indices = [
        i for i, s in enumerate(train_styles) if s == config.dominant_style
    ]
    minority_indices = [
        i for i, s in enumerate(train_styles) if s == config.minority_style
    ]

    print(f"  {config.dominant_style}: {len(dominant_indices)} samples")
    print(f"  {config.minority_style}: {len(minority_indices)} samples")

    # Load a processor to get metadata
    base_proc = GradientProcessor.load(index_path)

    # Compute per-module rank-1 preconditioners
    between_precs = {}
    print(f"  Computing per-module R_between for {len(module_names)} modules...")

    for name in tqdm(module_names):
        g_all = torch.from_numpy(_load_gradients_as_float(train_grads, name))

        # Get style-specific gradients
        g_dominant = g_all[dominant_indices]
        g_minority = g_all[minority_indices]

        # Compute means
        mu_dominant = g_dominant.mean(dim=0)
        mu_minority = g_minority.mean(dim=0)

        # Style direction
        delta = mu_dominant - mu_minority

        # Rank-1 preconditioner: outer product
        between_precs[name] = torch.outer(delta, delta)

    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    between_proc = GradientProcessor(
        normalizers=base_proc.normalizers,
        preconditioners=between_precs,
        preconditioners_eigen={},
        projection_dim=base_proc.projection_dim,
        projection_type=base_proc.projection_type,
        include_bias=base_proc.include_bias,
    )
    between_proc.save(output_path)
    print(f"Saved R_between preconditioner to {output_path}")

    return output_path


def score_asymmetric_eval_with_pca_projection(
    config: AsymmetricConfig,
    base_path: Path | str,
    style_subspace: dict[str, tuple],
    top_k: int = 10,
    preconditioner_name: str | None = None,
    damping_factor: float = 0.1,
    eval_prompt_column: str = "fact",
    eval_completion_column: str = "reworded",
) -> np.ndarray:
    """Score eval queries using PCA projection to remove style direction.

    Instead of using matrix-inverse preconditioning, this projects eval gradients
    onto the orthogonal complement of the style subspace before computing scores.
    Can optionally combine with a preconditioner applied after projection.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        style_subspace: Dictionary from compute_pca_style_subspace().
        top_k: Number of principal components used (for cache naming).
        preconditioner_name: Optional preconditioner to apply after projection.
        damping_factor: Damping factor for matrix inversion
            (default: 0.1).
        eval_prompt_column: Column to use as prompt for eval
            gradients (default: "fact").
        eval_completion_column: Column to use as completion for
            eval gradients (default: "reworded"). Set to
            "question"/"answer" for semantic-only attribution.

    Returns:
        Score matrix of shape (n_eval, n_train).
    """
    import json

    import torch
    from tqdm import tqdm

    from bergson.data import load_gradients
    from bergson.gradients import GradientProcessor
    from bergson.utils.math import damped_psd_power

    from .preconditioners import project_orthogonal_to_style_subspace

    base_path = Path(base_path)
    index_path = base_path / "index"
    data_path = base_path / "data"

    # Build cache path including preconditioner, damping factor, and eval columns
    damping_suffix = f"_d{damping_factor:.0e}" if damping_factor != 0.1 else ""
    eval_col_suffix = ""
    if eval_prompt_column != "fact" or eval_completion_column != "reworded":
        eval_col_suffix = f"_{eval_prompt_column}_{eval_completion_column}"
    if preconditioner_name:
        scores_path = (
            base_path / f"scores_pca_k{top_k}_{preconditioner_name}"
            f"{damping_suffix}{eval_col_suffix}"
        )
        precond_path = base_path / preconditioner_name
    else:
        scores_path = (
            base_path / f"scores_pca_k{top_k}{damping_suffix}{eval_col_suffix}"
        )
        precond_path = None

    # Return cached if exists
    if (scores_path / "scores.npy").exists():
        print(f"Loading cached scores from {scores_path}")
        return np.load(scores_path / "scores.npy")

    scores_path.mkdir(parents=True, exist_ok=True)

    # Load train and eval datasets
    train_ds = _load_hf_dataset(data_path / "train.hf")
    eval_ds = _load_hf_dataset(data_path / "eval.hf")

    n_train = len(train_ds)
    n_eval = len(eval_ds)

    print(
        f"Scoring {n_eval} eval queries against "
        f"{n_train} train samples (PCA projection)"
    )

    # Load train gradients
    print("Loading train gradients...")
    train_grads = load_gradients(index_path, structured=True)

    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Load preconditioner if specified
    h_inv = {}
    if precond_path and (precond_path / "preconditioners.pth").exists():
        print(f"Loading preconditioner from {precond_path} (damping={damping_factor})")
        proc = GradientProcessor.load(precond_path)
        device = torch.device("cuda:0")
        for name in tqdm(module_names, desc="Computing H^(-1)"):
            H = proc.preconditioners[name].to(device=device)
            h_inv[name] = damped_psd_power(H, power=-1, damping_factor=damping_factor)

    # Concatenate train gradients
    print("Preparing train gradients...")
    train_grad_list = []
    for name in tqdm(module_names, desc="Loading train grads"):
        g = _load_gradients_as_float(train_grads, name)
        train_grad_list.append(torch.from_numpy(g))
    train_grad_tensor = torch.cat(train_grad_list, dim=1)

    # Unit normalize train grads
    train_norms = train_grad_tensor.norm(dim=1, keepdim=True)
    train_grad_tensor = train_grad_tensor / (train_norms + 1e-8)
    train_grad_tensor = train_grad_tensor.cuda()

    # Compute eval gradients if needed
    print("Computing eval gradients...")
    if eval_prompt_column == "fact" and eval_completion_column == "reworded":
        eval_grads_path = base_path / "eval_grads"
    else:
        eval_grads_path = (
            base_path / f"eval_grads_{eval_prompt_column}_{eval_completion_column}"
        )

    if not eval_grads_path.exists():
        index_cfg = IndexConfig.load_yaml(index_path / "index_config.yaml")

        _run_bergson_build(
            eval_grads_path,
            model=index_cfg.model,
            dataset_path=data_path / "eval.hf",
            prompt_column=eval_prompt_column,
            completion_column=eval_completion_column,
            projection_dim=index_cfg.projection_dim or 16,
            label="eval",
        )

    # Load eval gradients and apply PCA projection
    eval_grads = load_gradients(eval_grads_path, structured=True)
    eval_grad_list = []

    # Track cumulative dimension for concatenation
    for name in tqdm(module_names, desc="Loading and projecting eval grads"):
        g = torch.from_numpy(_load_gradients_as_float(eval_grads, name))

        # Apply PCA projection if we have the subspace for this module
        if name in style_subspace:
            eigvecs, _ = style_subspace[name]
            g = g.cuda()
            eigvecs = eigvecs.cuda()
            g = project_orthogonal_to_style_subspace(g, eigvecs)
            # Apply preconditioning after projection if specified
            if h_inv:
                g = g @ h_inv[name]
            g = g.cpu()
        elif h_inv:
            # Apply preconditioning even without PCA projection
            g = (g.cuda() @ h_inv[name]).cpu()

        eval_grad_list.append(g)

    eval_grad_tensor = torch.cat(eval_grad_list, dim=1)

    # Unit normalize eval grads
    eval_norms = eval_grad_tensor.norm(dim=1, keepdim=True)
    eval_grad_tensor = eval_grad_tensor / (eval_norms + 1e-8)
    eval_grad_tensor = eval_grad_tensor.cuda()

    # Compute scores
    print("Computing scores...")
    scores = (eval_grad_tensor @ train_grad_tensor.T).cpu().numpy()

    # Save scores
    np.save(scores_path / "scores.npy", scores)
    print(f"Saved scores to {scores_path}")

    return scores


def compute_asymmetric_metrics_with_pca(
    config: AsymmetricConfig,
    base_path: Path | str,
    style_subspace: dict[str, tuple],
    top_k: int = 10,
    preconditioner_name: str | None = None,
    damping_factor: float = 0.1,
    eval_prompt_column: str = "fact",
    eval_completion_column: str = "reworded",
) -> "AsymmetricMetrics":
    """Compute metrics using PCA projection style suppression.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        style_subspace: Dictionary from compute_pca_style_subspace().
        top_k: Number of principal components.
        preconditioner_name: Optional preconditioner to combine
            with PCA.
        damping_factor: Damping factor for matrix inversion
            (default: 0.1).
        eval_prompt_column: Column to use as prompt for eval
            gradients (default: "fact").
        eval_completion_column: Column to use as completion for
            eval gradients (default: "reworded"). Set to
            "question"/"answer" for semantic-only attribution.

    Returns:
        AsymmetricMetrics dataclass.
    """
    base_path = Path(base_path)
    data_path = base_path / "data"

    # Load datasets
    train_ds = _load_hf_dataset(data_path / "train.hf")
    eval_ds = _load_hf_dataset(data_path / "eval.hf")

    # Load scores (using PCA projection)
    scores = score_asymmetric_eval_with_pca_projection(
        config,
        base_path,
        style_subspace,
        top_k,
        preconditioner_name,
        damping_factor=damping_factor,
        eval_prompt_column=eval_prompt_column,
        eval_completion_column=eval_completion_column,
    )

    return _compute_metrics_from_scores(
        scores, train_ds, eval_ds, config.minority_style
    )


def create_majority_style_eval(
    config: AsymmetricConfig,
    base_path: Path | str,
    force_regenerate: bool = False,
) -> tuple[Path, bool]:
    """Create eval set using majority style (control for style mismatch).

    Instead of using minority style queries, uses dominant style queries
    for the exclusive facts. This shows baseline performance without style mismatch.

    IMPORTANT: Uses templates >= train_template_cutoff to ensure NO overlap with
    training data. This provides a proper train/test split where eval majority
    style items test semantic matching (same fact, different surface form) rather
    than exact text matching.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        force_regenerate: If True, regenerate even if cached version exists.

    Returns:
        Tuple of (path to the majority style eval dataset, has_leakage flag).
        has_leakage is True if there's train/test overlap (e.g., from HF data).
    """
    base_path = Path(base_path)
    data_path = base_path / "data"
    majority_eval_path = data_path / "eval_majority_style.hf"

    # Check for existing cached version
    if majority_eval_path.exists() and not force_regenerate:
        print(f"Loading cached majority style eval from {majority_eval_path}")

        # Check for train/test leakage by comparing reworded texts
        train_ds = _load_hf_dataset(data_path / "train.hf")
        majority_eval_ds = _load_hf_dataset(majority_eval_path)

        train_reworded = set(train_ds["reworded"])  # type: ignore[index]
        eval_reworded = set(majority_eval_ds["reworded"])  # type: ignore[index]
        overlap = train_reworded & eval_reworded
        has_leakage = len(overlap) > 0

        if has_leakage:
            print(
                f"  WARNING: {len(overlap)}/{len(eval_reworded)} eval items have "
                "exact text match in train (train/test leakage)"
            )
            print("  Use force_regenerate=True with local data to fix")

        return majority_eval_path, has_leakage

    print("Creating majority style eval set (control)...")

    # Check if local styled datasets exist for proper template segregation
    local_styled_path = Path(
        f"data/facts_dataset_{config.dominant_style}-Qwen3-8B-Base.hf"
    )
    if not local_styled_path.exists():
        print(f"  WARNING: Local styled dataset not found at {local_styled_path}")
        print("  Cannot create properly segregated majority eval")
        print("  Using HF eval_majority_style (may have train/test leakage)")
        return majority_eval_path, True  # Return existing HF version with leakage flag

    # Load the minority style eval to get the semantic facts (identifier, field pairs)
    eval_ds = _load_hf_dataset(data_path / "eval.hf")

    # Get semantic facts from eval (identifier, field pairs)
    eval_semantic_facts = {(row["identifier"], row["field"]) for row in eval_ds}  # type: ignore[index]

    # Load dominant style dataset
    dominant_ds = _load_hf_dataset(local_styled_path)

    # Add back metadata columns from original
    original = _load_hf_dataset("data/facts_dataset.hf")
    fact_to_meta = {row["fact"]: row for row in original}  # type: ignore[index]

    for col in original.column_names:
        if col not in dominant_ds.column_names:
            restored_col = [fact_to_meta[row["fact"]][col] for row in dominant_ds]  # type: ignore[index]
            dominant_ds = dominant_ds.add_column(col, restored_col)

    # Select dominant style versions of eval semantic facts
    # Use templates >= cutoff to ensure NO overlap with training data
    dominant_eval_indices = [
        i
        for i, row in enumerate(dominant_ds)
        if (row["identifier"], row["field"]) in eval_semantic_facts  # type: ignore[index]
        and row["template"] >= config.train_template_cutoff  # type: ignore[index]
    ]
    majority_eval_ds = dominant_ds.select(dominant_eval_indices)

    print(
        f"  Using templates >= {config.train_template_cutoff} (no overlap with train)"
    )
    print(f"  Found {len(majority_eval_ds)} majority style eval samples")

    # Add style columns if not present
    if "style" not in majority_eval_ds.column_names:
        majority_eval_ds = majority_eval_ds.add_column(
            "style", [config.dominant_style] * len(majority_eval_ds)
        )
    if "expected_match_style" not in majority_eval_ds.column_names:
        majority_eval_ds = majority_eval_ds.add_column(
            "expected_match_style", [config.dominant_style] * len(majority_eval_ds)
        )

    majority_eval_ds.save_to_disk(str(majority_eval_path))
    print(f"Saved majority style eval to {majority_eval_path}")

    return majority_eval_path, False  # No leakage with proper segregation


def score_majority_style_eval(
    config: AsymmetricConfig,
    base_path: Path | str,
    preconditioner_name: str | None = None,
) -> np.ndarray:
    """Score majority style eval queries (control for style mismatch).

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        preconditioner_name: Name of preconditioner subdirectory (None for no precond).

    Returns:
        Score matrix of shape (n_eval, n_train).
    """
    import json

    import torch
    from tqdm import tqdm

    from bergson.data import load_gradients
    from bergson.gradients import GradientProcessor
    from bergson.utils.math import damped_psd_power

    base_path = Path(base_path)
    index_path = base_path / "index"
    data_path = base_path / "data"

    # Create majority style eval if needed
    _, has_leakage = create_majority_style_eval(config, base_path)
    if has_leakage:
        print(
            "  Note: Majority control may show inflated "
            "accuracy due to train/test leakage"
        )

    # Determine output path
    if preconditioner_name:
        scores_path = base_path / f"scores_majority_{preconditioner_name}"
        precond_path = base_path / preconditioner_name
    else:
        scores_path = base_path / "scores_majority_no_precond"
        precond_path = None

    # Return cached if exists
    if (scores_path / "scores.npy").exists():
        print(f"Loading cached scores from {scores_path}")
        return np.load(scores_path / "scores.npy")

    scores_path.mkdir(parents=True, exist_ok=True)

    # Load train and eval datasets
    train_ds = _load_hf_dataset(data_path / "train.hf")
    eval_ds = _load_hf_dataset(data_path / "eval_majority_style.hf")

    n_train = len(train_ds)
    n_eval = len(eval_ds)

    print(
        f"Scoring {n_eval} majority style eval queries against {n_train} train samples"
    )

    # Load train gradients
    print("Loading train gradients...")
    train_grads = load_gradients(index_path, structured=True)

    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Load preconditioner if specified
    h_inv = {}
    if precond_path and (precond_path / "preconditioners.pth").exists():
        print(f"Loading preconditioner from {precond_path}")
        proc = GradientProcessor.load(precond_path)
        device = torch.device("cuda:0")
        for name in tqdm(module_names, desc="Computing H^(-1)"):
            H = proc.preconditioners[name].to(device=device)
            h_inv[name] = damped_psd_power(H, power=-1)

    # Concatenate train gradients
    print("Preparing train gradients...")
    train_grad_list = []
    for name in tqdm(module_names, desc="Loading train grads"):
        g = _load_gradients_as_float(train_grads, name)
        train_grad_list.append(torch.from_numpy(g))
    train_grad_tensor = torch.cat(train_grad_list, dim=1)

    # Unit normalize train grads
    train_norms = train_grad_tensor.norm(dim=1, keepdim=True)
    train_grad_tensor = train_grad_tensor / (train_norms + 1e-8)
    train_grad_tensor = train_grad_tensor.cuda()

    # Compute eval gradients for majority style
    print("Computing majority style eval gradients...")
    majority_eval_grads_path = base_path / "eval_grads_majority"
    if not majority_eval_grads_path.exists():
        index_cfg = IndexConfig.load_yaml(index_path / "index_config.yaml")

        _run_bergson_build(
            majority_eval_grads_path,
            model=index_cfg.model,
            dataset_path=data_path / "eval_majority_style.hf",
            projection_dim=index_cfg.projection_dim or 16,
            label="majority eval",
        )

    # Load eval gradients
    eval_grads = load_gradients(majority_eval_grads_path, structured=True)
    eval_grad_list = []
    for name in tqdm(module_names, desc="Loading eval grads"):
        g = torch.from_numpy(_load_gradients_as_float(eval_grads, name))
        if h_inv:
            g = (g.cuda() @ h_inv[name]).cpu()
        eval_grad_list.append(g)
    eval_grad_tensor = torch.cat(eval_grad_list, dim=1)

    # Unit normalize eval grads
    eval_norms = eval_grad_tensor.norm(dim=1, keepdim=True)
    eval_grad_tensor = eval_grad_tensor / (eval_norms + 1e-8)
    eval_grad_tensor = eval_grad_tensor.cuda()

    # Compute scores
    print("Computing scores...")
    scores = (eval_grad_tensor @ train_grad_tensor.T).cpu().numpy()

    # Save scores
    np.save(scores_path / "scores.npy", scores)
    print(f"Saved scores to {scores_path}")

    return scores


def compute_majority_style_metrics(
    config: AsymmetricConfig,
    base_path: Path | str,
    preconditioner_name: str | None = None,
) -> AsymmetricMetrics:
    """Compute metrics for majority style eval (control).

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        preconditioner_name: Name of preconditioner to use.

    Returns:
        AsymmetricMetrics dataclass.
    """
    base_path = Path(base_path)
    data_path = base_path / "data"

    # Create majority style eval if needed
    _, _ = create_majority_style_eval(config, base_path)

    # Load datasets
    train_ds = _load_hf_dataset(data_path / "train.hf")
    eval_ds = _load_hf_dataset(data_path / "eval_majority_style.hf")

    # Load scores
    scores = score_majority_style_eval(config, base_path, preconditioner_name)

    return _compute_metrics_from_scores(
        scores, train_ds, eval_ds, config.minority_style
    )


def score_summed_eval(
    config: AsymmetricConfig,
    base_path: Path | str,
    preconditioner_name: str | None = None,
) -> np.ndarray:
    """Score using summed eval gradients (minority + majority style for each fact).

    Instead of using minority-style eval gradients alone, this sums the gradients
    from both style versions of each fact. This makes the query "style-neutral"
    since style-specific components should cancel while semantic components reinforce.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        preconditioner_name: Name of preconditioner subdirectory (None for no precond).

    Returns:
        Score matrix of shape (n_eval, n_train).
    """
    import json

    import torch
    from tqdm import tqdm

    from bergson.data import load_gradients
    from bergson.gradients import GradientProcessor
    from bergson.utils.math import damped_psd_power

    base_path = Path(base_path)
    index_path = base_path / "index"
    data_path = base_path / "data"

    # Determine output path
    if preconditioner_name:
        scores_path = base_path / f"scores_summed_eval_{preconditioner_name}"
        precond_path = base_path / preconditioner_name
    else:
        scores_path = base_path / "scores_summed_eval_no_precond"
        precond_path = None

    # Return cached if exists
    if (scores_path / "scores.npy").exists():
        print(f"Loading cached scores from {scores_path}")
        return np.load(scores_path / "scores.npy")

    scores_path.mkdir(parents=True, exist_ok=True)

    # Load train dataset
    train_ds = _load_hf_dataset(data_path / "train.hf")
    n_train = len(train_ds)

    # Load eval datasets (need both minority and majority style versions)
    eval_minority_ds = _load_hf_dataset(data_path / "eval.hf")

    # Create majority style eval if needed
    _, _ = create_majority_style_eval(config, base_path)
    eval_majority_ds = _load_hf_dataset(data_path / "eval_majority_style.hf")

    n_eval = len(eval_minority_ds)
    print(
        f"Scoring {n_eval} summed eval queries "
        f"(minority + majority) against {n_train} train samples"
    )

    # Build semantic fact mapping for alignment (identifier, field pairs)
    # This works even when templates differ between minority and majority eval
    minority_semantic_facts = [
        (row["identifier"], row["field"]) for row in eval_minority_ds  # type: ignore[index]
    ]
    majority_semantic_to_idx = {
        (row["identifier"], row["field"]): i for i, row in enumerate(eval_majority_ds)  # type: ignore[index]
    }

    # Verify alignment by semantic fact
    assert len(eval_minority_ds) == len(
        eval_majority_ds
    ), "Eval datasets must have same size"
    for sf in minority_semantic_facts:
        assert (
            sf in majority_semantic_to_idx
        ), f"Semantic fact {sf} not found in majority eval"

    # Load train gradients
    print("Loading train gradients...")
    train_grads = load_gradients(index_path, structured=True)

    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Load preconditioner if specified
    h_inv = {}
    if precond_path and (precond_path / "preconditioners.pth").exists():
        print(f"Loading preconditioner from {precond_path}")
        proc = GradientProcessor.load(precond_path)
        device = torch.device("cuda:0")
        for name in tqdm(module_names, desc="Computing H^(-1)"):
            H = proc.preconditioners[name].to(device=device)
            h_inv[name] = damped_psd_power(H, power=-1)

    # Concatenate train gradients
    print("Preparing train gradients...")
    train_grad_list = []
    for name in tqdm(module_names, desc="Loading train grads"):
        g = _load_gradients_as_float(train_grads, name)
        train_grad_list.append(torch.from_numpy(g))
    train_grad_tensor = torch.cat(train_grad_list, dim=1)

    # Unit normalize train grads
    train_norms = train_grad_tensor.norm(dim=1, keepdim=True)
    train_grad_tensor = train_grad_tensor / (train_norms + 1e-8)
    train_grad_tensor = train_grad_tensor.cuda()

    # Ensure both eval gradient sets exist
    eval_minority_grads_path = base_path / "eval_grads"
    eval_majority_grads_path = base_path / "eval_grads_majority"

    index_cfg = IndexConfig.load_yaml(index_path / "index_config.yaml")

    # Build minority eval grads if needed
    if not eval_minority_grads_path.exists():
        print("Computing minority style eval gradients...")
        _run_bergson_build(
            eval_minority_grads_path,
            model=index_cfg.model,
            dataset_path=data_path / "eval.hf",
            projection_dim=index_cfg.projection_dim or 16,
            label="minority eval",
        )

    # Build majority eval grads if needed
    if not eval_majority_grads_path.exists():
        print("Computing majority style eval gradients...")
        _run_bergson_build(
            eval_majority_grads_path,
            model=index_cfg.model,
            dataset_path=data_path / "eval_majority_style.hf",
            projection_dim=index_cfg.projection_dim or 16,
            label="majority eval",
        )

    # Load both eval gradient sets
    print("Loading eval gradients (minority + majority)...")
    minority_grads = load_gradients(eval_minority_grads_path, structured=True)
    majority_grads = load_gradients(eval_majority_grads_path, structured=True)

    # Sum gradients: for each eval fact, sum minority + majority style gradients
    # Align by semantic fact (identifier, field) since templates may differ
    summed_grad_list = []
    for name in tqdm(module_names, desc="Summing eval grads"):
        g_minority = torch.from_numpy(_load_gradients_as_float(minority_grads, name))
        g_majority = torch.from_numpy(_load_gradients_as_float(majority_grads, name))

        # Align majority grads to minority semantic fact order
        aligned_majority_indices = [
            majority_semantic_to_idx[sf] for sf in minority_semantic_facts
        ]
        g_majority_aligned = g_majority[aligned_majority_indices]

        # Sum the gradients
        g_summed = g_minority + g_majority_aligned

        # Apply preconditioning if specified
        if h_inv:
            g_summed = (g_summed.cuda() @ h_inv[name]).cpu()

        summed_grad_list.append(g_summed)

    eval_grad_tensor = torch.cat(summed_grad_list, dim=1)

    # Unit normalize summed eval grads
    eval_norms = eval_grad_tensor.norm(dim=1, keepdim=True)
    eval_grad_tensor = eval_grad_tensor / (eval_norms + 1e-8)
    eval_grad_tensor = eval_grad_tensor.cuda()

    # Compute scores
    print("Computing scores...")
    scores = (eval_grad_tensor @ train_grad_tensor.T).cpu().numpy()

    # Save scores
    np.save(scores_path / "scores.npy", scores)
    print(f"Saved scores to {scores_path}")

    return scores


def compute_summed_eval_metrics(
    config: AsymmetricConfig,
    base_path: Path | str,
    preconditioner_name: str | None = None,
) -> AsymmetricMetrics:
    """Compute metrics for summed eval gradient approach.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        preconditioner_name: Name of preconditioner to use.

    Returns:
        AsymmetricMetrics dataclass.
    """
    base_path = Path(base_path)
    data_path = base_path / "data"

    # Load datasets
    train_ds = _load_hf_dataset(data_path / "train.hf")
    eval_ds = _load_hf_dataset(data_path / "eval.hf")

    # Load scores (computed with summed gradients)
    scores = score_summed_eval(config, base_path, preconditioner_name)

    return _compute_metrics_from_scores(
        scores, train_ds, eval_ds, config.minority_style
    )


def sweep_pca_k(
    config: AsymmetricConfig | None = None,
    base_path: Path | str = "runs/asymmetric_style",
    k_values: list[int] | None = None,
    preconditioners: list[str | None] | None = None,
) -> dict[str, AsymmetricMetrics]:
    """Sweep over k values and preconditioners for PCA projection approach.

    Args:
        config: Experiment configuration (uses defaults if None).
        base_path: Base path for experiment outputs.
        k_values: List of k values to test (default: [1, 5, 10, 20, 50, 100]).
        preconditioners: List of preconditioner names to combine with PCA.
                        None means no preconditioner. Default: [None, "index"].

    Returns:
        Dictionary mapping strategy names to their metrics.
    """
    from .preconditioners import compute_pca_style_subspace

    if config is None:
        config = AsymmetricConfig()

    base_path = Path(base_path)

    if k_values is None:
        k_values = [1, 5, 10, 20, 50, 100]

    if preconditioners is None:
        preconditioners = [
            None,
            "index",
        ]  # None = no precond, "index" = train second moment

    # Check that style indices exist
    minority_idx = Path(config.style_index_path) / config.minority_style
    dominant_idx = Path(config.style_index_path) / config.dominant_style
    if not (minority_idx.exists() and dominant_idx.exists()):
        raise FileNotFoundError(
            f"Style-specific indices not found at {minority_idx} and {dominant_idx}. "
            "Run build_style_indices() first."
        )

    all_metrics: dict[str, AsymmetricMetrics] = {}

    # Compute style subspaces for each k value
    print("=" * 70)
    print("PCA K-VALUE AND PRECONDITIONER SWEEP")
    print("=" * 70)

    for k in k_values:
        print(f"\n--- Computing style subspace for k={k} ---")
        style_subspace = compute_pca_style_subspace(
            minority_idx, dominant_idx, base_path / "pca_subspace", top_k=k
        )

        for precond_name in preconditioners:
            precond_display = precond_name if precond_name else "no_precond"
            strategy_name = f"pca_k{k}_{precond_display}"

            print(f"\n--- Strategy: {strategy_name} ---")
            metrics = compute_asymmetric_metrics_with_pca(
                config,
                base_path,
                style_subspace,
                top_k=k,
                preconditioner_name=precond_name,
            )
            print(f"  Top-1 Semantic: {metrics.top1_semantic_accuracy:.2%}")
            print(f"  Top-1 Style Leak: {metrics.top1_style_leakage:.2%}")
            all_metrics[strategy_name] = metrics

    # Print summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)

    print(f"\n{'Strategy':<30} {'Top-1 Semantic':<15} {'Top-1 Style Leak':<17}")
    print("-" * 65)

    for name, m in sorted(all_metrics.items()):
        print(
            f"{name:<30} {m.top1_semantic_accuracy:<15.2%} "
            f"{m.top1_style_leakage:<17.2%}"
        )

    return all_metrics


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
) -> dict[str, AsymmetricMetrics]:
    """Run the full asymmetric style suppression experiment.

    Compares attribution quality with and without style suppression preconditioning.

    Args:
        config: Experiment configuration (uses defaults if None). Set config.hf_dataset
            to load data from HuggingFace instead of generating locally.
            PCA k values and style index path are configured via
            config.pca_k_values and config.style_index_path.
        base_path: Base path for experiment outputs.
        analysis_model: Model to use for gradient collection.
            Defaults to HF_ANALYSIS_MODEL.
        include_pca: Whether to include PCA projection strategy.
        include_summed_loss: Whether to include summed loss
            preconditioner strategy.
        include_second_moments: Whether to include train/eval/mixed
            second moment strategies.
        include_majority_control: Whether to include majority style
            eval as control.
        include_summed_eval: Whether to include summed eval gradient
            approach (minority + majority).
        include_semantic_eval: Whether to include semantic-only eval
            using question/answer columns. This tests attribution
            when gradients only come from the semantic content
            (answer tokens), ignoring style in the eval query.
        damping_factor: Damping factor for matrix inversion
            (default: 0.1).

    Returns:
        Dictionary mapping preconditioner names to their metrics.
    """
    from .preconditioners import (
        compute_eval_preconditioner,
        compute_pca_style_subspace,
        compute_summed_loss_preconditioner,
        compute_train_eval_mixed_preconditioner,
    )

    if config is None:
        config = AsymmetricConfig()

    base_path = Path(base_path)

    print("=" * 70)
    print("ASYMMETRIC STYLE SUPPRESSION EXPERIMENT")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Dominant style: {config.dominant_style} ({config.dominant_ratio:.0%})")
    print(f"  Minority style: {config.minority_style}")
    print(f"  Exclusive facts: {config.exclusive_ratio:.0%}")

    # Step 1: Create dataset and index
    print("\n" + "-" * 60)
    print("STEP 1: Creating asymmetric dataset and index")
    print("-" * 60)
    create_asymmetric_index(config, base_path, analysis_model)

    # Load eval facts to exclude from PCA computation (prevent data leakage)
    data_path = base_path / "data"
    eval_ds = _load_hf_dataset(data_path / "eval.hf")
    eval_facts_to_exclude: set[str] = set(eval_ds["fact"])  # type: ignore[arg-type]
    print(f"Loaded {len(eval_facts_to_exclude)} eval facts to exclude from PCA")

    # Step 2: Compute R_between preconditioner
    print("\n" + "-" * 60)
    print("STEP 2: Computing style suppression preconditioner (R_between)")
    print("-" * 60)
    compute_style_preconditioner(base_path, config)

    # Step 2b: Compute summed loss preconditioner if requested
    if include_summed_loss:
        print("\n" + "-" * 60)
        print("STEP 2b: Computing summed loss preconditioner")
        print("-" * 60)
        # We need the style-specific indices for this
        minority_idx = Path(config.style_index_path) / config.minority_style
        dominant_idx = Path(config.style_index_path) / config.dominant_style
        if minority_idx.exists() and dominant_idx.exists():
            summed_loss_path = base_path / "summed_loss"
            compute_summed_loss_preconditioner(
                minority_idx, dominant_idx, summed_loss_path
            )
        else:
            print(
                "  Style-specific indices not found, "
                "skipping summed loss preconditioner"
            )
            print(f"  (Expected: {minority_idx} and {dominant_idx})")
            include_summed_loss = False

    # Step 2c: Compute PCA style subspace if requested
    pca_top_k = config.pca_k_values[0]
    style_subspace = None
    if include_pca:
        print("\n" + "-" * 60)
        print(f"STEP 2c: Computing PCA style subspace (top_k={pca_top_k})")
        print("-" * 60)
        minority_idx = Path(config.style_index_path) / config.minority_style
        dominant_idx = Path(config.style_index_path) / config.dominant_style
        if minority_idx.exists() and dominant_idx.exists():
            style_subspace = compute_pca_style_subspace(
                minority_idx,
                dominant_idx,
                base_path / "pca_subspace",
                top_k=pca_top_k,
                exclude_facts=eval_facts_to_exclude,
            )
        else:
            print("  Style-specific indices not found, skipping PCA projection")
            print(f"  (Expected: {minority_idx} and {dominant_idx})")
            include_pca = False

    # Step 2d: Compute second moment preconditioners if requested
    if include_second_moments:
        print("\n" + "-" * 60)
        print("STEP 2d: Computing second moment preconditioners (train/eval/mixed)")
        print("-" * 60)

        index_path = base_path / "index"
        eval_grads_path = base_path / "eval_grads"

        # Note: train second moment is already computed during index build
        # We just need to use it directly from the index

        # Compute eval second moment
        if eval_grads_path.exists():
            compute_eval_preconditioner(
                eval_grads_path,
                base_path / "eval_second_moment",
                reference_proc_path=index_path,  # Use train index for metadata
            )

            # Compute 50:50 train-eval mixed
            compute_train_eval_mixed_preconditioner(
                index_path,
                eval_grads_path,
                base_path / "train_eval_mixed",
                train_weight=0.5,
            )
        else:
            print("  Eval grads not found, will compute during scoring")
            include_second_moments = False

    # Step 3: Score and evaluate with each strategy
    print("\n" + "-" * 60)
    print("STEP 3: Evaluating preconditioner strategies")
    print("-" * 60)

    # Basic strategies using matrix-inverse preconditioning
    strategies = [
        (None, "no_precond"),
        ("r_between", "r_between"),
    ]

    # Add summed loss if available
    if include_summed_loss:
        strategies.append(("summed_loss", "summed_loss"))

    # Add second moment strategies if available
    if include_second_moments:
        # Train second moment (use the index's preconditioner directly)
        strategies.append(("index", "train_second_moment"))
        # Eval second moment
        strategies.append(("eval_second_moment", "eval_second_moment"))
        # 50:50 train-eval mixed
        strategies.append(("train_eval_mixed", "train_eval_mixed"))

    all_metrics: dict[str, AsymmetricMetrics] = {}

    for precond_name, display_name in strategies:
        print(f"\n--- Strategy: {display_name} ---")
        metrics = compute_asymmetric_metrics(
            config, base_path, precond_name, damping_factor=damping_factor
        )
        print_metrics(metrics, display_name)
        all_metrics[display_name] = metrics

    # Evaluate PCA projection strategy (different approach - not matrix-inverse)
    if include_pca and style_subspace is not None:
        print(f"\n--- Strategy: pca_projection_k{pca_top_k} ---")
        metrics = compute_asymmetric_metrics_with_pca(
            config,
            base_path,
            style_subspace,
            top_k=pca_top_k,
            damping_factor=damping_factor,
        )
        print_metrics(metrics, f"pca_projection_k{pca_top_k}")
        all_metrics[f"pca_projection_k{pca_top_k}"] = metrics

    # Evaluate majority style control (no style mismatch)
    if include_majority_control:
        print("\n" + "-" * 60)
        print("MAJORITY STYLE CONTROL (no style mismatch)")
        print("-" * 60)
        print("\n--- Control: majority_style_no_precond ---")
        metrics = compute_majority_style_metrics(config, base_path, None)
        print_metrics(metrics, "majority_no_precond")
        all_metrics["majority_no_precond"] = metrics

    # Evaluate summed eval gradient approach (minority + majority style)
    if include_summed_eval:
        print("\n" + "-" * 60)
        print("SUMMED EVAL GRADIENTS (minority + majority style)")
        print("-" * 60)
        print("\n--- Strategy: summed_eval_no_precond ---")
        metrics = compute_summed_eval_metrics(config, base_path, None)
        print_metrics(metrics, "summed_eval")
        all_metrics["summed_eval"] = metrics

    # Evaluate semantic-only approach
    # (question/answer columns - gradients only from answer)
    if include_semantic_eval:
        print("\n" + "-" * 60)
        print("SEMANTIC-ONLY EVAL (gradients only from answer tokens)")
        print("-" * 60)

        semantic_strategies: list[tuple[str | None, str]] = [
            (None, "semantic_no_precond"),
            ("index", "semantic_index"),
            ("r_between", "semantic_r_between"),
        ]
        if include_summed_loss:
            semantic_strategies.append(("summed_loss", "semantic_summed_loss"))
        if include_second_moments:
            semantic_strategies.append(
                ("eval_second_moment", "semantic_eval_second_moment")
            )
            semantic_strategies.append(
                ("train_eval_mixed", "semantic_train_eval_mixed")
            )

        for precond_name, display_name in semantic_strategies:
            print(f"\n--- Strategy: {display_name} ---")
            metrics = compute_asymmetric_metrics(
                config,
                base_path,
                precond_name,
                damping_factor=damping_factor,
                eval_prompt_column="question",
                eval_completion_column="answer",
            )
            print_metrics(metrics, display_name)
            all_metrics[display_name] = metrics

        # Semantic + PCA projection sweep over config.pca_k_values
        if include_pca:
            minority_idx = Path(config.style_index_path) / config.minority_style
            dominant_idx = Path(config.style_index_path) / config.dominant_style
            if minority_idx.exists() and dominant_idx.exists():
                for k in config.pca_k_values:
                    style_subspace_k = compute_pca_style_subspace(
                        minority_idx,
                        dominant_idx,
                        base_path / "pca_subspace",
                        top_k=k,
                        exclude_facts=eval_facts_to_exclude,
                    )

                    pca_strategies: list[tuple[str | None, str]] = [
                        (None, f"semantic_pca_projection_k{k}"),
                        ("index", f"semantic_pca_k{k}_index"),
                    ]
                    for precond_name, display_name in pca_strategies:
                        print(f"\n--- Strategy: {display_name} ---")
                        metrics = compute_asymmetric_metrics_with_pca(
                            config,
                            base_path,
                            style_subspace_k,
                            top_k=k,
                            preconditioner_name=precond_name,
                            damping_factor=damping_factor,
                            eval_prompt_column="question",
                            eval_completion_column="answer",
                        )
                        print_metrics(metrics, display_name)
                        all_metrics[display_name] = metrics

    # Print summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print(f"\n{'Strategy':<35} {'Top-1 Semantic':<15} {'Top-1 Style Leak':<17}")
    print("-" * 70)

    for name, m in all_metrics.items():
        sem = m.top1_semantic_accuracy
        leak = m.top1_style_leakage
        print(f"{name:<35} {sem:<15.2%} {leak:<17.2%}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("\nSuccess criteria:")
    print("  - Higher semantic accuracy = preconditioner helps find correct facts")
    print("  - Lower style leakage = preconditioner reduces style matching")
    print("\nStrategies:")
    print("  - no_precond: Baseline without any style suppression")
    print("  - r_between: Rank-1 preconditioner from style mean difference")
    if include_summed_loss:
        print("  - summed_loss: Preconditioner from summed gradients across pairs")
    if include_second_moments:
        print("  - train_second_moment: Second moment matrix from train gradients")
        print("  - eval_second_moment: Second moment matrix from eval gradients")
        print("  - train_eval_mixed: 50:50 mixture of train and eval second moments")
    if include_pca:
        print(f"  - pca_projection_k{pca_top_k}: Project out top-{pca_top_k} style PCs")
    if include_majority_control:
        print(
            "  - majority_no_precond: Control using majority "
            "style for eval (no mismatch)"
        )
    if include_summed_eval:
        print(
            "  - summed_eval: Sum minority + majority style eval gradients "
            "(style-neutral query)"
        )
    if include_semantic_eval:
        print(
            "  - semantic_*: Eval gradients only from answer tokens "
            "(question/answer format)"
        )
        print(
            "    Tests if attribution works when query has no style information at all"
        )

    return all_metrics


# =============================================================================
# RAW INNER PRODUCT COMPARISON
# =============================================================================
# Compare cosine similarity vs raw inner product scoring


def score_with_inner_product(
    config: AsymmetricConfig,
    base_path: Path | str,
    eval_style: str = "minority",
    preconditioner_name: str | None = None,
) -> np.ndarray:
    """Score using raw inner product (no unit normalization).

    This matches bergson's default behavior where unit_normalize=False.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        eval_style: Which eval set to use ("minority", "majority", "summed").
        preconditioner_name: Name of preconditioner subdirectory.

    Returns:
        Score matrix of shape (n_eval, n_train).
    """
    import json

    import torch
    from tqdm import tqdm

    from bergson.data import load_gradients
    from bergson.gradients import GradientProcessor
    from bergson.utils.math import damped_psd_power

    base_path = Path(base_path)
    index_path = base_path / "index"
    data_path = base_path / "data"

    # Determine output path
    suffix = f"_innerproduct_{eval_style}"
    if preconditioner_name:
        scores_path = base_path / f"scores{suffix}_{preconditioner_name}"
        precond_path = base_path / preconditioner_name
    else:
        scores_path = base_path / f"scores{suffix}_no_precond"
        precond_path = None

    # Return cached if exists
    if (scores_path / "scores.npy").exists():
        print(f"Loading cached scores from {scores_path}")
        return np.load(scores_path / "scores.npy")

    scores_path.mkdir(parents=True, exist_ok=True)

    # Load train gradients
    print("Loading train gradients...")
    train_grads = load_gradients(index_path, structured=True)

    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Load preconditioner if specified
    h_inv = {}
    if precond_path and (precond_path / "preconditioners.pth").exists():
        print(f"Loading preconditioner from {precond_path}")
        proc = GradientProcessor.load(precond_path)
        device = torch.device("cuda:0")
        for name in tqdm(module_names, desc="Computing H^(-1)"):
            H = proc.preconditioners[name].to(device=device)
            h_inv[name] = damped_psd_power(H, power=-1)

    # Concatenate train gradients - NO NORMALIZATION
    print("Preparing train gradients (no normalization)...")
    train_grad_list = []
    for name in tqdm(module_names, desc="Loading train grads"):
        g = _load_gradients_as_float(train_grads, name)
        train_grad_list.append(torch.from_numpy(g))
    train_grad_tensor = torch.cat(train_grad_list, dim=1).cuda()

    # Load eval gradients based on style
    if eval_style == "minority":
        eval_grads_path = base_path / "eval_grads"
    elif eval_style == "majority":
        eval_grads_path = base_path / "eval_grads_majority"
    elif eval_style == "summed":
        # Need to sum minority + majority
        minority_grads = load_gradients(base_path / "eval_grads", structured=True)
        majority_grads = load_gradients(
            base_path / "eval_grads_majority", structured=True
        )

        # Load eval datasets for alignment
        eval_minority_ds = _load_hf_dataset(data_path / "eval.hf")
        eval_majority_ds = _load_hf_dataset(data_path / "eval_majority_style.hf")

        # Use semantic fact alignment (identifier, field) since templates may differ
        minority_semantic_facts = [
            (row["identifier"], row["field"]) for row in eval_minority_ds  # type: ignore[index]
        ]
        majority_semantic_to_idx = {
            (row["identifier"], row["field"]): i  # type: ignore[index]
            for i, row in enumerate(eval_majority_ds)
        }

        summed_grad_list = []
        for name in tqdm(module_names, desc="Summing eval grads"):
            g_minority = torch.from_numpy(
                _load_gradients_as_float(minority_grads, name)
            )
            g_majority = torch.from_numpy(
                _load_gradients_as_float(majority_grads, name)
            )

            aligned_majority_indices = [
                majority_semantic_to_idx[sf] for sf in minority_semantic_facts
            ]
            g_majority_aligned = g_majority[aligned_majority_indices]

            g_summed = g_minority + g_majority_aligned

            if h_inv:
                g_summed = (g_summed.cuda() @ h_inv[name]).cpu()

            summed_grad_list.append(g_summed)

        eval_grad_tensor = torch.cat(summed_grad_list, dim=1).cuda()

        # NO NORMALIZATION - raw inner product
        print("Computing scores (raw inner product)...")
        scores = (eval_grad_tensor @ train_grad_tensor.T).cpu().numpy()

        np.save(scores_path / "scores.npy", scores)
        print(f"Saved scores to {scores_path}")
        return scores
    else:
        raise ValueError(f"Unknown eval_style: {eval_style}")

    # Load eval gradients
    print(f"Loading {eval_style} eval gradients...")
    eval_grads = load_gradients(eval_grads_path, structured=True)

    eval_grad_list = []
    for name in tqdm(module_names, desc="Loading eval grads"):
        g = torch.from_numpy(_load_gradients_as_float(eval_grads, name))
        if h_inv:
            g = (g.cuda() @ h_inv[name]).cpu()
        eval_grad_list.append(g)

    eval_grad_tensor = torch.cat(eval_grad_list, dim=1).cuda()

    # NO NORMALIZATION - raw inner product
    print("Computing scores (raw inner product)...")
    scores = (eval_grad_tensor @ train_grad_tensor.T).cpu().numpy()

    np.save(scores_path / "scores.npy", scores)
    print(f"Saved scores to {scores_path}")

    return scores


def run_inner_product_comparison(
    config: AsymmetricConfig | None = None,
    base_path: Path | str = "runs/asymmetric_style",
) -> dict[str, "AsymmetricMetrics"]:
    """Compare key strategies using raw inner product instead of cosine similarity.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.

    Returns:
        Dictionary mapping strategy names to their metrics.
    """
    if config is None:
        config = AsymmetricConfig()

    base_path = Path(base_path)
    data_path = base_path / "data"

    print("=" * 70)
    print("INNER PRODUCT VS COSINE SIMILARITY COMPARISON")
    print("=" * 70)
    print("\nRunning key strategies with raw inner product (bergson default)")
    print()

    # Load datasets for metrics computation
    train_ds = _load_hf_dataset(data_path / "train.hf")
    eval_ds = _load_hf_dataset(data_path / "eval.hf")

    n_eval = len(eval_ds)
    train_styles = train_ds["style"]  # type: ignore[index]
    train_identifiers = train_ds["identifier"]  # type: ignore[index]
    train_fields = train_ds["field"]  # type: ignore[index]
    eval_identifiers = eval_ds["identifier"]  # type: ignore[index]
    eval_fields = eval_ds["field"]  # type: ignore[index]

    def compute_metrics_from_scores(scores):
        top_indices = np.argsort(-scores, axis=1)[:, :10]

        sem_top1 = sem_top5 = leak_top1 = 0

        for i in range(n_eval):
            top_k_idx = top_indices[i]

            # Check semantic matching (same identifier AND field = same underlying fact)
            for k, idx in enumerate(top_k_idx):
                if (
                    train_identifiers[idx] == eval_identifiers[i]
                    and train_fields[idx] == eval_fields[i]
                ):
                    if k == 0:
                        sem_top1 += 1
                    if k < 5:
                        sem_top5 += 1
                        break

            if train_styles[top_k_idx[0]] == config.minority_style:
                leak_top1 += 1

        return AsymmetricMetrics(
            top1_semantic_accuracy=sem_top1 / n_eval,
            top5_semantic_recall=sem_top5 / n_eval,
            top10_semantic_recall=0,  # Not computed
            top1_style_leakage=leak_top1 / n_eval,
            top5_style_leakage=0,
            top10_style_leakage=0,
            top1_subject_accuracy=0,
            top1_field_accuracy=0,
        )

    all_metrics = {}

    strategies = [
        ("minority_no_precond", "minority", None),
        ("majority_no_precond", "majority", None),
        ("summed_no_precond", "summed", None),
        ("minority_index", "minority", "index"),
    ]

    for name, eval_style, precond in strategies:
        print(f"\n--- Strategy: {name} (inner product) ---")
        scores = score_with_inner_product(config, base_path, eval_style, precond)
        metrics = compute_metrics_from_scores(scores)
        print(f"  Top-1 Semantic: {metrics.top1_semantic_accuracy:.2%}")
        print(f"  Top-1 Style Leak: {metrics.top1_style_leakage:.2%}")
        all_metrics[f"ip_{name}"] = metrics

    # Print comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON: Inner Product vs Cosine Similarity")
    print("=" * 70)

    print(f"\n{'Strategy':<30} {'Cosine Top-1':<15} {'InnerProd Top-1':<15}")
    print("-" * 60)

    # Load cosine results for comparison
    cosine_results = {
        "minority_no_precond": 0.87,
        "majority_no_precond": 100.0,
        "summed_no_precond": 92.71,
        "minority_index": 1.04,
    }

    for name, _, _ in strategies:
        cosine = cosine_results.get(name, 0)
        ip = all_metrics[f"ip_{name}"].top1_semantic_accuracy * 100
        print(f"{name:<30} {cosine:<15.2f}% {ip:<15.2f}%")

    return all_metrics


# =============================================================================
# REWRITE ABLATION EXPERIMENT
# =============================================================================
# Test: what if we sum two rewrite styles (shakespeare + pirate) that are both
# different from training? This tests whether summed_eval works because of
# general style cancellation or because one component matches training.


def create_original_style_eval(
    config: AsymmetricConfig,
    base_path: Path | str,
) -> Path:
    """Create eval set using original un-stylized facts.

    Creates a dataset where both prompt and completion use the original fact text
    (no stylization). This represents the "true" eval data before any rewriting.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.

    Returns:
        Path to the original style eval dataset.
    """
    base_path = Path(base_path)
    data_path = base_path / "data"
    original_eval_path = data_path / "eval_original_style.hf"

    if original_eval_path.exists():
        print(f"Loading cached original style eval from {original_eval_path}")
        return original_eval_path

    print("Creating original style eval set...")

    # Load the minority style eval to get the facts we need
    eval_ds = _load_hf_dataset(data_path / "eval.hf")

    eval_facts = list(eval_ds["fact"])  # type: ignore[index]

    # Load original facts dataset to get metadata
    original = _load_hf_dataset("data/facts_dataset.hf")
    fact_to_row = {row["fact"]: row for row in original}  # type: ignore[index]

    # Build original style eval dataset (fact = reworded = original text)
    rows = []
    for fact in eval_facts:
        if fact not in fact_to_row:
            print(f"Warning: fact not found in original dataset: {fact[:50]}...")
            continue
        row = dict(fact_to_row[fact])
        row["reworded"] = fact  # Use original fact as "reworded" too
        row["style"] = "original"
        row["expected_match_style"] = config.dominant_style
        rows.append(row)

    original_eval_ds = Dataset.from_list(rows)
    original_eval_ds.save_to_disk(str(original_eval_path))
    print(
        f"Saved original style eval ({len(original_eval_ds)} samples)"
        f" to {original_eval_path}"
    )

    return original_eval_path


def create_pirate_style_eval(
    config: AsymmetricConfig,
    base_path: Path | str,
) -> Path:
    """Create eval set using pirate style for the exclusive facts.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.

    Returns:
        Path to the pirate style eval dataset.
    """
    base_path = Path(base_path)
    data_path = base_path / "data"
    pirate_eval_path = data_path / "eval_pirate_style.hf"

    if pirate_eval_path.exists():
        print(f"Loading cached pirate style eval from {pirate_eval_path}")
        return pirate_eval_path

    print("Creating pirate style eval set...")

    # Load the minority style eval to get the facts we need
    eval_ds = _load_hf_dataset(data_path / "eval.hf")

    eval_facts = set(eval_ds["fact"])  # type: ignore[index]

    # Load pirate dataset
    pirate_ds = _load_hf_dataset("data/facts_dataset_pirate-Qwen3-8B-Base.hf")

    # Add back metadata columns from original
    original = _load_hf_dataset("data/facts_dataset.hf")
    fact_to_meta = {row["fact"]: row for row in original}  # type: ignore[index]

    for col in original.column_names:
        if col not in pirate_ds.column_names:
            restored_col = [fact_to_meta[row["fact"]][col] for row in pirate_ds]  # type: ignore[index]
            pirate_ds = pirate_ds.add_column(col, restored_col)

    # Select only the exclusive facts (same facts as in minority eval)
    pirate_eval_indices = [
        i for i, row in enumerate(pirate_ds) if row["fact"] in eval_facts  # type: ignore[index]
    ]
    pirate_eval_ds = pirate_ds.select(pirate_eval_indices)

    # Add style columns
    pirate_eval_ds = pirate_eval_ds.add_column(
        "style", ["pirate"] * len(pirate_eval_ds)
    )
    pirate_eval_ds = pirate_eval_ds.add_column(
        "expected_match_style", [config.dominant_style] * len(pirate_eval_ds)
    )

    pirate_eval_ds.save_to_disk(str(pirate_eval_path))
    print(
        f"Saved pirate style eval ({len(pirate_eval_ds)} samples) to {pirate_eval_path}"
    )

    return pirate_eval_path


def score_summed_rewrites(
    config: AsymmetricConfig,
    base_path: Path | str,
    preconditioner_name: str | None = None,
) -> np.ndarray:
    """Score using summed rewrite gradients (shakespeare + pirate).

    Tests whether summing two different rewrite styles helps with style invariance,
    even when neither rewrite matches the training distribution.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        preconditioner_name: Name of preconditioner subdirectory (None for no precond).

    Returns:
        Score matrix of shape (n_eval, n_train).
    """
    import json

    import torch
    from tqdm import tqdm

    from bergson.data import load_gradients
    from bergson.gradients import GradientProcessor
    from bergson.utils.math import damped_psd_power

    base_path = Path(base_path)
    index_path = base_path / "index"
    data_path = base_path / "data"

    # Determine output path
    if preconditioner_name:
        scores_path = base_path / f"scores_summed_rewrites_{preconditioner_name}"
        precond_path = base_path / preconditioner_name
    else:
        scores_path = base_path / "scores_summed_rewrites_no_precond"
        precond_path = None

    # Return cached if exists
    if (scores_path / "scores.npy").exists():
        print(f"Loading cached scores from {scores_path}")
        return np.load(scores_path / "scores.npy")

    scores_path.mkdir(parents=True, exist_ok=True)

    # Load train dataset
    train_ds = _load_hf_dataset(data_path / "train.hf")
    n_train = len(train_ds)

    # Create shakespeare and pirate eval datasets if needed
    # Shakespeare is already in eval.hf (minority style)
    create_pirate_style_eval(config, base_path)

    # Load eval datasets
    shakespeare_eval_ds = _load_hf_dataset(data_path / "eval.hf")

    pirate_eval_ds = _load_hf_dataset(data_path / "eval_pirate_style.hf")

    n_eval = len(shakespeare_eval_ds)
    print(
        f"Scoring {n_eval} summed rewrite queries (shakespeare + pirate)"
        f" against {n_train} train"
    )

    # Build fact-to-index mapping for alignment
    shakespeare_facts = shakespeare_eval_ds["fact"]
    pirate_facts = pirate_eval_ds["fact"]
    pirate_fact_to_idx = {f: i for i, f in enumerate(pirate_facts)}

    # Verify alignment
    for f in shakespeare_facts:
        assert f in pirate_fact_to_idx, f"Fact {f} not found in pirate eval"

    # Load train gradients
    print("Loading train gradients...")
    train_grads = load_gradients(index_path, structured=True)

    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Load preconditioner if specified
    h_inv = {}
    if precond_path and (precond_path / "preconditioners.pth").exists():
        print(f"Loading preconditioner from {precond_path}")
        proc = GradientProcessor.load(precond_path)
        device = torch.device("cuda:0")
        for name in tqdm(module_names, desc="Computing H^(-1)"):
            H = proc.preconditioners[name].to(device=device)
            h_inv[name] = damped_psd_power(H, power=-1)

    # Concatenate train gradients
    print("Preparing train gradients...")
    train_grad_list = []
    for name in tqdm(module_names, desc="Loading train grads"):
        g = _load_gradients_as_float(train_grads, name)
        train_grad_list.append(torch.from_numpy(g))
    train_grad_tensor = torch.cat(train_grad_list, dim=1)

    # Unit normalize train grads
    train_norms = train_grad_tensor.norm(dim=1, keepdim=True)
    train_grad_tensor = train_grad_tensor / (train_norms + 1e-8)
    train_grad_tensor = train_grad_tensor.cuda()

    # Build gradient paths
    shakespeare_grads_path = base_path / "eval_grads"  # minority = shakespeare
    pirate_grads_path = base_path / "eval_grads_pirate"

    index_cfg = IndexConfig.load_yaml(index_path / "index_config.yaml")

    # Build shakespeare eval grads if needed
    if not shakespeare_grads_path.exists():
        print("Computing shakespeare style eval gradients...")
        _run_bergson_build(
            shakespeare_grads_path,
            model=index_cfg.model,
            dataset_path=data_path / "eval.hf",
            projection_dim=index_cfg.projection_dim or 16,
            label="shakespeare eval",
        )

    # Build pirate eval grads if needed
    if not pirate_grads_path.exists():
        print("Computing pirate style eval gradients...")
        _run_bergson_build(
            pirate_grads_path,
            model=index_cfg.model,
            dataset_path=data_path / "eval_pirate_style.hf",
            projection_dim=index_cfg.projection_dim or 16,
            label="pirate eval",
        )

    # Load both eval gradient sets
    print("Loading eval gradients (shakespeare + pirate)...")
    shakespeare_grads = load_gradients(shakespeare_grads_path, structured=True)
    pirate_grads = load_gradients(pirate_grads_path, structured=True)

    # Sum gradients: for each eval fact, sum shakespeare + pirate style gradients
    summed_grad_list = []
    for name in tqdm(module_names, desc="Summing rewrite grads"):
        g_shakespeare = torch.from_numpy(
            _load_gradients_as_float(shakespeare_grads, name)
        )
        g_pirate = torch.from_numpy(_load_gradients_as_float(pirate_grads, name))

        # Align pirate grads to shakespeare fact order
        aligned_pirate_indices = [pirate_fact_to_idx[f] for f in shakespeare_facts]
        g_pirate_aligned = g_pirate[aligned_pirate_indices]

        # Sum the gradients
        g_summed = g_shakespeare + g_pirate_aligned

        # Apply preconditioning if specified
        if h_inv:
            g_summed = (g_summed.cuda() @ h_inv[name]).cpu()

        summed_grad_list.append(g_summed)

    eval_grad_tensor = torch.cat(summed_grad_list, dim=1)

    # Unit normalize summed eval grads
    eval_norms = eval_grad_tensor.norm(dim=1, keepdim=True)
    eval_grad_tensor = eval_grad_tensor / (eval_norms + 1e-8)
    eval_grad_tensor = eval_grad_tensor.cuda()

    # Compute scores
    print("Computing scores...")
    scores = (eval_grad_tensor @ train_grad_tensor.T).cpu().numpy()

    # Save scores
    np.save(scores_path / "scores.npy", scores)
    print(f"Saved scores to {scores_path}")

    return scores


def score_original_style_eval(
    config: AsymmetricConfig,
    base_path: Path | str,
    preconditioner_name: str | None = None,
) -> np.ndarray:
    """Score using original un-stylized eval gradients.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        preconditioner_name: Name of preconditioner subdirectory (None for no precond).

    Returns:
        Score matrix of shape (n_eval, n_train).
    """
    import json

    import torch
    from tqdm import tqdm

    from bergson.data import load_gradients
    from bergson.gradients import GradientProcessor
    from bergson.utils.math import damped_psd_power

    base_path = Path(base_path)
    index_path = base_path / "index"
    data_path = base_path / "data"

    # Determine output path
    if preconditioner_name:
        scores_path = base_path / f"scores_original_style_{preconditioner_name}"
        precond_path = base_path / preconditioner_name
    else:
        scores_path = base_path / "scores_original_style_no_precond"
        precond_path = None

    # Return cached if exists
    if (scores_path / "scores.npy").exists():
        print(f"Loading cached scores from {scores_path}")
        return np.load(scores_path / "scores.npy")

    scores_path.mkdir(parents=True, exist_ok=True)

    # Create original style eval dataset if needed
    create_original_style_eval(config, base_path)

    # Load datasets
    train_ds = _load_hf_dataset(data_path / "train.hf")
    n_train = len(train_ds)

    original_eval_ds = _load_hf_dataset(data_path / "eval_original_style.hf")
    n_eval = len(original_eval_ds)

    print(
        f"Scoring {n_eval} original style eval queries against {n_train} train samples"
    )

    # Load train gradients
    print("Loading train gradients...")
    train_grads = load_gradients(index_path, structured=True)

    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Load preconditioner if specified
    h_inv = {}
    if precond_path and (precond_path / "preconditioners.pth").exists():
        print(f"Loading preconditioner from {precond_path}")
        proc = GradientProcessor.load(precond_path)
        device = torch.device("cuda:0")
        for name in tqdm(module_names, desc="Computing H^(-1)"):
            H = proc.preconditioners[name].to(device=device)
            h_inv[name] = damped_psd_power(H, power=-1)

    # Concatenate train gradients
    print("Preparing train gradients...")
    train_grad_list = []
    for name in tqdm(module_names, desc="Loading train grads"):
        g = _load_gradients_as_float(train_grads, name)
        train_grad_list.append(torch.from_numpy(g))
    train_grad_tensor = torch.cat(train_grad_list, dim=1)

    # Unit normalize train grads
    train_norms = train_grad_tensor.norm(dim=1, keepdim=True)
    train_grad_tensor = train_grad_tensor / (train_norms + 1e-8)
    train_grad_tensor = train_grad_tensor.cuda()

    # Build original style eval grads if needed
    original_grads_path = base_path / "eval_grads_original"

    index_cfg = IndexConfig.load_yaml(index_path / "index_config.yaml")

    if not original_grads_path.exists():
        print("Computing original style eval gradients...")
        _run_bergson_build(
            original_grads_path,
            model=index_cfg.model,
            dataset_path=data_path / "eval_original_style.hf",
            projection_dim=index_cfg.projection_dim or 16,
            label="original style eval",
        )

    # Load original eval gradients
    print("Loading original style eval gradients...")
    original_grads = load_gradients(original_grads_path, structured=True)

    eval_grad_list = []
    for name in tqdm(module_names, desc="Loading original eval grads"):
        g = torch.from_numpy(_load_gradients_as_float(original_grads, name))

        # Apply preconditioning if specified
        if h_inv:
            g = (g.cuda() @ h_inv[name]).cpu()

        eval_grad_list.append(g)

    eval_grad_tensor = torch.cat(eval_grad_list, dim=1)

    # Unit normalize eval grads
    eval_norms = eval_grad_tensor.norm(dim=1, keepdim=True)
    eval_grad_tensor = eval_grad_tensor / (eval_norms + 1e-8)
    eval_grad_tensor = eval_grad_tensor.cuda()

    # Compute scores
    print("Computing scores...")
    scores = (eval_grad_tensor @ train_grad_tensor.T).cpu().numpy()

    # Save scores
    np.save(scores_path / "scores.npy", scores)
    print(f"Saved scores to {scores_path}")

    return scores


def compute_rewrite_ablation_metrics(
    config: AsymmetricConfig,
    base_path: Path | str,
    strategy: str,
    preconditioner_name: str | None = None,
) -> "AsymmetricMetrics":
    """Compute metrics for rewrite ablation strategies.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        strategy: One of "original", "summed_rewrites",
            "shakespeare_only", "pirate_only".
        preconditioner_name: Name of preconditioner subdirectory (None for no precond).

    Returns:
        AsymmetricMetrics with accuracy measurements.
    """
    base_path = Path(base_path)
    data_path = base_path / "data"

    # Load train dataset for ground truth mapping
    train_ds = _load_hf_dataset(data_path / "train.hf")

    # Load eval dataset (use minority style for fact mapping)
    eval_ds = _load_hf_dataset(data_path / "eval.hf")

    # Get scores based on strategy
    if strategy == "original":
        scores = score_original_style_eval(config, base_path, preconditioner_name)
    elif strategy == "summed_rewrites":
        scores = score_summed_rewrites(config, base_path, preconditioner_name)
    elif strategy == "shakespeare_only":
        # Shakespeare is the minority style, use standard scoring
        scores = score_asymmetric_eval(config, base_path, preconditioner_name)
    elif strategy == "pirate_only":
        # Score using pirate eval gradients only
        scores = _score_single_style_eval(
            config, base_path, "pirate", preconditioner_name
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return _compute_metrics_from_scores(
        scores, train_ds, eval_ds, config.minority_style
    )


def _score_single_style_eval(
    config: AsymmetricConfig,
    base_path: Path | str,
    style: str,
    preconditioner_name: str | None = None,
) -> np.ndarray:
    """Score using a single style's eval gradients.

    Helper function for scoring with pirate-only or other single styles.
    """
    import json

    import torch
    from tqdm import tqdm

    from bergson.data import load_gradients
    from bergson.gradients import GradientProcessor
    from bergson.utils.math import damped_psd_power

    base_path = Path(base_path)
    index_path = base_path / "index"
    data_path = base_path / "data"

    # Determine paths
    if style == "pirate":
        create_pirate_style_eval(config, base_path)
        eval_path = data_path / "eval_pirate_style.hf"
        grads_path = base_path / "eval_grads_pirate"
    else:
        raise ValueError(f"Unsupported style: {style}")

    if preconditioner_name:
        scores_path = base_path / f"scores_{style}_only_{preconditioner_name}"
        precond_path = base_path / preconditioner_name
    else:
        scores_path = base_path / f"scores_{style}_only_no_precond"
        precond_path = None

    # Return cached if exists
    if (scores_path / "scores.npy").exists():
        print(f"Loading cached scores from {scores_path}")
        return np.load(scores_path / "scores.npy")

    scores_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_ds = _load_hf_dataset(data_path / "train.hf")
    n_train = len(train_ds)

    eval_ds = _load_hf_dataset(eval_path)
    n_eval = len(eval_ds)

    print(
        f"Scoring {n_eval} {style} style eval queries against {n_train} train samples"
    )

    # Load train gradients
    train_grads = load_gradients(index_path, structured=True)

    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Load preconditioner if specified
    h_inv = {}
    if precond_path and (precond_path / "preconditioners.pth").exists():
        proc = GradientProcessor.load(precond_path)
        device = torch.device("cuda:0")
        for name in tqdm(module_names, desc="Computing H^(-1)"):
            H = proc.preconditioners[name].to(device=device)
            h_inv[name] = damped_psd_power(H, power=-1)

    # Prepare train gradients
    train_grad_list = []
    for name in tqdm(module_names, desc="Loading train grads"):
        g = _load_gradients_as_float(train_grads, name)
        train_grad_list.append(torch.from_numpy(g))
    train_grad_tensor = torch.cat(train_grad_list, dim=1)
    train_norms = train_grad_tensor.norm(dim=1, keepdim=True)
    train_grad_tensor = train_grad_tensor / (train_norms + 1e-8)
    train_grad_tensor = train_grad_tensor.cuda()

    # Build eval grads if needed
    index_cfg = IndexConfig.load_yaml(index_path / "index_config.yaml")

    if not grads_path.exists():
        print(f"Computing {style} style eval gradients...")
        _run_bergson_build(
            grads_path,
            model=index_cfg.model,
            dataset_path=eval_path,
            projection_dim=index_cfg.projection_dim or 16,
            label=f"{style} eval",
        )

    # Load eval gradients
    eval_grads = load_gradients(grads_path, structured=True)
    eval_grad_list = []
    for name in tqdm(module_names, desc=f"Loading {style} eval grads"):
        g = torch.from_numpy(_load_gradients_as_float(eval_grads, name))
        if h_inv:
            g = (g.cuda() @ h_inv[name]).cpu()
        eval_grad_list.append(g)

    eval_grad_tensor = torch.cat(eval_grad_list, dim=1)
    eval_norms = eval_grad_tensor.norm(dim=1, keepdim=True)
    eval_grad_tensor = eval_grad_tensor / (eval_norms + 1e-8)
    eval_grad_tensor = eval_grad_tensor.cuda()

    # Compute scores
    scores = (eval_grad_tensor @ train_grad_tensor.T).cpu().numpy()
    np.save(scores_path / "scores.npy", scores)
    print(f"Saved scores to {scores_path}")

    return scores


def run_rewrite_ablation_experiment(
    config: AsymmetricConfig | None = None,
    base_path: Path | str = "runs/asymmetric_style",
) -> dict[str, "AsymmetricMetrics"]:
    """Run the rewrite ablation experiment.

    Compares:
    - original: Score with un-stylized eval gradients
    - summed_rewrites: Sum of shakespeare + pirate eval gradients
    - shakespeare_only: Just shakespeare eval gradients (baseline)
    - pirate_only: Just pirate eval gradients
    - summed_eval (reference): Sum of minority + majority style (from main experiment)

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.

    Returns:
        Dictionary mapping strategy names to their metrics.
    """
    if config is None:
        config = AsymmetricConfig()

    base_path = Path(base_path)

    print("=" * 70)
    print("REWRITE ABLATION EXPERIMENT")
    print("=" * 70)
    print("\nThis tests whether summing two different rewrite styles helps,")
    print("even when neither rewrite matches the training distribution.")
    print("\nSetup:")
    print(f"  - Training: {config.dominant_style} style (majority)")
    print("  - Eval strategies:")
    print("    - original: un-stylized facts")
    print("    - shakespeare_only: just shakespeare rewrite")
    print("    - pirate_only: just pirate rewrite")
    print("    - summed_rewrites: shakespeare + pirate rewrites summed")
    print("    - summed_eval (reference): minority + majority style summed")
    print()

    all_metrics: dict[str, AsymmetricMetrics] = {}

    strategies = [
        ("original", "original"),
        ("shakespeare_only", "shakespeare_only"),
        ("pirate_only", "pirate_only"),
        ("summed_rewrites", "summed_rewrites"),
    ]

    for name, strategy in strategies:
        print(f"\n--- Strategy: {name} ---")
        metrics = compute_rewrite_ablation_metrics(config, base_path, strategy)
        print(f"  Top-1 Semantic: {metrics.top1_semantic_accuracy:.2%}")
        print(f"  Top-1 Style Leak: {metrics.top1_style_leakage:.2%}")
        all_metrics[name] = metrics

    # Add summed_eval reference
    print("\n--- Strategy: summed_eval (reference) ---")
    summed_metrics = compute_summed_eval_metrics(config, base_path)
    print(f"  Top-1 Semantic: {summed_metrics.top1_semantic_accuracy:.2%}")
    print(f"  Top-1 Style Leak: {summed_metrics.top1_style_leakage:.2%}")
    all_metrics["summed_eval_reference"] = summed_metrics

    # Print summary
    print("\n" + "=" * 70)
    print("REWRITE ABLATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Strategy':<25} {'Top-1 Semantic':<15} {'Top-1 Style Leak':<17}")
    print("-" * 60)

    for name, m in all_metrics.items():
        print(
            f"{name:<25} {m.top1_semantic_accuracy:<15.2%} "
            f"{m.top1_style_leakage:<17.2%}"
        )

    return all_metrics


if __name__ == "__main__":
    run_asymmetric_experiment()
