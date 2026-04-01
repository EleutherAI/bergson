"""Preconditioner computation and comparison utilities for semantic experiments."""

import subprocess
from pathlib import Path

import ml_dtypes  # noqa: F401  # registers bfloat16 dtype with numpy
import numpy as np
import torch
from tqdm import tqdm

from bergson.data import load_gradients
from bergson.gradients import GradientProcessor
from bergson.utils.utils import numpy_to_tensor

from .data import create_qwen_only_dataset


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


def build_style_indices(analysis_model: str = "tmp/checkpoint-282") -> None:
    """Build separate indices for pirate and shakespeare to
    get separate preconditioners.

    Args:
        analysis_model: Model to use for gradient collection.
    """
    base_path = Path("runs/precond_comparison")
    base_path.mkdir(parents=True, exist_ok=True)

    styles = [
        ("data/facts_dataset_pirate-Qwen3-8B-Base.hf", "pirate"),
        ("data/facts_dataset_shakespeare-Qwen3-8B-Base.hf", "shakespeare"),
    ]

    for dataset_path, style_name in styles:
        run_path = base_path / style_name
        if run_path.exists():
            print(f"Index already exists at {run_path}, skipping...")
            continue

        print(f"Building index for {style_name}...")
        cmd = [
            "bergson",
            "build",
            str(run_path),
            "--model",
            analysis_model,
            "--dataset",
            dataset_path,
            "--drop_columns",
            "False",
            "--prompt_column",
            "fact",
            "--completion_column",
            "reworded",
            "--fsdp",
            "--projection_dim",
            "16",
            "--token_batch_size",
            "6000",
            # NOTE: Do NOT skip preconditioners - we need them!
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"bergson build failed for {style_name}")
        print(result.stdout)

    # Also build combined index on merged Qwen-only dataset
    combined_path = base_path / "combined"
    if not combined_path.exists():
        # Ensure Qwen-only dataset exists
        qwen_dataset_path = create_qwen_only_dataset()

        print("Building combined index...")
        cmd = [
            "bergson",
            "build",
            str(combined_path),
            "--model",
            analysis_model,
            "--dataset",
            str(qwen_dataset_path),
            "--drop_columns",
            "False",
            "--prompt_column",
            "fact",
            "--completion_column",
            "reworded",
            "--fsdp",
            "--projection_dim",
            "16",
            "--token_batch_size",
            "6000",
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError("bergson build failed for combined")
        print(result.stdout)
    else:
        print(f"Combined index already exists at {combined_path}, skipping...")


def compute_between_preconditioner_covariance(
    pirate_path: Path | str,
    shakespeare_path: Path | str,
    combined_path: Path | str,
    output_path: Path | str,
) -> GradientProcessor:
    """Compute R_between = R_combined - (R_pirate + R_shakespeare) / 2.

    Mathematical reasoning:
    - R_pirate and R_shakespeare capture within-class variance only
    - R_combined captures within-class + between-class variance
    - R_between = R_combined - R_within isolates the between-class component

    This captures the "style" direction that differs between pirate and shakespeare.
    Preconditioning with this should downweight the style direction.

    Args:
        pirate_path: Path to pirate style preconditioner.
        shakespeare_path: Path to shakespeare style preconditioner.
        combined_path: Path to combined preconditioner.
        output_path: Path to save the between-class preconditioner.

    Returns:
        The computed GradientProcessor.
    """
    output_path = Path(output_path)

    # Check cache first
    if (output_path / "preconditioners.pth").exists():
        print(f"Loading cached R_between (covariance) from {output_path}")
        return GradientProcessor.load(output_path)

    print("Computing R_between preconditioner (covariance method)...")
    pirate_proc = GradientProcessor.load(Path(pirate_path))
    shakespeare_proc = GradientProcessor.load(Path(shakespeare_path))
    combined_proc = GradientProcessor.load(Path(combined_path))

    between_precs = {}
    for name in pirate_proc.preconditioners:
        R_pirate = pirate_proc.preconditioners[name]
        R_shakespeare = shakespeare_proc.preconditioners[name]
        R_combined = combined_proc.preconditioners[name]

        # R_within = average of within-class covariances
        R_within = 0.5 * R_pirate + 0.5 * R_shakespeare

        # R_between = R_combined - R_within (isolates between-class variance)
        between_precs[name] = R_combined - R_within

    # Create processor with required fields from one of the source processors
    between_proc = GradientProcessor(
        normalizers=pirate_proc.normalizers,
        preconditioners=between_precs,
        preconditioners_eigen={},
        projection_dim=pirate_proc.projection_dim,
        projection_type=pirate_proc.projection_type,
        include_bias=pirate_proc.include_bias,
    )
    between_proc.save(output_path)
    print(f"Saved R_between preconditioner to {output_path}")
    return between_proc


def compute_between_preconditioner_means(
    pirate_index_path: Path | str,
    shakespeare_index_path: Path | str,
    output_path: Path | str,
) -> GradientProcessor:
    """Compute R_between =
        (mu_pirate - mu_shakespeare)(mu_pirate - mu_shakespeare)^T per module.

    This creates a rank-1 preconditioner from the difference in class means.
    More targeted than the covariance method - captures exactly the "style direction".

    Works per-module to avoid OOM from creating the full outer product.

    Args:
        pirate_index_path: Path to pirate gradient index.
        shakespeare_index_path: Path to shakespeare gradient index.
        output_path: Path to save the between-class preconditioner.

    Returns:
        The computed GradientProcessor.
    """
    output_path = Path(output_path)

    # Check cache first
    if (output_path / "preconditioners.pth").exists():
        print(f"Loading cached R_between (means) from {output_path}")
        return GradientProcessor.load(output_path)

    print("Computing R_between preconditioner (class means method)...")

    pirate_path = Path(pirate_index_path)
    shakespeare_path = Path(shakespeare_index_path)

    # Load structured gradients (per-module) instead of flattened
    print("  Loading pirate gradients (structured)...")
    pirate_grads = load_gradients(pirate_path, structured=True)

    print("  Loading shakespeare gradients (structured)...")
    shakespeare_grads = load_gradients(shakespeare_path, structured=True)

    # Load a processor to get module names and metadata
    pirate_proc = GradientProcessor.load(pirate_path)

    # Compute per-module rank-1 preconditioners
    between_precs = {}
    module_names = list(pirate_proc.preconditioners.keys())

    print(f"  Computing per-module R_between for {len(module_names)} modules...")
    for name in tqdm(module_names):
        # Get gradients for this module (numpy structured array access)
        pirate_mod = numpy_to_tensor(pirate_grads[name]).float()
        shakespeare_mod = numpy_to_tensor(shakespeare_grads[name]).float()

        # Compute means
        mu_pirate = pirate_mod.mean(dim=0)
        mu_shakespeare = shakespeare_mod.mean(dim=0)

        # Style direction for this module
        delta = mu_pirate - mu_shakespeare

        # Rank-1 preconditioner: outer product
        between_precs[name] = torch.outer(delta, delta)

    between_proc = GradientProcessor(
        normalizers=pirate_proc.normalizers,
        preconditioners=between_precs,
        preconditioners_eigen={},
        projection_dim=pirate_proc.projection_dim,
        projection_type=pirate_proc.projection_type,
        include_bias=pirate_proc.include_bias,
    )
    between_proc.save(output_path)
    print(f"Saved R_between preconditioner (means) to {output_path}")
    return between_proc


# Default to the means-based approach as it's more targeted
compute_between_preconditioner = compute_between_preconditioner_means


def compute_mixed_preconditioner(
    pirate_path: Path | str,
    shakespeare_path: Path | str,
    output_path: Path | str,
) -> GradientProcessor:
    """Compute R_mixed = 0.5 * R_pirate + 0.5 * R_shakespeare.

    Args:
        pirate_path: Path to pirate style preconditioner.
        shakespeare_path: Path to shakespeare style preconditioner.
        output_path: Path to save the mixed preconditioner.

    Returns:
        The computed GradientProcessor.
    """
    output_path = Path(output_path)

    # Check cache first
    if (output_path / "preconditioners.pth").exists():
        print(f"Loading cached mixed preconditioner from {output_path}")
        return GradientProcessor.load(output_path)

    print("Computing mixed 50-50 preconditioner...")
    pirate_proc = GradientProcessor.load(Path(pirate_path))
    shakespeare_proc = GradientProcessor.load(Path(shakespeare_path))

    mixed_precs = {}
    for name in pirate_proc.preconditioners:
        mixed_precs[name] = (
            0.5 * pirate_proc.preconditioners[name]
            + 0.5 * shakespeare_proc.preconditioners[name]
        )

    mixed_proc = GradientProcessor(
        normalizers=pirate_proc.normalizers,
        preconditioners=mixed_precs,
        preconditioners_eigen={},
        projection_dim=pirate_proc.projection_dim,
        projection_type=pirate_proc.projection_type,
        include_bias=pirate_proc.include_bias,
    )
    mixed_proc.save(output_path)
    print(f"Saved mixed preconditioner to {output_path}")
    return mixed_proc


def compute_summed_loss_preconditioner(
    pirate_index_path: Path | str,
    shakespeare_index_path: Path | str,
    output_path: Path | str,
) -> GradientProcessor:
    """Compute preconditioner from summed loss across style contrastive pairs.

    Instead of computing gradients separately and then averaging, this approach
    conceptually sums the loss across contrastive pairs before computing gradients.
    For paired samples with the same underlying fact but different styles:
    - g_summed = g_pirate + g_shakespeare (for same fact)
    - R_summed = sum over pairs of outer(g_summed, g_summed)

    This captures the common (semantic) direction by reinforcing what's shared.

    Args:
        pirate_index_path: Path to pirate gradient index.
        shakespeare_index_path: Path to shakespeare gradient index.
        output_path: Path to save the preconditioner.

    Returns:
        The computed GradientProcessor.
    """
    from datasets import load_from_disk

    output_path = Path(output_path)

    # Check cache first
    if (output_path / "preconditioners.pth").exists():
        print(f"Loading cached summed loss preconditioner from {output_path}")
        return GradientProcessor.load(output_path)

    print("Computing summed loss preconditioner from style contrastive pairs...")

    pirate_path = Path(pirate_index_path)
    shakespeare_path = Path(shakespeare_index_path)

    # Load structured gradients
    print("  Loading pirate gradients...")
    pirate_grads = load_gradients(pirate_path, structured=True)
    print("  Loading shakespeare gradients...")
    shakespeare_grads = load_gradients(shakespeare_path, structured=True)

    # Load datasets to match facts
    pirate_ds = load_from_disk(
        str(pirate_path.parent / "pirate" / "dataset")
        if (pirate_path.parent / "pirate" / "dataset").exists()
        else "data/facts_dataset_pirate-Qwen3-8B-Base.hf"
    )
    shakespeare_ds = load_from_disk(
        str(shakespeare_path.parent / "shakespeare" / "dataset")
        if (shakespeare_path.parent / "shakespeare" / "dataset").exists()
        else "data/facts_dataset_shakespeare-Qwen3-8B-Base.hf"
    )

    if hasattr(pirate_ds, "keys"):
        pirate_ds = pirate_ds["train"]
    if hasattr(shakespeare_ds, "keys"):
        shakespeare_ds = shakespeare_ds["train"]

    # Build fact -> index mapping
    pirate_facts = pirate_ds["fact"]  # type: ignore[index]
    shakespeare_facts = shakespeare_ds["fact"]  # type: ignore[index]

    pirate_fact_to_idx = {f: i for i, f in enumerate(pirate_facts)}
    shakespeare_fact_to_idx = {f: i for i, f in enumerate(shakespeare_facts)}

    # Find common facts (contrastive pairs) and build aligned index arrays
    common_facts = list(
        set(pirate_fact_to_idx.keys()) & set(shakespeare_fact_to_idx.keys())
    )
    pirate_indices = [pirate_fact_to_idx[f] for f in common_facts]
    shakespeare_indices = [shakespeare_fact_to_idx[f] for f in common_facts]
    print(f"  Found {len(common_facts)} contrastive pairs")

    # Load a processor to get metadata
    pirate_proc = GradientProcessor.load(pirate_path)
    module_names = list(pirate_proc.preconditioners.keys())

    # Compute per-module preconditioners from summed gradients (batched)
    summed_precs = {}
    print(f"  Computing per-module preconditioners for {len(module_names)} modules...")

    for name in tqdm(module_names):
        pirate_mod = numpy_to_tensor(pirate_grads[name]).float()
        shakespeare_mod = numpy_to_tensor(shakespeare_grads[name]).float()

        # Extract aligned pairs using fancy indexing (batched)
        g_pirate_aligned = pirate_mod[pirate_indices]  # [n_pairs, d]
        g_shakespeare_aligned = shakespeare_mod[shakespeare_indices]  # [n_pairs, d]

        # Sum gradients across contrastive pairs
        g_summed = g_pirate_aligned + g_shakespeare_aligned  # [n_pairs, d]

        # Compute covariance: (1/n) * G^T @ G = sum of outer products / n
        R = g_summed.T @ g_summed / len(common_facts)  # [d, d]
        summed_precs[name] = R

    summed_proc = GradientProcessor(
        normalizers=pirate_proc.normalizers,
        preconditioners=summed_precs,
        preconditioners_eigen={},
        projection_dim=pirate_proc.projection_dim,
        projection_type=pirate_proc.projection_type,
        include_bias=pirate_proc.include_bias,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    summed_proc.save(output_path)
    print(f"Saved summed loss preconditioner to {output_path}")
    return summed_proc


def compute_pca_style_subspace(
    pirate_index_path: Path | str,
    shakespeare_index_path: Path | str,
    output_path: Path | str,
    top_k: int = 10,
    exclude_facts: set[str] | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Compute the style subspace from pairwise gradient differences using PCA.

    For each contrastive pair (same fact, different styles):
    - Δg = g_pirate - g_shakespeare
    Stacks all Δg into a matrix and computes PCA to find the top-k principal
    components that capture the "style direction".

    Args:
        pirate_index_path: Path to pirate gradient index.
        shakespeare_index_path: Path to shakespeare gradient index.
        output_path: Path to save the style subspace.
        top_k: Number of top principal components to keep.
        exclude_facts: Optional set of fact strings to exclude from PCA computation.
            Use this to prevent data leakage when eval facts overlap with PCA data.

    Returns:
        Dictionary mapping module names to (eigenvectors, eigenvalues) tuples.
        eigenvectors has shape [d, k] where columns are the top-k style directions.
    """
    from datasets import load_from_disk

    output_path = Path(output_path)
    # Use different cache file when excluding facts to avoid mixing leaked/non-leaked
    cache_suffix = "_noleak" if exclude_facts else ""
    cache_file = output_path / f"style_subspace_k{top_k}{cache_suffix}.pth"

    if cache_file.exists():
        print(f"Loading cached style subspace from {cache_file}")
        return torch.load(cache_file, weights_only=True)

    print(f"Computing style subspace via PCA (top_k={top_k})...")

    pirate_path = Path(pirate_index_path)
    shakespeare_path = Path(shakespeare_index_path)

    # Load structured gradients
    print("  Loading pirate gradients...")
    pirate_grads = load_gradients(pirate_path, structured=True)
    print("  Loading shakespeare gradients...")
    shakespeare_grads = load_gradients(shakespeare_path, structured=True)

    # Load datasets to match facts
    pirate_ds = load_from_disk("data/facts_dataset_pirate-Qwen3-8B-Base.hf")
    shakespeare_ds = load_from_disk("data/facts_dataset_shakespeare-Qwen3-8B-Base.hf")

    if hasattr(pirate_ds, "keys"):
        pirate_ds = pirate_ds["train"]
    if hasattr(shakespeare_ds, "keys"):
        shakespeare_ds = shakespeare_ds["train"]

    # Build fact -> index mapping
    pirate_facts = pirate_ds["fact"]  # type: ignore[index]
    shakespeare_facts = shakespeare_ds["fact"]  # type: ignore[index]

    pirate_fact_to_idx = {f: i for i, f in enumerate(pirate_facts)}
    shakespeare_fact_to_idx = {f: i for i, f in enumerate(shakespeare_facts)}

    # Find common facts and build aligned index arrays
    common_facts = set(pirate_fact_to_idx.keys()) & set(shakespeare_fact_to_idx.keys())

    # Exclude eval facts to prevent data leakage
    if exclude_facts:
        n_before = len(common_facts)
        common_facts = common_facts - exclude_facts
        n_excluded = n_before - len(common_facts)
        print(f"  Excluded {n_excluded} facts to prevent data leakage")

    common_facts = list(common_facts)
    pirate_indices = [pirate_fact_to_idx[f] for f in common_facts]
    shakespeare_indices = [shakespeare_fact_to_idx[f] for f in common_facts]
    print(f"  Found {len(common_facts)} contrastive pairs")

    # Get module names from processor
    pirate_proc = GradientProcessor.load(pirate_path)
    module_names = list(pirate_proc.preconditioners.keys())

    style_subspace = {}
    variance_pcts: list[float] = []
    n_capped = 0
    print(f"  Computing PCA for {len(module_names)} modules...")

    for name in tqdm(module_names):
        pirate_mod = numpy_to_tensor(pirate_grads[name]).float()
        shakespeare_mod = numpy_to_tensor(shakespeare_grads[name]).float()

        # Extract aligned pairs using fancy indexing (batched)
        g_pirate_aligned = pirate_mod[pirate_indices]  # [n_pairs, d]
        g_shakespeare_aligned = shakespeare_mod[shakespeare_indices]  # [n_pairs, d]

        # Compute gradient differences (batched)
        diff_matrix = g_pirate_aligned - g_shakespeare_aligned  # [n_pairs, d]

        # Center the differences (mean-subtract)
        diff_centered = diff_matrix - diff_matrix.mean(dim=0, keepdim=True)

        # Compute covariance matrix: (1/n) * D^T @ D
        n = diff_centered.shape[0]
        cov = diff_centered.T @ diff_centered / n  # [d, d]

        # Eigendecomposition (sorted ascending)
        eigvals, eigvecs = torch.linalg.eigh(cov)

        # Get top-k (largest eigenvalues are at the end)
        d = eigvals.shape[0]
        k = min(top_k, d)
        top_eigvals = eigvals[-k:].flip(0)  # Descending order
        top_eigvecs = eigvecs[:, -k:].flip(
            1
        )  # [d, k], columns are principal components

        style_subspace[name] = (top_eigvecs, top_eigvals)

        # Track variance explained for reporting
        total_var = eigvals.sum().item()
        explained_var = top_eigvals.sum().item()
        pct = explained_var / total_var * 100 if total_var > 0 else 0.0
        variance_pcts.append(pct)
        if k >= d:
            n_capped += 1

    # Print variance explained summary
    if variance_pcts:
        mean_var = sum(variance_pcts) / len(variance_pcts)
        sorted_pcts = sorted(variance_pcts)
        mid = len(sorted_pcts) // 2
        median_var = (
            sorted_pcts[mid]
            if len(sorted_pcts) % 2 == 1
            else (sorted_pcts[mid - 1] + sorted_pcts[mid]) / 2
        )
        print(f"\n  PCA variance explained (k={top_k}):")
        print(f"    Mean across modules:   {mean_var:.1f}%")
        print(f"    Median across modules: {median_var:.1f}%")
        print(f"    Modules where k >= dim (capped): {n_capped}/{len(variance_pcts)}")

    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(style_subspace, cache_file)
    print(f"Saved style subspace to {cache_file}")
    return style_subspace


def report_pca_variance(
    pirate_index_path: Path | str,
    shakespeare_index_path: Path | str,
    output_path: Path | str,
    k_values: list[int],
    exclude_facts: set[str] | None = None,
) -> dict[int, dict[str, float]]:
    """Compute all eigenvalues once and report variance explained for each k.

    This avoids recomputing PCA for each k value. Computes the full
    eigendecomposition once per module and reports what fraction of variance
    each k captures.

    Args:
        pirate_index_path: Path to pirate gradient index.
        shakespeare_index_path: Path to shakespeare gradient index.
        output_path: Path for cache/output (unused for caching here).
        k_values: List of k values to report variance for.
        exclude_facts: Optional set of fact strings to exclude.

    Returns:
        Dictionary mapping k -> {"mean_pct": float, "median_pct": float,
        "n_capped": int, "n_modules": int}.
    """
    from datasets import load_from_disk

    pirate_path = Path(pirate_index_path)
    shakespeare_path = Path(shakespeare_index_path)

    print("Computing PCA variance analysis...")
    print("  Loading pirate gradients...")
    pirate_grads = load_gradients(pirate_path, structured=True)
    print("  Loading shakespeare gradients...")
    shakespeare_grads = load_gradients(shakespeare_path, structured=True)

    pirate_ds = load_from_disk("data/facts_dataset_pirate-Qwen3-8B-Base.hf")
    shakespeare_ds = load_from_disk("data/facts_dataset_shakespeare-Qwen3-8B-Base.hf")

    if hasattr(pirate_ds, "keys"):
        pirate_ds = pirate_ds["train"]
    if hasattr(shakespeare_ds, "keys"):
        shakespeare_ds = shakespeare_ds["train"]

    pirate_facts = pirate_ds["fact"]  # type: ignore[index]
    shakespeare_facts = shakespeare_ds["fact"]  # type: ignore[index]

    pirate_fact_to_idx = {f: i for i, f in enumerate(pirate_facts)}
    shakespeare_fact_to_idx = {f: i for i, f in enumerate(shakespeare_facts)}

    common_facts = set(pirate_fact_to_idx.keys()) & set(shakespeare_fact_to_idx.keys())
    if exclude_facts:
        n_before = len(common_facts)
        common_facts = common_facts - exclude_facts
        print(
            f"  Excluded {n_before - len(common_facts)} facts to prevent data leakage"
        )

    common_facts_list = list(common_facts)
    pirate_indices = [pirate_fact_to_idx[f] for f in common_facts_list]
    shakespeare_indices = [shakespeare_fact_to_idx[f] for f in common_facts_list]
    print(f"  Found {len(common_facts_list)} contrastive pairs")

    pirate_proc = GradientProcessor.load(pirate_path)
    module_names = list(pirate_proc.preconditioners.keys())

    # For each k, track per-module variance explained percentages
    results: dict[int, dict[str, float]] = {}
    per_k_pcts: dict[int, list[float]] = {k: [] for k in k_values}
    per_k_capped: dict[int, int] = {k: 0 for k in k_values}

    print(f"  Computing eigendecomposition for {len(module_names)} modules...")
    for name in tqdm(module_names):
        pirate_mod = numpy_to_tensor(pirate_grads[name]).float()
        shakespeare_mod = numpy_to_tensor(shakespeare_grads[name]).float()

        g_pirate_aligned = pirate_mod[pirate_indices]
        g_shakespeare_aligned = shakespeare_mod[shakespeare_indices]
        diff_matrix = g_pirate_aligned - g_shakespeare_aligned
        diff_centered = diff_matrix - diff_matrix.mean(dim=0, keepdim=True)

        n = diff_centered.shape[0]
        cov = diff_centered.T @ diff_centered / n
        eigvals, _ = torch.linalg.eigh(cov)

        total_var = eigvals.sum().item()
        d = eigvals.shape[0]

        for k in k_values:
            k_actual = min(k, d)
            top_k_var = eigvals[-k_actual:].sum().item()
            pct = top_k_var / total_var * 100 if total_var > 0 else 0.0
            per_k_pcts[k].append(pct)
            if k >= d:
                per_k_capped[k] += 1

    # Summarize
    print(f"\n{'='*60}")
    print("PCA VARIANCE EXPLAINED ANALYSIS")
    print(f"{'='*60}")
    print(f"  Modules: {len(module_names)}, Per-module dimension: {d}")
    print(f"  Contrastive pairs: {len(common_facts_list)}")

    print(f"\n  {'k':<8} {'Mean %':<12} {'Median %':<12} {'Capped':<15}")
    print(f"  {'-'*47}")

    for k in k_values:
        pcts = per_k_pcts[k]
        mean_pct = sum(pcts) / len(pcts)
        sorted_pcts = sorted(pcts)
        mid = len(sorted_pcts) // 2
        median_pct = (
            sorted_pcts[mid]
            if len(sorted_pcts) % 2 == 1
            else (sorted_pcts[mid - 1] + sorted_pcts[mid]) / 2
        )
        capped = per_k_capped[k]
        print(
            f"  {k:<8} {mean_pct:<12.1f} {median_pct:<12.1f} "
            f"{capped}/{len(module_names)}"
        )
        results[k] = {
            "mean_pct": mean_pct,
            "median_pct": median_pct,
            "n_capped": capped,
            "n_modules": len(module_names),
        }

    return results


def project_orthogonal_to_style_subspace(
    grads: torch.Tensor,
    style_eigenvecs: torch.Tensor,
) -> torch.Tensor:
    """Project gradients onto the orthogonal complement of the style subspace.

    Given gradients g and style subspace basis V (columns are principal components),
    computes: g_projected = g - V @ V^T @ g

    This removes the component of g that lies in the style subspace.

    Args:
        grads: Gradient tensor of shape [n, d].
        style_eigenvecs: Eigenvectors defining style subspace, shape [d, k].

    Returns:
        Projected gradients of shape [n, d].
    """
    # V @ V^T is the projection matrix onto the style subspace
    # I - V @ V^T is the projection onto the orthogonal complement
    # g_proj = g - V @ (V^T @ g)
    style_component = grads @ style_eigenvecs @ style_eigenvecs.T
    return grads - style_component


def apply_pca_projection_to_eval_grads(
    eval_grads: dict[str, torch.Tensor],
    style_subspace: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Apply PCA style projection to evaluation gradients.

    Projects eval gradients onto the orthogonal complement of the style subspace,
    effectively removing the style direction before computing influence.

    Args:
        eval_grads: Dictionary mapping module names to gradient tensors [n, d].
        style_subspace: Dictionary mapping module names to (eigenvecs, eigvals) tuples.
        device: Device to use for computation.

    Returns:
        Dictionary of projected gradients.
    """
    projected = {}
    for name, grads in eval_grads.items():
        if name in style_subspace:
            eigvecs, _ = style_subspace[name]
            if device is not None:
                grads = grads.to(device)
                eigvecs = eigvecs.to(device)
            projected[name] = project_orthogonal_to_style_subspace(grads, eigvecs)
        else:
            projected[name] = grads
    return projected


def compute_eval_preconditioner(
    eval_grads_path: Path | str,
    output_path: Path | str,
    reference_proc_path: Path | str | None = None,
) -> GradientProcessor:
    """Compute second moment matrix from eval gradients.

    R_eval = (1/n) * G_eval^T @ G_eval

    Args:
        eval_grads_path: Path to eval gradients index.
        output_path: Path to save the preconditioner.
        reference_proc_path: Path to a reference processor
        for module names (if eval has none).

    Returns:
        The computed GradientProcessor.
    """
    import json

    output_path = Path(output_path)

    # Check cache first
    if (output_path / "preconditioners.pth").exists():
        print(f"Loading cached eval preconditioner from {output_path}")
        return GradientProcessor.load(output_path)

    print("Computing eval second moment preconditioner...")

    eval_path = Path(eval_grads_path)

    # Load structured gradients
    print("  Loading eval gradients...")
    eval_grads = load_gradients(eval_path, structured=True)

    # Get module names from info.json
    with open(eval_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Load a reference processor to get metadata
    # (use reference if eval doesn't have precs)
    if reference_proc_path:
        base_proc = GradientProcessor.load(Path(reference_proc_path))
    else:
        base_proc = GradientProcessor.load(eval_path)

    # Compute per-module second moment matrices
    eval_precs = {}
    print(f"  Computing per-module preconditioners for {len(module_names)} modules...")

    for name in tqdm(module_names):
        g = torch.from_numpy(_load_gradients_as_float(eval_grads, name))
        n = g.shape[0]
        # Second moment: (1/n) * G^T @ G
        R = g.T @ g / n
        eval_precs[name] = R

    eval_proc = GradientProcessor(
        normalizers=base_proc.normalizers,
        preconditioners=eval_precs,
        preconditioners_eigen={},
        projection_dim=base_proc.projection_dim,
        projection_type=base_proc.projection_type,
        include_bias=base_proc.include_bias,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    eval_proc.save(output_path)
    print(f"Saved eval preconditioner to {output_path}")
    return eval_proc


def compute_train_eval_mixed_preconditioner(
    train_index_path: Path | str,
    eval_grads_path: Path | str,
    output_path: Path | str,
    train_weight: float = 0.5,
) -> GradientProcessor:
    """Compute 50:50 (or custom weighted) mixture of train and eval second moments.

    R_mixed = train_weight * R_train + (1 - train_weight) * R_eval

    Args:
        train_index_path: Path to train gradients index.
        eval_grads_path: Path to eval gradients index.
        output_path: Path to save the preconditioner.
        train_weight: Weight for train preconditioner (default 0.5).

    Returns:
        The computed GradientProcessor.
    """
    import json

    output_path = Path(output_path)

    # Check cache first
    if (output_path / "preconditioners.pth").exists():
        print(f"Loading cached train-eval mixed preconditioner from {output_path}")
        return GradientProcessor.load(output_path)

    print(
        f"Computing train-eval mixed preconditioner ({train_weight:.0%} "
        f"train, {1-train_weight:.0%} eval)..."
    )

    train_path = Path(train_index_path)
    eval_path = Path(eval_grads_path)

    # Load structured gradients
    print("  Loading train gradients...")
    train_grads = load_gradients(train_path, structured=True)
    print("  Loading eval gradients...")
    eval_grads = load_gradients(eval_path, structured=True)

    # Load a processor to get metadata and module names
    base_proc = GradientProcessor.load(train_path)

    # Get module names from info.json
    with open(train_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Compute per-module mixed second moment matrices
    mixed_precs = {}
    print(f"  Computing per-module preconditioners for {len(module_names)} modules...")

    for name in tqdm(module_names):
        g_train = torch.from_numpy(_load_gradients_as_float(train_grads, name))
        g_eval = torch.from_numpy(_load_gradients_as_float(eval_grads, name))

        n_train = g_train.shape[0]
        n_eval = g_eval.shape[0]

        # Second moments
        R_train = g_train.T @ g_train / n_train
        R_eval = g_eval.T @ g_eval / n_eval

        # Weighted mixture
        R_mixed = train_weight * R_train + (1 - train_weight) * R_eval
        mixed_precs[name] = R_mixed

    mixed_proc = GradientProcessor(
        normalizers=base_proc.normalizers,
        preconditioners=mixed_precs,
        preconditioners_eigen={},
        projection_dim=base_proc.projection_dim,
        projection_type=base_proc.projection_type,
        include_bias=base_proc.include_bias,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    mixed_proc.save(output_path)
    print(f"Saved train-eval mixed preconditioner to {output_path}")
    return mixed_proc
