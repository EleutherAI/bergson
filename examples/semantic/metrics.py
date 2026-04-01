"""Similarity metrics computation for semantic experiments."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk

from bergson import IndexConfig, load_gradient_dataset
from bergson.data import load_gradients

from .scoring import compute_scores_with_bergson, load_scores_matrix


def build_style_lookup(include_llama: bool = False) -> dict[tuple[str, str], str]:
    """Build a lookup from (fact, reworded) -> style name.

    Args:
        include_llama: Whether to include Llama-generated styles.

    Returns:
        Dictionary mapping (fact, reworded) tuples to style names.
    """
    style_lookup: dict[tuple[str, str], str] = {}
    style_datasets = [
        ("data/facts_dataset_shakespeare-Qwen3-8B-Base.hf", "shakespeare"),
        ("data/facts_dataset_pirate-Qwen3-8B-Base.hf", "pirate"),
    ]
    if include_llama:
        style_datasets.extend(
            [
                (
                    "data/facts_dataset_shakespeare-Meta-Llama-3-8B.hf",
                    "shakespeare-llama",
                ),
                ("data/facts_dataset_pirate-Meta-Llama-3-8B.hf", "pirate-llama"),
            ]
        )
    for path, style_name in style_datasets:
        ds = load_from_disk(path)
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
        for row in ds:
            style_lookup[(row["fact"], row["reworded"])] = style_name  # type: ignore[index]
    return style_lookup


def compute_metrics_groupwise(
    index_path: Path | str,
    group_by: str = "field",  # "field" or "style"
    unit_normalize: bool = True,
) -> dict[str, Any]:
    """Compute intra/inter similarities using group-aggregated gradients.

    Groups by either field (birthdate, employer, etc.) or style (shakespeare, pirate).
    Only uses Qwen styles (excludes Llama).

    Args:
        index_path: Path to the gradient index.
        group_by: "field" or "style" - what to group by.
        unit_normalize: Whether to unit normalize gradients.

    Returns:
        Dictionary with groups, similarities matrix, and group counts.
    """
    index_path = Path(index_path)

    # Load gradient dataset with metadata
    print("Loading gradient dataset...")
    grad_ds = load_gradient_dataset(index_path, structured=True)
    print(f"  Loaded {len(grad_ds)} rows")

    # Get gradient column names
    with open(index_path / "info.json") as f:
        info = json.load(f)
    grad_columns = info["dtype"]["names"]
    print(f"  Gradient columns: {len(grad_columns)} modules")

    # Build style lookup (Qwen only, no Llama)
    print("Building style lookup (Qwen only)...")
    style_lookup = build_style_lookup(include_llama=False)

    # Use batch column access for speed
    facts = grad_ds["fact"]
    reworded = grad_ds["reworded"]
    fields = grad_ds["field"]

    print("Mapping styles...")
    styles: list[str | None] = [
        style_lookup.get((f, r), None) for f, r in zip(facts, reworded)
    ]

    # Filter to only Qwen styles (exclude Llama and unknown)
    print("Filtering to Qwen styles only...")
    keep_indices = [i for i, s in enumerate(styles) if s is not None]
    grad_ds = grad_ds.select(keep_indices)
    styles = [styles[i] for i in keep_indices]
    fields = [fields[i] for i in keep_indices]
    print(f"  Keeping {len(grad_ds)} rows")

    # Build group keys based on group_by parameter
    print(f"Building groups by {group_by}...")
    if group_by == "field":
        group_keys = fields
    elif group_by == "style":
        group_keys = styles
    else:
        raise ValueError(f"group_by must be 'field' or 'style', got {group_by}")

    # Filter out None values before sorting
    unique_groups = sorted(g for g in set(group_keys) if g is not None)
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    row_to_group = torch.tensor([group_to_idx[g] for g in group_keys if g is not None])
    print(f"  Found {len(unique_groups)} unique groups: {unique_groups}")

    # Load gradients directly from memmap (much faster than HF dataset)
    print("Loading gradients from memmap...")
    grad_mmap = load_gradients(index_path, structured=False)
    # Select only the kept rows
    all_grads = torch.from_numpy(grad_mmap[keep_indices].copy()).float()
    print(f"  Gradient tensor shape: {all_grads.shape}")

    # Compute mean gradient per group
    print("Computing mean gradients per group...")
    num_groups = len(unique_groups)
    group_grads = torch.zeros(num_groups, all_grads.shape[1], dtype=torch.float32)
    group_counts = torch.zeros(num_groups, dtype=torch.float32)

    for g_idx in range(num_groups):
        mask = row_to_group == g_idx
        group_grads[g_idx] = all_grads[mask].sum(dim=0)
        group_counts[g_idx] = mask.sum().float()

    # Average
    group_grads = group_grads / group_counts.unsqueeze(1)

    # Unit normalize if requested
    if unit_normalize:
        norms = group_grads.norm(dim=1, keepdim=True)
        group_grads = group_grads / (norms + 1e-8)

    # Compute pairwise similarities between groups
    print("Computing pairwise similarities...")
    group_grads = group_grads.cuda()
    similarities = group_grads @ group_grads.T
    similarities = similarities.cpu()
    print(f"  Similarity matrix shape: {similarities.shape}")

    # Report results
    print("\n" + "=" * 60)
    print(f"SIMILARITY MATRIX (grouped by {group_by})")
    print("=" * 60)

    # Print the full similarity matrix since it's small
    print(f"\nGroups: {unique_groups}")
    print("\nSimilarity matrix:")
    for i, g1 in enumerate(unique_groups):
        row_str = "  " + str(g1).ljust(15) + ": "
        row_str += " ".join(f"{similarities[i, j]:.3f}" for j in range(num_groups))
        print(row_str)

    # Compute intra vs inter group stats
    n = num_groups
    row_idx, col_idx = torch.triu_indices(n, n, offset=1)
    off_diag_sims = similarities[row_idx, col_idx]
    diag_sims = similarities.diag()

    print(f"\nDiagonal (self-similarity): {diag_sims.mean():.4f}")
    print(f"Off-diagonal (cross-group): {off_diag_sims.mean():.4f}")
    print(f"Difference: {diag_sims.mean() - off_diag_sims.mean():.4f}")

    return {
        "groups": unique_groups,
        "similarities": similarities,
        "group_counts": group_counts,
    }


def compute_metrics(
    index_path: Path | str,
    scores_path: Path | str | None = None,
    exclude_llama: bool = False,
    query_preconditioner_path: str | None = None,
    index_preconditioner_path: str | None = None,
) -> dict[str, float]:
    """Compute intra/inter similarities for subject (identifier) and style.

    Uses bergson score_dataset to compute pairwise similarities instead of
    custom gradient inner product implementation.

    If both query_preconditioner_path and index_preconditioner_path are given,
    they are mixed internally before scoring.

    Args:
        index_path: Path to the gradient index.
        scores_path: Optional path to precomputed scores.
        exclude_llama: Whether to exclude Llama-generated samples.
        query_preconditioner_path: Optional path to query preconditioner.
        index_preconditioner_path: Optional path to index preconditioner.

    Returns:
        Dictionary of similarity statistics.
    """
    index_path = Path(index_path)

    # Determine scores path
    if scores_path is None:
        scores_path = index_path.parent / "scores"
    else:
        scores_path = Path(scores_path)

    # Compute scores using bergson if not already done
    compute_scores_with_bergson(
        index_path,
        scores_path,
        query_preconditioner_path=query_preconditioner_path,
        index_preconditioner_path=index_preconditioner_path,
    )

    # Load metadata from HF dataset (fast)
    print("Loading metadata...")
    # Get dataset path from index config
    index_cfg = IndexConfig.load_yaml(index_path / "index_config.yaml")
    dataset_path = index_cfg.data.dataset or str(index_path / "data.hf")
    meta_ds = load_from_disk(dataset_path)
    if isinstance(meta_ds, DatasetDict):
        meta_ds = meta_ds["train"]

    # Build style lookup from individual datasets
    print("Building style lookup...")
    style_lookup: dict[tuple[str, str], str] = {}
    style_datasets = [
        ("data/facts_dataset_shakespeare-Qwen3-8B-Base.hf", "shakespeare-qwen"),
        ("data/facts_dataset_pirate-Qwen3-8B-Base.hf", "pirate-qwen"),
        ("data/facts_dataset_shakespeare-Meta-Llama-3-8B.hf", "shakespeare-llama"),
        ("data/facts_dataset_pirate-Meta-Llama-3-8B.hf", "pirate-llama"),
    ]

    for path, style_name in style_datasets:
        ds = load_from_disk(path)
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
        for row in ds:
            style_lookup[(row["fact"], row["reworded"])] = style_name  # type: ignore[index]

    # Extract metadata
    identifiers = meta_ds["identifier"]
    fields = meta_ds["field"]
    templates = meta_ds["template"]
    facts = meta_ds["fact"]
    reworded = meta_ds["reworded"]

    # Map each row to its style
    styles = [style_lookup.get((f, r), "unknown") for f, r in zip(facts, reworded)]

    # Load scores matrix from bergson output
    print("Loading scores matrix...")
    scores = load_scores_matrix(scores_path)
    n = len(scores)
    print(f"  Scores shape: {scores.shape}")

    # Filter out llama data if requested
    if exclude_llama:
        print("Excluding Llama data...")
        keep_indices = [i for i, s in enumerate(styles) if "llama" not in s]
        print(f"  Keeping {len(keep_indices)} / {len(styles)} samples")
        identifiers = [identifiers[i] for i in keep_indices]
        fields = [fields[i] for i in keep_indices]
        templates = [templates[i] for i in keep_indices]
        facts = [facts[i] for i in keep_indices]
        reworded = [reworded[i] for i in keep_indices]
        styles = [styles[i] for i in keep_indices]
        # Filter scores matrix (both rows and columns)
        scores = scores[np.ix_(keep_indices, keep_indices)]
        n = len(keep_indices)

    # Convert to torch for GPU-accelerated analysis
    print("Transferring scores to GPU...")
    similarities = torch.from_numpy(scores).cuda()

    print(f"Computing statistics for {n} samples...")

    # Convert metadata to CPU tensors
    identifiers_t = torch.tensor(identifiers)
    templates_t = torch.tensor(templates)
    field_to_idx = {f: i for i, f in enumerate(set(fields))}
    style_to_idx = {s: i for i, s in enumerate(set(styles))}
    fields_t = torch.tensor([field_to_idx[f] for f in fields])
    styles_t = torch.tensor([style_to_idx[s] for s in styles])

    # Build masks for upper triangle
    # (i < j to avoid double counting and self-similarity)
    row_idx, col_idx = torch.triu_indices(n, n, offset=1)

    # Get similarities for upper triangle pairs
    upper_sims = similarities[row_idx, col_idx].cpu()

    # Build condition masks for the pairs
    same_subject = identifiers_t[row_idx] == identifiers_t[col_idx]
    same_field = fields_t[row_idx] == fields_t[col_idx]
    same_template = templates_t[row_idx] == templates_t[col_idx]
    same_style = styles_t[row_idx] == styles_t[col_idx]

    def compute_mean(mask: torch.Tensor) -> float:
        if mask.sum() == 0:
            return 0.0
        return upper_sims[mask].mean().item()

    # Compute statistics
    stats = {
        "intra_subject": compute_mean(same_subject),
        "inter_subject": compute_mean(~same_subject),
        "intra_fact": compute_mean(same_subject & same_field),
        "inter_fact_same_subject": compute_mean(same_subject & ~same_field),
        "intra_field": compute_mean(same_field),
        "inter_field": compute_mean(~same_field),
        "intra_template": compute_mean(same_template),
        "inter_template": compute_mean(~same_template),
        "intra_style": compute_mean(same_style),
        "inter_style": compute_mean(~same_style),
    }

    # Report results
    print("\n" + "=" * 60)
    print("SEMANTIC SIMILARITY RESULTS")
    print("=" * 60)

    print("\nSubject (same person vs different person):")
    print(f"  Intra-subject mean: {stats['intra_subject']:.4f}")
    print(f"  Inter-subject mean: {stats['inter_subject']:.4f}")
    print(f"  Difference: {stats['intra_subject'] - stats['inter_subject']:.4f}")

    print("\nFact (same person+field = same underlying fact):")
    print(f"  Intra-fact mean: {stats['intra_fact']:.4f}")
    print(
        f"  Inter-fact (same person, diff field): "
        f"{stats['inter_fact_same_subject']:.4f}"
    )
    print(f"  Difference: {stats['intra_fact'] - stats['inter_fact_same_subject']:.4f}")

    print("\nField (same field type, e.g. birthdate, employer):")
    print(f"  Intra-field mean: {stats['intra_field']:.4f}")
    print(f"  Inter-field mean: {stats['inter_field']:.4f}")
    print(f"  Difference: {stats['intra_field'] - stats['inter_field']:.4f}")

    print("\nTemplate (same original phrasing template):")
    print(f"  Intra-template mean: {stats['intra_template']:.4f}")
    print(f"  Inter-template mean: {stats['inter_template']:.4f}")
    print(f"  Difference: {stats['intra_template'] - stats['inter_template']:.4f}")

    print("\nStyle (same rewording style):")
    print(f"  Intra-style mean: {stats['intra_style']:.4f}")
    print(f"  Inter-style mean: {stats['inter_style']:.4f}")
    print(f"  Difference: {stats['intra_style'] - stats['inter_style']:.4f}")

    # Interpretation:
    # - High fact difference = embeddings capture semantic content
    # - Low template difference = embeddings see through phrasing variations
    # - Low style difference = embeddings see through rewording styles
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("If embeddings capture semantics well:")
    print("  - Fact difference should be HIGH (same fact clusters)")
    print("  - Template difference should be LOW (phrasing doesn't matter)")
    print("  - Style difference should be LOW (rewording doesn't matter)")

    return stats
