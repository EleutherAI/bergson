"""Backward-compatible wrapper for semantic experiments.

This module re-exports all functions from the semantic package for backward
compatibility. New code should import directly from
examples.semantic_experiment instead.
"""

# Re-export everything from the semantic package
from examples.semantic_experiment import (
    build_style_indices,
    build_style_lookup,
    compute_between_hessian,
    compute_between_hessian_covariance,
    compute_between_hessian_means,
    compute_metrics,
    compute_metrics_groupwise,
    compute_mixed_hessian,
    compute_scores_fast,
    compute_scores_with_bergson,
    create_data,
    create_index,
    create_qwen_only_dataset,
    finetune,
    load_scores_matrix,
    main,
    reword,
    run_hessian_comparison,
)

__all__ = [
    # Data creation
    "reword",
    "create_data",
    "create_qwen_only_dataset",
    # Scoring
    "load_scores_matrix",
    "compute_scores_fast",
    "compute_scores_with_bergson",
    # Metrics
    "build_style_lookup",
    "compute_metrics_groupwise",
    "compute_metrics",
    # Hessians
    "build_style_indices",
    "compute_between_hessian_covariance",
    "compute_between_hessian_means",
    "compute_between_hessian",
    "compute_mixed_hessian",
    # Experiment
    "create_index",
    "finetune",
    "run_hessian_comparison",
    "main",
]

if __name__ == "__main__":
    main()
