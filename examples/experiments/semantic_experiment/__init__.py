"""Semantic experiments for analyzing gradient-based embeddings.

This package provides tools for:
- Creating reworded datasets in different styles (Shakespeare, Pirate)
- Computing pairwise similarity scores from gradient embeddings
- Analyzing semantic vs stylistic similarity patterns
- Comparing different preconditioning strategies
- Asymmetric style distribution experiments for style suppression validation
"""

from .asymmetric import (
    AsymmetricConfig,
    AsymmetricMetrics,
    compute_asymmetric_metrics,
    compute_style_hessian,
    create_asymmetric_dataset,
    create_asymmetric_index,
    print_metrics,
    run_asymmetric_experiment,
    score_asymmetric_eval,
)
from .attribute_preservation import (
    AttributePreservationConfig,
    AttributePreservationMetrics,
    compute_attribute_metrics,
    create_attribute_dataset,
    create_attribute_index,
    create_styled_datasets,
    print_attribute_metrics,
    run_attribute_preservation_experiment,
    score_attribute_eval,
)
from .data import create_data, create_qwen_only_dataset, reword
from .experiment import (
    create_index,
    finetune,
    main,
    run_hessian_comparison,
)
from .hessians import (
    build_style_indices,
    compute_between_hessian,
    compute_between_hessian_covariance,
    compute_between_hessian_means,
    compute_mixed_hessian,
)
from .metrics import build_style_lookup, compute_metrics, compute_metrics_groupwise
from .scoring import (
    compute_scores_fast,
    compute_scores_with_bergson,
    load_scores_matrix,
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
    # Asymmetric style experiment
    "AsymmetricConfig",
    "AsymmetricMetrics",
    "create_asymmetric_dataset",
    "create_asymmetric_index",
    "score_asymmetric_eval",
    "compute_asymmetric_metrics",
    "compute_style_hessian",
    "print_metrics",
    "run_asymmetric_experiment",
    # Attribute preservation experiment
    "AttributePreservationConfig",
    "AttributePreservationMetrics",
    "create_attribute_dataset",
    "create_attribute_index",
    "create_styled_datasets",
    "score_attribute_eval",
    "compute_attribute_metrics",
    "print_attribute_metrics",
    "run_attribute_preservation_experiment",
]
