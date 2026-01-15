"""Test EKFAC computation against ground truth."""

import json
import os

import torch


def test_total_processed_examples(
    ground_truth_covariances_path: str, ekfac_results_path: str
) -> None:
    """Test that total processed examples match between ground truth and computed."""
    total_processed_ground_truth_path = os.path.join(
        ground_truth_covariances_path, "stats.json"
    )
    total_processed_run_path = os.path.join(ekfac_results_path, "total_processed.pt")

    with open(total_processed_ground_truth_path, "r") as f:
        ground_truth_data = json.load(f)
        total_processed_ground_truth = ground_truth_data["total_processed_global"]

    total_processed_run = torch.load(total_processed_run_path, weights_only=True).item()

    assert total_processed_ground_truth == total_processed_run, (
        f"Total processed examples do not match! "
        f"Ground truth: {total_processed_ground_truth}, Run: {total_processed_run}"
    )
    print(f"✓ Total processed examples match: {total_processed_ground_truth}")
