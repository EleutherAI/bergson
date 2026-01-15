import os

import pytest
import torch
from safetensors.torch import load_file

from tests.ekfac_tests.test_utils import load_sharded_covariances


@pytest.mark.parametrize("covariance_type", ["activation", "gradient"])
def test_covariances(
    ekfac_results_path: str,
    ground_truth_covariances_path: str,
    covariance_type: str,
) -> None:
    """Test covariances against ground truth."""
    print(f"\nTesting {covariance_type} covariances...")

    covariances_ground_truth_path = os.path.join(
        ground_truth_covariances_path, f"{covariance_type}_covariance.safetensors"
    )
    covariances_run_path = os.path.join(
        ekfac_results_path, f"{covariance_type}_sharded"
    )

    ground_truth_covariances = load_file(covariances_ground_truth_path)
    run_covariances = load_sharded_covariances(covariances_run_path)

    rtol = 1e-10
    atol = 0
    all_match = True
    error_details = []

    for k in ground_truth_covariances:
        gt = ground_truth_covariances[k]
        run = run_covariances[k]

        if not torch.allclose(gt, run, rtol=rtol, atol=atol):
            all_match = False
            diff = (gt - run).abs()
            rel_diff = diff / (gt.abs() + 1e-10)
            error_details.append(
                f"  {k}: max_rel_diff={100 * rel_diff.max():.3f}%, "
                f"mean={100 * rel_diff.mean():.3f}%"
            )

    if all_match:
        print(f"{covariance_type} covariances match within tolerance (rtol={rtol})")
    else:
        error_msg = (
            f"{covariance_type} covariances do not match (rtol={rtol})!\n"
            + "\n".join(error_details)
        )
        assert False, error_msg

    print("-*" * 50)
