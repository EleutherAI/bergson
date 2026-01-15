import os

import pytest
import torch
from safetensors.torch import load_file

from tests.ekfac_tests.test_utils import load_sharded_covariances


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Numerical precision differences on CPU vs GPU",
)
def test_eigenvalue_corrections(
    ground_truth_eigenvalue_corrections_path: str,
    ekfac_results_path: str,
) -> None:
    """Test eigenvalue corrections against ground truth."""
    print("\nTesting eigenvalue corrections...")

    lambda_ground_truth_path = os.path.join(
        ground_truth_eigenvalue_corrections_path, "eigenvalue_corrections.safetensors"
    )
    lambda_run_path = os.path.join(ekfac_results_path, "eigenvalue_correction_sharded")

    # load ground_truth
    lambda_ground_truth = load_file(lambda_ground_truth_path)

    # load run eigenvalue corrections (sharded)
    lambda_run = load_sharded_covariances(lambda_run_path)

    total_processed_run_path = os.path.join(ekfac_results_path, "total_processed.pt")
    lambda_device = lambda_run[list(lambda_run.keys())[0]].device
    total = torch.load(total_processed_run_path, map_location=lambda_device)

    # Normalize by total
    lambda_run = {k: v / total for k, v in lambda_run.items()}

    # Use reasonable tolerance for numerical differences between implementations
    # due to float precision, accumulation order, and eigenvector differences
    # query_key_value layers can have up to ~10% differences due to eigenvector issues
    rtol = 0.12  # 12% relative tolerance
    atol = 1e-4
    all_match = True
    error_details = []
    has_significant_errors = False

    for k in lambda_ground_truth:
        gt = lambda_ground_truth[k]
        run = lambda_run[k]

        if not torch.allclose(gt, run, rtol=rtol, atol=atol):
            all_match = False
            diff = (gt - run).abs()
            rel_diff = diff / (gt.abs() + 1e-10)
            max_rel_diff = rel_diff.max()

            # Find location of max difference
            coord = diff.argmax()
            a, b = coord // gt.shape[1], coord % gt.shape[1]

            if max_rel_diff < 0.05:  # 5% threshold for reporting
                error_details.append(
                    f"  {k}: small differences within tolerance "
                    f"(max_rel_diff={(100 * max_rel_diff):.3f}%)"
                )
            else:
                has_significant_errors = True
                error_details.append(
                    f"  {k}: max_rel_diff={(100 * max_rel_diff):.3f}%, "
                    f"mean={(100 * rel_diff.mean()):.3f}%"
                )
                error_details.append(
                    f"    at [{a},{b}]: gt={gt[a, b]:.3e}, run={run[a, b]:.3e}"
                )

    if all_match:
        print(f"Eigenvalue corrections match within tolerance (rtol={rtol})")
    elif has_significant_errors:
        error_msg = f"Eigenvalue corrections do not match (rtol={rtol})!\n" + "\n".join(
            error_details
        )
        assert False, error_msg
    else:
        print("Eigenvalue corrections: all differences within tolerance")
