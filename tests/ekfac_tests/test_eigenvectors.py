import os

import pytest
import torch
from safetensors.torch import load_file

from tests.ekfac_tests.test_utils import load_sharded_covariances


@pytest.mark.parametrize("eigenvector_type", ["activation", "gradient"])
def test_eigenvectors(
    ekfac_results_path: str,
    ground_truth_eigenvectors_path: str,
    eigenvector_type: str,
) -> None:
    """Test eigenvectors against ground truth."""
    print(f"\nTesting {eigenvector_type} eigenvectors...")

    eigenvectors_ground_truth_path = os.path.join(
        ground_truth_eigenvectors_path, f"eigenvectors_{eigenvector_type}s.safetensors"
    )
    eigenvectors_run_path = os.path.join(
        ekfac_results_path, f"eigen_{eigenvector_type}_sharded"
    )

    # load ground_truth
    ground_truth_eigenvectors = load_file(eigenvectors_ground_truth_path)

    # load run eigenvectors (sharded) and concatenate
    run_eigenvectors = load_sharded_covariances(eigenvectors_run_path)

    rtol = 1e-5
    atol = 1e-7
    all_match = True
    error_details = []

    for k in ground_truth_eigenvectors:
        gt = ground_truth_eigenvectors[k]
        run = run_eigenvectors[k]

        if not torch.allclose(gt, run, rtol=rtol, atol=atol):
            all_match = False
            diff = (gt - run).abs()
            max_diff_val = diff.max()

            # Find location of max difference
            max_diff_flat_idx = torch.argmax(diff)
            max_diff_idx = torch.unravel_index(max_diff_flat_idx, diff.shape)
            relative_diff = 100 * max_diff_val / (gt[max_diff_idx].abs() + 1e-10)

            error_details.append(
                f"  {k}: abs_diff={max_diff_val:.2e}, rel_diff={relative_diff:.2e}%"
            )

    if all_match:
        print(f"{eigenvector_type} eigenvectors match (rtol={rtol}, atol={atol})")
    else:
        error_msg = f"{eigenvector_type} eigenvectors do not match!\n" + "\n".join(
            error_details
        )
        assert False, error_msg

    print("-*" * 50)
