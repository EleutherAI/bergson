import os

import pytest
from safetensors.torch import load_file

from bergson.hessians.utils import TensorDict
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
        ekfac_results_path, f"{covariance_type}_covariance_sharded"
    )

    ground_truth_covariances = TensorDict(load_file(covariances_ground_truth_path))
    run_covariances = TensorDict(load_sharded_covariances(covariances_run_path))

    diff = (
        ground_truth_covariances.sub(run_covariances)
        .div(ground_truth_covariances)
        .abs()
    )

    rtol = 1e-10
    atol = 0
    equal_dict = ground_truth_covariances.allclose(
        run_covariances, rtol=rtol, atol=atol
    )

    if all(equal_dict.values()):
        print(f"{covariance_type} covariances match")
    else:
        max_diff = diff.max()
        # Collect error details for assertion message
        error_details = []
        for k, v in equal_dict.items():
            if not v:
                error_details.append(
                    f"  {k}: max_rel_diff={(100 * max_diff[k]):.3f}%, "
                    f"mean={(100 * diff[k].mean()):.3f}%"
                )

        error_msg = f"{covariance_type} covariances do not match!\n" + "\n".join(
            error_details
        )
        assert False, error_msg

    print("-*" * 50)
