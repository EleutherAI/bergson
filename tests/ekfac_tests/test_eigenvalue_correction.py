import os

import pytest
import torch
from safetensors.torch import load_file

from bergson.hessians.utils import TensorDict


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
    lambda_ground_truth = TensorDict(load_file(lambda_ground_truth_path))

    world_size = len(os.listdir(lambda_run_path))  # number of shards
    lambda_run_shards_path = [
        os.path.join(lambda_run_path, f"shard_{rank}.safetensors")
        for rank in range(world_size)
    ]
    lambda_list_shards = [
        (load_file(shard_path)) for shard_path in lambda_run_shards_path
    ]
    lambda_run = {}
    for k, v in lambda_list_shards[0].items():
        if len(v.shape) == 0:
            lambda_run[k] = v
        else:
            lambda_run[k] = torch.cat([shard[k] for shard in lambda_list_shards], dim=0)

    lambda_run = TensorDict(lambda_run)

    total_processed_run_path = os.path.join(
        ekfac_results_path, "total_processed_lambda_correction.pt"
    )
    lambda_device = lambda_run[list(lambda_run.keys())[0]].device
    total = torch.load(total_processed_run_path, map_location=lambda_device)
    lambda_run.div_(total)
    rtol = 1e-10
    equal_dict = lambda_ground_truth.allclose(lambda_run, rtol=rtol)

    if all(equal_dict.values()):
        print("Eigenvalue corrections match!")
    else:
        diff = lambda_ground_truth.sub(lambda_run).div(lambda_ground_truth).abs()
        max_diff = diff.max()
        # Collect error details for assertion message
        error_details = []
        has_significant_errors = False

        for k, v in equal_dict.items():
            if not v:
                # Find location of max difference
                coord = diff[k].argmax()
                a, b = (
                    coord // lambda_ground_truth[k].shape[1],
                    coord % lambda_ground_truth[k].shape[1],
                )

                if max_diff[k] < 1e-3:
                    error_details.append(
                        f"  {k}: small differences within tolerance (max_rel_diff={(100 * max_diff[k]):.3f}%)"
                    )
                else:
                    has_significant_errors = True
                    error_details.append(
                        f"  {k}: max_rel_diff={(100 * max_diff[k]):.3f}%, "
                        f"mean={(100 * diff[k].mean()):.3f}%"
                    )
                    error_details.append(
                        f"    at [{a},{b}]: gt={lambda_ground_truth[k][a, b]:.3e}, "
                        f"run={lambda_run[k][a, b]:.3e}"
                    )

        if has_significant_errors:
            error_msg = "Eigenvalue corrections do not match!\n" + "\n".join(
                error_details
            )
            assert False, error_msg
        else:
            print("✓ Eigenvalue corrections: all differences within tolerance")
