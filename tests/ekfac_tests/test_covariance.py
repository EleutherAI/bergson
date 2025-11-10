import os
from typing import Literal

import torch
from safetensors.torch import load_file

from bergson.hessians.utils import TensorDict


def test_covariances(
    ground_truth_path,
    run_path,
    covariance_type: Literal["activation", "gradient"] = "activation",
):
    covariances_ground_truth_path = os.path.join(
        ground_truth_path, f"covariances/{covariance_type}_covariance.safetensors"
    )
    covariances_run_path = os.path.join(
        run_path, f"{covariance_type}_covariance_sharded"
    )

    # load ground_truth
    ground_truth_covariances = TensorDict(load_file(covariances_ground_truth_path))

    world_size = len(os.listdir(covariances_run_path))  # number of shards
    # load run covariances shards and concatenate them

    run_covariances_shards = [
        os.path.join(covariances_run_path, f"shard_{rank}.safetensors")
        for rank in range(world_size)
    ]
    run_covariances_list = [(load_file(shard)) for shard in run_covariances_shards]
    run_covariances = {}
    for k, v in run_covariances_list[0].items():
        run_covariances[k] = torch.cat(
            [shard[k] for shard in run_covariances_list], dim=0
        )

    run_covariances = TensorDict(run_covariances)

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
