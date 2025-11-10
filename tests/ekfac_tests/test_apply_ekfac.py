"""Test EKFAC application against ground truth."""

import json
import os
from typing import Optional

import pytest
import torch
from safetensors.torch import load_file

from bergson.data import DataConfig, IndexConfig, load_gradients
from bergson.distributed import distributed_computing
from bergson.hessians.ekfac_apply import ekfac_apply_worker


@pytest.fixture(scope="module")
def ekfac_apply_gradient_path(
    test_dir: str,
    ground_truth_path: str,
    world_size: int,
    overwrite: bool,
    ekfac_results_path: str,
    use_fsdp: bool,
    gradient_path: Optional[str],
    gradient_batch_size: int,
) -> str:
    """Setup EKFAC application configuration and run if needed.

    ground_truth_path fixture ensures all required files exist (covariances, eigenvectors, etc).
    ekfac_results_path ensures EKFAC computation has run.
    """
    # Load configuration
    with open(os.path.join(ground_truth_path, "index_config.json"), "r") as f:
        cfg_json = json.load(f)

    if gradient_path is None:
        pytest.skip(
            "No --gradient-path argument provided, skipping EKFAC application tests."
        )
        return ""

    cfg = IndexConfig(**cfg_json)
    cfg.data = DataConfig(**(cfg_json["data"]))
    cfg.run_path = test_dir + "/run"
    cfg.debug = True
    cfg.fsdp = use_fsdp
    cfg.world_size = world_size
    cfg.ekfac = True
    cfg.gradient_path = gradient_path
    cfg.gradient_batch_size = gradient_batch_size

    results_path = gradient_path + "_ekfac"

    if os.path.exists(results_path) and not overwrite:
        print(f"Using existing {results_path}.")
    else:
        print(f"\nRunning EKFAC application in {results_path}...")
        distributed_computing(
            cfg=cfg,
            worker_fn=ekfac_apply_worker,
            setup_data=False,
            setup_model=False,
            setup_processor=False,
        )
        print("EKFAC application completed successfully in {results_path}.")

    return results_path


def test_gradients_after_ekfac(test_dir: str, ekfac_apply_gradient_path: str) -> None:
    """Test gradients after EKFAC application against ground truth."""

    ground_truth_path = test_dir + "/test_gradients/gradients_after_ekfac"

    ground_truth = load_file(
        os.path.join(ground_truth_path, "gradients.safetensors"), device="cuda"
    )
    computed_mmap = load_gradients(ekfac_apply_gradient_path)

    for k in ground_truth.keys():
        ground_truth_tensor = ground_truth[k].to(dtype=torch.float32)

        computed_tensor = (
            torch.from_numpy(computed_mmap[k].copy())
            .to(device="cuda")
            .view(-1, *ground_truth_tensor.shape[1:])
        ).to(dtype=torch.float32)

        if not (ground_truth_tensor.shape == computed_tensor.shape):
            raise ValueError(
                f"Shape mismatch for key {k}: {ground_truth_tensor.shape} vs {computed_tensor.shape}"
            )

        if not torch.allclose(ground_truth_tensor, computed_tensor, rtol=1e-3, atol=0):
            abs_diff = torch.abs(ground_truth_tensor - computed_tensor)
            rel_diff = abs_diff / (torch.abs(ground_truth_tensor) + 1e-12)

            max_abs_diff = torch.max(abs_diff).item()
            max_rel_diff = torch.max(rel_diff).item()
            argmax_idx = torch.argmax(rel_diff)
            coords = torch.unravel_index(argmax_idx, ground_truth_tensor.shape)

            gt_val = ground_truth_tensor.flatten()[argmax_idx].item()
            comp_val = computed_tensor.flatten()[argmax_idx].item()

            print(
                f"Mismatch '{k}': max_abs={max_abs_diff:.2e}, max_rel={max_rel_diff:.2e}"
            )
            print(f"  At {tuple(coords)}: gt={gt_val:.2e}, comp={comp_val:.2e}")

    print("\n✓ All gradient tests passed\n")
