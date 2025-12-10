"""Pytest configuration and fixtures for EKFAC tests."""

import json
import os
from typing import Any, Optional

import pytest
from compute_ekfac_ground_truth import (
    combine_covariances_step,
    combine_eigenvalue_corrections_step,
    compute_covariances_step,
    compute_eigenvalue_corrections_step,
    compute_eigenvectors_step,
    load_dataset_step,
    load_model_step,
    setup_paths_and_config,
    tokenize_and_allocate_step,
)
from test_utils import set_all_seeds

from bergson.data import DataConfig, IndexConfig, Precision
from bergson.distributed import distributed_computing
from bergson.hessians.compute_all import compute_all_factors


def pytest_addoption(parser) -> None:
    """Add custom command-line options for EKFAC tests."""
    parser.addoption(
        "--gradient_batch_size",
        action="store",
        type=int,
        default=1,
        help="Batch size for gradient computation (default: 1)",
    )
    parser.addoption(
        "--gradient_path",
        action="store",
        default=None,
        help="Path to the gradient file",
    )
    parser.addoption(
        "--model_name",
        action="store",
        type=str,
        default="EleutherAI/Pythia-14m",
        help="Model name to use for ground truth generation (default: EleutherAI/Pythia-14m)",
    )
    parser.addoption(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing run directory",
    )
    parser.addoption(
        "--precision",
        action="store",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "int4", "int8"],
        help="Model precision for ground truth generation (default: fp32)",
    )
    parser.addoption(
        "--sample",
        action="store_true",
        default=False,
        help="Enable sampling mode",
    )
    parser.addoption(
        "--test_dir",
        action="store",
        default=None,
        help="Directory containing test data. If not provided, generates test data using compute_ekfac_ground_truth.py.",
    )
    parser.addoption(
        "--use_fsdp",
        action="store_true",
        default=False,
        help="Use Fully Sharded Data Parallel (FSDP)",
    )
    parser.addoption(
        "--world_size",
        action="store",
        type=int,
        default=1,
        help="World size for distributed training (default: 1)",
    )


@pytest.fixture(autouse=True)
def setup_test() -> None:
    """Setup logic run before each test."""
    set_all_seeds(seed=42)


@pytest.fixture(scope="session")
def gradient_batch_size(request) -> int:
    return request.config.getoption("--gradient_batch_size")


@pytest.fixture(scope="session")
def gradient_path(request) -> Optional[str]:
    return request.config.getoption("--gradient_path")


@pytest.fixture(scope="session")
def model_name(request) -> str:
    return request.config.getoption("--model_name")


@pytest.fixture(scope="session")
def overwrite(request) -> bool:
    return request.config.getoption("--overwrite")


@pytest.fixture(scope="session")
def precision(request) -> Precision:
    return request.config.getoption("--precision")


@pytest.fixture(scope="session")
def sample(request) -> bool:
    return request.config.getoption("--sample")


@pytest.fixture(scope="session")
def use_fsdp(request) -> bool:
    return request.config.getoption("--use_fsdp")


@pytest.fixture(scope="session")
def world_size(request) -> int:
    return request.config.getoption("--world_size")


@pytest.fixture(scope="session")
def test_dir(request, tmp_path_factory) -> str:
    """Get or create test directory (does not generate ground truth data)."""
    # Check if test directory was provided
    test_dir = request.config.getoption("--test_dir")
    if test_dir is not None:
        return test_dir

    # Create temporary directory for auto-generated test data
    tmp_dir = tmp_path_factory.mktemp("ekfac_test_data")
    return str(tmp_dir)


def ground_truth_base_path(test_dir: str) -> str:
    return os.path.join(test_dir, "ground_truth")


@pytest.fixture(scope="session")
def ground_truth_setup(
    request, test_dir: str, precision: Precision, overwrite: bool
) -> dict[str, Any]:
    # Setup for generation
    model_name = request.config.getoption("--model_name")
    world_size = request.config.getoption("--world_size")

    print(f"\n{'='*60}")
    print("Generating ground truth test data")
    print(f"Model: {model_name}")
    print(f"Precision: {precision}")
    print(f"World size: {world_size}")
    print(f"{'='*60}\n")

    cfg, workers, device, target_modules, dtype = setup_paths_and_config(
        precision=precision,
        test_path=ground_truth_base_path(test_dir),
        model_name=model_name,
        world_size=world_size,
        overwrite=overwrite,
    )

    model = load_model_step(cfg, dtype)
    ds = load_dataset_step(cfg)
    data, batches_world, tokenizer = tokenize_and_allocate_step(ds, cfg, workers)

    return {
        "cfg": cfg,
        "workers": workers,
        "device": device,
        "target_modules": target_modules,
        "dtype": dtype,
        "model": model,
        "data": data,
        "batches_world": batches_world,
    }


@pytest.fixture(scope="session")
def ground_truth_covariances_path(
    ground_truth_setup: dict[str, Any], test_dir: str
) -> str:
    """Ensure ground truth covariances exist and return path."""
    base_path = ground_truth_base_path(test_dir)
    covariances_path = os.path.join(base_path, "covariances")

    if os.path.exists(covariances_path) and not overwrite:
        print("Using existing covariances")
        return covariances_path

    setup = ground_truth_setup
    covariance_test_path = compute_covariances_step(
        setup["model"],
        setup["data"],
        setup["batches_world"],
        setup["device"],
        setup["target_modules"],
        setup["workers"],
        base_path,
    )
    combine_covariances_step(covariance_test_path, setup["workers"], setup["device"])
    print("Covariances computed")
    return covariances_path


@pytest.fixture(scope="session")
def ground_truth_eigenvectors_path(
    ground_truth_covariances_path: str,
    ground_truth_setup: dict[str, Any],
    test_dir: str,
) -> str:
    """Ensure ground truth eigenvectors exist and return path."""
    base_path = ground_truth_base_path(test_dir)
    eigenvectors_path = os.path.join(base_path, "eigenvectors")

    if os.path.exists(eigenvectors_path) and not overwrite:
        print("Using existing eigenvectors")
        return eigenvectors_path

    setup = ground_truth_setup
    compute_eigenvectors_step(base_path, setup["device"], setup["dtype"])
    print("Eigenvectors computed")
    return eigenvectors_path


@pytest.fixture(scope="session")
def ground_truth_eigenvalue_corrections_path(
    ground_truth_eigenvectors_path: str,
    ground_truth_setup: dict[str, Any],
    test_dir: str,
) -> str:
    """Ensure ground truth eigenvalue corrections exist and return path."""
    base_path = ground_truth_base_path(test_dir)
    eigenvalue_corrections_path = os.path.join(base_path, "eigenvalue_corrections")

    if os.path.exists(eigenvalue_corrections_path) and not overwrite:
        print("Using existing eigenvalue corrections")
        return eigenvalue_corrections_path

    setup = ground_truth_setup
    eigenvalue_correction_test_path, total_processed_global_lambda = (
        compute_eigenvalue_corrections_step(
            setup["model"],
            setup["data"],
            setup["batches_world"],
            setup["device"],
            setup["target_modules"],
            setup["workers"],
            base_path,
        )
    )
    combine_eigenvalue_corrections_step(
        eigenvalue_correction_test_path,
        setup["workers"],
        setup["device"],
        total_processed_global_lambda,
    )
    print("Eigenvalue corrections computed")
    print("\n=== Ground Truth Computation Complete ===")
    print(f"Results saved to: {base_path}")
    return eigenvalue_corrections_path


@pytest.fixture(scope="session")
def ground_truth_path(
    ground_truth_eigenvalue_corrections_path: str, test_dir: str
) -> str:
    """Get ground truth base path with all data guaranteed to exist.

    Depends on ground_truth_eigenvalue_corrections_path to ensure all ground truth data exists.
    """
    return ground_truth_base_path(test_dir)


@pytest.fixture(scope="session")
def ekfac_results_path(
    test_dir: str,
    ground_truth_path: str,
    world_size: int,
    overwrite: bool,
    use_fsdp: bool,
    sample: bool,
) -> str:
    """Setup EKFAC configuration, run computation if needed, and return results path.

    ground_truth_path fixture ensures all ground truth data exists.
    """
    results_path = os.path.join(test_dir, "run/influence_results")

    # Load configuration
    with open(os.path.join(ground_truth_path, "index_config.json"), "r") as f:
        cfg_json = json.load(f)

    cfg = IndexConfig(**cfg_json)
    cfg.data = DataConfig(**(cfg_json["data"]))
    assert isinstance(cfg.fsdp, bool)  # for type checker
    cfg.run_path = test_dir + "/run"
    cfg.debug = True
    cfg.fsdp = use_fsdp
    cfg.world_size = world_size
    cfg.sample = sample

    if os.path.exists(results_path) and not overwrite:
        print(f"Using existing EKFAC results in {results_path}.")
    else:
        print(f"\nRunning EKFAC computation in {results_path}...")
        distributed_computing(
            cfg=cfg,
            worker_fn=compute_all_factors,
        )
        print(f"EKFAC computation completed in {results_path}.")

    return results_path
