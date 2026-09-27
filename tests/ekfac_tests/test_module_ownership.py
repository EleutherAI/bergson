"""Fitting on several ranks, where each module belongs to one rank, must give
the row shards a single process would."""

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import PreTrainedModel

from bergson.config import DataConfig, HessianConfig, IndexConfig
from bergson.data import allocate_batches
from bergson.gradients import GradientProcessor
from bergson.hessians.eigenvectors import LambdaCollector
from bergson.hessians.hessian_approximations import (
    approximate_hessians,
    collect_hessians,
)
from bergson.hessians.kfac import CovarianceCollector
from bergson.hessians.sharded_computation import assign_module_owners, shard_bounds
from bergson.utils.worker_utils import setup_data_pipeline, setup_model_and_peft

needs_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Needs two GPUs"
)


# Model, filter_modules and moe_experts for each case.
MODELS = {
    "dense": ("EleutherAI/pythia-14m", "embed_out", ""),
    # Fused experts see their own [N, S], unlike the batch's.
    "moe": (
        "trl-internal-testing/tiny-GptOssForCausalLM",
        "lm_head",
        "model.layers.*.mlp.experts",
    ),
}


def index_cfg(run_path: Path, nproc: int, model: str) -> IndexConfig:
    name, filter_modules, moe_experts = MODELS[model]
    cfg = IndexConfig(
        run_path=str(run_path),
        data=DataConfig(
            dataset="NeelNanda/pile-10k", split="train[:8]", truncation=True
        ),
        # Every document fills its batch alone at 64 tokens, so both fits see the
        # same batches. Padding shorter documents together changes the
        # gradients below the last layer by up to ~1% on main as well.
        token_batch_size=64,
        precision="fp32",
        model=name,
        filter_modules=filter_modules,
        moe_experts=moe_experts,
    )
    cfg.distributed.nproc_per_node = nproc
    return cfg


# Dataset labels: sampled labels would differ between the fits.
HESSIAN_CFG = HessianConfig(method="kfac", ev_correction=True, use_dataset_labels=True)


def full_matrices(sub_path: str, world_size: int) -> dict[str, torch.Tensor]:
    shards = [
        load_file(os.path.join(sub_path, f"shard_{rank}.safetensors"))
        for rank in range(world_size)
    ]
    return {key: torch.cat([shard[key] for shard in shards]) for key in shards[0]}


def assert_close_to_scale(actual: torch.Tensor, expected: torch.Tensor):
    torch.testing.assert_close(
        actual, expected, rtol=0, atol=1e-5 * expected.abs().max().item()
    )


def test_assign_module_owners_balances_elements():
    device = torch.device("cpu")
    target_info = {
        "big": (device, torch.Size([8, 8]), False),
        "mid": (device, torch.Size([6, 6]), False),
        "small_a": (device, torch.Size([4, 4]), False),
        "small_b": (device, torch.Size([4, 4]), False),
    }
    assert assign_module_owners(target_info, 2) == {
        "big": 0,
        "mid": 1,
        "small_a": 1,
        "small_b": 1,
    }


@needs_two_gpus
@pytest.mark.parametrize("model", MODELS)
def test_distributed_covariances_match_one_process(tmp_path: Path, model: str):
    single = approximate_hessians(index_cfg(tmp_path / "single", 1, model), HESSIAN_CFG)
    split = approximate_hessians(index_cfg(tmp_path / "split", 2, model), HESSIAN_CFG)

    for sub in ("activation_sharded", "gradient_sharded"):
        expected = full_matrices(os.path.join(single, sub), 1)
        actual = full_matrices(os.path.join(split, sub), 2)
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_close_to_scale(actual[key], expected[key])


@needs_two_gpus
@pytest.mark.parametrize("model", MODELS)
def test_distributed_eigenvalue_corrections_match_one_process(
    tmp_path: Path, model: str
):
    split = approximate_hessians(index_cfg(tmp_path / "split", 2, model), HESSIAN_CFG)

    # Eigenvectors are only unique up to rotation within repeated eigenvalues,
    # so correct against the two-rank fit's own, joined into one shard.
    merged = tmp_path / "merged"
    for sub in ("eigen_activation_sharded", "eigen_gradient_sharded"):
        os.makedirs(merged / sub)
        save_file(
            full_matrices(os.path.join(split, sub), 2),
            str(merged / sub / "shard_0.safetensors"),
        )

    cfg = index_cfg(tmp_path / "single", 1, model)
    cfg.partial_run_path.mkdir(parents=True)
    data, _ = setup_data_pipeline(cfg)
    network, target_modules = setup_model_and_peft(cfg)
    assert isinstance(network, PreTrainedModel)
    collect_hessians(
        network,
        data,
        cfg,
        batches=allocate_batches(data["length"][:], cfg.token_batch_size),
        target_modules=target_modules,
        hessian_cfg=HESSIAN_CFG,
        ev_correction=True,
        eigen_path=str(merged),
        path=str(tmp_path / "single"),
    )

    sub = "eigenvalue_correction_sharded"
    expected = full_matrices(str(tmp_path / "single" / sub), 1)
    actual = full_matrices(os.path.join(split, sub), 2)
    assert expected.keys() == actual.keys()
    for key in expected:
        assert_close_to_scale(actual[key], expected[key])


# Three ranks split odd widths unevenly (rank 0 takes the remainder), and each rank
# sees batches of its own shapes with several documents each.
CPU_RANKS = 3
WIDTHS = (5, 7, 4, 3)


def small_network() -> nn.Sequential:
    torch.manual_seed(0)
    layers = []
    for in_dim, out_dim in zip(WIDTHS, WIDTHS[1:]):
        layers += [nn.Linear(in_dim, out_dim), nn.Tanh()]
    return nn.Sequential(*layers[:-1])


def rank_batches(rank: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(100 + rank)
    batches = []
    for b in range(3):
        n, s = 1 + (rank + b) % 3, 2 + (2 * rank + b) % 4
        x = torch.randn(n, s, WIDTHS[0], generator=generator)
        batches.append((x, torch.rand(n, s, generator=generator) > 0.3))
    return batches


def write_eigenvectors(root: Path, world_size: int):
    """Random orthonormal eigenvectors, in ``world_size`` row shards."""
    torch.manual_seed(1)
    network = small_network()
    for sub, dim in (
        ("eigen_activation_sharded", lambda layer: layer.in_features + 1),
        ("eigen_gradient_sharded", lambda layer: layer.out_features),
    ):
        full = {
            str(i): torch.linalg.qr(torch.randn(dim(layer), dim(layer)))[0].double()
            for i, layer in enumerate(network)
            if isinstance(layer, nn.Linear)
        }
        os.makedirs(root / sub, exist_ok=True)
        for rank in range(world_size):
            shard = {
                key: value[
                    slice(*shard_bounds(len(value), rank, world_size))
                ].contiguous()
                for key, value in full.items()
            }
            save_file(shard, str(root / sub / f"shard_{rank}.safetensors"))


def fit_small(rank: int, world_size: int, port: int, root: str):
    """Covariances and eigenvalue corrections over ``world_size`` gloo ranks, or
    over every rank's batches in one process when ``world_size`` is 1."""
    if world_size > 1:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        batches = rank_batches(rank)
    else:
        batches = [b for r in range(CPU_RANKS) for b in rank_batches(r)]

    out = os.path.join(root, f"fit{world_size}")
    eigen_path = os.path.join(root, f"eigen{world_size}")
    for ev_correction in (False, True):
        network = small_network()
        processor = GradientProcessor(include_bias=True)
        collector = (
            LambdaCollector(
                model=network,
                path=out,
                dtype=torch.float64,
                processor=processor,
                eigen_path=eigen_path,
            )
            if ev_correction
            else CovarianceCollector(
                model=network, path=out, dtype=torch.float64, processor=processor
            )
        )
        for x, mask in batches:
            # with_batch returns the collector, whose context registers the hooks.
            with collector.with_batch(mask):
                (network(x) ** 2 * mask[..., None]).sum().backward()
            network.zero_grad()
        collector.teardown()

    if world_size > 1:
        dist.destroy_process_group()


def test_gloo_ranks_match_one_process_on_cpu(tmp_path: Path, monkeypatch):
    # Collectors place their factors on the GPU when one is visible, and gloo
    # can't gather CUDA tensors.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for world_size in (1, CPU_RANKS):
        write_eigenvectors(tmp_path / f"eigen{world_size}", world_size)
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    mp.spawn(fit_small, args=(1, port, str(tmp_path)), nprocs=1)
    mp.spawn(fit_small, args=(CPU_RANKS, port, str(tmp_path)), nprocs=CPU_RANKS)

    for sub in (
        "activation_sharded",
        "gradient_sharded",
        "eigenvalue_correction_sharded",
    ):
        expected = full_matrices(str(tmp_path / "fit1" / sub), 1)
        actual = full_matrices(str(tmp_path / f"fit{CPU_RANKS}" / sub), CPU_RANKS)
        assert expected.keys() == actual.keys()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
