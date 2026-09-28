"""Modules reading the same input share one activation covariance, which must
leave every saved factor as it would be without sharing."""

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file

from bergson.collector.collector import HookCollectorBase
from bergson.config import DataConfig, DistributedConfig, HessianConfig, IndexConfig
from bergson.hessians.hessian_approximations import (
    FACTOR_SUBDIRS,
    approximate_hessians,
    partition_modules,
)
from bergson.hessians.kfac import SharedInputCheck, find_shared_inputs
from bergson.hessians.sharded_computation import assign_module_owners


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(4, 6)
        self.norm = nn.LayerNorm(6)
        self.q = nn.Linear(6, 6, bias=False)
        self.k = nn.Linear(6, 6, bias=False)
        self.v = nn.Linear(6, 6, bias=False)
        self.biased = nn.Linear(6, 6)
        self.dropped = nn.Linear(6, 6, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.twice = nn.Linear(6, 6, bias=False)
        self.out = nn.Linear(6, 6, bias=False)

    def forward(self, x):
        h = self.norm(self.embed(x))
        return self.out(
            self.q(h)
            + self.k(h)
            + self.v(h)
            + self.biased(h)
            + self.dropped(self.dropout(h))
            + self.twice(self.twice(h))
        )


def toy_shared_inputs(include_bias: bool, train: bool = False) -> dict[str, str]:
    model = ToyModel().train(train)
    target_info = HookCollectorBase.discover_targets(model, include_bias=include_bias)
    return find_shared_inputs(model, target_info)


def test_find_shared_inputs():
    # In eval mode the dropout returns its input; a module called twice is left out.
    assert toy_shared_inputs(include_bias=False) == {
        "k": "q",
        "v": "q",
        "biased": "q",
        "dropped": "q",
    }
    # The ones column makes the biased module's activations differ.
    assert toy_shared_inputs(include_bias=True) == {
        "k": "q",
        "v": "q",
        "dropped": "q",
    }
    # Active dropout gives the module its own input.
    assert toy_shared_inputs(include_bias=False, train=True) == {
        "k": "q",
        "v": "q",
        "biased": "q",
    }


def test_shared_input_check_rejects_other_input():
    check = SharedInputCheck({"k": "q", "v": "q"})
    x = torch.randn(1, 2, 6)
    assert not check("q", x)
    assert check("k", x)
    assert check("v", x)
    assert not check.inputs, "released after the last module reads it"

    assert not check("q", x)
    with pytest.raises(RuntimeError, match="same input"):
        check("k", x.clone())

    assert not check("q", x)
    x.add_(1)
    with pytest.raises(RuntimeError, match="same input"):
        check("k", x)


def test_assign_module_owners_keeps_shared_inputs_together():
    device = torch.device("cpu")
    target_info = {
        name: (device, torch.Size([8, 8]), False) for name in ("q", "k", "v", "o")
    }
    owners = assign_module_owners(target_info, 2, {"k": "q", "v": "q"})
    assert owners["q"] == owners["k"] == owners["v"] != owners["o"]


def test_partition_modules_keeps_shared_inputs_together():
    names = [f"m{i}" for i in range(7)]
    assert partition_modules(names, 3, {}) == partition_modules(names, 3)

    groups = partition_modules(names, 3, {"m3": "m2", "m6": "m2"})
    assert groups == [["m0", "m1", "m2", "m3", "m6"], ["m4"], ["m5"]]


def fit(
    tmp_path: Path, model: str, share: bool, partitions: int = 1
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, str]]:
    """The fit's factors and the modules found to share an input."""
    found = []

    def find(*args):
        found.append(find_shared_inputs(*args) if share else {})
        return found[-1]

    name = f"{'shared' if share else 'separate'}_p{partitions}"
    cfg = IndexConfig(
        run_path=str(tmp_path / name),
        model=model,
        data=DataConfig(
            dataset="NeelNanda/pile-10k", split="train[:8]", truncation=True
        ),
        token_batch_size=512,
        precision="fp32",
        filter_modules="embed_out,lm_head",
        distributed=DistributedConfig(nproc_per_node=1),
    )
    # Dataset labels: sampled labels would differ between the fits.
    hessian_cfg = HessianConfig(
        method="kfac",
        ev_correction=True,
        use_dataset_labels=True,
        module_partitions=partitions,
    )
    with pytest.MonkeyPatch.context() as m:
        m.setattr("bergson.hessians.hessian_approximations.find_shared_inputs", find)
        path = approximate_hessians(cfg, hessian_cfg)

    factors = {
        sub: load_file(os.path.join(path, sub, "shard_0.safetensors"))
        for sub in FACTOR_SUBDIRS
    }
    (shared_inputs,) = found
    return factors, shared_inputs


def assert_same_factors(actual, expected):
    for sub in FACTOR_SUBDIRS:
        assert actual[sub].keys() == expected[sub].keys(), sub
        for key in expected[sub]:
            if sub == "eigen_activation_sharded":
                # Eigenvectors aren't unique; compare the eigenvalues instead.
                continue
            assert torch.equal(actual[sub][key], expected[sub][key]), f"{sub}/{key}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("partitions", [1, 3])
def test_sharing_matches_separate_fits(tmp_path: Path, partitions: int):
    # Separate q_proj, k_proj and v_proj.
    model = "trl-internal-testing/tiny-GptOssForCausalLM"
    shared, shared_inputs = fit(tmp_path, model, share=True, partitions=partitions)
    separate, _ = fit(tmp_path, model, share=False)
    assert shared_inputs, "expected q_proj, k_proj and v_proj to share an input"

    assert_same_factors(shared, separate)
    for name, source in shared_inputs.items():
        for sub in ("activation_sharded", "eigen_activation_sharded", "factor_eig_a"):
            assert torch.equal(shared[sub][name], shared[sub][source])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_no_shared_inputs_leaves_fit_unchanged(tmp_path: Path):
    # Fused query_key_value, so no two modules read the same input.
    model = "EleutherAI/pythia-14m"
    probed, shared_inputs = fit(tmp_path, model, share=True)
    separate, _ = fit(tmp_path, model, share=False)
    assert shared_inputs == {}
    assert_same_factors(probed, separate)
