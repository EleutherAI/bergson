"""Gradient collection for MoE models whose experts share one fused parameter.

gpt-oss and Mixtral are the reference pair: transposed vs not, biased vs not,
interleaved vs concatenated gate. Models are built from configs in-process, so
the suite stays offline.
"""

import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    GptOssConfig,
    MixtralConfig,
    OlmoeConfig,
    Qwen3MoeConfig,
)

from bergson.collector.collector import HookCollectorBase
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig
from bergson.gradients import GradientProcessor, LayerAdapter
from bergson.hessians.kfac import CovarianceCollector
from bergson.moe import ExpertLinear, expand_moe

PATTERN = "model.layers.*.mlp.experts"
FAMILIES = ("gpt_oss", "mixtral")
NUM_LAYERS = 2
NUM_EXPERTS = 4
TOP_K = 2
SEQ_LEN = 7

# include_bias is a no-op on Mixtral, which has no biased module at all, while
# every module gpt-oss tracks carries one.
BIAS_CASES = [("gpt_oss", False), ("gpt_oss", True), ("mixtral", False)]

SHARED = dict(
    hidden_size=32,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=64,
    max_position_embeddings=64,
    num_experts_per_tok=TOP_K,
)

# Qwen3-MoE and OLMoE vary the block wiring around the same fused layout.
# DeepSeek-V3 is left out: bergson cannot hook its attention at all, fused
# experts or not, because kv_b_proj takes a 4D input.
CONFIGS = {
    "gpt_oss": lambda: GptOssConfig(
        intermediate_size=16, num_local_experts=NUM_EXPERTS, head_dim=8, **SHARED
    ),
    "mixtral": lambda: MixtralConfig(
        intermediate_size=16, num_local_experts=NUM_EXPERTS, **SHARED
    ),
    "qwen3_moe": lambda: Qwen3MoeConfig(
        intermediate_size=32,
        moe_intermediate_size=16,
        num_experts=NUM_EXPERTS,
        head_dim=8,
        **SHARED,
    ),
    "olmoe": lambda: OlmoeConfig(
        intermediate_size=16, num_experts=NUM_EXPERTS, **SHARED
    ),
}


def build_model(family: str) -> nn.Module:
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(CONFIGS[family](), dtype=torch.float32)
    model.eval()
    return model


def expert_names(target_info) -> list[str]:
    """Expert projections among the discovered targets."""
    return [n for n in target_info if ".experts.expert_" in n]


def make_collector(model, *, processor=None, **cfg_kwargs) -> GradientCollector:
    return GradientCollector(
        model=model,
        cfg=IndexConfig(run_path="/tmp/bergson-moe-test", **cfg_kwargs),
        data=Dataset.from_dict({"input_ids": [[1] * SEQ_LEN]}),
        processor=processor or GradientProcessor(),
        skip_index=True,
    )


def backward_pass(model, x: torch.Tensor) -> None:
    model.zero_grad()
    (model(input_ids=x).logits ** 2).sum().backward()


def autograd_gradient(model, name: str, include_bias: bool) -> torch.Tensor:
    """``name``'s weight gradient from the last backward, read off the *fused*
    parameter and oriented to [out, in]. An independent reference, not a
    restatement of the collector's arithmetic."""
    layer = model.get_submodule(name)
    grad = getattr(layer._experts, layer.weight_name).grad[layer.expert_idx]
    if LayerAdapter.weight_transposed(layer):
        grad = grad.T

    fused_bias = getattr(layer._experts, f"{layer.weight_name}_bias", None)
    if include_bias and fused_bias is not None:
        grad = torch.cat([grad, fused_bias.grad[layer.expert_idx, :, None]], dim=1)
    return grad.flatten()


def collect_and_compare(model, batch_size: int, include_bias: bool) -> None:
    """Check each collected per-example gradient against a per-example backward."""
    expand_moe(model, PATTERN)
    model.requires_grad_(True)
    x = torch.randint(0, 64, (batch_size, SEQ_LEN))

    collector = make_collector(
        model, processor=GradientProcessor(include_bias=include_bias)
    )
    names = expert_names(collector.target_info)
    assert len(names) == NUM_LAYERS * NUM_EXPERTS * 2

    with collector:
        backward_pass(model, x)
    collected = {k: v.clone() for k, v in collector.mod_grads.items()}

    # Builder concatenates every module in shapes(), so none may be missing,
    # including experts that happened to receive no tokens.
    assert set(collected) == set(collector.shapes())

    for example in range(batch_size):
        backward_pass(model, x[example : example + 1])
        for name in names:
            torch.testing.assert_close(
                collected[name][example].float(),
                autograd_gradient(model, name, include_bias).float(),
                atol=1e-5,
                rtol=1e-4,
                msg=f"{name}, example {example}",
            )
    assert max(g.abs().max() for g in collected.values()) > 0, "all-zero gradients"


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("family,include_bias", BIAS_CASES)
def test_per_example_gradients_match_autograd(family, batch_size, include_bias):
    collect_and_compare(build_model(family), batch_size, include_bias)


@pytest.mark.parametrize("family", ["qwen3_moe", "olmoe"])
def test_other_families_match_autograd(family):
    """The same pattern and forward carry past the reference pair."""
    collect_and_compare(build_model(family), batch_size=2, include_bias=False)


@pytest.mark.parametrize("family", FAMILIES)
def test_expansion_covers_every_fused_expert(family):
    """Expansion adds exactly the fused expert weights, and nothing else."""
    model = build_model(family)

    def tracked() -> int:
        total = 0
        for name in HookCollectorBase.discover_targets(model):
            layer = model.get_submodule(name)
            total += getattr(layer, LayerAdapter.in_attr(layer)) * getattr(
                layer, LayerAdapter.out_attr(layer)
            )
        return total

    before = tracked()
    expand_moe(model, PATTERN)
    after = tracked()

    fused = sum(p.numel() for p in model.parameters() if p.ndim == 3)
    assert fused and after - before == fused

    total = sum(p.numel() for p in model.parameters())
    assert before / total < 0.5 < after / total


@pytest.mark.parametrize("family", FAMILIES)
def test_expansion_is_transparent(family):
    """Expansion changes what is visible, not what the model computes."""
    model = build_model(family)
    x = torch.randint(0, 64, (3, SEQ_LEN))
    with torch.no_grad():
        reference = model(input_ids=x).logits.clone()

    parameters = {n for n, _ in model.named_parameters()}
    state_dict = set(model.state_dict())
    modules = set(dict(model.named_modules()))

    added = expand_moe(model, PATTERN)
    assert len(added) == NUM_LAYERS * NUM_EXPERTS * 2
    assert expand_moe(model, PATTERN) == added, "expansion should be idempotent"

    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=x).logits, reference)
    assert {n for n, _ in model.named_parameters()} == parameters
    assert set(model.state_dict()) == state_dict
    assert modules < set(dict(model.named_modules()))


@pytest.mark.parametrize("family", FAMILIES)
def test_covariance_honors_the_collection_mask(family, tmp_path):
    """The batch's collection mask reaches an expert's routed rows, so masking
    every position out leaves EK-FAC's activation covariance empty."""
    model = build_model(family)
    expand_moe(model, PATTERN)
    model.requires_grad_(True)
    x = torch.randint(0, 64, (3, SEQ_LEN))

    for keep in (True, False):
        collector = CovarianceCollector(
            model=model, dtype=torch.float32, path=str(tmp_path)
        )
        names = expert_names(collector.target_info)
        assert len(names) == NUM_LAYERS * NUM_EXPERTS * 2

        with collector.with_batch(torch.full((3, SEQ_LEN), keep)):
            backward_pass(model, x)

        for name in names:
            size = model.get_submodule(name).in_features
            assert collector.A_cov_dict[name].shape == (size, size)

        collected = sum(collector.A_cov_dict[n].abs().sum() for n in names)
        assert bool(collected > 0) is keep


def test_pattern_matching_nothing_raises():
    model = build_model("gpt_oss")
    with pytest.raises(ValueError, match="matched no module"):
        expand_moe(model, "model.layers.*.mlp.wizards")


def test_pattern_matching_a_dense_module_raises():
    model = build_model("gpt_oss")
    with pytest.raises(ValueError, match="not an MoE layer"):
        expand_moe(model, "model.layers.*.mlp")


def test_experts_are_untracked_without_the_pattern():
    """Tracking is opt-in: an unexpanded model discovers no expert."""
    model = build_model("gpt_oss")
    assert expert_names(HookCollectorBase.discover_targets(model)) == []
    assert not any(isinstance(m, ExpertLinear) for m in model.modules())


def test_attribute_tokens_is_rejected():
    model = build_model("gpt_oss")
    expand_moe(model, PATTERN)
    collector = make_collector(model, attribute_tokens=True)

    with pytest.raises(ValueError, match="attribute_tokens is incompatible"):
        with collector:
            pass
