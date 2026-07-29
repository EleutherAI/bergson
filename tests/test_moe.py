"""Gradient collection for MoE models with fused expert and router parameters.

gpt-oss and Mixtral are the reference pair: transposed vs not, biased vs not,
interleaved vs concatenated gate. Models are built from configs in-process, so
the suite stays offline.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    DeepseekV3Config,
    GptOssConfig,
    MixtralConfig,
    OlmoeConfig,
    Qwen3MoeConfig,
)
from transformers.integrations import use_experts_implementation

from bergson.collector.collector import HookCollectorBase
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig
from bergson.gradients import AdamNormalizer, GradientProcessor, LayerAdapter
from bergson.hessians.kfac import CovarianceCollector
from bergson.moe import ExpertLinear, expand_moe, restore_moe, tracked_router

FAMILIES = ("gpt_oss", "mixtral")
NUM_EXPERTS = 4
TOP_K = 2
SEQ_LEN = 7

# include_bias is a no-op on Mixtral, which has no biased module at all, while
# every module gpt-oss tracks carries one.
BIAS_CASES = [("gpt_oss", False), ("gpt_oss", True), ("mixtral", False)]

SHARED = dict(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=64,
    max_position_embeddings=64,
    num_experts_per_tok=TOP_K,
)

# DeepSeek-V3 brings a sigmoid grouped-top-k router with a score-correction bias
# and a shared expert; the other two vary the block wiring.
CONFIGS = {
    "gpt_oss": lambda: GptOssConfig(
        intermediate_size=16, num_local_experts=NUM_EXPERTS, head_dim=8, **SHARED
    ),
    "mixtral": lambda: MixtralConfig(
        intermediate_size=16, num_local_experts=NUM_EXPERTS, **SHARED
    ),
    "deepseek_v3": lambda: DeepseekV3Config(
        intermediate_size=32,
        moe_intermediate_size=16,
        n_routed_experts=NUM_EXPERTS,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        n_shared_experts=1,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        kv_lora_rank=8,
        q_lora_rank=8,
        **{**SHARED, "num_key_value_heads": 4},
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


def moe_names(target_info) -> list[str]:
    """Expert and router modules among the discovered targets."""
    return [
        n
        for n in target_info
        if "experts.expert_" in n or n.rpartition(".")[2] in ("gate", "router")
    ]


def make_collector(model, *, processor=None, track_moe_experts=True, **cfg_kwargs):
    return GradientCollector(
        model=model.base_model,
        cfg=IndexConfig(run_path="/tmp/bergson-moe-test", **cfg_kwargs),
        data=Dataset.from_dict({"input_ids": [[1] * SEQ_LEN]}),
        processor=processor or GradientProcessor(),
        skip_index=True,
        track_moe_experts=track_moe_experts,
    )


def backward_pass(model, x: torch.Tensor) -> None:
    model.zero_grad()
    (model(input_ids=x).logits ** 2).sum().backward()


def autograd_gradient(base, name: str, include_bias: bool) -> torch.Tensor:
    """``name``'s weight gradient from the last backward, read off the *fused*
    parameter and oriented to [out, in]. An independent reference, not a
    restatement of the collector's arithmetic."""
    layer = base.get_submodule(name)
    if isinstance(layer, ExpertLinear):
        grad = getattr(layer._experts, layer.weight_name).grad[layer.expert_idx]
        if LayerAdapter.weight_transposed(layer):
            grad = grad.T
        fused_bias = getattr(layer._experts, f"{layer.weight_name}_bias", None)
        if include_bias and fused_bias is not None:
            grad = torch.cat([grad, fused_bias.grad[layer.expert_idx, :, None]], dim=1)
    else:
        grad = layer.weight.grad
        if include_bias and getattr(layer, "bias", None) is not None:
            grad = torch.cat([grad, layer.bias.grad[:, None]], dim=1)
    return grad.flatten()


def assert_matches_autograd(
    base,
    names,
    collected,
    run_example,
    *,
    include_bias=False,
    normalizers=None,
    tol=None,
):
    """Check each collected per-example gradient against a per-sample backward.

    ``run_example(i)`` must leave the gradients holding example ``i``'s alone.
    With ``normalizers``, the reference is normalized as the collector would,
    catching an expert paired with the wrong normalizer or orientation.
    """
    atol, rtol = tol or (1e-5, 1e-4)
    for example in range(len(next(iter(collected.values())))):
        run_example(example)
        for name in names:
            expected = autograd_gradient(base, name, include_bias)
            if normalizers is not None:
                n = normalizers[name]
                expected = n.normalize_weight(
                    expected.view(n.weight_avg_sq.shape).clone()
                ).flatten()
            torch.testing.assert_close(
                collected[name][example].float(),
                expected.float(),
                atol=atol,
                rtol=rtol,
                msg=f"{name}, example {example}",
            )
    assert max(g.abs().max() for g in collected.values()) > 0, "all-zero gradients"


def collect_and_compare(model, batch_size: int, include_bias: bool, tol=None):
    model.requires_grad_(True)
    x = torch.randint(0, 64, (batch_size, SEQ_LEN), device=model.device)

    collector = make_collector(
        model, processor=GradientProcessor(include_bias=include_bias)
    )
    names = moe_names(collector.target_info)
    assert names, "no fused MoE experts or routers discovered"

    with collector:
        backward_pass(model, x)
    collected = {k: v.clone() for k, v in collector.mod_grads.items()}

    # Builder concatenates every module in shapes(), so none may be missing,
    # including experts that happened to receive no tokens.
    assert set(collected) == set(collector.shapes())

    assert_matches_autograd(
        model.base_model,
        names,
        collected,
        lambda i: backward_pass(model, x[i : i + 1]),
        include_bias=include_bias,
        tol=tol,
    )


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("family,include_bias", BIAS_CASES)
def test_per_example_gradients_match_autograd(family, batch_size, include_bias):
    collect_and_compare(build_model(family), batch_size, include_bias)


@pytest.mark.parametrize("family", ["deepseek_v3", "qwen3_moe", "olmoe"])
def test_other_families_match_autograd(family):
    """Detection generalizes past the reference pair."""
    collect_and_compare(build_model(family), batch_size=2, include_bias=False)


@pytest.mark.parametrize("family", FAMILIES)
def test_expansion_covers_every_fused_parameter(family):
    """Expansion adds exactly the fused expert and router weights."""
    base = build_model(family).base_model

    def tracked() -> int:
        total = 0
        for name in HookCollectorBase.discover_targets(base):
            layer = base.get_submodule(name)
            total += getattr(layer, LayerAdapter.in_attr(layer)) * getattr(
                layer, LayerAdapter.out_attr(layer)
            )
        return total

    before = tracked()
    expand_moe(base)
    after = tracked()

    experts = sum(p.numel() for p in base.parameters() if p.ndim == 3)
    routers = sum(
        m.weight.numel() for m in base.modules() if tracked_router(m) is not None
    )
    assert experts and routers
    assert after - before == experts + routers

    total = sum(p.numel() for p in base.parameters())
    assert before / total < 0.5 < after / total


@pytest.mark.parametrize("family", FAMILIES)
def test_expansion_is_transparent_and_reversible(family):
    """Expansion changes what is visible, not what the model computes."""
    model = build_model(family)
    x = torch.randint(0, 64, (3, SEQ_LEN))
    with torch.no_grad():
        reference = model(input_ids=x).logits.clone()

    parameters = {n for n, _ in model.named_parameters()}
    state_dict = set(model.state_dict())
    modules = set(dict(model.named_modules()))

    added = expand_moe(model)
    assert len(added) == 2 * (2 * NUM_EXPERTS + 1)  # two projections + a router
    assert expand_moe(model) == added, "expansion should be idempotent"

    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=x).logits, reference)
    assert {n for n, _ in model.named_parameters()} == parameters
    assert set(model.state_dict()) == state_dict
    assert modules < set(dict(model.named_modules()))

    restore_moe(model)
    assert set(dict(model.named_modules())) == modules
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=x).logits, reference)


@pytest.mark.parametrize("family", FAMILIES)
def test_expansion_state_is_root_independent(family):
    """Collection runs on base_model while callers hold model, so undo state
    keyed to one root would leave the other thinking nothing was expanded."""
    model = build_model(family)
    roots = ((model, model.base_model), (model.base_model, model))
    for expand_root, other_root in roots:
        added = expand_moe(expand_root)
        assert any(isinstance(m, ExpertLinear) for m in model.modules())
        # Names come back relative to the root asked, so compare counts.
        assert len(expand_moe(other_root)) == len(added)

        restore_moe(other_root)
        assert not any(isinstance(m, ExpertLinear) for m in model.modules())
        assert not any(tracked_router(m) is not None for m in model.modules())


def test_experts_are_untracked_by_default_and_warn():
    """Tracking is opt-in, and skipping experts is announced, not silent."""
    model = build_model("gpt_oss")
    with pytest.warns(UserWarning, match="fused MoE expert modules are not"):
        collector = GradientCollector(
            model=model.base_model,
            cfg=IndexConfig(run_path="/tmp/bergson-moe-test"),
            data=Dataset.from_dict({"input_ids": [[1] * SEQ_LEN]}),
            processor=GradientProcessor(),
            skip_index=True,
        )
    assert moe_names(collector.target_info) == []


def test_opt_out_is_authoritative_over_an_expanded_model():
    model = build_model("gpt_oss")
    expand_moe(model.base_model)
    collector = make_collector(model, track_moe_experts=False)
    assert moe_names(collector.target_info) == []


@pytest.mark.parametrize("family,include_bias", BIAS_CASES)
def test_projection_shapes_and_determinism(family, include_bias):
    """Every MoE module yields a deterministic [N, p*p] block."""
    model = build_model(family)
    p = 4
    x = torch.randint(0, 64, (2, SEQ_LEN))
    collector = make_collector(
        model, processor=GradientProcessor(projection_dim=p, include_bias=include_bias)
    )
    names = moe_names(collector.target_info)

    grads = []
    for _ in range(2):
        with collector:
            backward_pass(model, x)
        grads.append({n: collector.mod_grads[n].clone() for n in names})

    for name in names:
        assert collector.shapes()[name] == torch.Size((p, p))
        assert grads[0][name].shape == (2, p * p)
        torch.testing.assert_close(grads[0][name], grads[1][name])
    assert max(g.abs().max() for g in grads[0].values()) > 0


def test_global_projection_absorbs_experts():
    """projection_target='global' sums all modules into one vector per example,
    so expert tracking costs nothing extra in the index."""
    model = build_model("gpt_oss")
    p = 16
    collector = make_collector(
        model, processor=GradientProcessor(projection_dim=p, projection_target="global")
    )
    assert moe_names(collector.target_info)
    assert collector.shapes() == {"gradients": torch.Size((p,))}

    with collector:
        backward_pass(model, torch.randint(0, 64, (3, SEQ_LEN)))

    grads = collector.mod_grads["gradients"]
    assert grads.shape == (3, p)
    assert torch.isfinite(grads).all() and grads.abs().max() > 0


def test_shapes_match_collected_widths():
    """shapes() is what Builder sizes the index from. Run on gpt-oss, where a
    bias column widens every gradient."""
    model = build_model("gpt_oss")
    collector = make_collector(model, processor=GradientProcessor(include_bias=True))
    with collector:
        backward_pass(model, torch.randint(0, 64, (2, SEQ_LEN)))

    for name, shape in collector.shapes().items():
        assert collector.mod_grads[name].shape == (2, math.prod(shape)), name


def test_attribute_tokens_rejected():
    """Under top-k routing one token feeds several experts, so per-token rows
    cannot line up with token positions."""
    collector = make_collector(build_model("gpt_oss"), attribute_tokens=True)
    with pytest.raises(ValueError, match="attribute_tokens is incompatible"):
        collector.__enter__()


@pytest.mark.parametrize("family", FAMILIES)
def test_ekfac_covariance_over_experts(family, tmp_path):
    """EK-FAC factors use each expert's routed-row mask, not the batch mask."""
    model = build_model(family)
    x = torch.randint(0, 64, (2, SEQ_LEN))
    collector = CovarianceCollector(
        model=model.base_model,
        processor=GradientProcessor(),
        dtype=torch.float32,
        path=str(tmp_path),
        track_moe_experts=True,
    )
    names = moe_names(collector.target_info)
    assert names, "no MoE modules discovered, the checks below would be vacuous"

    with collector.with_batch(torch.ones(x.shape, dtype=torch.bool)):
        backward_pass(model, x)

    for name in names:
        layer = model.base_model.get_submodule(name)
        i = getattr(layer, LayerAdapter.in_attr(layer))
        o = getattr(layer, LayerAdapter.out_attr(layer))
        assert collector.A_cov_dict[name].shape[-1] == i, name
        assert collector.S_cov_dict[name].shape[-1] == o, name


@pytest.mark.parametrize("family", FAMILIES)
def test_normalized_gradients_match_autograd(family):
    """Per-expert normalizers reach the right expert in the right orientation."""
    model = build_model(family)
    model.requires_grad_(True)
    base = model.base_model
    x = torch.randint(0, 64, (2, SEQ_LEN))

    collector = make_collector(model)
    names = moe_names(collector.target_info)

    torch.manual_seed(1)
    normalizers = {}
    for name in names:
        layer = base.get_submodule(name)
        shape = (
            getattr(layer, LayerAdapter.out_attr(layer)),
            getattr(layer, LayerAdapter.in_attr(layer)),
        )
        normalizers[name] = AdamNormalizer(torch.rand(shape) + 0.1)
    collector.processor = GradientProcessor(normalizers=normalizers)

    with collector:
        backward_pass(model, x)
    collected = {k: v.clone() for k, v in collector.mod_grads.items()}

    assert_matches_autograd(
        base,
        names,
        collected,
        lambda i: backward_pass(model, x[i : i + 1]),
        normalizers=normalizers,
    )


@use_experts_implementation(has_gate=False)
class _NonGatedExperts(nn.Module):
    """A non-gated fused experts module, as nemotron_h declares one: up_proj
    instead of gate_up_proj, a plain activation instead of gating. No released
    model small enough to build offline has that pair. No forward, because the
    collector always substitutes bergson_experts_forward."""

    def __init__(self, config, hidden: int, intermediate: int):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.up_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, intermediate, hidden) * 0.1
        )
        self.down_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, hidden, intermediate) * 0.1
        )
        self.act_fn = nn.GELU()


class _NonGatedMoEBlock(nn.Module):
    def __init__(self, hidden: int = 16, intermediate: int = 8):
        super().__init__()
        self.hidden = hidden
        self.gate = nn.Module()
        self.gate.num_experts = NUM_EXPERTS
        self.gate.weight = nn.Parameter(torch.randn(NUM_EXPERTS, hidden) * 0.1)
        self.gate.forward = self._route  # type: ignore[method-assign]
        self.experts = _NonGatedExperts(
            SimpleNamespace(_experts_implementation="eager"), hidden, intermediate
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def _route(self, hidden_states):
        logits = nn.functional.linear(hidden_states, self.gate.weight)
        weights, index = torch.topk(logits.softmax(-1), TOP_K, dim=-1)
        return logits, weights, index

    def forward(self, x):
        batch, seq, hidden = x.shape
        flat = x.reshape(-1, hidden)
        _, weights, index = self.gate(flat)
        return self.experts(flat, index, weights).reshape(batch, seq, hidden)


def test_non_gated_experts_gradients_match_autograd():
    """The up_proj path is collected as correctly as the gated one."""
    torch.manual_seed(0)
    block = _NonGatedMoEBlock()
    x = torch.randn(3, SEQ_LEN, block.hidden)

    collector = GradientCollector(
        model=block,
        cfg=IndexConfig(run_path="/tmp/bergson-moe-test"),
        data=Dataset.from_dict({"input_ids": [[1] * SEQ_LEN] * 3}),
        processor=GradientProcessor(),
        skip_index=True,
        track_moe_experts=True,
    )
    names = moe_names(collector.target_info)
    assert len(names) == 2 * NUM_EXPERTS + 1
    assert all(
        collector.model.get_submodule(n).weight_name == "up_proj"
        for n in names
        if n.endswith("up_proj")
    )

    def run(example: int) -> None:
        block.zero_grad()
        (block(x[example : example + 1]) ** 2).sum().backward()

    with collector:
        block.zero_grad()
        (block(x) ** 2).sum().backward()
    collected = {k: v.clone() for k, v in collector.mod_grads.items()}

    assert_matches_autograd(block, names, collected, run)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gpu_gradients_and_transparency(family, dtype):
    """The padded grid, its scatter and the index_add all run in the model's
    dtype, so reduced precision is where a dtype mistake surfaces. On GPU the
    unexpanded model also dispatches to a fused grouped-matmul kernel, so this
    is the only check of fidelity to that kernel rather than to eager."""
    model = build_model(family).to(device="cuda", dtype=dtype)
    x = torch.randint(0, 64, (3, SEQ_LEN), device="cuda")

    with torch.no_grad():
        reference = model(input_ids=x).logits.clone()
    expand_moe(model)
    with torch.no_grad():
        torch.testing.assert_close(
            model(input_ids=x).logits, reference, atol=5e-2, rtol=5e-2
        )
    restore_moe(model)

    # Loose: bf16 carries ~3 decimal digits, and the collector reassociates the
    # sum differently from a per-sample backward.
    collect_and_compare(model, batch_size=3, include_bias=False, tol=(5e-2, 5e-2))
