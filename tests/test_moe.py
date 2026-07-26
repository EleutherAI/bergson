"""Gradient collection for MoE models with fused-parameter experts and routers.

Covers both storage conventions of transformers' ``use_experts_implementation``:
gpt-oss (``is_transposed``, biased, interleaved gate) and Mixtral
(non-transposed, unbiased, concatenated gate). Models are built from configs
constructed in-process so the suite stays offline.
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
from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientProcessor,
    LayerAdapter,
)
from bergson.hessians.kfac import CovarianceCollector
from bergson.moe import ExpertLinear, expand_moe, is_bare_linear, restore_moe
from bergson.utils.load_from_optimizer import get_normalizers

FAMILIES = ("gpt_oss", "mixtral")
BIAS_CASES = [("gpt_oss", False), ("gpt_oss", True), ("mixtral", False)]
"""``(family, include_bias)`` worth running: every one of gpt-oss's tracked
modules carries a bias, while Mixtral has none at all, so ``include_bias=True``
on Mixtral would re-run the ``False`` case."""
NUM_EXPERTS = 4
TOP_K = 2
SEQ_LEN = 7


def build_model(family: str) -> nn.Module:
    """A tiny MoE causal LM of the given family, on CPU in fp32."""
    kwargs = dict(
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        vocab_size=64,
        max_position_embeddings=64,
    )
    if family == "gpt_oss":
        config = GptOssConfig(head_dim=8, **kwargs)
    else:
        config = MixtralConfig(**kwargs)

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.eval()
    return model


def moe_module_names(target_info) -> list[str]:
    """The expert and router module names among discovered targets."""
    return [
        name
        for name in target_info
        if "experts.expert_" in name or name.rpartition(".")[2] in ("gate", "router")
    ]


def make_collector(model, *, processor=None, track_moe_experts=True, **cfg_kwargs):
    """A ``GradientCollector`` over ``model.base_model``, as collection builds it."""
    return GradientCollector(
        model=model.base_model,
        cfg=IndexConfig(run_path="/tmp/bergson-moe-test", **cfg_kwargs),
        data=Dataset.from_dict({"input_ids": [[1] * SEQ_LEN]}),
        processor=processor or GradientProcessor(),
        skip_index=True,
        track_moe_experts=track_moe_experts,
    )


def backward_pass(model, x: torch.Tensor) -> None:
    """Populate gradients from a scalar loss, as the collectors expect."""
    model.zero_grad()
    (model(input_ids=x).logits ** 2).sum().backward()


def autograd_gradient(base, name: str, include_bias: bool) -> torch.Tensor:
    """The flattened gradient of ``name``'s weight from the last backward pass.

    Read off the *fused* parameter and sliced/oriented back to the ``[out, in]``
    layout the collector reports, so this is an independent reference rather than
    a restatement of the collector's own arithmetic.
    """
    layer = base.get_submodule(name)
    if isinstance(layer, ExpertLinear):
        fused = getattr(layer._experts, layer.weight_name)
        assert fused.grad is not None
        grad = fused.grad[layer.expert_idx]
        if LayerAdapter.weight_transposed(layer):
            grad = grad.T
        fused_bias = getattr(layer._experts, f"{layer.weight_name}_bias", None)
        if include_bias and fused_bias is not None:
            assert fused_bias.grad is not None
            grad = torch.cat([grad, fused_bias.grad[layer.expert_idx, :, None]], dim=1)
    else:
        assert layer.weight.grad is not None
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
    include_bias: bool = False,
    normalizers: dict | None = None,
):
    """Check each collected per-example gradient against a per-sample backward.

    ``run_example(i)`` must leave the model's gradients holding example ``i``'s
    alone. When ``normalizers`` is given the reference is normalized the same
    way the collector would, which is what catches an expert being paired with
    the wrong normalizer or orientation.
    """
    for example in range(len(next(iter(collected.values())))):
        run_example(example)
        for name in names:
            expected = autograd_gradient(base, name, include_bias)
            if normalizers is not None:
                normalizer = normalizers[name]
                expected = normalizer.normalize_weight(
                    expected.view(normalizer.weight_avg_sq.shape).clone()
                ).flatten()
            torch.testing.assert_close(
                collected[name][example],
                expected,
                atol=1e-5,
                rtol=1e-4,
                msg=f"{name}, example {example}",
            )

    # Guard against a vacuous pass on all-zero gradients.
    assert max(grad.abs().max() for grad in collected.values()) > 0


def collect_and_compare(model, batch_size: int, include_bias: bool):
    """Collect per-example MoE gradients and check them against autograd."""
    model.requires_grad_(True)
    x = torch.randint(0, 64, (batch_size, SEQ_LEN))

    collector = make_collector(
        model, processor=GradientProcessor(include_bias=include_bias)
    )
    names = moe_module_names(collector.target_info)
    assert names, "no fused MoE experts or routers were discovered"

    with collector:
        backward_pass(model, x)
    collected = {name: grad.clone() for name, grad in collector.mod_grads.items()}

    # Builder concatenates every module in shapes(), so none may be missing —
    # including experts that happened to receive no tokens.
    assert set(collected) == set(collector.shapes())

    assert_matches_autograd(
        model.base_model,
        names,
        collected,
        lambda i: backward_pass(model, x[i : i + 1]),
        include_bias=include_bias,
    )


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("family,include_bias", BIAS_CASES)
def test_per_example_gradients_match_autograd(family, batch_size, include_bias):
    """Per-example expert and router gradients equal per-sample autograd."""
    collect_and_compare(build_model(family), batch_size, include_bias)


# Families whose router or MoE block differs from the two reference layouts.
# DeepSeek-V3 brings a sigmoid grouped-top-k router with a score-correction bias
# and a shared expert; the other two vary the block wiring.
OTHER_FAMILY_CONFIGS = {
    "deepseek_v3": lambda **shared: DeepseekV3Config(
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
        **{**shared, "num_key_value_heads": 4},
    ),
    "qwen3_moe": lambda **shared: Qwen3MoeConfig(
        intermediate_size=32,
        moe_intermediate_size=16,
        num_experts=NUM_EXPERTS,
        head_dim=8,
        **shared,
    ),
    "olmoe": lambda **shared: OlmoeConfig(
        intermediate_size=16, num_experts=NUM_EXPERTS, **shared
    ),
}


@pytest.mark.parametrize("family", sorted(OTHER_FAMILY_CONFIGS))
def test_other_families_match_autograd(family):
    """Capability-based detection generalizes past the two reference families."""
    config = OTHER_FAMILY_CONFIGS[family](
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
        num_experts_per_tok=TOP_K,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.eval()
    collect_and_compare(model, batch_size=2, include_bias=False)


@pytest.mark.parametrize("family", FAMILIES)
def test_expansion_covers_every_fused_parameter(family):
    """Expansion adds exactly the fused expert and router weights to the
    trackable set — the parameters that were silently unreachable before."""
    model = build_model(family)
    base = model.base_model

    def tracked_weights() -> int:
        total = 0
        for name in HookCollectorBase.discover_targets(base):
            layer = base.get_submodule(name)
            total += getattr(layer, LayerAdapter.in_attr(layer)) * getattr(
                layer, LayerAdapter.out_attr(layer)
            )
        return total

    before = tracked_weights()
    expand_moe(base)
    after = tracked_weights()

    fused_experts = sum(p.numel() for p in base.parameters() if p.ndim == 3)
    routers = sum(m.weight.numel() for m in base.modules() if is_bare_linear(m))
    assert fused_experts and routers
    assert after - before == fused_experts + routers

    # The fused parameters dominate an MoE model, so the tracked share should
    # go from a small minority to nearly all of it.
    model_total = sum(p.numel() for p in base.parameters())
    assert before / model_total < 0.5 < after / model_total


@pytest.mark.parametrize("family", FAMILIES)
def test_expansion_is_transparent_and_reversible(family):
    """Expansion changes what is *visible*, never what the model computes."""
    model = build_model(family)
    x = torch.randint(0, 64, (3, SEQ_LEN))
    with torch.no_grad():
        reference = model(input_ids=x).logits.clone()

    parameters = {name for name, _ in model.named_parameters()}
    state_dict = set(model.state_dict())
    modules = set(dict(model.named_modules()))

    added = expand_moe(model)
    # Two projections per expert per layer, plus one router per layer.
    assert len(added) == 2 * (2 * NUM_EXPERTS + 1)
    assert expand_moe(model) == added, "expansion should be idempotent"

    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=x).logits, reference)

    assert {name for name, _ in model.named_parameters()} == parameters
    assert set(model.state_dict()) == state_dict
    # Only additive, and every reported name is really there. The added set also
    # contains the per-expert containers, which hold no trackable weight.
    expanded = set(dict(model.named_modules()))
    assert modules < expanded and set(added) < expanded

    restore_moe(model)
    assert {name for name, _ in model.named_modules()} == modules
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=x).logits, reference)


def test_experts_are_untracked_by_default_and_warn():
    """Tracking is opt-in, and skipping the experts is announced.

    Skipping leaves only attention and ``lm_head`` attributed. The old failure
    mode was that nothing said so, so the warning is part of the contract.
    """
    model = build_model("gpt_oss")
    with pytest.warns(UserWarning, match="fused MoE expert modules are not"):
        collector = GradientCollector(
            model=model.base_model,
            cfg=IndexConfig(run_path="/tmp/bergson-moe-test"),
            data=Dataset.from_dict({"input_ids": [[1] * SEQ_LEN]}),
            processor=GradientProcessor(),
            skip_index=True,
        )
    assert moe_module_names(collector.target_info) == []


def test_opt_out_is_authoritative_over_an_expanded_model():
    """``track_moe_experts=False`` un-expands, wherever the expansion came from.

    The restore path does not depend on the family's storage layout, so one
    family covers it.
    """
    model = build_model("gpt_oss")
    expand_moe(model.base_model)
    collector = make_collector(model, track_moe_experts=False)
    assert moe_module_names(collector.target_info) == []


@pytest.mark.parametrize("family,include_bias", BIAS_CASES)
def test_projection_shapes_and_determinism(family, include_bias):
    """With projection, every MoE module yields a deterministic [N, p*p] block."""
    model = build_model(family)
    projection_dim = 4
    x = torch.randint(0, 64, (2, SEQ_LEN))

    collector = make_collector(
        model,
        processor=GradientProcessor(
            projection_dim=projection_dim, include_bias=include_bias
        ),
    )
    names = moe_module_names(collector.target_info)

    grads = []
    for _ in range(2):
        with collector:
            backward_pass(model, x)
        grads.append({name: collector.mod_grads[name].clone() for name in names})

    shapes = collector.shapes()
    for name in names:
        assert shapes[name] == torch.Size((projection_dim, projection_dim))
        assert grads[0][name].shape == (2, projection_dim**2)
        torch.testing.assert_close(grads[0][name], grads[1][name])
    assert max(g.abs().max() for g in grads[0].values()) > 0


def test_global_projection_absorbs_experts():
    """``projection_target='global'`` keeps expert tracking free in the index.

    Every module's projected gradient sums into one vector per example, so the
    index does not grow with the expert count — the mitigation the README points
    users at. The sum is over modules, so one family covers it.
    """
    model = build_model("gpt_oss")
    projection_dim = 16
    collector = make_collector(
        model,
        processor=GradientProcessor(
            projection_dim=projection_dim, projection_target="global"
        ),
    )
    assert moe_module_names(collector.target_info)
    assert collector.shapes() == {"gradients": torch.Size((projection_dim,))}

    with collector:
        backward_pass(model, torch.randint(0, 64, (3, SEQ_LEN)))

    grads = collector.mod_grads["gradients"]
    assert grads.shape == (3, projection_dim)
    assert torch.isfinite(grads).all() and grads.abs().max() > 0


def test_shapes_match_collected_widths():
    """``shapes()`` is the contract Builder sizes the index from.

    Run on gpt-oss, the family where a bias column widens every gradient and so
    the width is easiest to get wrong.
    """
    model = build_model("gpt_oss")
    collector = make_collector(model, processor=GradientProcessor(include_bias=True))
    with collector:
        backward_pass(model, torch.randint(0, 64, (2, SEQ_LEN)))

    for name, shape in collector.shapes().items():
        assert collector.mod_grads[name].shape == (2, math.prod(shape)), name


def test_attribute_tokens_rejected():
    """Per-token attribution cannot be aligned with top-k expert routing.

    The guard keys on the module type, so one family covers it.
    """
    model = build_model("gpt_oss")
    collector = make_collector(model, attribute_tokens=True)
    with pytest.raises(ValueError, match="attribute_tokens is incompatible"):
        collector.__enter__()


@pytest.mark.parametrize("family", FAMILIES)
def test_ekfac_covariance_over_experts(family, tmp_path):
    """EK-FAC factors use each expert's own routed-row mask, not the batch mask."""
    model = build_model(family)
    x = torch.randint(0, 64, (2, SEQ_LEN))
    collector = CovarianceCollector(
        model=model.base_model,
        processor=GradientProcessor(),
        dtype=torch.float32,
        path=str(tmp_path),
        track_moe_experts=True,
    )
    names = moe_module_names(collector.target_info)
    assert names, "no MoE modules discovered, the checks below would be vacuous"

    mask = torch.ones(x.shape, dtype=torch.bool)
    with collector.with_batch(mask):
        backward_pass(model, x)

    for name in names:
        layer = model.base_model.get_submodule(name)
        i = getattr(layer, LayerAdapter.in_attr(layer))
        o = getattr(layer, LayerAdapter.out_attr(layer))
        assert collector.A_cov_dict[name].shape[-1] == i, name
        assert collector.S_cov_dict[name].shape[-1] == o, name


def fake_optimizer_state(model, factored: bool) -> tuple[dict, dict]:
    """An ``optimizer.pt``-shaped second-moment state for every 2D and 3D param."""
    state, index_to_name = {}, {}
    for idx, (name, param) in enumerate(model.named_parameters()):
        index_to_name[idx] = name
        if param.ndim not in (2, 3):
            continue
        moments = torch.rand_like(param) + 0.1
        state[idx] = (
            {"exp_avg_sq_row": moments.mean(-1), "exp_avg_sq_col": moments.mean(-2)}
            if factored and param.ndim == 3
            else {"exp_avg_sq": moments}
        )
    return state, index_to_name


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("factored", [False, True])
def test_normalizers_from_fused_optimizer_state(family, factored):
    """A 3D expert second moment becomes one normalizer per expert.

    ``get_normalizers`` drops any moment that is not 2D, so without per-expert
    slicing ``--optimizer_state`` would silently normalize nothing on a MoE
    model.
    """
    model = build_model(family)
    expand_moe(model)
    state, index_to_name = fake_optimizer_state(model, factored)

    normalizers = get_normalizers(
        {"state": state},
        index_to_name,
        None,
        "",
        False,
        torch.device("cpu"),
        base_prefix="model.",
        model=model,
    )

    expected_type = AdafactorNormalizer if factored else AdamNormalizer
    shims = {
        name: module
        for name, module in model.base_model.named_modules()
        if isinstance(module, ExpertLinear)
    }
    assert len(shims) == 2 * 2 * NUM_EXPERTS  # two projections, per expert, per layer

    for name, shim in shims.items():
        normalizer = normalizers[name]
        assert isinstance(normalizer, expected_type), name
        if factored:
            assert normalizer.row.shape == (shim.out_features,), name
            assert normalizer.col.shape == (shim.in_features,), name
        else:
            assert normalizer.weight_avg_sq.shape == (
                shim.out_features,
                shim.in_features,
            ), name

    # The router's 2D weight keeps going through the ordinary path.
    router = "layers.0.mlp." + ("router" if family == "gpt_oss" else "gate")
    assert is_bare_linear(model.base_model.get_submodule(router))
    assert isinstance(normalizers[router], AdamNormalizer)


@use_experts_implementation(has_gate=False)
class _NonGatedExperts(nn.Module):
    """A non-gated fused experts module, as ``nemotron_h`` declares one.

    ``has_gate=False`` swaps ``gate_up_proj`` for ``up_proj`` and the gating
    mechanism for a plain activation. No released model small enough to build
    offline exercises that pair, so it is declared here with the same decorator
    the real ones use. No ``forward``: the collector always substitutes
    ``bergson_experts_forward``, which is the path under test.
    """

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
    """Router plus non-gated experts, wired like a transformers MoE block."""

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
    """The ``up_proj`` path is collected as correctly as the gated one."""
    torch.manual_seed(0)
    block = _NonGatedMoEBlock()
    batch = 3
    x = torch.randn(batch, SEQ_LEN, block.hidden)

    collector = GradientCollector(
        model=block,
        cfg=IndexConfig(run_path="/tmp/bergson-moe-test"),
        data=Dataset.from_dict({"input_ids": [[1] * SEQ_LEN] * batch}),
        processor=GradientProcessor(),
        skip_index=True,
        track_moe_experts=True,
    )
    names = moe_module_names(collector.target_info)
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
    collected = {name: grad.clone() for name, grad in collector.mod_grads.items()}

    assert_matches_autograd(block, names, collected, run)


@pytest.mark.parametrize("family", FAMILIES)
def test_normalized_gradients_match_autograd(family):
    """Per-expert normalizers are applied to the right expert and orientation."""
    model = build_model(family)
    model.requires_grad_(True)
    base = model.base_model
    x = torch.randint(0, 64, (2, SEQ_LEN))

    collector = make_collector(model)
    names = moe_module_names(collector.target_info)

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
    collected = {name: grad.clone() for name, grad in collector.mod_grads.items()}

    assert_matches_autograd(
        base,
        names,
        collected,
        lambda i: backward_pass(model, x[i : i + 1]),
        normalizers=normalizers,
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_expansion_state_is_root_independent(family):
    """Expanding and restoring work from any root containing the experts.

    Collection runs on ``model.base_model`` while callers hold ``model``, so
    undo state keyed to one root would leave the other believing nothing was
    expanded — and ``--track_moe_experts false`` would silently keep tracking.
    """
    model = build_model(family)
    for expand_root, restore_root in (
        (model, model.base_model),
        (model.base_model, model),
    ):
        added = expand_moe(expand_root)
        assert any(isinstance(m, ExpertLinear) for m in model.modules())

        # Expanding again from the other root must not double-attach. Names come
        # back relative to the root asked, so compare counts, not strings.
        assert len(expand_moe(restore_root)) == len(added)
        assert len(expand_moe(expand_root)) == len(added)

        restore_moe(restore_root)
        assert not any(isinstance(m, ExpertLinear) for m in model.modules())
        assert not any(is_bare_linear(m) for m in model.modules())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gpu_low_precision_gradients_match_autograd(family, dtype):
    """Per-example gradients survive the GPU low-precision path.

    The padded grid, the scatter into it and the ``index_add`` accumulation all
    run in the model's dtype, so reduced precision is where a dtype or
    accumulation mistake would surface.
    """
    model = build_model(family).to(device="cuda", dtype=dtype)
    model.requires_grad_(True)
    base = model.base_model
    x = torch.randint(0, 64, (3, SEQ_LEN), device="cuda")

    collector = make_collector(model)
    names = moe_module_names(collector.target_info)
    assert names

    with collector:
        backward_pass(model, x)
    collected = {name: grad.clone() for name, grad in collector.mod_grads.items()}
    assert all(torch.isfinite(g).all() for g in collected.values())

    for example in range(x.shape[0]):
        backward_pass(model, x[example : example + 1])
        for name in names:
            expected = autograd_gradient(base, name, include_bias=False)
            # Loose bounds: bf16 carries ~3 decimal digits, and the collector
            # reassociates the sum differently from a per-sample backward.
            torch.testing.assert_close(
                collected[name][example].float(),
                expected.float(),
                atol=5e-2,
                rtol=5e-2,
                msg=f"{name}, example {example}, {dtype}",
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("family", FAMILIES)
def test_gpu_expansion_is_transparent_in_bf16(family):
    """Replacing the experts' forward must not move the model's own outputs.

    On GPU the unexpanded model dispatches to a fused grouped-matmul kernel, so
    this is the check that the replacement stays faithful to the kernel it
    displaces, not merely to the eager reference.
    """
    model = build_model(family).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randint(0, 64, (3, SEQ_LEN), device="cuda")

    with torch.no_grad():
        reference = model(input_ids=x).logits.clone()
    expand_moe(model)
    with torch.no_grad():
        expanded = model(input_ids=x).logits

    torch.testing.assert_close(expanded, reference, atol=5e-2, rtol=5e-2)
