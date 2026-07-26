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


def make_collector(model, *, projection_dim=None, include_bias=False, **kwargs):
    """A ``GradientCollector`` over ``model.base_model``, as collection builds it."""
    tokens = [[1] * SEQ_LEN]
    return GradientCollector(
        model=model.base_model,
        cfg=IndexConfig(run_path="/tmp/bergson-moe-test", **kwargs),
        data=Dataset.from_dict({"input_ids": tokens}),
        processor=GradientProcessor(
            projection_dim=projection_dim, include_bias=include_bias
        ),
        skip_index=True,
    )


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


def collect_and_compare(model, batch_size: int, include_bias: bool):
    """Collect per-example MoE gradients and check them against autograd."""
    model.requires_grad_(True)
    base = model.base_model
    x = torch.randint(0, 64, (batch_size, SEQ_LEN))

    collector = make_collector(model, include_bias=include_bias)
    names = moe_module_names(collector.target_info)
    assert names, "no fused MoE experts or routers were discovered"

    with collector:
        model.zero_grad()
        (model(input_ids=x).logits ** 2).sum().backward()
    collected = {name: grad.clone() for name, grad in collector.mod_grads.items()}

    # Builder concatenates every module in shapes(), so none may be missing —
    # including experts that happened to receive no tokens.
    assert set(collected) == set(collector.shapes())

    for example in range(batch_size):
        model.zero_grad()
        (model(input_ids=x[example : example + 1]).logits ** 2).sum().backward()
        for name in names:
            expected = autograd_gradient(base, name, include_bias)
            torch.testing.assert_close(
                collected[name][example],
                expected,
                atol=1e-5,
                rtol=1e-4,
                msg=f"{name}, example {example}",
            )

    # Guard against a vacuous pass on all-zero gradients.
    assert max(grad.abs().max() for grad in collected.values()) > 0


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("include_bias", [False, True])
def test_per_example_gradients_match_autograd(family, batch_size, include_bias):
    """Per-example expert and router gradients equal per-sample autograd."""
    collect_and_compare(build_model(family), batch_size, include_bias)


def _other_family_config(family: str):
    """Configs for families whose router or block differs from the two above."""
    shared = dict(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
        num_experts_per_tok=TOP_K,
    )
    if family == "deepseek_v3":
        # Sigmoid + grouped top-k router with a score-correction bias, plus a
        # shared expert alongside the routed ones.
        return DeepseekV3Config(
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
            num_key_value_heads=4,
            **{k: v for k, v in shared.items() if k != "num_key_value_heads"},
        )
    if family == "qwen3_moe":
        return Qwen3MoeConfig(
            intermediate_size=32,
            moe_intermediate_size=16,
            num_experts=NUM_EXPERTS,
            head_dim=8,
            **shared,
        )
    return OlmoeConfig(intermediate_size=16, num_experts=NUM_EXPERTS, **shared)


@pytest.mark.parametrize("family", ["deepseek_v3", "qwen3_moe", "olmoe"])
def test_other_families_match_autograd(family):
    """Capability-based detection generalizes past the two reference families."""
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        _other_family_config(family), dtype=torch.float32
    )
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
    assert expand_moe(model) == [], "expansion should be idempotent"

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


@pytest.mark.parametrize("family", FAMILIES)
def test_track_moe_experts_false_skips_expansion(family):
    """The opt-out restores the previous, expert-blind behaviour.

    Authoritative even when something has already expanded the model, so the
    flag means the same thing wherever it is read.
    """
    model = build_model(family)
    expand_moe(model.base_model)
    collector = GradientCollector(
        model=model.base_model,
        cfg=IndexConfig(run_path="/tmp/bergson-moe-test"),
        data=Dataset.from_dict({"input_ids": [[1] * SEQ_LEN]}),
        processor=GradientProcessor(),
        skip_index=True,
        track_moe_experts=False,
    )
    assert moe_module_names(collector.target_info) == []


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("include_bias", [False, True])
def test_projection_shapes_and_determinism(family, include_bias):
    """With projection, every MoE module yields a deterministic [N, p*p] block."""
    model = build_model(family)
    projection_dim = 4
    x = torch.randint(0, 64, (2, SEQ_LEN))

    collector = make_collector(
        model, projection_dim=projection_dim, include_bias=include_bias
    )
    names = moe_module_names(collector.target_info)

    grads = []
    for _ in range(2):
        with collector:
            model.zero_grad()
            (model(input_ids=x).logits ** 2).sum().backward()
        grads.append({name: collector.mod_grads[name].clone() for name in names})

    shapes = collector.shapes()
    for name in names:
        assert shapes[name] == torch.Size((projection_dim, projection_dim))
        assert grads[0][name].shape == (2, projection_dim**2)
        torch.testing.assert_close(grads[0][name], grads[1][name])
    assert max(g.abs().max() for g in grads[0].values()) > 0


@pytest.mark.parametrize("family", FAMILIES)
def test_global_projection_absorbs_experts(family):
    """``projection_target='global'`` keeps expert tracking free in the index.

    Every module's projected gradient sums into one vector per example, so the
    index does not grow with the expert count — the mitigation the README points
    users at.
    """
    model = build_model(family)
    projection_dim = 16
    collector = GradientCollector(
        model=model.base_model,
        cfg=IndexConfig(run_path="/tmp/bergson-moe-test"),
        data=Dataset.from_dict({"input_ids": [[1] * SEQ_LEN]}),
        processor=GradientProcessor(
            projection_dim=projection_dim, projection_target="global"
        ),
        skip_index=True,
    )
    assert moe_module_names(collector.target_info)
    assert collector.shapes() == {"gradients": torch.Size((projection_dim,))}

    with collector:
        model.zero_grad()
        (
            model(input_ids=torch.randint(0, 64, (3, SEQ_LEN))).logits ** 2
        ).sum().backward()

    grads = collector.mod_grads["gradients"]
    assert grads.shape == (3, projection_dim)
    assert torch.isfinite(grads).all() and grads.abs().max() > 0


@pytest.mark.parametrize("family", FAMILIES)
def test_shapes_match_collected_widths(family):
    """``shapes()`` is the contract Builder sizes the index from."""
    model = build_model(family)
    collector = make_collector(model, include_bias=True)
    with collector:
        model.zero_grad()
        (
            model(input_ids=torch.randint(0, 64, (2, SEQ_LEN))).logits ** 2
        ).sum().backward()

    for name, shape in collector.shapes().items():
        assert collector.mod_grads[name].shape == (2, math.prod(shape)), name


@pytest.mark.parametrize("family", FAMILIES)
def test_attribute_tokens_rejected(family):
    """Per-token attribution cannot be aligned with top-k expert routing."""
    model = build_model(family)
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
    )
    names = moe_module_names(collector.target_info)

    mask = torch.ones(x.shape, dtype=torch.bool)
    with collector.with_batch(mask):
        model.zero_grad()
        (model(input_ids=x).logits ** 2).sum().backward()

    for name in names:
        layer = model.base_model.get_submodule(name)
        i = getattr(layer, LayerAdapter.in_attr(layer))
        o = getattr(layer, LayerAdapter.out_attr(layer))
        assert collector.A_cov_dict[name].shape[-1] == i, name
        assert collector.S_cov_dict[name].shape[-1] == o, name


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

    state, index_to_name = {}, {}
    for idx, (name, param) in enumerate(model.named_parameters()):
        index_to_name[idx] = name
        if param.ndim == 3:
            moments = torch.rand_like(param) + 0.1
            state[idx] = (
                {
                    "exp_avg_sq_row": moments.mean(-1),
                    "exp_avg_sq_col": moments.mean(-2),
                }
                if factored
                else {"exp_avg_sq": moments}
            )
        elif param.ndim == 2:
            state[idx] = {"exp_avg_sq": torch.rand_like(param) + 0.1}

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

    experts = model.base_model.layers[0].mlp.experts
    expected_type = AdafactorNormalizer if factored else AdamNormalizer
    for expert_idx in range(NUM_EXPERTS):
        for leaf in ("gate_up_proj", "down_proj"):
            name = f"layers.0.mlp.experts.expert_{expert_idx}.{leaf}"
            normalizer = normalizers[name]
            assert isinstance(normalizer, expected_type), name

            shim = getattr(getattr(experts, f"expert_{expert_idx}"), leaf)
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
    batch, seq = 3, SEQ_LEN
    x = torch.randn(batch, seq, block.hidden)

    collector = GradientCollector(
        model=block,
        cfg=IndexConfig(run_path="/tmp/bergson-moe-test"),
        data=Dataset.from_dict({"input_ids": [[1] * seq] * batch}),
        processor=GradientProcessor(),
        skip_index=True,
    )
    names = moe_module_names(collector.target_info)
    assert len(names) == 2 * NUM_EXPERTS + 1
    assert all(
        collector.model.get_submodule(n).weight_name == "up_proj"
        for n in names
        if n.endswith("up_proj")
    )

    with collector:
        block.zero_grad()
        (block(x) ** 2).sum().backward()
    collected = {name: grad.clone() for name, grad in collector.mod_grads.items()}

    for example in range(batch):
        block.zero_grad()
        (block(x[example : example + 1]) ** 2).sum().backward()
        for name in names:
            torch.testing.assert_close(
                collected[name][example],
                autograd_gradient(block, name, include_bias=False),
                atol=1e-5,
                rtol=1e-4,
                msg=f"{name}, example {example}",
            )
    assert max(grad.abs().max() for grad in collected.values()) > 0


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
    shapes = {}
    for name in names:
        layer = base.get_submodule(name)
        shapes[name] = (
            getattr(layer, LayerAdapter.out_attr(layer)),
            getattr(layer, LayerAdapter.in_attr(layer)),
        )
    normalizers = {
        name: AdamNormalizer(torch.rand(s) + 0.1) for name, s in shapes.items()
    }
    collector.processor = GradientProcessor(normalizers=normalizers)

    with collector:
        model.zero_grad()
        (model(input_ids=x).logits ** 2).sum().backward()
    collected = {name: grad.clone() for name, grad in collector.mod_grads.items()}

    for example in range(x.shape[0]):
        model.zero_grad()
        (model(input_ids=x[example : example + 1]).logits ** 2).sum().backward()
        for name in names:
            expected = autograd_gradient(base, name, include_bias=False)
            # normalize_weight mutates its argument, so hand it a fresh view.
            expected = normalizers[name].normalize_weight(
                expected.view(shapes[name]).clone()
            )
            torch.testing.assert_close(
                collected[name][example],
                expected.flatten(),
                atol=1e-5,
                rtol=1e-4,
                msg=f"{name}, example {example}",
            )
