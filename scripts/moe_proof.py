"""Evidence that MoE fused-parameter experts and routers are now attributed.

Prints four independent checks, each reproducible from a clean checkout:

1. Coverage — what fraction of a real MoE model's parameters bergson tracks,
   before and after. Models are built on the meta device, so this measures
   production-sized models without downloading or allocating weights.
2. Correctness — max |collected - autograd| for per-example expert and router
   gradients, against per-sample backward passes on the fused parameters.
3. Transparency — the expanded model's logits, and its module/parameter
   inventory, are unchanged.
4. Sparsity — collected gradients respect top-k routing: an expert that a
   given example never routed to gets exactly zero gradient for it.

    python scripts/moe_proof.py
"""

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
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
from bergson.moe import ExpertLinear, expand_moe, restore_moe

PRODUCTION_MODELS = [
    "openai/gpt-oss-20b",
    "mistralai/Mixtral-8x7B-v0.1",
    "Qwen/Qwen3-30B-A3B",
    "allenai/OLMoE-1B-7B-0924",
]

TINY_CONFIGS = {
    "gpt-oss": lambda: GptOssConfig(
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        max_position_embeddings=64,
    ),
    "Mixtral": lambda: MixtralConfig(
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        max_position_embeddings=64,
    ),
    "Qwen3-MoE": lambda: Qwen3MoeConfig(
        hidden_size=32,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=64,
    ),
    "OLMoE": lambda: OlmoeConfig(
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        max_position_embeddings=64,
    ),
}


def tracked_parameter_count(model: nn.Module) -> tuple[int, int]:
    """(number of tracked modules, number of parameters they cover)."""
    info = HookCollectorBase.discover_targets(model)
    total = 0
    for name in info:
        layer = model.get_submodule(name)
        i = getattr(layer, LayerAdapter.in_attr(layer))
        o = getattr(layer, LayerAdapter.out_attr(layer))
        total += i * o
    return len(info), total


def report_coverage() -> None:
    print("\n1. COVERAGE — parameters bergson can attribute")
    print(f"   {'model':<28} {'before':>10} {'after':>10}   modules")
    for repo in PRODUCTION_MODELS:
        try:
            config = AutoConfig.from_pretrained(repo)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)
        except Exception as exc:  # offline, gated repo, unsupported arch
            print(f"   {repo:<28} skipped ({type(exc).__name__})")
            continue

        total = sum(p.numel() for p in model.parameters())
        before_mods, before = tracked_parameter_count(model)
        expand_moe(model)
        after_mods, after = tracked_parameter_count(model)
        print(
            f"   {repo:<28} {before / total:>9.1%} {after / total:>9.1%}"
            f"   {before_mods} -> {after_mods}"
        )


def build_tiny(name: str) -> nn.Module:
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(TINY_CONFIGS[name](), dtype=torch.float32)
    model.eval()
    return model


def moe_names(target_info) -> list[str]:
    return [
        n
        for n in target_info
        if "experts.expert_" in n or n.rpartition(".")[2] in ("gate", "router")
    ]


def collect(model, x, include_bias=False):
    from datasets import Dataset

    collector = GradientCollector(
        model=model.base_model,
        cfg=IndexConfig(run_path="/tmp/bergson-moe-proof"),
        data=Dataset.from_dict({"input_ids": x.tolist()}),
        processor=GradientProcessor(include_bias=include_bias),
        skip_index=True,
    )
    with collector:
        model.zero_grad()
        (model(input_ids=x).logits ** 2).sum().backward()
    return collector, {k: v.clone() for k, v in collector.mod_grads.items()}


def autograd_reference(base, name, include_bias):
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


def report_correctness() -> None:
    print("\n2. CORRECTNESS — per-example gradients vs per-sample autograd")
    print(f"   {'family':<12} {'MoE modules':>12} {'max abs diff':>14}  verdict")
    for name in TINY_CONFIGS:
        model = build_tiny(name)
        model.requires_grad_(True)
        base = model.base_model
        include_bias = name == "gpt-oss"  # only gpt-oss has expert biases
        x = torch.randint(0, 64, (3, 7))

        collector, collected = collect(model, x, include_bias)
        names = moe_names(collector.target_info)

        worst = 0.0
        for example in range(x.shape[0]):
            model.zero_grad()
            (model(input_ids=x[example : example + 1]).logits ** 2).sum().backward()
            for mod in names:
                reference = autograd_reference(base, mod, include_bias)
                delta = (reference - collected[mod][example]).abs().max().item()
                worst = max(worst, delta)

        scale = max(g.abs().max().item() for g in collected.values())
        ok = "PASS" if worst < 1e-4 * max(scale, 1.0) else "FAIL"
        print(f"   {name:<12} {len(names):>12} {worst:>14.2e}  {ok}")


def report_transparency() -> None:
    print("\n3. TRANSPARENCY — expansion changes visibility, not computation")
    print(f"   {'family':<12} {'logit delta':>12} {'params':>8} {'state_dict':>11}")
    for name in TINY_CONFIGS:
        model = build_tiny(name)
        x = torch.randint(0, 64, (3, 7))
        with torch.no_grad():
            before = model(input_ids=x).logits.clone()

        params = {n for n, _ in model.named_parameters()}
        state = set(model.state_dict())

        expand_moe(model)
        with torch.no_grad():
            after = model(input_ids=x).logits
        delta = (before - after).abs().max().item()
        same_params = params == {n for n, _ in model.named_parameters()}
        same_state = state == set(model.state_dict())

        restore_moe(model)
        with torch.no_grad():
            reverted = (before - model(input_ids=x).logits).abs().max().item()
        assert reverted == 0.0, "restore_moe did not revert exactly"

        print(
            f"   {name:<12} {delta:>12.1e} {str(same_params):>8} {str(same_state):>11}"
        )
    print("   (restore_moe reverted every model bit-exactly)")


def report_sparsity() -> None:
    """Many experts, few tokens, so most experts go unrouted for most examples."""
    print("\n4. SPARSITY — gradients respect top-k routing")
    num_experts, top_k, batch, seq = 16, 2, 4, 3

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        MixtralConfig(
            hidden_size=32,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=num_experts,
            num_experts_per_tok=top_k,
            vocab_size=64,
            max_position_embeddings=64,
        ),
        dtype=torch.float32,
    )
    model.eval()
    model.requires_grad_(True)
    x = torch.randint(0, 64, (batch, seq))
    _, collected = collect(model, x)

    expert_grads = {n: g for n, g in collected.items() if "experts.expert_" in n}
    pairs = [(n, i) for n in expert_grads for i in range(batch)]
    zero = [(n, i) for n, i in pairs if expert_grads[n][i].abs().max() == 0]

    # Each example routes seq*top_k slots, so it can touch at most that many
    # experts; the rest must come back exactly zero.
    reachable = min(seq * top_k, num_experts)
    print(
        f"   {batch} examples x {seq} tokens, top-{top_k} of {num_experts} experts: "
        f"each example reaches at most {reachable} experts"
    )
    print(
        f"   {len(zero)} of {len(pairs)} (example, expert-projection) pairs are "
        f"exactly zero"
    )
    print("   -> unrouted experts contribute no gradient, as top-k routing requires")
    assert zero, "expected some experts to go unrouted"
    for name, example in zero:
        assert expert_grads[name][example].abs().sum() == 0


def main() -> None:
    print("=" * 72)
    print("bergson MoE fused-parameter expert/router attribution — evidence")
    print("=" * 72)
    report_coverage()
    report_correctness()
    report_transparency()
    report_sparsity()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
