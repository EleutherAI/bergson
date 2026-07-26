"""Measure what MoE expert tracking costs in the expert GEMMs.

Tracking pads each expert's routed tokens into an ``[N, L]`` grid, where ``L``
is the widest example's row count. The padded grid multiplies more rows than the
routing requires, by ``N * L / rows_routed`` — the load imbalance *across
examples in one batch*, which depends only on batch shape, expert count and
top-k, so it is exactly computable without a GPU.

What this does NOT measure: wall-clock on GPU. Tracking replaces the fused
grouped-matmul kernel the model would otherwise dispatch to, and that cost is
separate from, and likely larger than, the padding overhead measured here. Run
``--wall_clock`` for an eager-mode timing ratio on whatever device is present.

    python benchmarks/moe_padding_overhead.py
    python benchmarks/moe_padding_overhead.py --wall_clock
"""

import argparse
import time
from dataclasses import dataclass

import torch

from bergson.moe import _Grid


@dataclass
class Shape:
    """One realistic batch geometry."""

    label: str
    num_examples: int
    seq_len: int
    num_experts: int
    top_k: int


SHAPES = [
    Shape("1 long doc      ", 1, 2048, 32, 4),
    Shape("4 x 512         ", 4, 512, 32, 4),
    Shape("16 x 128        ", 16, 128, 32, 4),
    Shape("64 x 32         ", 64, 32, 32, 4),
    Shape("8 x 256, 128 exp", 8, 256, 128, 8),
]


def padding_ratio(shape: Shape, seed: int = 0) -> float:
    """Padded rows divided by routed rows, averaged over the experts."""
    generator = torch.Generator().manual_seed(seed)
    num_tokens = shape.num_examples * shape.seq_len
    # Uniform routing: the balanced case, and so the optimistic one.
    top_k_index = torch.stack(
        [
            torch.randperm(shape.num_experts, generator=generator)[: shape.top_k]
            for _ in range(num_tokens)
        ]
    )

    padded = routed = 0
    for expert in range(shape.num_experts):
        token_idx, _ = torch.where(top_k_index == expert)
        grid = _Grid.build(token_idx, shape.num_examples, shape.seq_len)
        padded += shape.num_examples * grid.width
        routed += token_idx.numel()
    return padded / max(routed, 1)


def wall_clock_ratio(shape: Shape) -> str:
    """Eager forward+backward with tracking vs without, on the local device."""
    from transformers import AutoModelForCausalLM, MixtralConfig

    from bergson.moe import expand_moe, restore_moe

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        MixtralConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            num_local_experts=shape.num_experts,
            num_experts_per_tok=shape.top_k,
            vocab_size=512,
            max_position_embeddings=shape.seq_len,
        ),
        dtype=dtype,
    ).to(device)
    x = torch.randint(0, 512, (shape.num_examples, shape.seq_len), device=device)

    def timed() -> float:
        for _ in range(2):  # warm up
            model.zero_grad()
            (model(input_ids=x).logits ** 2).sum().backward()
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(5):
            model.zero_grad()
            (model(input_ids=x).logits ** 2).sum().backward()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) / 5

    baseline = timed()
    expand_moe(model)
    tracked = timed()
    restore_moe(model)
    return f"{tracked / baseline:.2f}x ({device})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall_clock", action="store_true")
    args = parser.parse_args()

    print("MoE expert tracking — padded rows / routed rows in the expert GEMMs")
    print(f"  {'batch':<18} {'experts':>8} {'top-k':>6} {'padding':>9}", end="")
    print(f" {'wall clock':>18}" if args.wall_clock else "")

    for shape in SHAPES:
        row = (
            f"  {shape.label:<18} {shape.num_experts:>8} {shape.top_k:>6} "
            f"{padding_ratio(shape):>8.2f}x"
        )
        if args.wall_clock:
            row += f" {wall_clock_ratio(shape):>18}"
        print(row)

    print(
        "\nPadding overhead is load imbalance across examples, so it falls as the\n"
        "batch holds fewer, longer documents. Wall clock also carries the loss of\n"
        "the fused grouped-matmul kernel, which the padding figure does not cover."
    )


if __name__ == "__main__":
    main()
