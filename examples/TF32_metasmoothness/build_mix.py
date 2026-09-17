"""Build the training data for the TF32 metasmoothness example.

Mixes documents from cais/wmdp-bio-forget-corpus with documents that passed the
combined filter in the EleutherAI filtering annealing mix, in an 85/15 token
ratio, up to a total token budget.

The annealing mix is sampled as contiguous blocks at random offsets because a
shuffled pass over the full table reads pages across the whole file and runs
out of memory.

Usage:
    python examples/TF32_metasmoothness/build_mix.py
"""

import argparse
import random

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

FORGET_DATASET = "cais/wmdp-bio-forget-corpus"
PRETRAIN_DATASET = "EleutherAI/filtering-annealing-mix_20250226-011545"


def count_tokens(texts: list[str], tokenizer) -> list[int]:
    """Token counts per text, matching tokenize_and_chunk's counting."""
    encodings = tokenizer(texts, add_special_tokens=False, truncation=False)
    return [len(ids) for ids in encodings["input_ids"]]


def take_to_budget(
    rows,
    tokenizer,
    token_budget: int,
    keep_row=None,
    batch_size: int = 1024,
) -> tuple[list[str], int]:
    """Accumulate texts from ``rows`` until ``token_budget`` is reached.

    ``keep_row`` optionally filters rows before counting. The final document
    that crosses the budget is included, so the result slightly overshoots.
    """
    texts: list[str] = []
    total = 0
    pending: list[str] = []

    def drain():
        nonlocal total
        for text, n in zip(pending, count_tokens(pending, tokenizer)):
            if total >= token_budget:
                break
            texts.append(text)
            total += n
        pending.clear()

    for row in rows:
        if keep_row is not None and not keep_row(row):
            continue
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            continue
        pending.append(text)
        if len(pending) == batch_size:
            drain()
            if total >= token_budget:
                break
    if pending and total < token_budget:
        drain()
    return texts, total


def random_block_rows(ds: Dataset, seed: int, block_size: int = 1024):
    """Yield rows from contiguous blocks at shuffled offsets.

    Keeps arrow reads sequential within each block so sampling a small
    fraction of a huge dataset does not touch pages across the whole file.
    """
    rng = random.Random(seed)
    starts = list(range(0, len(ds), block_size))
    rng.shuffle(starts)
    for start in starts:
        block = ds.select(range(start, min(start + block_size, len(ds))))
        for row in block:
            yield row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/TF32_metasmoothness/mix_85_15")
    parser.add_argument(
        "--tokenizer", default="EleutherAI/deep-ignorance-e2e-strong-filter"
    )
    parser.add_argument("--total_tokens", type=int, default=130_000_000)
    parser.add_argument("--pretrain_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    forget_budget = int(args.total_tokens * (1 - args.pretrain_fraction))
    pretrain_budget = int(args.total_tokens * args.pretrain_fraction)

    forget_ds = load_dataset(FORGET_DATASET, split="train")
    forget_texts, forget_tokens = take_to_budget(
        forget_ds.shuffle(seed=args.seed), tokenizer, forget_budget
    )
    print(
        f"forget: {len(forget_texts)}/{len(forget_ds)} docs, "
        f"{forget_tokens:,} tokens (budget {forget_budget:,})",
        flush=True,
    )
    if forget_tokens < forget_budget:
        print(
            "Warning: forget corpus is smaller than its budget; "
            "the mix will undershoot total_tokens.",
            flush=True,
        )
        pretrain_budget = int(
            forget_tokens * args.pretrain_fraction / (1 - args.pretrain_fraction)
        )

    pretrain_ds = load_dataset(PRETRAIN_DATASET, split="train")
    pretrain_texts, pretrain_tokens = take_to_budget(
        random_block_rows(pretrain_ds, args.seed),
        tokenizer,
        pretrain_budget,
        # The annealing mix stores filter verdicts per row; False means the
        # row survived filtering and was trained on.
        keep_row=lambda row: str(row["combined_filter"]) == "False",
    )
    print(
        f"pretrain: {len(pretrain_texts)} docs, "
        f"{pretrain_tokens:,} tokens (budget {pretrain_budget:,})",
        flush=True,
    )

    mixed = concatenate_datasets(
        [
            Dataset.from_dict(
                {"text": forget_texts, "source": ["forget"] * len(forget_texts)}
            ),
            Dataset.from_dict(
                {"text": pretrain_texts, "source": ["pretrain"] * len(pretrain_texts)}
            ),
        ]
    ).shuffle(seed=args.seed)

    total = forget_tokens + pretrain_tokens
    print(
        f"total: {len(mixed)} docs, {total:,} tokens "
        f"({pretrain_tokens / total:.1%} pretrain), "
        f"~{total // (128 * 1024)} steps at batch 128 x 1024",
        flush=True,
    )
    mixed.save_to_disk(args.output_dir)
    print(f"saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
