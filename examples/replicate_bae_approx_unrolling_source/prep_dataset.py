"""Reproduce the WikiText-2 / GPT-2 512-token-chunk dataset for the replication.

Production rule is the standard HF ``run_clm`` recipe (also what kronfluence's
``examples/wikitext`` uses): tokenize raw wikitext-2-raw-v1 with the GPT-2
tokenizer, then ``group_texts`` -- within each 1000-row map batch, concatenate
tokens, drop the remainder, and slice into fixed 512-token blocks (one block ==
one document). This is a public preprocessing recipe, not a borrowed ground
truth, and it reproduces the exact 4656 train / 481 validation chunking used by
the replication configs.

The pushed copy is ``EleutherAI/bergson-wikitext-2-4656-chunks`` (columns
``input_ids`` and ``length``), which the replication YAMLs load with
``chunk_length: 0``.

    # Dry run (prints split sizes); add --push <repo> to push to the hub.
    python examples/replicate_bae_approx_unrolling_source/prep_dataset.py
"""

import argparse

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

BLOCK = 512
DEFAULT_REPO = "EleutherAI/bergson-wikitext-2-4656-chunks"


def group_texts(examples: dict) -> dict:
    concat = sum(examples["input_ids"], [])
    total = (len(concat) // BLOCK) * BLOCK
    return {"input_ids": [concat[i : i + BLOCK] for i in range(0, total, BLOCK)]}


def build() -> DatasetDict:
    tok = AutoTokenizer.from_pretrained("gpt2")
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    out = {}
    for split in ("train", "validation"):
        toks = raw[split].map(
            lambda b: tok(b["text"]),
            batched=True,
            remove_columns=raw[split].column_names,
            desc=f"tokenize {split}",
        )
        blocks = toks.map(
            group_texts,
            batched=True,
            remove_columns=[c for c in toks.column_names if c != "input_ids"],
            desc=f"chunk {split}",
        )
        out[split] = blocks.map(lambda x: {"length": len(x["input_ids"])})
    return DatasetDict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--push",
        nargs="?",
        const=DEFAULT_REPO,
        default="",
        help=f"push to this hub repo id (default {DEFAULT_REPO} when bare)",
    )
    args = ap.parse_args()

    ds = build()
    for split, d in ds.items():
        print(f"{split}: {len(d)} chunks, cols={d.column_names}")

    if args.push:
        print(f"pushing to {args.push} ...")
        ds.push_to_hub(args.push)
        print("pushed.")


if __name__ == "__main__":
    main()
