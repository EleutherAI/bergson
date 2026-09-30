"""Chunk-level scores from a per-token score store.

    python examples/compare_wikitext/chunks/chunk_scores.py \
        <token scores> <out> {fixed64,natural}

Every training row gets the sum of its chunk's per-token scores, so the per-token
proponent filter masks whole chunks. ``fixed64`` cuts each training chunk into
64-token windows; ``natural`` cuts after sentence ends (" . ") and line breaks and
merges the pieces forward until each has at least 32 tokens. Row ``t`` holds the
loss on token ``t + 1``, so a piece over token positions ``[a, b)`` owns rows
``[max(a - 1, 0), b - 1)``.
"""

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from bergson.data import load_scores
from bergson.score.score_writer import save_token_scores

DATASET = "EleutherAI/bergson-wikitext-512-chunks"
MIN_TOKENS = 32
BOUNDARY = re.compile(r" \. |\n")


def fixed_bounds(n_tok: int) -> list[int]:
    return sorted(set(range(0, n_tok, 64)) | {n_tok})


def natural_bounds(text: str, n_tok: int, tokenizer) -> list[int]:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    starts = [s for s, _ in enc["offset_mapping"][:n_tok]]
    # A boundary sits at the first token starting at or after each break.
    bounds, j = {0, n_tok}, 0
    for end in [m.end() for m in BOUNDARY.finditer(text)] + [len(text) + 1]:
        while j < n_tok and starts[j] < end:
            j += 1
        bounds.add(j)
    bounds = sorted(bounds)
    merged, acc = [0], 0
    for b, size in zip(bounds[1:], np.diff(bounds)):
        acc += size
        if acc >= MIN_TOKENS:
            merged.append(b)
            acc = 0
    if merged[-1] != n_tok:
        if len(merged) > 1:
            merged[-1] = n_tok  # the short tail joins the last piece
        else:
            merged.append(n_tok)
    return merged


def chunk_of_row(n_rows: np.ndarray, bounds: list[list[int]]) -> tuple[np.ndarray, int]:
    out, cid = [], 0
    for n, b in zip(n_rows, bounds):
        rows = np.full(n, -1, dtype=np.int64)
        for lo, hi in zip(b[:-1], b[1:]):
            lo, hi = max(lo - 1, 0), min(hi - 1, n)
            if hi > lo:
                rows[lo:hi] = cid
                cid += 1
        assert (rows >= 0).all()
        out.append(rows)
    return np.concatenate(out), cid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scores", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("chunking", choices=["fixed64", "natural"])
    args = p.parse_args()

    store = load_scores(args.scores)
    offsets = store.offsets
    n_rows = np.diff(offsets)
    raw = np.asarray(store[:], dtype=np.float64)

    if args.chunking == "fixed64":
        bounds = [fixed_bounds(int(n) + 1) for n in n_rows]
    else:
        texts = load_dataset(DATASET, split="train")["text"]
        assert len(texts) == len(n_rows)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        bounds = [
            natural_bounds(t, int(n) + 1, tokenizer) for t, n in zip(texts, n_rows)
        ]
    rows, n_chunks = chunk_of_row(n_rows, bounds)

    sums = np.zeros((n_chunks, raw.shape[1]))
    np.add.at(sums, rows, raw)
    save_token_scores(args.out, sums[rows].astype(np.float32), offsets)
    # The config carries the store's sign convention.
    for f in ("config.yaml", "processor_config.yaml"):
        if (args.scores / f).exists():
            shutil.copy(args.scores / f, args.out / f)
    print(f"{args.chunking}: {n_chunks} chunks over {len(n_rows)} training chunks")


if __name__ == "__main__":
    main()
