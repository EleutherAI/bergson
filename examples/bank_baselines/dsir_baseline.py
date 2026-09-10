"""Baseline: DSIR importance weights (Xie et al., 2023).

Data Selection with Importance Resampling scores a raw document by how much
more likely it is under a hashed-n-gram model of the target set than under the
same model of the raw corpus. Each document is a bag of unigrams and bigrams
hashed into ``--buckets`` bins; the target and raw distributions are the
normalized bucket counts, and a document's importance weight is the dot product
of its bucket counts with ``log(target) - log(raw)``. DSIR then resamples
proportionally to the weights; here the weights themselves are the scores, so
the top-k by weight is the selected set.

The target set is one query document at a time and the raw corpus is the
bank's training set. A high weight means the doc looks like the query and is
predicted influential, so the loss-diff-convention score is ``-weight``.

Run with (builds the default bank if --bank is omitted):
    python -m examples.bank_baselines.dsir_baseline --bank runs/retrain_bank_path
"""

import argparse
import re
import zlib
from pathlib import Path

import numpy as np

from . import common

TOKEN = re.compile(r"\w+|[^\w\s]")


def hashed_ngram_counts(texts: list[str], buckets: int) -> np.ndarray:
    """``[len(texts), buckets]`` counts of hashed unigrams and bigrams."""
    counts = np.zeros((len(texts), buckets), dtype=np.float32)
    for row, text in enumerate(texts):
        toks = TOKEN.findall(text.lower())
        grams = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
        for g in grams:
            counts[row, zlib.crc32(g.encode()) % buckets] += 1
    return counts


def dsir_weights(
    train_counts: np.ndarray, query_counts: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Importance weight of every train doc under every query -> [docs, queries]."""
    raw = train_counts.sum(0)
    raw = raw / raw.sum()
    target = query_counts / query_counts.sum(1, keepdims=True)
    log_diff = np.log(target + eps) - np.log(raw + eps)  # [queries, buckets]
    return train_counts @ log_diff.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=None, help="re-train bank dir; built if omitted")
    ap.add_argument("--query_split", default=common.DEFAULT_QUERY_SPLIT)
    ap.add_argument(
        "--query_dataset",
        default=None,
        help="query dataset; default = bank train dataset",
    )
    ap.add_argument("--out", default=str(common.REPO / "runs" / "bank_baselines"))
    ap.add_argument("--buckets", type=int, default=10_000)
    args = ap.parse_args()

    bank = common.ensure_bank(args.bank)
    spec = common.read_bank_spec(bank)
    query_dataset = args.query_dataset or spec.dataset
    out_dir = Path(args.out)

    train_texts, query_texts = common.load_texts(spec, query_dataset, args.query_split)
    print(f"DSIR over {len(train_texts)} train docs, {len(query_texts)} queries ...")
    train_counts = hashed_ngram_counts(train_texts, args.buckets)
    query_counts = hashed_ngram_counts(query_texts, args.buckets)
    scores = -dsir_weights(train_counts, query_counts)  # loss-diff convention
    score_path = common.save_scores(scores, out_dir, "dsir_scores")

    rhos = common.evaluate_lds(
        bank,
        score_path,
        out_dir / "dsir_validate",
        spec,
        query_dataset,
        args.query_split,
    )
    common.report("DSIR importance weight", rhos)


if __name__ == "__main__":
    main()
