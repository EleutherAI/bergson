"""Baseline: BM25 lexical overlap.

Scores each training doc against each query by Okapi BM25 -- pure surface-form
term overlap, no model and no learned embedding. A lexical proxy for influence:
docs sharing query terms are predicted influential, so the loss-diff-convention
score is ``-bm25``. Often a stronger influence proxy than deep-semantic
embedders for small LMs, and essentially free.

Run with (builds the default bank if --bank is omitted):
    python -m examples.bank_baselines.bm25_baseline --bank runs/retrain_bank_path
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from . import common


def bm25_scores(
    train_texts: list[str], query_texts: list[str], k1: float = 1.5, b: float = 0.75
) -> np.ndarray:
    """Okapi BM25 score of every train doc against every query -> [docs, queries]."""
    vectorizer = CountVectorizer()
    tf = vectorizer.fit_transform(train_texts).astype(np.float64).tocsr()  # [D, V]
    n_docs, _ = tf.shape

    doc_len = np.asarray(tf.sum(axis=1)).ravel()
    avgdl = doc_len.mean()
    df = np.asarray((tf > 0).sum(axis=0)).ravel()
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    # Per-nonzero BM25 term weight: idf * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl)).
    rows, cols = tf.nonzero()
    denom_bias = k1 * (1 - b + b * doc_len[rows] / avgdl)
    weighted = tf.copy()
    weighted.data = idf[cols] * tf.data * (k1 + 1.0) / (tf.data + denom_bias)

    # Query terms as a binary term set (Okapi BM25 ignores query term frequency).
    q_bin = (vectorizer.transform(query_texts) > 0).astype(np.float64)  # [Q, V]
    return np.asarray((weighted @ q_bin.T).todense())  # [D, Q]


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
    args = ap.parse_args()

    bank = common.ensure_bank(args.bank)
    spec = common.read_bank_spec(bank)
    query_dataset = args.query_dataset or spec.dataset
    out_dir = Path(args.out)

    train_texts, query_texts = common.load_texts(spec, query_dataset, args.query_split)
    print(f"BM25 over {len(train_texts)} train docs, {len(query_texts)} queries ...")
    scores = -bm25_scores(train_texts, query_texts)  # loss-diff convention
    score_path = common.save_scores(scores, out_dir, "bm25_scores")

    rhos = common.evaluate_lds(
        bank,
        score_path,
        out_dir / "bm25_validate",
        spec,
        query_dataset,
        args.query_split,
    )
    common.report("BM25 lexical overlap", rhos)


if __name__ == "__main__":
    main()
