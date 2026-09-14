"""Baseline: hybrid BM25 + Qwen3-Embedding-8B semantic search.

Fuses the score stores written by ``bm25_baseline.py`` and ``qwen3_baseline.py``.
Per query, each similarity is min-max scaled over the training docs and the two
are mixed as ``alpha * semantic + (1 - alpha) * bm25``; the loss-diff-convention
score is the negated mixture.

Run with:
    python -m examples.gradient_free_baselines.hybrid_baseline \
        --bank runs/retrain_bank_path \
        --bm25 runs/.../bm25_scores/scores --qwen3 runs/.../qwen3_scores/scores \
        --alpha 0.5
"""

import argparse
from pathlib import Path

import numpy as np

from bergson.data import load_scores

from . import common


def minmax_per_query(sim: np.ndarray) -> np.ndarray:
    """Scale each column of a ``[docs, queries]`` matrix to [0, 1]."""
    lo, hi = sim.min(axis=0, keepdims=True), sim.max(axis=0, keepdims=True)
    return (sim - lo) / np.where(hi > lo, hi - lo, 1.0)


def hybrid_scores(bm25: Path, qwen3: Path, alpha: float) -> np.ndarray:
    """Mixed similarity of every train doc against every query -> [docs, queries]."""
    bm25_sim = -np.asarray(load_scores(bm25)[:], dtype=np.float64)
    sem_sim = -np.asarray(load_scores(qwen3)[:], dtype=np.float64)
    assert bm25_sim.shape == sem_sim.shape, (bm25_sim.shape, sem_sim.shape)
    return alpha * minmax_per_query(sem_sim) + (1 - alpha) * minmax_per_query(bm25_sim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=None, help="re-train bank dir; built if omitted")
    ap.add_argument("--bm25", required=True, help="bm25_baseline.py score dir")
    ap.add_argument("--qwen3", required=True, help="qwen3_baseline.py score dir")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on semantic")
    ap.add_argument("--query_split", default=common.DEFAULT_QUERY_SPLIT)
    ap.add_argument(
        "--query_dataset",
        default=None,
        help="query dataset; default = bank train dataset",
    )
    ap.add_argument(
        "--out", default=str(common.REPO / "runs" / "gradient_free_baselines")
    )
    args = ap.parse_args()

    bank = common.ensure_bank(args.bank)
    spec = common.read_bank_spec(bank)
    query_dataset = args.query_dataset or spec.dataset
    out_dir = Path(args.out)
    name = f"hybrid_a{round(args.alpha * 100):02d}"

    scores = -hybrid_scores(Path(args.bm25), Path(args.qwen3), args.alpha)
    score_path = common.save_scores(scores, out_dir, f"{name}_scores")
    rhos = common.evaluate_lds(
        bank,
        score_path,
        out_dir / f"{name}_validate",
        spec,
        query_dataset,
        args.query_split,
    )
    common.report(f"Hybrid BM25 + Qwen3 (alpha={args.alpha})", rhos)


if __name__ == "__main__":
    main()
