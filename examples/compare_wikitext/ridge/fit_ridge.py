"""Ridge over EK-FAC's module x band contributions, fitted to MAGIC scores.

    python examples/compare_wikitext/ridge/fit_ridge.py \
        --contributions runs/compare_wikitext/ridge/contributions.npy \
        --magic runs/compare_wikitext/random/scores \
        --out runs/compare_wikitext/ekfac_ridge

The features of each (training doc, query) pair are the 48 x 7 contributions of
``band_contributions.py``, divided by the query's EK-FAC score standard
deviation; the target is the query's MAGIC score, standardised. Queries are
split into ``--folds`` folds by index, and each query is scored by weights fitted
on the other folds' queries only. The ridge penalty is ``--strength`` times the
mean diagonal of the Gram matrix. Writes a loss-signed score store
(``<out>/scores``) and the per-fold weights (``<out>/weights.npy``).
"""

import argparse
from pathlib import Path

import numpy as np

from bergson.data import load_scores_loss_signed
from bergson.score.score_writer import save_sequence_scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contributions", required=True, type=Path)
    ap.add_argument("--magic", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--strength", type=float, default=1e-3)
    args = ap.parse_args()

    c = np.load(args.contributions).astype(
        np.float64
    )  # [docs, modules, bands, queries]
    n_docs, n_mod, n_band, n_q = c.shape
    ekfac = c.sum((1, 2))
    x = (c / ekfac.std(0)).reshape(n_docs, n_mod * n_band, n_q)
    # MAGIC stores are loss-signed; the contributions are proponent-positive.
    magic = -load_scores_loss_signed(str(args.magic))[0].numpy().astype(np.float64)
    assert magic.shape == (n_docs, n_q), magic.shape
    y = (magic - magic.mean(0)) / magic.std(0)

    folds = np.arange(n_q) % args.folds
    pred = np.zeros((n_docs, n_q))
    weights = np.zeros((args.folds, n_mod * n_band))
    for k in range(args.folds):
        train_q = np.flatnonzero(folds != k)
        xt = np.concatenate([x[:, :, q] for q in train_q])
        yt = np.concatenate([y[:, q] for q in train_q])
        gram = xt.T @ xt
        penalty = args.strength * np.trace(gram) / len(gram)
        weights[k] = np.linalg.solve(gram + penalty * np.eye(len(gram)), xt.T @ yt)
        for q in np.flatnonzero(folds == k):
            pred[:, q] = x[:, :, q] @ weights[k]

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "weights.npy", weights.reshape(args.folds, n_mod, n_band))
    save_sequence_scores(args.out / "scores", -pred)
    print(f"Saved ridge scores {pred.shape} -> {args.out / 'scores'}")


if __name__ == "__main__":
    main()
