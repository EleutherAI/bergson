"""Rank uncertainty scores for PAC labeling by how much of the corpus they let
the cheap labeler keep.

Usage:
    python examples/pac_labeling/hillclimb.py --expert <magic>/scores \
        --cheap <run>/ekfac_scores/scores --out <dir> \
        --feature bm25=<run>/bm25_scores --feature semantic=<dir>/semantic_scores \
        --doc-feature doc_loss=<dir>/doc_loss.npy ...

Every ``--feature`` is a loss-signed score directory with one column per
query; every ``--doc-feature`` a per-document ``.npy``. For each query the
cheap score is calibrated as in ``pac_attribution.py`` and the same losses
are formed. Candidate uncertainty scores are the rank of each feature (its
own proponents first), its rank disagreement with the cheap score, and
gradient-boosted regressions of the loss on all features, fit on the
calibration samples of every query (leave-one-query-out for the query being
scored plus its own calibration sample).

Two numbers rank a candidate. The oracle save at ``eps`` sorts documents by
uncertainty and keeps the longest prefix whose mean loss is at most ``eps``:
what PAC labeling would certify with unlimited expert samples. The PAC save
is what ``pac_label`` certifies with ``m`` samples, averaged over trials.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from bergson.data import load_scores_loss_signed
from bergson.pac import pac_label, sample_indices
from examples.pac_labeling.pac_attribution import (
    TOP_FRACTION,
    cheap_cutoff,
    rank01,
    task_loss,
)

EPS = {"proponent": [0.005, 0.01], "recall": [0.002, 0.005], "score": [0.05]}


def oracle_save(u: np.ndarray, loss: np.ndarray, eps: float) -> float:
    order = np.argsort(u, kind="stable")
    cum = np.cumsum(loss[order]) / len(loss)
    ok = np.flatnonzero(cum <= eps)
    return (ok[-1] + 1) / len(loss) if len(ok) else 0.0


def load_feature(path: str) -> np.ndarray:
    scores, _ = load_scores_loss_signed(path)
    return scores.numpy().astype(np.float64)


def orient(feature: np.ndarray, expert: np.ndarray, cal: list[np.ndarray]) -> float:
    """Sign making the feature loss-signed, from the calibration samples."""
    rho = np.mean(
        [spearmanr(feature[c, q], expert[c, q]).correlation for q, c in enumerate(cal)]
    )
    return 1.0 if rho >= 0 else -1.0


def design(
    q: int,
    yhat: np.ndarray,
    tau: float,
    features: dict[str, np.ndarray],
    doc_features: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    cut = cheap_cutoff(yhat)
    cols += [yhat / abs(tau), rank01(yhat), rank01(-np.abs(yhat - cut))]
    names += ["cheap", "cheap_rank", "cheap_boundary"]
    for name, f in features.items():
        z = (f[:, q] - f[:, q].mean()) / (f[:, q].std() + 1e-12)
        r = rank01(f[:, q])
        cols += [z, r, np.abs(r - rank01(yhat))]
        names += [name, f"{name}_rank", f"{name}_dis"]
    for name, f in doc_features.items():
        cols.append(rank01(f))
        names.append(name)
    return np.stack(cols, 1), names


def learned(
    q: int,
    designs: list[tuple[np.ndarray, list[str]]],
    losses: list[np.ndarray],
    cal: list[np.ndarray],
    columns: list[int],
    seed: int,
) -> np.ndarray:
    x = np.concatenate([designs[j][0][cal[j]][:, columns] for j in range(len(cal))])
    y = np.concatenate([losses[j][cal[j]] for j in range(len(cal))])
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=seed
    )
    model.fit(x, y)
    pred = model.predict(designs[q][0][:, columns])
    # Ties inside a leaf are broken by the cheap score's own rank.
    return pred + 1e-9 * designs[q][0][:, 1]


def pac_trials(args) -> list[dict]:
    q, task, name, u, loss, n_expert_cal, cfg = args
    n = len(loss)
    rows = []
    for trial in range(cfg["trials"]):
        rng = np.random.default_rng([cfg["seed"], q, trial, 7])
        for m in cfg["m"]:
            sample = sample_indices(n, m, rng)
            for method in cfg["methods"]:
                for eps in EPS[task]:
                    res = pac_label(u, sample, loss[sample], eps, cfg["alpha"], method)
                    expert = res.expert.copy()
                    n_exp = expert.sum() + n_expert_cal
                    rows.append(
                        {
                            "query": q,
                            "task": task,
                            "uncertainty": name,
                            "eps": eps,
                            "m": m,
                            "method": method,
                            "trial": trial,
                            "save": 1 - n_exp / n,
                            "error": float((loss * ~expert).mean()),
                        }
                    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expert", required=True)
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--feature", action="append", default=[])
    ap.add_argument("--doc-feature", action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=["proponent", "recall", "score"])
    ap.add_argument("--m-cal", type=int, default=1000)
    ap.add_argument("--m", type=int, nargs="+", default=[2000])
    ap.add_argument("--methods", nargs="+", default=["betting"])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--pac-top", type=int, default=8, help="candidates per task")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    expert = load_feature(args.expert)
    cheap = load_feature(args.cheap)
    n, nq = expert.shape
    cal = [
        np.random.default_rng([args.seed, q, 0]).choice(n, args.m_cal, replace=False)
        for q in range(nq)
    ]
    features = {}
    for spec in args.feature:
        name, path = spec.split("=", 1)
        f = load_feature(path)
        features[name] = f * orient(f, expert, cal)
    doc_features = {}
    for spec in args.doc_feature:
        name, path = spec.split("=", 1)
        doc_features[name] = np.load(path).astype(np.float64)

    calib = []
    for q in range(nq):
        a, b = np.polyfit(cheap[cal[q], q], expert[cal[q], q], 1)
        calib.append(
            (a * cheap[:, q] + b, float(np.quantile(expert[cal[q], q], TOP_FRACTION)))
        )
    designs = [
        design(q, yhat, tau, features, doc_features)
        for q, (yhat, tau) in enumerate(calib)
    ]
    names = designs[0][1]
    groups = {
        "cheap": [i for i, c in enumerate(names) if c.startswith("cheap")],
        "cheap+retrieval": [
            i
            for i, c in enumerate(names)
            if c.startswith(("cheap", "bm25", "semantic", "activation", "doc_loss"))
        ],
        "all": list(range(len(names))),
    }
    gradient_cols = [
        i
        for i, c in enumerate(names)
        if c.startswith(("grad", "train", "ekfac", "update", "epoch"))
    ]
    if gradient_cols:
        groups["cheap+gradients"] = groups["cheap"] + gradient_cols

    oracle_rows, candidates = [], {}
    for task in args.tasks:
        losses = [
            task_loss(task, expert[:, q], calib[q][0], calib[q][1]) for q in range(nq)
        ]
        for q in range(nq):
            yhat, tau = calib[q]
            cut = cheap_cutoff(yhat)
            us = {"random": np.random.default_rng([args.seed, q, 1]).uniform(size=n)}
            us["cheap_rank"] = rank01(-yhat)
            us["cheap_boundary"] = rank01(-np.abs(yhat - cut))
            for name, f in features.items():
                us[f"{name}_rank"] = rank01(-f[:, q])
                us[f"{name}_dis"] = np.abs(rank01(f[:, q]) - rank01(yhat))
            for name, f in doc_features.items():
                us[name] = rank01(f)
                us[f"{name}_neg"] = rank01(-f)
            for gname, cols in groups.items():
                us[f"learned[{gname}]"] = learned(
                    q, designs, losses, cal, cols, args.seed
                )
            for name, u in us.items():
                candidates[(task, q, name)] = u
                for eps in EPS[task]:
                    oracle_rows.append(
                        {
                            "task": task,
                            "query": q,
                            "uncertainty": name,
                            "eps": eps,
                            "oracle_save": oracle_save(u, losses[q], eps),
                            "cheap_loss": float(losses[q].mean()),
                        }
                    )
        print(f"{task}: oracle saves done")
    oracle = pd.DataFrame(oracle_rows)
    oracle.to_csv(args.out / "oracle.csv", index=False)
    board = (
        oracle.groupby(["task", "uncertainty", "eps"]).oracle_save.mean().unstack("eps")
    )
    pd.set_option("display.width", 200)
    print(board.round(3).to_string())

    cfg = {
        "trials": args.trials,
        "seed": args.seed,
        "m": args.m,
        "methods": args.methods,
        "alpha": args.alpha,
    }
    jobs = []
    for task in args.tasks:
        losses = [
            task_loss(task, expert[:, q], calib[q][0], calib[q][1]) for q in range(nq)
        ]
        mean_save = (
            oracle[oracle.task == task].groupby("uncertainty").oracle_save.mean()
        )
        top = list(mean_save.sort_values(ascending=False).index[: args.pac_top])
        for name in {"random", "cheap_rank", *top}:
            for q in range(nq):
                jobs.append(
                    (
                        q,
                        task,
                        name,
                        candidates[(task, q, name)],
                        losses[q],
                        args.m_cal,
                        cfg,
                    )
                )
    with Pool(args.workers) as pool:
        rows = [r for out in pool.imap_unordered(pac_trials, jobs) for r in out]
    trials = pd.DataFrame(rows)
    trials.to_csv(args.out / "pac_trials.csv", index=False)
    trials["violation"] = trials.error > trials.eps
    keys = ["task", "uncertainty", "eps", "m", "method"]
    summary = (
        trials.groupby(keys)
        .agg(
            pac_save=("save", "mean"),
            error_q=("error", lambda e: e.quantile(1 - args.alpha)),
            violations=("violation", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(
        oracle.groupby(["task", "uncertainty", "eps"]).oracle_save.mean().reset_index()
    )
    order = ["task", "eps", "m", "method", "pac_save"]
    summary = summary.sort_values(order, ascending=[True, True, True, True, False])
    summary.to_csv(args.out / "leaderboard.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    (args.out / "features.json").write_text(
        json.dumps(
            {
                "features": list(features),
                "doc_features": list(doc_features),
                "design": names,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
