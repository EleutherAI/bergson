"""PAC labeling of attribution scores with MAGIC as the expert and a cheap
scorer (EK-FAC) as the AI.

Usage:
    python examples/pac_labeling/pac_attribution.py \
        --expert <magic run>/scores --cheap <run>/ekfac_scores/scores \
        --bank <run>/bank_from_filter --out <dir>

Each query is one labeling problem over the training documents. A calibration
sample of ``m_cal`` expert scores fits the cheap scores onto the expert scale
by least squares and fixes the loss; a second sample of ``m`` expert scores
sets the PAC threshold. Two losses are run:

* ``proponent``: 0-1 loss on membership in the strongest 1% of scores. The
  expert cutoff is the 1% quantile of the calibration sample's expert scores;
  the cheap label is membership in the cheap scorer's own strongest 1%.
* ``recall``: 0-1 loss on missing a member of that set, so ``eps`` is the
  missed fraction of proponents times 1%.
* ``score``: squared error of the calibrated cheap score in units of that
  cutoff, clipped to 1.

Uncertainty scores are ranks of the calibrated cheap score (``magnitude``,
``proponent``, ``boundary``), a ``learned`` score that bins the cheap score
by rank and takes each bin's mean loss on the calibration sample, ordering
within a bin by the task's rank score, and ``random``. Every trial records
the fraction of documents that keep the cheap score, the realized loss against
the expert scores, and for the hybrid vector (expert where labeled, cheap
elsewhere) its overlap with the expert's top 1% and its LDS against the
retrain bank.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from bergson.data import load_scores_loss_signed
from bergson.pac import pac_label, sample_indices

BIN_EDGES = np.array(
    [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]
    + [0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 1.0]
)
TOP_FRACTION = 0.01


def load_bank(bank: Path) -> tuple[list[np.ndarray], np.ndarray]:
    subsets = [np.asarray(s) for s in json.loads((bank / "subsets.json").read_text())]
    val = pd.read_csv(bank / "validation_merged.csv")
    diffs = val.pivot(index="subset", columns="query", values="diff").to_numpy()
    return subsets[: diffs.shape[0]], diffs


def lds(scores: np.ndarray, subsets: list[np.ndarray], diffs_q: np.ndarray) -> float:
    """Spearman between each subset's summed loss-signed scores and the loss
    change measured when the subset is removed."""
    sums = np.array([scores[s].sum() for s in subsets])
    return float(spearmanr(sums, diffs_q).correlation)


def top_set(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(scores)[:k]


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    return len(set(a) & set(b)) / len(a)


def rank01(x: np.ndarray) -> np.ndarray:
    return (rankdata(x) - 1) / (len(x) - 1)


def learned_uncertainty(
    cheap: np.ndarray,
    cal: np.ndarray,
    cal_loss: np.ndarray,
    within: np.ndarray,
    prior: float = 2.0,
) -> np.ndarray:
    """Mean calibration loss per rank bin of the cheap score, shrunk toward
    the overall mean by ``prior`` pseudo-counts, with ``within`` breaking ties
    inside a bin."""
    r = rank01(cheap)
    bins = np.searchsorted(BIN_EDGES, r, side="right") - 1
    bins = np.clip(bins, 0, len(BIN_EDGES) - 2)
    mean = cal_loss.mean()
    u = np.empty(len(BIN_EDGES) - 1)
    for b in range(len(u)):
        sel = bins[cal] == b
        u[b] = (cal_loss[sel].sum() + prior * mean) / (sel.sum() + prior)
    return u[bins] + 1e-6 * within


def uncertainties(
    task: str, yhat: np.ndarray, tau: float, cal: np.ndarray, cal_loss: np.ndarray, rng
) -> dict[str, np.ndarray]:
    if task in ("proponent", "recall"):
        own = ("boundary", rank01(-np.abs(yhat - cheap_cutoff(yhat))))
    else:
        own = ("magnitude", rank01(np.abs(yhat)))
    return {
        "proponent": rank01(-yhat),
        own[0]: own[1],
        "learned": learned_uncertainty(yhat, cal, cal_loss, own[1]),
        "random": rng.uniform(size=len(yhat)),
    }


def cheap_cutoff(yhat: np.ndarray) -> float:
    return float(np.quantile(yhat, TOP_FRACTION))


def task_loss(task: str, y: np.ndarray, yhat: np.ndarray, tau: float) -> np.ndarray:
    if task == "proponent":
        return ((y <= tau) != (yhat <= cheap_cutoff(yhat))).astype(np.float64)
    if task == "recall":
        return ((y <= tau) & (yhat > cheap_cutoff(yhat))).astype(np.float64)
    return np.minimum(1.0, ((y - yhat) / tau) ** 2)


def run_query(args) -> list[dict]:
    q, y, x, subsets, diffs_q, cfg = args
    n = len(y)
    k = int(TOP_FRACTION * n)
    expert_top = top_set(y, k)
    rows = []
    for trial in range(cfg["trials"]):
        rng = np.random.default_rng([cfg["seed"], q, trial])
        cal = rng.choice(n, cfg["m_cal"], replace=False)
        a, b = np.polyfit(x[cal], y[cal], 1)
        yhat = a * x + b
        sample = sample_indices(n, cfg["m"], rng)
        tau = float(np.quantile(y[cal], TOP_FRACTION))
        for task in cfg["tasks"]:
            loss = task_loss(task, y, yhat, tau)
            us = uncertainties(task, yhat, tau, cal, loss[cal], rng)
            for name, u in us.items():
                for eps in cfg["eps"][task]:
                    res = pac_label(u, sample, loss[sample], eps, cfg["alpha"])
                    expert = res.expert.copy()
                    expert[cal] = True
                    hybrid = np.where(expert, y, yhat)
                    rows.append(
                        {
                            "query": q,
                            "trial": trial,
                            "task": task,
                            "uncertainty": name,
                            "eps": eps,
                            "u_hat": res.u_hat,
                            "n_expert": int(expert.sum()),
                            "budget_save": 1 - expert.mean(),
                            "error": float((loss * ~expert).mean()),
                            "cheap_error": float(loss.mean()),
                            "top1_overlap": overlap(top_set(hybrid, k), expert_top),
                            "lds": lds(hybrid, subsets, diffs_q),
                        }
                    )
        if trial == 0:
            rows.append(
                {
                    "query": q,
                    "trial": -1,
                    "task": "baseline",
                    "uncertainty": "cheap_only",
                    "eps": np.nan,
                    "u_hat": np.nan,
                    "n_expert": 0,
                    "budget_save": 1.0,
                    "error": np.nan,
                    "cheap_error": np.nan,
                    "top1_overlap": overlap(top_set(yhat, k), expert_top),
                    "lds": lds(yhat, subsets, diffs_q),
                }
            )
            rows.append(
                {
                    "query": q,
                    "trial": -1,
                    "task": "baseline",
                    "uncertainty": "expert_only",
                    "eps": np.nan,
                    "u_hat": np.nan,
                    "n_expert": n,
                    "budget_save": 0.0,
                    "error": 0.0,
                    "cheap_error": np.nan,
                    "top1_overlap": 1.0,
                    "lds": lds(y, subsets, diffs_q),
                }
            )
    return rows


def export_expert_set(
    y: np.ndarray, x: np.ndarray, q: int, cfg: dict, out: Path
) -> None:
    """Write the documents trial 0 sends to the expert under the export
    setting, the set a subset MAGIC run scores."""
    n = len(y)
    rng = np.random.default_rng([cfg["seed"], q, 0])
    cal = rng.choice(n, cfg["m_cal"], replace=False)
    a, b = np.polyfit(x[cal], y[cal], 1)
    yhat = a * x + b
    sample = sample_indices(n, cfg["m"], rng)
    tau = float(np.quantile(y[cal], TOP_FRACTION))
    task, name, eps = cfg["export"]
    loss = task_loss(task, y, yhat, tau)
    u = uncertainties(task, yhat, tau, cal, loss[cal], rng)[name]
    res = pac_label(u, sample, loss[sample], eps, cfg["alpha"])
    expert = res.expert.copy()
    expert[cal] = True
    np.save(out / f"expert_set_q{q}.npy", np.flatnonzero(expert))
    meta = {
        "query": q,
        "task": task,
        "uncertainty": name,
        "eps": eps,
        "u_hat": res.u_hat,
        "n_expert": int(expert.sum()),
        "n_threshold": int((u >= res.u_hat).sum()),
        "calibration": {"slope": float(a), "intercept": float(b), "tau": tau},
    }
    (out / f"expert_set_q{q}.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expert", type=Path, required=True)
    ap.add_argument("--cheap", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--queries", type=int, nargs="*")
    ap.add_argument("--m", type=int, default=2000)
    ap.add_argument("--m-cal", type=int, default=1000)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.002, 0.005, 0.01, 0.02])
    ap.add_argument("--score-eps", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    ap.add_argument(
        "--recall-eps", type=float, nargs="+", default=[0.001, 0.002, 0.005]
    )
    ap.add_argument("--tasks", nargs="+", default=["proponent", "recall", "score"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--export",
        nargs=3,
        metavar=("TASK", "UNCERTAINTY", "EPS"),
        default=["proponent", "learned", "0.005"],
        help="setting whose trial-0 expert set is written per query",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    y_all, _ = load_scores_loss_signed(str(args.expert))
    x_all, _ = load_scores_loss_signed(str(args.cheap))
    y_all, x_all = y_all.numpy().astype(np.float64), x_all.numpy().astype(np.float64)
    subsets, diffs = load_bank(args.bank)
    queries = args.queries or list(range(y_all.shape[1]))
    cfg = {
        "trials": args.trials,
        "seed": args.seed,
        "m": args.m,
        "m_cal": args.m_cal,
        "alpha": args.alpha,
        "eps": {
            "proponent": args.eps,
            "recall": args.recall_eps,
            "score": args.score_eps,
        },
        "tasks": args.tasks,
        "export": (args.export[0], args.export[1], float(args.export[2])),
    }
    config = json.dumps({**cfg, **vars(args)}, default=str)
    (args.out / "config.json").write_text(config)

    jobs = [(q, y_all[:, q], x_all[:, q], subsets, diffs[:, q], cfg) for q in queries]
    with Pool(args.workers) as pool:
        rows = [r for out in pool.imap_unordered(run_query, jobs) for r in out]
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "trials.csv", index=False)

    for q in queries:
        export_expert_set(y_all[:, q], x_all[:, q], q, cfg, args.out)

    pac = df[df.trial >= 0]
    summary = (
        pac.groupby(["task", "uncertainty", "eps"])
        .agg(
            budget_save=("budget_save", "mean"),
            budget_save_sd=("budget_save", "std"),
            error_q=("error", lambda e: e.quantile(1 - args.alpha)),
            error_mean=("error", "mean"),
            cheap_error=("cheap_error", "mean"),
            top1_overlap=("top1_overlap", "mean"),
            lds=("lds", "mean"),
            n=("error", "size"),
        )
        .reset_index()
    )
    summary["violations"] = (
        pac.assign(v=pac.error > pac.eps)
        .groupby(["task", "uncertainty", "eps"])["v"]
        .mean()
        .to_numpy()
    )
    summary.to_csv(args.out / "summary.csv", index=False)
    base = df[df.trial < 0].groupby("uncertainty")[["top1_overlap", "lds"]].mean()
    base.to_csv(args.out / "baselines.csv")
    pd.set_option("display.width", 200)
    print(base)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
