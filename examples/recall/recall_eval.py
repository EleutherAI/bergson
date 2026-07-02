"""Evaluate factual-entailment recall of attribution scores.

Prototype for a future ``bergson recall`` command (interface mirrors
``bergson validate``: point it at saved attribution scores and the datasets
they refer to).

Given per-query attribution scores of a training dataset (built with
``bergson ekfac --query_aggregation none``, giving one score column per query
example), measures how highly each query's ground-truth training document —
the statement entailing the answer to that query — is ranked. The ground
truth for query q is the train row with the same ``(identifier, field)``
(see examples/recall/prepare_recall_data.py).

Reports recall@k (fraction of queries whose ground-truth document is in the
top-k), MRR, and mean/median rank — overall and per question field — for
every score directory found (``scores`` or ``scores_lam_*`` for a damping
sweep), and writes them to ``<run_path>/recall.csv``.

Example:
    python -m examples.recall.recall_eval runs/recall/ekfac \
        --train_dataset runs/recall/data/train \
        --query_dataset runs/recall/data/query
"""

import csv
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

import numpy as np
from simple_parsing import field, parse

from bergson.config.config import ScoreConfig
from bergson.config.config_io import load_subconfig
from bergson.data import load_data_string, load_scores


@dataclass
class RecallEvalConfig:
    """Config for the recall evaluation."""

    run_path: str = field(positional=True)
    """An ekfac run directory containing ``scores``/``scores_lam_*``
    subdirectories, or a single scores directory."""

    train_dataset: str = ""
    """The attributed (training) dataset, with identifier/field columns."""

    query_dataset: str = ""
    """The query dataset, with identifier/field columns; row order must match
    the score columns."""

    ks: list[int] = dc_field(default_factory=lambda: [1, 5, 100])
    """Ranks at which to report recall@k."""

    out_path: str = ""
    """Where to write the results CSV. Defaults to <run_path>/recall.csv."""


def find_score_dirs(run_path: Path) -> dict[str, Path]:
    """Map a label (damping tag or 'scores') to each scores directory."""
    if (run_path / "scores.bin").exists():
        return {run_path.name: run_path}

    dirs = {}
    for sub in sorted(run_path.glob("scores*")):
        if (sub / "scores.bin").exists():
            label = sub.name.removeprefix("scores").removeprefix("_") or "scores"
            dirs[label] = sub
    if not dirs:
        raise FileNotFoundError(f"No scores directories found under {run_path}")
    return dirs


def ground_truth_rows(train_ds, query_ds) -> list[int]:
    """Index of each query's ground-truth train row, matched on
    (identifier, field)."""
    train_index: dict[tuple[int, str], list[int]] = {}
    for i, (identifier, fld) in enumerate(
        zip(train_ds["identifier"], train_ds["field"])
    ):
        train_index.setdefault((identifier, fld), []).append(i)

    rows = []
    for identifier, fld in zip(query_ds["identifier"], query_ds["field"]):
        matches = train_index.get((identifier, fld), [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly 1 train row for query ({identifier}, {fld}), "
                f"got {len(matches)}"
            )
        rows.append(matches[0])
    return rows


def query_ranks(scores: np.ndarray, gt_rows: list[int]) -> np.ndarray:
    """Competition rank of each query's ground-truth document (1 = best)."""
    ranks = []
    for q, gt in enumerate(gt_rows):
        col = scores[:, q]
        ranks.append(1 + int((col > col[gt]).sum()))
    return np.array(ranks)


def summarize(ranks: np.ndarray, ks: list[int]) -> dict[str, float]:
    metrics = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    metrics["mrr"] = float((1.0 / ranks).mean())
    metrics["mean_rank"] = float(ranks.mean())
    metrics["median_rank"] = float(np.median(ranks))
    return metrics


def main():
    run_cfg = parse(RecallEvalConfig)
    assert run_cfg.train_dataset, "--train_dataset must be provided"
    assert run_cfg.query_dataset, "--query_dataset must be provided"

    run_path = Path(run_cfg.run_path)
    score_dirs = find_score_dirs(run_path)

    train_ds = load_data_string(run_cfg.train_dataset)
    query_ds = load_data_string(run_cfg.query_dataset)
    gt_rows = ground_truth_rows(train_ds, query_ds)
    fields = query_ds["field"]

    num_train = len(train_ds)
    print(
        f"{len(gt_rows)} queries against {num_train} training docs "
        f"(chance recall@k = k/{num_train})"
    )

    results = []
    for label, score_dir in score_dirs.items():
        scores = np.asarray(load_scores(score_dir)[:], dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != len(gt_rows):
            raise ValueError(
                f"{score_dir} has score shape {scores.shape}; expected "
                f"({num_train}, {len(gt_rows)}). Was the ekfac run built with "
                "--query_aggregation none?"
            )

        # More negative = more influential for MAGIC scores; IF pipelines
        # record higher_is_better=True. Flip so that higher = more relevant.
        score_cfg = load_subconfig(score_dir, "score_cfg", ScoreConfig)
        if score_cfg is not None and not score_cfg.higher_is_better:
            scores = -scores

        ranks = query_ranks(scores, gt_rows)
        for fld in ["all"] + sorted(set(fields)):
            mask = (
                np.ones(len(ranks), dtype=bool)
                if fld == "all"
                else np.array([f == fld for f in fields])
            )
            results.append(
                {
                    "scores": label,
                    "field": fld,
                    "n_queries": int(mask.sum()),
                    **summarize(ranks[mask], run_cfg.ks),
                }
            )

    out_path = run_cfg.out_path or str(run_path / "recall.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} rows to {out_path}\n")

    header = list(results[0].keys())
    print(" | ".join(f"{h:>12}" for h in header))
    for row in results:
        if row["field"] != "all":
            continue
        print(
            " | ".join(
                f"{row[h]:>12.4f}" if isinstance(row[h], float) else f"{row[h]!s:>12}"
                for h in header
            )
        )


if __name__ == "__main__":
    main()
