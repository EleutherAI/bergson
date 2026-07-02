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
import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
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

    lexical_baseline: bool = False
    """Also evaluate a TF-IDF token-overlap baseline (scores each train doc
    by summed idf of tokens shared with the query text). If attribution only
    matches this, it is doing surface-form retrieval, not influence."""

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


def entity_metrics(
    scores: np.ndarray, gt_rows: list[int], query_ds, train_ds
) -> tuple[np.ndarray, np.ndarray]:
    """Disentangle entity retrieval from fact discrimination.

    Returns per-query (entity_rank, within_entity_hit): the best competition
    rank among any of the query entity's train rows, and whether the
    ground-truth row outscores the entity's other rows (i.e. the right fact
    wins once the right entity is found). A method that only matches the
    alias surface form gets good entity ranks but ~chance (0.25)
    within-entity hits."""
    entity_rows: dict[int, list[int]] = {}
    for i, identifier in enumerate(train_ds["identifier"]):
        entity_rows.setdefault(identifier, []).append(i)

    ent_ranks, within_hits = [], []
    for q, (identifier, gt) in enumerate(zip(query_ds["identifier"], gt_rows)):
        col = scores[:, q]
        rows = entity_rows[identifier]
        best = max(col[r] for r in rows)
        ent_ranks.append(1 + int((col > best).sum()))
        within_hits.append(all(col[gt] >= col[r] for r in rows))
    return np.array(ent_ranks), np.array(within_hits)


def summarize(
    ranks: np.ndarray,
    ks: list[int],
    num_train: int,
    entity_ranks: np.ndarray,
    within_entity_hits: np.ndarray,
) -> dict[str, float]:
    metrics = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    metrics["mrr"] = float((1.0 / ranks).mean())
    metrics["mean_rank"] = float(ranks.mean())
    metrics["median_rank"] = float(np.median(ranks))
    # Scale-invariant: comparable across training set sizes, unlike recall@k.
    metrics["mean_pct_rank"] = float(((ranks - 1) / (num_train - 1)).mean())
    metrics["entity_recall@1"] = float((entity_ranks <= 1).mean())
    metrics["within_entity_acc"] = float(within_entity_hits.mean())
    return metrics


def lexical_baseline_scores(train_ds, query_ds) -> np.ndarray:
    """TF-IDF token-overlap scores: (num_train, num_queries).

    Note this is a strong foil, not a straw man: queries contain the answer
    string (the gradient is of the answer tokens), and the entailing
    statement states that answer verbatim, so answer-token overlap alone
    solves within-entity discrimination on this dataset."""

    def tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9<|>]+", text.lower()))

    doc_tokens = [tokens(text) for text in train_ds["text"]]
    num_docs = len(doc_tokens)

    doc_freq: dict[str, int] = {}
    postings: dict[str, list[int]] = {}
    for i, doc_toks in enumerate(doc_tokens):
        for tok in doc_toks:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
            postings.setdefault(tok, []).append(i)

    scores = np.zeros((num_docs, len(query_ds)))
    for q, text in enumerate(query_ds["text"]):
        for tok in tokens(text):
            if tok in postings:
                idf = np.log(num_docs / doc_freq[tok])
                scores[postings[tok], q] += idf
    return scores


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

    doc_lengths = np.array([len(text.split()) for text in train_ds["text"]])
    results = []

    def evaluate(label: str, scores: np.ndarray) -> None:
        ranks = query_ranks(scores, gt_rows)
        ent_ranks, within_hits = entity_metrics(scores, gt_rows, query_ds, train_ds)

        # Length bias diagnostic: with sum-reduced losses, long documents have
        # large-norm gradients and can displace the ground truth for length
        # rather than relevance reasons.
        rho = spearmanr(np.abs(scores).mean(axis=1), doc_lengths).statistic
        print(f"[{label}] spearman(mean |score|, doc length) = {rho:.3f}")

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
                    **summarize(
                        ranks[mask],
                        run_cfg.ks,
                        num_train,
                        ent_ranks[mask],
                        within_hits[mask],
                    ),
                }
            )

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

        evaluate(label, scores)

    if run_cfg.lexical_baseline:
        evaluate("lexical_tfidf", lexical_baseline_scores(train_ds, query_ds))

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
