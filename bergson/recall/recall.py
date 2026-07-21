"""Evaluate attribution scores by synthetic factual recall (MRR, Recall@k).

For each question we rank every training statement by its precomputed
attribution score and locate the gold statements — those stating the same
person+field fact the question asks about. A question is a hit if any gold
statement (any paraphrase of the fact) ranks in the top k.
"""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk

from bergson.config.config import RecallConfig, ScoreConfig
from bergson.config.config_io import load_subconfig
from bergson.data import load_scores
from bergson.recall.generate import ensure_recall_datasets
from bergson.utils.csv_writer import CSVWriter
from bergson.utils.utils import assert_type


def gold_ranks(scores_col: np.ndarray, gold_idx: np.ndarray) -> np.ndarray:
    """1-indexed ranks of ``gold_idx`` when ``scores_col`` is sorted
    descending, with ties broken by original row order (stable sort)."""
    ranks = np.empty(len(gold_idx), dtype=np.int64)
    for j, i in enumerate(gold_idx):
        higher = int((scores_col > scores_col[i]).sum())
        tied_before = int((scores_col[:i] == scores_col[i]).sum())
        ranks[j] = higher + tied_before + 1

    return ranks


def resolve_higher_is_better(scores_path: str, explicit: bool | None) -> bool:
    """Score orientation to use when ranking ``scores_path``.

    ``explicit`` wins when set. Otherwise use the ``score_cfg.higher_is_better``
    recorded in the score directory.
    """
    if explicit is not None:
        return explicit

    score_cfg = (
        load_subconfig(scores_path, "score_cfg", ScoreConfig)
        if scores_path and os.path.isdir(scores_path)
        else None
    )
    if score_cfg is None:
        return True

    print(
        f"Using higher_is_better={score_cfg.higher_is_better} recorded in "
        f"{scores_path}; set recall_cfg.higher_is_better to override."
    )
    return score_cfg.higher_is_better


def run_recall(recall_cfg: RecallConfig) -> dict[str, float]:
    """Compute MRR and Recall@k of attribution scores against the gold
    entailing statements, writing ``recall.csv`` and ``summary.csv`` to
    ``recall_cfg.run_path``."""
    statements_path, questions_path = ensure_recall_datasets(recall_cfg.data)
    statements = assert_type(Dataset, load_from_disk(str(statements_path)))
    questions = assert_type(Dataset, load_from_disk(str(questions_path)))

    scores = load_scores(Path(recall_cfg.scores))
    if len(scores) != len(statements):
        raise ValueError(
            f"Scores at {recall_cfg.scores} cover {len(scores)} items but "
            f"{statements_path} has {len(statements)} statements. Was the "
            f"index built from a different dataset?"
        )
    if scores.num_scores != len(questions):
        raise ValueError(
            f"Scores at {recall_cfg.scores} have {scores.num_scores} queries "
            f"but {questions_path} has {len(questions)} questions. Were the "
            f"query gradients built from a different dataset?"
        )
    if not scores.is_written():
        print(
            "Warning: not all score entries have been written; "
            "results may be incomplete."
        )

    # Gold statements for a question are all rows stating the same
    # person+field fact.
    gold: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i, (identifier, fact_field) in enumerate(
        zip(statements["identifier"], statements["field"])
    ):
        gold[(identifier, fact_field)].append(i)

    higher_is_better = resolve_higher_is_better(
        recall_cfg.scores, recall_cfg.higher_is_better
    )

    k = recall_cfg.k
    os.makedirs(recall_cfg.run_path, exist_ok=True)
    recall_csv_path = os.path.join(recall_cfg.run_path, "recall.csv")
    recall_csv_writer = CSVWriter(
        recall_csv_path,
        columns=[
            "question_idx",
            "identifier",
            "field",
            "answer",
            "first_gold_rank",
            "reciprocal_rank",
            f"hit_at_{k}",
            f"strict_recall_at_{k}",
            "num_gold",
        ],
    )

    identifiers = questions["identifier"]
    fields = questions["field"]
    answers = questions["answer"]

    reciprocal_ranks = []
    hits = []
    strict_recalls = []
    for q_idx in range(len(questions)):
        col = np.asarray(scores.get(slice(None), q_idx), dtype=np.float64)
        if not higher_is_better:
            col = -col

        gold_idx = np.asarray(gold[(identifiers[q_idx], fields[q_idx])])
        ranks = gold_ranks(col, gold_idx)

        first_gold_rank = int(ranks.min())
        reciprocal_rank = 1.0 / first_gold_rank
        hit = int(first_gold_rank <= k)
        strict_recall = float((ranks <= k).sum() / len(ranks))

        recall_csv_writer.writerow(
            q_idx,
            identifiers[q_idx],
            fields[q_idx],
            answers[q_idx],
            first_gold_rank,
            reciprocal_rank,
            hit,
            strict_recall,
            len(gold_idx),
        )
        reciprocal_ranks.append(reciprocal_rank)
        hits.append(hit)
        strict_recalls.append(strict_recall)

    recall_csv_writer.close()
    print(f"Saved per-question recall data to {recall_csv_path}")

    mrr = float(np.mean(reciprocal_ranks))
    recall_at_k = float(np.mean(hits))
    strict_recall_at_k = float(np.mean(strict_recalls))

    summary_csv_path = os.path.join(recall_cfg.run_path, "summary.csv")
    summary_csv_writer = CSVWriter(
        summary_csv_path,
        columns=[
            "MRR",
            f"recall_at_{k}",
            f"strict_recall_at_{k}",
            "N",
            "num_people",
        ],
    )
    summary_csv_writer.writerow(
        mrr,
        recall_at_k,
        strict_recall_at_k,
        len(questions),
        recall_cfg.data.num_people,
    )
    summary_csv_writer.close()

    print(
        f"MRR: {mrr:.4f}  Recall@{k}: {recall_at_k:.4f}  "
        f"(N={len(questions)} questions, {len(statements)} statements)"
    )
    print(f"Saved summary to {summary_csv_path}")

    return {
        "mrr": mrr,
        f"recall_at_{k}": recall_at_k,
        f"strict_recall_at_{k}": strict_recall_at_k,
    }
