"""Score only the rows an earlier run ranked highest."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from bergson.config.config import CandidateConfig, ScoreConfig
from bergson.config.config_io import load_subconfig
from bergson.data import load_scores


def candidate_count(cfg: CandidateConfig, num_rows: int) -> int:
    """Rows kept per query column."""
    if (cfg.top_k > 0) == (cfg.fraction > 0):
        raise ValueError("Set exactly one of candidates.top_k and candidates.fraction.")
    if cfg.top_k > 0:
        return min(cfg.top_k, num_rows)
    if not 0.0 < cfg.fraction <= 1.0:
        raise ValueError(f"candidates.fraction must be in (0, 1]; got {cfg.fraction}.")
    return max(1, round(cfg.fraction * num_rows))


def earlier_higher_is_better(cfg: CandidateConfig) -> bool:
    if cfg.higher_is_better is not None:
        return cfg.higher_is_better
    score_cfg = load_subconfig(cfg.scores, "score_cfg", ScoreConfig)
    if score_cfg is None:
        raise ValueError(
            f"{cfg.scores} has no saved score_cfg; set candidates.higher_is_better."
        )
    return score_cfg.higher_is_better


def load_earlier_scores(cfg: CandidateConfig, num_rows: int) -> np.ndarray:
    """The earlier run's ``[num_rows, num_scores]`` scores."""
    loaded = load_scores(Path(cfg.scores))
    if loaded.offsets is not None:
        raise ValueError(f"{cfg.scores} holds per-token scores.")
    if len(loaded) != num_rows:
        raise ValueError(
            f"{cfg.scores} has {len(loaded)} rows; this run's dataset has {num_rows}."
        )
    if not loaded.is_written():
        raise ValueError(f"{cfg.scores} is incomplete.")
    return np.asarray(loaded[:]).astype(np.float64)


def select_from_query_record(cfg: CandidateConfig, num_rows: int) -> np.ndarray:
    """Sorted union over the queries a ``query --record`` CSV holds of each
    query's kept rows."""
    if cfg.fraction > 0:
        raise ValueError("candidates.fraction does not apply to a query record.")
    direction = "Top" if cfg.direction == "proponents" else "Bottom"

    kept: dict[str, list[int]] = defaultdict(list)
    with open(cfg.scores, newline="") as f:
        for row in csv.DictReader(f):
            if row["direction"] == direction:
                kept[row["query"]].append(int(row["result_index"]))
    if not kept:
        raise ValueError(f"{cfg.scores} has no {direction} rows.")

    rows = [i for per_query in kept.values() for i in per_query[: cfg.top_k or None]]
    candidates = np.unique(np.asarray(rows, dtype=np.int64))
    if candidates[-1] >= num_rows:
        raise ValueError(
            f"{cfg.scores} refers to row {candidates[-1]}; this run's dataset has "
            f"{num_rows} rows."
        )
    return candidates


def select_candidates(cfg: CandidateConfig, num_rows: int) -> np.ndarray:
    """Sorted union over query columns of each column's kept rows."""
    if cfg.scores.endswith(".csv"):
        return select_from_query_record(cfg, num_rows)

    scores = load_earlier_scores(cfg, num_rows)
    count = candidate_count(cfg, num_rows)
    if count >= num_rows:
        return np.arange(num_rows)

    keep_high = earlier_higher_is_better(cfg) == (cfg.direction == "proponents")
    ordered = -scores if keep_high else scores
    return np.unique(np.argpartition(ordered, count - 1, axis=0)[:count])
