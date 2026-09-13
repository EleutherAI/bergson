"""Score only the rows an earlier run ranked highest.

The store keeps every row: candidates carry this run's scores, the rest keep
the earlier run's, oriented to this run, scaled to the candidates' spread and
shifted past the weakest candidate.
"""

import json
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


def select_candidates(cfg: CandidateConfig, num_rows: int) -> np.ndarray:
    """Sorted union over query columns of each column's kept rows."""
    scores = load_earlier_scores(cfg, num_rows)
    count = candidate_count(cfg, num_rows)
    if count >= num_rows:
        return np.arange(num_rows)

    keep_high = earlier_higher_is_better(cfg) == (cfg.direction == "proponents")
    ordered = -scores if keep_high else scores
    return np.unique(np.argpartition(ordered, count - 1, axis=0)[:count])


def merge_scores(
    candidate_scores: np.ndarray,
    earlier: np.ndarray,
    candidates: np.ndarray,
    *,
    keep_high: bool,
    flip: bool,
) -> np.ndarray:
    """Fill the rows outside ``candidates`` from ``earlier``.

    ``earlier`` has this run's column count or one column; ``flip`` negates
    it. The filled rows keep their order, scaled per column to the
    candidates' spread so float32 keeps them apart, with the best one float32
    step past the weakest candidate.
    """
    num_rows, earlier_cols = earlier.shape
    num_scores = candidate_scores.shape[1]
    if earlier_cols not in (1, num_scores):
        raise ValueError(
            f"The earlier run has {earlier_cols} score columns; this run has "
            f"{num_scores}."
        )

    earlier = np.broadcast_to(-earlier if flip else earlier, (num_rows, num_scores))
    merged = np.array(earlier, dtype=np.float64)
    merged[candidates] = candidate_scores

    rest = np.ones(num_rows, dtype=bool)
    rest[candidates] = False
    if rest.any():
        sign = 1.0 if keep_high else -1.0
        own = sign * candidate_scores
        filled = sign * earlier[rest]
        own_span = own.max(axis=0) - own.min(axis=0)
        span = filled.max(axis=0) - filled.min(axis=0)
        scale = np.where(
            (own_span > 0) & (span > 0), own_span / np.maximum(span, 1e-300), 1.0
        )
        boundary = np.nextafter(own.min(axis=0).astype(np.float32), np.float32(-np.inf))
        merged[rest] = sign * (boundary - (filled.max(axis=0) - filled) * scale)
    return merged


def write_merged_scores(
    path: Path, cfg: CandidateConfig, candidates: np.ndarray, higher_is_better: bool
) -> None:
    """Fill the non-candidate rows of the store at ``path`` from the earlier run."""
    with open(path / "info.json") as f:
        info = json.load(f)
    store = np.memmap(
        path / "scores.bin", dtype=info["dtype"], mode="r+", shape=(info["num_items"],)
    )
    fields = [f"score_{i}" for i in range(info["num_scores"])]

    merged = merge_scores(
        np.stack([store[f][candidates] for f in fields], axis=1).astype(np.float64),
        load_earlier_scores(cfg, len(store)),
        candidates,
        keep_high=higher_is_better == (cfg.direction == "proponents"),
        flip=earlier_higher_is_better(cfg) != higher_is_better,
    )

    rest = np.ones(len(store), dtype=bool)
    rest[candidates] = False
    for i, f in enumerate(fields):
        store[f][rest] = merged[rest, i]
        store[f"written_{i}"][rest] = True
    store.flush()
    np.save(path / "candidates.npy", candidates)
