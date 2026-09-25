from copy import deepcopy
from pathlib import Path

import numpy as np

from ..build import build_query
from ..config.config import IndexConfig, PreprocessConfig, ScoreConfig, TracInConfig
from ..config.config_io import save_run_config
from ..data import load_scores
from ..distributed import parent_barrier
from ..score.score import score_dataset
from ..score.score_writer import save_sequence_scores, save_token_scores
from ..utils.worker_utils import validate_run_path
from .commands import Score
from .trackstar import _step_complete


def tracin(index_cfg: IndexConfig, tracin_cfg: TracInConfig):
    """Run TracIn: build the query index and score the training set at each
    checkpoint, then sum the scores weighted by the checkpoint learning rates."""
    checkpoints = tracin_cfg.checkpoints
    if not checkpoints:
        raise ValueError("TracIn needs at least one entry in tracin_cfg.checkpoints.")
    if len(tracin_cfg.lr_list) != len(checkpoints):
        raise ValueError(
            f"tracin_cfg.lr_list has {len(tracin_cfg.lr_list)} entries but there "
            f"are {len(checkpoints)} checkpoints; give one learning rate each."
        )
    if tracin_cfg.query.path:
        raise ValueError(
            "TracIn builds the query gradients at every checkpoint, so it can't "
            "use an existing query index; leave tracin_cfg.query.path unset."
        )

    resume = tracin_cfg.resume
    preprocess_cfg = PreprocessConfig()

    def _validate(cfg: IndexConfig):
        if resume and cfg.partial_run_path.exists():
            return
        validate_run_path(cfg)

    score_dirs = []
    for i, model in enumerate(checkpoints):
        print(f"[TracIn] checkpoint {i + 1}/{len(checkpoints)}: {model}")
        ckpt_path = f"{index_cfg.run_path}/checkpoint_{i}"
        query_path = f"{ckpt_path}/query"
        scores_path = f"{ckpt_path}/scores"

        if not _step_complete(query_path, resume):
            query_cfg = deepcopy(index_cfg)
            query_cfg.model = model
            query_cfg.run_path = query_path
            _validate(query_cfg)
            build_query(query_cfg, tracin_cfg.query, preprocess_cfg)

        if not _step_complete(scores_path, resume):
            score_index_cfg = deepcopy(index_cfg)
            score_index_cfg.model = model
            score_index_cfg.run_path = scores_path
            score_cfg = deepcopy(tracin_cfg.score_cfg)
            score_cfg.query_path = query_path
            score_cfg.higher_is_better = True
            _validate(score_index_cfg)
            save_run_config(
                Score(score_cfg, score_index_cfg, preprocess_cfg),
                score_index_cfg.partial_run_path,
            )
            score_dataset(score_index_cfg, score_cfg, preprocess_cfg)
        score_dirs.append(Path(scores_path))
    parent_barrier(index_cfg.distributed)

    if index_cfg.distributed._node_rank == 0:
        total = np.sum(
            [
                lr * load_scores(d)[:].astype(np.float64)
                for lr, d in zip(tracin_cfg.lr_list, score_dirs)
            ],
            axis=0,
        )
        out_path = Path(index_cfg.run_path) / "scores"
        if index_cfg.attribute_tokens:
            offsets = np.load(score_dirs[0] / "offsets.npy")
            save_token_scores(out_path, total, offsets)
        else:
            save_sequence_scores(out_path, total)

        out_index_cfg = deepcopy(index_cfg)
        out_index_cfg.run_path = str(out_path)
        save_run_config(
            Score(ScoreConfig(higher_is_better=True), out_index_cfg, preprocess_cfg),
            out_path,
        )
    parent_barrier(index_cfg.distributed)
