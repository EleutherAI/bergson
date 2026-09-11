"""TRAK over Bergson's projected-gradient index.

TRAK (Park et al., 2023, https://arxiv.org/abs/2303.14186) scores a training
example ``z_i`` for a query ``z_q`` as

    phi(z_q)^T (Phi^T Phi + lambda I)^-1 phi(z_i) * (1 - p_i)

where ``phi`` is a random projection of the per-example gradient, ``Phi``
stacks the projected training gradients and ``p_i`` is the model's
probability of ``z_i``'s labels. ``phi`` is one random projection of the
whole gradient (``projection_target="global"``: every module's flattened
gradient is projected with its own block of a single ``k x d`` Rademacher
matrix and the blocks are summed) and the Gram is the ``autocorrelation``
Hessian with ``scope="joint"`` over that sketch. The damped inverse is applied
to the query side, so the pipeline is the trackstar one without Hessian
mixing or unit normalization, plus the ``(1 - p_i)`` weighting and an
optional average over independently trained checkpoints.
"""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..build import build
from ..config.config import HessianConfig, IndexConfig, TrakConfig
from ..config.config_io import save_run_config
from ..data import pad_and_tensor
from ..distributed import parent_barrier
from ..hessians.hessian_approximations import approximate_hessians
from ..score.score import score_dataset
from ..utils.worker_utils import (
    setup_data_pipeline,
    setup_model_and_peft,
    validate_run_path,
)
from .commands import Build, Hessian, Score
from .trackstar import _limit_split_for_hess, _step_complete


def _train_label_probs(index_cfg: IndexConfig, batch_size: int) -> np.ndarray:
    """Return ``p_i = exp(-mean token CE)`` of every training row under the model.

    The geometric-mean token probability stands in for the classification
    ``p_i`` of Park et al.; ``1 - p_i`` is TRAK's per-example weight.
    """
    ds, _ = setup_data_pipeline(index_cfg)
    cfg = deepcopy(index_cfg)
    cfg.fsdp = False
    model, _ = setup_model_and_peft(cfg, apply_fsdp=False)
    model.eval()
    device = next(model.parameters()).device
    probs = np.zeros(len(ds), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            batch = ds[start : start + batch_size]
            x, y, _, _ = pad_and_tensor(
                batch["input_ids"],
                labels=batch.get("labels"),
                device=device,
                sync_max_len=False,
            )
            logits = model(input_ids=x).logits[:, :-1].float()
            target = y[:, 1:]
            token_ce = F.cross_entropy(
                logits.flatten(0, 1),
                target.flatten(),
                reduction="none",
                ignore_index=-100,
            ).view(target.shape)
            valid = (target != -100).float()
            mean_ce = (token_ce * valid).sum(1) / valid.sum(1).clamp(min=1)
            probs[start : start + len(mean_ce)] = torch.exp(-mean_ce).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return probs


def _open_scores(scores_dir: Path, mode: str) -> tuple[np.memmap, dict]:
    info = json.loads((scores_dir / "info.json").read_text())
    if info.get("attribute_tokens"):
        raise ValueError("TRAK weighting supports per-sequence scores only")
    mmap = np.memmap(
        scores_dir / "scores.bin",
        dtype=info["dtype"],
        mode=mode,
        shape=(info["num_rows"],),
    )
    return mmap, info


def _weight_rows(scores_dir: Path, weights: np.ndarray) -> None:
    """Multiply every query's score column by the per-row ``weights`` in place."""
    mmap, info = _open_scores(scores_dir, "r+")
    assert len(weights) == info["num_rows"], (len(weights), info["num_rows"])
    for q in range(info["num_scores"]):
        col = f"score_{q}"
        mmap[col] = (mmap[col].astype(np.float64) * weights).astype(mmap[col].dtype)
    mmap.flush()
    np.save(scores_dir / "trak_weights.npy", weights)


def _average_scores(member_dirs: list[Path], out_dir: Path) -> None:
    """Write the mean of the members' score stores to ``out_dir``."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(member_dirs[0], out_dir)
    out, info = _open_scores(out_dir, "r+")
    for q in range(info["num_scores"]):
        col = f"score_{q}"
        acc = np.zeros(info["num_rows"], dtype=np.float64)
        for d in member_dirs:
            member, _ = _open_scores(d, "r")
            acc += member[col].astype(np.float64)
        out[col] = (acc / len(member_dirs)).astype(out[col].dtype)
    out.flush()


def _trak_single(index_cfg: IndexConfig, trak_cfg: TrakConfig) -> Path:
    """Gram -> preconditioned query index -> score -> (1 - p) weighting."""
    run_path = index_cfg.run_path
    gram_path = f"{run_path}/train_hessian"
    query_path = f"{run_path}/query"
    scores_path = f"{run_path}/scores"
    resume = trak_cfg.resume
    preprocess_cfg = deepcopy(trak_cfg.preprocess_cfg)
    # TRAK whitens with the Gram inverse; unit normalization would turn that
    # into the trackstar H^-1/2 split and is not part of the method.
    preprocess_cfg.unit_normalize = False
    preprocess_cfg.hessian_path = gram_path

    def _validate(cfg: IndexConfig):
        if resume and cfg.partial_run_path.exists():
            return
        validate_run_path(cfg)

    print("Step 1/4: Fitting the projected-gradient Gram...")
    if not _step_complete(gram_path, resume):
        gram_cfg = deepcopy(index_cfg)
        gram_cfg.run_path = gram_path
        _limit_split_for_hess(gram_cfg, trak_cfg.stats_sample_size)
        _validate(gram_cfg)
        hess_cfg = HessianConfig(method="autocorrelation", scope="joint")
        save_run_config(
            Hessian(hessian_cfg=hess_cfg, index_cfg=gram_cfg),
            gram_cfg.partial_run_path,
        )
        approximate_hessians(gram_cfg, hess_cfg)
    parent_barrier(index_cfg.distributed)

    print("Step 2/4: Building the Gram-whitened query index...")
    if not _step_complete(query_path, resume):
        query_cfg = deepcopy(index_cfg)
        query_cfg.run_path = query_path
        query_cfg.data = deepcopy(trak_cfg.query)
        if preprocess_cfg.aggregation != "none" and query_cfg.attribute_tokens:
            query_cfg.attribute_tokens = False
        _validate(query_cfg)
        save_run_config(Build(query_cfg, preprocess_cfg), query_cfg.partial_run_path)
        build(query_cfg, preprocess_cfg)

    print("Step 3/4: Scoring the training set...")
    if not _step_complete(scores_path, resume):
        score_index_cfg = deepcopy(index_cfg)
        score_index_cfg.run_path = scores_path
        score_cfg = deepcopy(trak_cfg.score_cfg)
        score_cfg.query_path = query_path
        score_cfg.higher_is_better = True
        _validate(score_index_cfg)
        save_run_config(
            Score(score_cfg, score_index_cfg, preprocess_cfg),
            score_index_cfg.partial_run_path,
        )
        score_dataset(score_index_cfg, score_cfg, preprocess_cfg)
    parent_barrier(index_cfg.distributed)

    print("Step 4/4: Weighting training rows by (1 - p_i)...")
    weights_file = Path(scores_path) / "trak_weights.npy"
    if (
        trak_cfg.q_weighting == "one_minus_p"
        and index_cfg.distributed._node_rank == 0
        and not weights_file.exists()
    ):
        if index_cfg.attribute_tokens:
            raise ValueError("TRAK's (1 - p) weighting needs per-sequence scores")
        probs = _train_label_probs(index_cfg, trak_cfg.loss_batch_size)
        _weight_rows(Path(scores_path), 1.0 - probs)
    parent_barrier(index_cfg.distributed)
    return Path(scores_path)


def trak(index_cfg: IndexConfig, trak_cfg: TrakConfig):
    """Run TRAK, averaging over ``trak_cfg.checkpoints`` when given."""
    if index_cfg.projection_dim == 0:
        raise ValueError(
            "TRAK scores random-projected gradients and requires a nonzero "
            "index_cfg.projection_dim; got 0."
        )
    if index_cfg.projection_target != "global":
        raise ValueError(
            "TRAK projects the whole gradient with one random matrix; set "
            "index_cfg.projection_target='global' (got "
            f"{index_cfg.projection_target!r}). Per-module projections "
            "concatenated over modules are a different sketch; use "
            "`bergson trackstar` for a per-module pipeline."
        )
    if not trak_cfg.checkpoints:
        _trak_single(index_cfg, trak_cfg)
        return

    members = []
    for i, model in enumerate(trak_cfg.checkpoints):
        print(f"[TRAK] checkpoint {i + 1}/{len(trak_cfg.checkpoints)}: {model}")
        member_cfg = deepcopy(index_cfg)
        member_cfg.model = model
        member_cfg.run_path = f"{index_cfg.run_path}/checkpoint_{i}"
        members.append(_trak_single(member_cfg, trak_cfg))
    if index_cfg.distributed._node_rank == 0:
        _average_scores(members, Path(index_cfg.run_path) / "scores")
    parent_barrier(index_cfg.distributed)
