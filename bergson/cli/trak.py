import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from ..build import build
from ..config.config import HessianConfig, IndexConfig, InversionConfig, TrakConfig
from ..config.config_io import save_run_config
from ..data import pad_and_tensor
from ..distributed import parent_barrier
from ..gradients import GradientProcessor
from ..hessians.hessian_approximations import approximate_hessians
from ..hessians.inversion import invert_psd_matrix
from ..score.score import score_dataset
from ..utils.worker_utils import (
    setup_data_pipeline,
    setup_model_and_peft,
    validate_run_path,
)
from .commands import Build, Hessian, Score
from .trackstar import _limit_split_for_hess, _step_complete


def _train_label_probs(index_cfg: IndexConfig, batch_size: int) -> np.ndarray:
    """Return ``p_i``, the mean label-token probability of every training row
    under the model, so ``1 - p_i`` is the row's TRAK ``Q`` term: the
    derivative of the loss w.r.t. the negative log-odds output, averaged over tokens.
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
            mean_p = (torch.exp(-token_ce) * valid).sum(1) / valid.sum(1).clamp(min=1)
            probs[start : start + len(mean_p)] = mean_p.cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return probs


def _open_scores(scores_dir: Path, mode: Literal["r", "r+"]) -> tuple[np.memmap, dict]:
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


def _inverse_gram_scale(gram_path: str, inversion_cfg: InversionConfig) -> float:
    """Mean absolute entry of the inverse Gram at ``gram_path``. The reference
    implementation divides the inverse by it so that ensemble members' scores
    share a scale."""
    processor = GradientProcessor.load(Path(gram_path), map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h_inv = invert_psd_matrix(
        processor.hessians["joint"].to(device=device, dtype=torch.float32),
        inversion=inversion_cfg.inversion,
        damping_factor=inversion_cfg.damping_factor,
        power=-1.0,
    )
    return float(h_inv.abs().mean())


def _average_scores(member_dirs: list[Path], out_dir: Path) -> None:
    """Write the ensemble score store to ``out_dir``. Each member's unweighted
    scores divided by its ``trak_scale``, averaged over members, times the
    members' mean ``1 - p_i`` weights."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(member_dirs[0], out_dir)
    weights = np.mean([np.load(d / "trak_weights.npy") for d in member_dirs], axis=0)
    scales = np.array([np.load(d / "trak_scale.npy") for d in member_dirs])
    out, info = _open_scores(out_dir, "r+")
    for q in range(info["num_scores"]):
        col = f"score_{q}"
        acc = np.zeros(info["num_rows"], dtype=np.float64)
        for d, scale in zip(member_dirs, scales):
            member, _ = _open_scores(d, "r")
            acc += member[col].astype(np.float64) / scale
        out[col] = (acc / len(member_dirs) * weights).astype(out[col].dtype)
    out.flush()
    np.save(out_dir / "trak_weights.npy", weights)
    np.save(out_dir / "trak_scale.npy", scales)


def _trak_single(
    index_cfg: IndexConfig, trak_cfg: TrakConfig, member: bool = False
) -> Path:
    """Gram -> preconditioned query index -> score -> (1 - p) weighting.

    An ensemble ``member`` saves its weights and scale but leaves its scores
    unweighted for :func:`_average_scores`.
    """
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
        # The Gram is over the training examples' own negative log-odds gradients.
        hess_cfg = HessianConfig(
            method="autocorrelation", structure="joint", use_dataset_labels=True
        )
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
    scores_dir = Path(scores_path)
    weights_file = scores_dir / "trak_weights.npy"
    if index_cfg.distributed._node_rank == 0 and not weights_file.exists():
        if index_cfg.attribute_tokens:
            raise ValueError("TRAK's (1 - p) weighting needs per-sequence scores")
        weights = 1.0 - _train_label_probs(index_cfg, trak_cfg.loss_batch_size)
        scale = _inverse_gram_scale(gram_path, preprocess_cfg.inversion_cfg)
        if not member:
            _weight_rows(scores_dir, weights / scale)
        np.save(scores_dir / "trak_scale.npy", scale)
        np.save(weights_file, weights)
    parent_barrier(index_cfg.distributed)
    return scores_dir


def trak(index_cfg: IndexConfig, trak_cfg: TrakConfig):
    """Run TRAK, averaging over ``trak_cfg.checkpoints`` when given."""
    if index_cfg.projection_dim == 0:
        raise ValueError(
            "TRAK scores random-projected gradients and requires a nonzero "
            "index_cfg.projection_dim; got 0."
        )
    if index_cfg.loss_fn != "log_odds":
        raise ValueError(
            "TRAK's features are gradients of the log odds output function; set "
            f"index_cfg.loss_fn='log_odds' (got {index_cfg.loss_fn!r})."
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
        member_cfg.projection_seed = index_cfg.projection_seed + i
        member_cfg.run_path = f"{index_cfg.run_path}/checkpoint_{i}"
        members.append(_trak_single(member_cfg, trak_cfg, member=True))
    if index_cfg.distributed._node_rank == 0:
        _average_scores(members, Path(index_cfg.run_path) / "scores")
    parent_barrier(index_cfg.distributed)
