import hashlib
import json
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path

from ..build import build
from ..config import (
    HessianConfig,
    HessianPipelineConfig,
    IndexConfig,
    PreprocessConfig,
    ScoreConfig,
)
from ..distributed import launch_distributed_run
from ..score.score import score_dataset
from ..utils.worker_utils import validate_run_path
from .apply_hessian import EkfacConfig, apply_worker
from .hessian_approximations import approximate_hessians


def _fingerprint(*values) -> str:
    payload = [asdict(v) if is_dataclass(v) else v for v in values]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _step_complete(path: str, resume: bool, fingerprint: str) -> bool:
    """A directory alone is not evidence that a distributed stage finished."""
    if not resume:
        return False
    marker = Path(path) / "PIPELINE_COMPLETE.json"
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text())
    if saved["fingerprint"] != fingerprint:
        raise ValueError(f"Completed stage has different inputs: {path}")
    for filename, size in saved["outputs"].items():
        output = Path(path) / filename
        if not output.is_file() or output.stat().st_size != size:
            raise ValueError(f"Completed stage output is missing or changed: {output}")
    print(f"  Reusing completed stage at {path}")
    return True


def _mark_complete(path: str, fingerprint: str) -> None:
    directory = Path(path)
    outputs = {
        str(p.relative_to(directory)): p.stat().st_size
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "PIPELINE_COMPLETE.json"
    }
    if not outputs:
        raise RuntimeError(f"Stage returned without output files: {path}")
    marker = directory / "PIPELINE_COMPLETE.json"
    temporary = marker.with_suffix(".partial")
    temporary.write_text(
        json.dumps({"fingerprint": fingerprint, "outputs": outputs}, sort_keys=True)
    )
    temporary.replace(marker)


def _query_config(index_cfg: IndexConfig, cfg: HessianPipelineConfig, path: str):
    query_cfg = deepcopy(index_cfg)
    query_cfg.run_path = path
    query_cfg.data = deepcopy(cfg.query)
    query_cfg.loss_fn = cfg.query_loss_fn or index_cfg.loss_fn
    query_cfg.loss_reduction = cfg.query_loss_reduction or index_cfg.loss_reduction
    query_cfg.example_weight_column = cfg.query_weight_column
    if cfg.query_dpo_beta is not None:
        query_cfg.dpo_beta = cfg.query_dpo_beta
    query_cfg.projection_dim = 0
    query_cfg.skip_hessians = True
    return query_cfg


@contextmanager
def _timed(label: str, durations: dict[str, float]):
    """Time a pipeline step and print the wall-clock duration on exit."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        durations[label] = elapsed
        print(f"  [{label}] took {elapsed:.1f}s")


def hessian_pipeline(
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
    hessian_pipeline_cfg: HessianPipelineConfig,
):
    """Run the full Hessian-preconditioned influence pipeline.

    1. Build mean query gradient.
    2. Fit Hessian factors (kfac, tkfac, shampoo) on the training dataset.
    3. Apply the inverse Hessian to the mean query gradient.
    4. Score each training example against the transformed query gradient.
    """
    run_path = index_cfg.run_path
    method = hessian_cfg.method
    query_path = f"{run_path}/query"
    hessian_path = hessian_pipeline_cfg.hessian_run_path or f"{run_path}/hessian"
    transformed_query_path = f"{run_path}/{method}_query"
    scores_path = f"{run_path}/scores"
    resume = hessian_pipeline_cfg.resume
    provenance = (
        json.loads(Path(hessian_pipeline_cfg.provenance_path).read_text())
        if hessian_pipeline_cfg.provenance_path
        else {}
    )
    query_cfg = _query_config(index_cfg, hessian_pipeline_cfg, query_path)
    query_preprocess_cfg = PreprocessConfig(
        aggregation="mean",
        aggregation_device="cpu" if query_cfg.offload_gradients else "cuda",
    )
    query_hash = _fingerprint(query_cfg, query_preprocess_cfg, provenance)
    hessian_index_cfg = deepcopy(index_cfg)
    hessian_index_cfg.run_path = hessian_path
    hessian_hash = _fingerprint(hessian_index_cfg, hessian_cfg, provenance)
    transformed_hash = _fingerprint(
        query_hash, hessian_hash, hessian_pipeline_cfg.lambda_damp_factor
    )
    score_index_cfg = deepcopy(index_cfg)
    score_index_cfg.run_path = scores_path
    score_index_cfg.projection_dim = 0
    score_index_cfg.skip_hessians = True
    score_cfg = deepcopy(score_cfg)
    score_cfg.query_path = transformed_query_path
    score_cfg.higher_is_better = True
    score_hash = _fingerprint(
        score_index_cfg, score_cfg, preprocess_cfg, transformed_hash
    )

    def _validate(cfg: IndexConfig):
        if resume:
            # Preserve interrupted artifacts for inspection, then retry the
            # unfinished stage from a clean path. Never trust partial contents.
            for path in (Path(cfg.run_path), cfg.partial_run_path):
                if path.exists():
                    archived = Path(str(path) + f".incomplete-{time.time_ns()}")
                    path.rename(archived)
        validate_run_path(cfg)

    durations: dict[str, float] = {}

    # ── Step 1: Build mean query gradient ─────────────────────────────────
    print("Step 1/4: Building mean query gradient...")
    if not _step_complete(query_path, resume, query_hash):
        with _timed("step1_build_query", durations):
            _validate(query_cfg)
            build(query_cfg, query_preprocess_cfg)
            _mark_complete(query_path, query_hash)

    # ── Step 2: Fit Hessian factors on training data ──────────────────────
    print(f"Step 2/4: Fitting {method} factors on training data...")
    if not _step_complete(hessian_path, resume, hessian_hash):
        with _timed("step2_fit_hessian", durations):
            _validate(hessian_index_cfg)
            approximate_hessians(hessian_index_cfg, hessian_cfg)
            _mark_complete(hessian_path, hessian_hash)

    # ── Step 3: Apply inverse Hessian to the mean query gradient ──────────
    print(f"Step 3/4: Applying {method} inverse Hessian to mean query gradient...")
    if not _step_complete(transformed_query_path, resume, transformed_hash):
        apply_index_cfg = deepcopy(index_cfg)
        apply_index_cfg.run_path = transformed_query_path
        _validate(apply_index_cfg)
        hessian_method_path = f"{hessian_path}/{method}"
        ekfac_cfg = EkfacConfig(
            hessian_method_path=hessian_method_path,
            gradient_path=query_path,
            run_path=transformed_query_path,
            ev_correction=hessian_cfg.ev_correction,
            lambda_damp_factor=hessian_pipeline_cfg.lambda_damp_factor,
        )
        launch_distributed_run(
            "apply_hessian",
            apply_worker,
            [ekfac_cfg],
            index_cfg.distributed,
        )
        _mark_complete(transformed_query_path, transformed_hash)

    if hessian_pipeline_cfg.skip_scoring:
        print(f"Prepared transformed query at: {transformed_query_path}")
        return

    # ── Step 4: Score training examples ───────────────────────────────────
    print("Step 4/4: Scoring training data against transformed query...")
    if not _step_complete(scores_path, resume, score_hash):
        _validate(score_index_cfg)

        score_dataset(score_index_cfg, score_cfg, preprocess_cfg)
        _mark_complete(scores_path, score_hash)

    print(f"Done! Scores saved to: {scores_path}")
    if durations:
        total = sum(durations.values())
        print(f"Step timings (s): {durations} | total {total:.1f}s")
