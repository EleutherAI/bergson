import json
import shutil
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file

from bergson.config.config import DistributedConfig
from bergson.distributed import init_dist, launch_distributed_run
from bergson.hessians.eigenvectors import compute_eigendecomposition
from bergson.utils.logger import get_logger
from bergson.utils.utils import get_device

_SHARD_KINDS = ("activation_sharded", "gradient_sharded")

LAMBDA_COUNTS_FILENAME = "counts.json"

FisherNormalization = Literal["document", "token", "none"]


def write_lambda_counts(lambda_dir: Path, *, documents: int, tokens: int) -> None:
    """Record the denominators available for normalizing a checkpoint's lambdas.

    ``LambdaCollector`` accumulates a sum over documents; ``total_processed``
    counts tokens. Both are recorded so the choice is made at aggregation time.
    """
    with open(lambda_dir / LAMBDA_COUNTS_FILENAME, "w") as f:
        json.dump({"documents": documents, "tokens": tokens}, f)


def lambda_denominator(
    input_dirs: list[Path], normalization: FisherNormalization
) -> float:
    """Pooled count over a segment's checkpoints.

    The lambdas are summed over checkpoints, so the denominator sums too: the
    result is the mean over every (checkpoint, document) pair, which is what
    kronfluence's ``lambda_matrix / num_lambda_processed`` computes for a single
    checkpoint.
    """
    if normalization == "none":
        return 1.0

    key = "documents" if normalization == "document" else "tokens"
    total = 0
    for d in input_dirs:
        counts_path = d / LAMBDA_COUNTS_FILENAME
        if not counts_path.exists():
            raise FileNotFoundError(
                f"Missing {counts_path}, needed to normalize the segment "
                f"eigenvalues by {key}. It is written alongside the lambda "
                "shards; re-run the per-checkpoint lambda step, or set "
                "`fisher_normalization: none` to keep the unnormalized sum."
            )
        with open(counts_path) as f:
            total += json.load(f)[key]
    return float(total)


def sum_sharded_dirs(
    input_dirs: list[Path],
    output_dir: Path,
    distributed: DistributedConfig,
    divisor: float = 1.0,
) -> None:
    """Sum per-rank shards across ``input_dirs`` into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    launch_distributed_run(
        "sum_sharded_dirs",
        _sum_sharded_dirs_worker,
        [input_dirs, output_dir, divisor],
        distributed,
    )


def _sum_sharded_dirs_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    input_dirs: list[Path],
    output_dir: Path,
    divisor: float,
) -> None:
    init_dist(rank, local_rank, world_size)
    device = get_device(local_rank)
    shard_name = f"shard_{rank}.safetensors"
    in_paths = [d / shard_name for d in input_dirs]
    _sum_my_shard(in_paths, output_dir / shard_name, device=device, divisor=divisor)
    if world_size > 1:
        dist.barrier()


def _sum_my_shard(
    in_paths: list[Path],
    out_path: Path,
    device: str,
    divisor: float = 1.0,
) -> None:
    """Sum all tensor dicts in ``in_paths`` and write to ``out_path``."""
    acc: dict[str, torch.Tensor] = {}
    for c, p in enumerate(in_paths):
        with safe_open(p, framework="pt", device=device) as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if c == 0:
                    acc[k] = t.clone()
                else:
                    acc[k].add_(t)
    if divisor != 1.0:
        for v in acc.values():
            v.div_(divisor)
    save_file({k: v.cpu() for k, v in acc.items()}, out_path)


def aggregate_segment_covariances(
    run_path: str | Path,
    method: str,
    n_segments: int,
    per_segment: int,
    distributed: DistributedConfig,
    *,
    resume: bool = False,
) -> None:
    """Sum per-checkpoint covariances per segment, then eigendecompose.

    Output layout::

        <run_path>/segment_{i}/<method>/
            activation_sharded/shard_*.safetensors
            gradient_sharded/shard_*.safetensors
            eigen_activation_sharded/shard_*.safetensors
            eigen_gradient_sharded/shard_*.safetensors
            total_processed.pt
    """
    logger = get_logger("aggregate_segment_covariances")
    base_run = Path(run_path)

    segments_to_process: list[int] = []
    for seg in range(n_segments):
        seg_dir = base_run / f"segment_{seg}"
        out_dir = seg_dir / method
        cov_done = (out_dir / "activation_sharded/shard_0.safetensors").exists()
        eigen_done = all((out_dir / f"eigen_{kind}").exists() for kind in _SHARD_KINDS)

        if resume and cov_done and eigen_done:
            logger.info(f"[seg {seg}] skip — cov + eigvecs both exist")
            continue
        if not resume and out_dir.exists():
            shutil.rmtree(out_dir)

        for i in range(per_segment):
            d = seg_dir / f"ckpt_{i}" / method
            if not d.exists():
                raise FileNotFoundError(
                    f"Expected per-checkpoint covariance dir {d} not found. "
                    "Did step 1 finish for this checkpoint?"
                )
        segments_to_process.append(seg)

    if not segments_to_process:
        logger.info("All segments already aggregated; nothing to do.")
        return

    launch_distributed_run(
        "aggregate_segment_covariances",
        _aggregate_cov_worker,
        [base_run, method, per_segment, segments_to_process],
        distributed,
    )


def _aggregate_cov_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    base_run: Path,
    method: str,
    per_segment: int,
    segments_to_process: list[int],
) -> None:
    """Sum cov shards + total_processed, then eigendecompose, in one launch."""
    init_dist(rank, local_rank, world_size)
    logger = get_logger("aggregate_segment_covariances")
    device = get_device(local_rank)
    shard_name = f"shard_{rank}.safetensors"

    for seg in segments_to_process:
        seg_dir = base_run / f"segment_{seg}"
        out_dir = seg_dir / method
        ckpt_method_dirs = [seg_dir / f"ckpt_{i}" / method for i in range(per_segment)]
        out_dir.mkdir(parents=True, exist_ok=True)

        cov_done = (out_dir / "total_processed.pt").exists()
        if not cov_done:
            logger.info(f"[seg {seg} rank {rank}] summing covariances -> {out_dir}")
            for kind in _SHARD_KINDS:
                (out_dir / kind).mkdir(parents=True, exist_ok=True)
                in_paths = [d / kind / shard_name for d in ckpt_method_dirs]
                _sum_my_shard(in_paths, out_dir / kind / shard_name, device=device)

            if rank == 0:
                total = None
                for d in ckpt_method_dirs:
                    t = torch.load(
                        d / "total_processed.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    total = t if total is None else total + t
                torch.save(total, out_dir / "total_processed.pt")

            if world_size > 1:
                dist.barrier()

        logger.info(f"[seg {seg} rank {rank}] eigendecomposing -> {out_dir}")
        total_processed = torch.load(
            out_dir / "total_processed.pt",
            map_location="cpu",
            weights_only=False,
        )
        for kind in _SHARD_KINDS:
            compute_eigendecomposition(
                str(out_dir / kind),
                total_processed=total_processed,
            )

        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.barrier()


def aggregate_segment_lambdas(
    run_path: str | Path,
    method: str,
    n_segments: int,
    per_segment: int,
    distributed: DistributedConfig,
    *,
    resume: bool = False,
    normalization: FisherNormalization = "document",
    input_subdir: str = "averaged_ev_correct_sharded",
    output_subdir: str = "eigenvalue_correction_sharded",
) -> None:
    """Sum per-checkpoint lambdas into per-segment lambda.

    ``normalization`` divides the sum by the pooled document or token count, so
    the segment eigenvalues are an expected Fisher rather than a total that
    grows with dataset size. See ``ApproxUnrollingConfig.fisher_normalization``.
    """
    logger = get_logger("aggregate_segment_lambdas")
    base_run = Path(run_path)

    for seg in range(n_segments):
        seg_dir = base_run / f"segment_{seg}"
        out_dir = seg_dir / method / output_subdir

        if out_dir.exists():
            if resume:
                logger.info(f"[seg {seg}] skip — exists at {out_dir}")
                continue
            shutil.rmtree(out_dir)

        input_dirs = [
            seg_dir / f"ckpt_{i}" / method / input_subdir for i in range(per_segment)
        ]
        for d in input_dirs:
            if not d.exists():
                raise FileNotFoundError(
                    f"Missing per-ckpt lambda dir {d}; did step 3 finish?"
                )

        divisor = lambda_denominator(input_dirs, normalization)
        logger.info(
            f"[seg {seg}] summing lambdas -> {out_dir} "
            f"(normalization={normalization}, divisor={divisor:g})"
        )
        sum_sharded_dirs(input_dirs, out_dir, distributed, divisor=divisor)
