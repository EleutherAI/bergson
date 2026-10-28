"""LiSSA (Agarwal et al. 2017, https://arxiv.org/abs/1602.03943) estimates of
damped Gauss-Newton inverse-Hessian-vector products, the influence estimator of
Bae et al. (2022, https://arxiv.org/abs/2209.05364)."""

import json
import math
import shutil
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from bergson.build import build_query
from bergson.cli.commands import Score
from bergson.config import IndexConfig, LissaConfig, PreprocessConfig, ScoreConfig
from bergson.config.config_io import save_run_config
from bergson.data import column_offsets, create_index, load_gradients
from bergson.distributed import launch_distributed_run
from bergson.hessians.astra import GaussNewtonProduct
from bergson.hessians.pipeline import _step_complete, _timed
from bergson.score.score import score_dataset
from bergson.utils.logger import get_logger
from bergson.utils.step_state import partial_path, promote_step
from bergson.utils.utils import get_device, numpy_to_tensor
from bergson.utils.worker_utils import (
    setup_data_pipeline,
    setup_model_and_peft,
    validate_run_path,
)


def lissa(
    product: GaussNewtonProduct,
    q: dict[str, Tensor],
    num_docs: int,
    cfg: LissaConfig,
    gen: torch.Generator,
    label: str = "",
) -> dict[str, Tensor]:
    """Mean over repeats of ``sum_t (I - (G + damping I) / scale)^t q / scale``."""
    logger = get_logger("Lissa")
    out = {n: torch.zeros_like(v) for n, v in q.items()}
    for repeat in range(cfg.repeats):
        h = {n: v.clone() for n, v in q.items()}
        for step in range(cfg.depth):
            indices = torch.randperm(num_docs, generator=gen)[: cfg.batch_size]
            gh = product(h, indices.tolist())
            for n in h:
                h[n] += q[n] - (gh[n] + cfg.damping * h[n]) / cfg.scale
            if (step + 1) % 500 == 0:
                norm = torch.stack([v.norm() for v in h.values()]).norm()
                logger.info(
                    f"{label} repeat {repeat} step {step + 1}: "
                    f"estimate norm {norm.item() / cfg.scale:.6g}"
                )
        for n in out:
            out[n] += h[n] / (cfg.scale * cfg.repeats)
    return out


class Lissa:
    """Estimates a subset of the queries' inverse-Hessian-vector products."""

    def __init__(
        self,
        query_path: str,
        run_path: str,
        index_cfg: IndexConfig,
        lissa_cfg: LissaConfig,
        device: str,
    ):
        self.query_path = query_path
        self.run_path = run_path
        self.cfg = lissa_cfg
        self.device = device

        with open(Path(query_path) / "info.json") as f:
            grad_sizes = json.load(f)["grad_sizes"]
        self.names = list(grad_sizes)
        self.offsets = column_offsets(grad_sizes)

        model, _ = setup_model_and_peft(index_cfg, attn_implementation="eager")
        model.eval()
        data, _ = setup_data_pipeline(index_cfg)
        self.hvp = GaussNewtonProduct(
            model,
            data,
            index_cfg,
            self.names,
            micro_batch_size=lissa_cfg.micro_batch_size,
        )
        self.num_docs = len(data)

    def run(self, rows: range):
        query = load_gradients(self.query_path)
        out = np.memmap(
            Path(self.run_path) / "gradients.bin",
            dtype=query.dtype,
            mode="r+",
            shape=query.shape,
        )
        for row in rows:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="The given NumPy array is not writable"
                )
                q = {
                    n: numpy_to_tensor(query[row, slice(*self.offsets[n])]).to(
                        self.device, torch.float32
                    )
                    for n in self.names
                }
            gen = torch.Generator().manual_seed(self.cfg.seed * 1_000_003 + row)
            x = lissa(self.hvp, q, self.num_docs, self.cfg, gen, f"query {row}")
            for n in self.names:
                out[row, slice(*self.offsets[n])] = x[n].cpu().numpy()
            out.flush()


def lissa_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    query_path: str,
    run_path: str,
    index_cfg: IndexConfig,
    lissa_cfg: LissaConfig,
):
    """Solve this rank's share of the queries; no process group is needed."""
    device = get_device(local_rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    num_queries = load_gradients(query_path).shape[0]
    per_rank = math.ceil(num_queries / world_size)
    rows = range(rank * per_rank, min((rank + 1) * per_rank, num_queries))
    Lissa(query_path, run_path, index_cfg, lissa_cfg, device).run(rows)


def lissa_pipeline(
    index_cfg: IndexConfig,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
    lissa_cfg: LissaConfig,
):
    """Build the query gradients, estimate their damped Gauss-Newton
    inverse-Hessian-vector products with LiSSA, and score the training data
    against them."""
    if preprocess_cfg.unit_normalize:
        raise ValueError("LiSSA does not support preprocess_cfg.unit_normalize.")
    if index_cfg.projection_dim > 0:
        raise ValueError("LiSSA needs projection_dim=0.")
    if index_cfg.distributed.nnode > 1:
        raise ValueError("LiSSA runs on a single node.")
    if index_cfg.distributed.device_map_auto:
        raise ValueError("LiSSA doesn't support device_map_auto.")

    run_path = index_cfg.run_path
    query_path = f"{run_path}/query"
    solution_path = f"{run_path}/lissa_query"
    scores_path = f"{run_path}/scores"
    resume = lissa_cfg.resume
    durations: dict[str, float] = {}

    def _validate(cfg: IndexConfig):
        if resume and cfg.partial_run_path.exists():
            return
        validate_run_path(cfg)

    print(f"Step 1/3: Building query gradient(s) ({lissa_cfg.query.aggregation})...")
    if lissa_cfg.query.path:
        query_path = lissa_cfg.query.path
    elif not _step_complete(query_path, resume):
        with _timed("step1_build_query", durations):
            query_cfg = deepcopy(index_cfg)
            query_cfg.run_path = query_path
            _validate(query_cfg)
            build_query(query_cfg, lissa_cfg.query, PreprocessConfig())

    print(f"Step 2/3: LiSSA ({lissa_cfg.repeats} x {lissa_cfg.depth} terms)...")
    if not _step_complete(solution_path, resume):
        with _timed("step2_lissa", durations):
            query = load_gradients(query_path)
            with open(f"{query_path}/info.json") as f:
                grad_sizes = json.load(f)["grad_sizes"]
            part = partial_path(solution_path)
            create_index(part, query.shape[0], grad_sizes, query.dtype)
            shutil.copy(f"{query_path}/processor_config.yaml", part)
            launch_distributed_run(
                "lissa",
                lissa_worker,
                [query_path, str(part), index_cfg, lissa_cfg],
                index_cfg.distributed,
            )
            promote_step(solution_path)

    print("Step 3/3: Scoring training data against the LiSSA solutions...")
    if not _step_complete(scores_path, resume):
        score_index_cfg = deepcopy(index_cfg)
        score_index_cfg.run_path = scores_path
        score_cfg.query_path = solution_path
        score_cfg.higher_is_better = True
        _validate(score_index_cfg)
        save_run_config(
            Score(score_cfg, score_index_cfg, preprocess_cfg),
            score_index_cfg.partial_run_path,
        )
        score_dataset(score_index_cfg, score_cfg, preprocess_cfg)

    print(f"Done! Scores saved to: {scores_path}")
    if durations:
        print(f"Step timings (s): {durations}")
