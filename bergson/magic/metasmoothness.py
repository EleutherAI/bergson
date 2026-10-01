"""Empirical metasmoothness of the MAGIC training routine.

Implements Definition 2 of "Optimizing ML Training with Metagradient
Descent" (arXiv 2503.13751): train three models with data weights ``1``,
``1 + h*v`` and ``1 + 2h*v`` where ``v ~ N(0, I)`` over training docs, then
score the movement-weighted sign agreement between the two consecutive
finite-difference derivatives of the final parameters. A score near 1 means
the parameter response to data-weight perturbations is locally linear
(metasmooth); near 0 or negative means metagradients will predict poorly.

This costs three trainings and no leave-k-out bank, which is what makes it
usable as a search objective over smoothness knobs (eps_root, adam_eps, lr,
batch size) before committing to full LDS validation runs.
"""

import json
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer

from ..config.config import MetasmoothnessConfig
from ..distributed import DIST_TIMEOUT, launch_distributed_run
from ..utils.utils import dist_backend, dist_device_id, get_device, get_device_index
from ..utils.worker_utils import setup_data_pipeline
from .cli import attach_doc_ids_if_missing, shuffled_epochs
from .data_stream import DataStream, pad_dataset_to_batch_size
from .trainer import prepare_trainer


def metasmoothness_score(
    theta0: torch.Tensor,
    theta_h: torch.Tensor,
    theta_2h: torch.Tensor,
    *,
    reduce: bool = False,
) -> float:
    """Movement-weighted sign agreement of consecutive finite differences.

    ``sign(theta_2h - theta_h) . diag(d / |d|_1) . sign(theta_h - theta0)``
    with ``d = |theta_2h - theta0|`` (Def. 2 of arXiv 2503.13751; the ``/h``
    factors cancel inside ``sign``).

    The weighted sum and ``|d|_1`` are both plain sums over coordinates, so
    with ``reduce`` the arguments may be one rank's disjoint shard and the
    all-reduce returns exactly the value the unsharded vectors would give.
    """
    d = (theta_2h - theta0).abs()
    s1 = torch.sign(theta_h - theta0)
    s2 = torch.sign(theta_2h - theta_h)
    packed = torch.stack([(d * s1 * s2).sum(), d.sum()])
    if reduce:
        dist.all_reduce(packed)
    weighted, total = packed[0], packed[1]
    if total == 0:
        return 1.0
    return float(weighted / total)


def total_movement_l1(
    theta0: torch.Tensor, theta_2h: torch.Tensor, *, reduce: bool = False
) -> float:
    """``|theta_2h - theta0|_1``, summed across shards when ``reduce``."""
    moved = (theta_2h - theta0).abs().sum()
    if reduce:
        dist.all_reduce(moved)
    return float(moved)


def metasmoothness_worker(
    global_rank: int,
    rank: int,
    world_size: int,
    train_dataset,
    num_train_docs: int,
    run_cfg: MetasmoothnessConfig,
):
    if torch.cuda.is_available():
        torch.cuda.set_device(get_device_index(rank))

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            dist_backend(mixed=True),
            init_method=f"tcp://{addr}:{port}",
            device_id=dist_device_id(rank),
            rank=global_rank,
            timeout=DIST_TIMEOUT,
            world_size=world_size,
        )

    assert not getattr(run_cfg, "per_token", False)

    if run_cfg.fsdp and run_cfg.save_models:
        raise ValueError(
            "metasmoothness save_models is not supported with fsdp: the "
            "parameters are sharded DTensors and each rank would write its "
            "own slice as if it were the whole adapter."
        )

    assert run_cfg.batch_size % world_size == 0

    train_dataset, num_train_docs, padding = pad_dataset_to_batch_size(
        train_dataset, run_cfg.batch_size, num_train_docs, "Train", global_rank
    )

    stream = DataStream(
        train_dataset,
        run_cfg.batch_size,
        device=get_device(rank),
        input_key=run_cfg.data.prompt_column,
        weight_shape=(num_train_docs,),
    )
    schedule = run_cfg.lr_schedule.get_schedule(len(stream))

    # Same v on every rank; pad slots are never perturbed.
    gen = torch.Generator().manual_seed(run_cfg.direction_seed)
    v = torch.randn(num_train_docs, generator=gen)
    padding.zero_weights(v)

    thetas: list[torch.Tensor] = []
    sharded = False
    for k in range(3):
        weights = 1.0 + run_cfg.fd_step * k * v
        padding.zero_weights(weights)
        stream.weights.data.copy_(weights.to(stream.weights.device))

        torch.manual_seed(run_cfg.seed)
        torch.cuda.manual_seed_all(run_cfg.seed)

        trainer, fwd_state, model = prepare_trainer(run_cfg, rank, schedule)
        fwd_state.detach_()
        fwd_state = trainer.train(
            fwd_state,
            stream,
            inplace=True,
            fsdp=run_cfg.fsdp,
            max_grad_norm=run_cfg.max_grad_norm,
            grad_accum_steps=run_cfg.grad_accum_steps,
        )

        # Every rank keeps its own parameters. Under fsdp those are disjoint
        # shards, and the score reduces across them, so all ranks take part.
        sharded = any(isinstance(p, DTensor) for p in fwd_state.params.values())
        theta = torch.cat(
            [
                (p.to_local() if isinstance(p, DTensor) else p)
                .detach()
                .float()
                .flatten()
                for p in fwd_state.params.values()
            ]
        )
        # Shards are small; a replicated vector does not need to sit in VRAM.
        thetas.append(theta if sharded else theta.cpu())

        if global_rank == 0:
            print(f"[metasmoothness] finished training {k + 1}/3 (w = 1 + {k}*h*v)")

        if k == 0 and run_cfg.save_models and global_rank == 0:
            out_dir = os.path.join(run_cfg.run_path, "model")
            os.makedirs(out_dir, exist_ok=True)
            with fwd_state.activate(model), torch.no_grad():
                model.save_pretrained(out_dir, safe_serialization=True)
            AutoTokenizer.from_pretrained(
                run_cfg.tokenizer or run_cfg.model
            ).save_pretrained(out_dir)
        del trainer, fwd_state, model

    # Both reductions are collectives, so they run on every rank.
    score = metasmoothness_score(*thetas, reduce=sharded)
    movement = total_movement_l1(thetas[0], thetas[2], reduce=sharded)

    if global_rank == 0:
        result = {
            "score": score,
            "fd_step": run_cfg.fd_step,
            "direction_seed": run_cfg.direction_seed,
            "total_movement_l1": movement,
        }
        print(f"[metasmoothness] score = {score:.4f} (h={run_cfg.fd_step})")
        os.makedirs(run_cfg.run_path, exist_ok=True)
        with open(os.path.join(run_cfg.run_path, "metasmoothness.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"[metasmoothness] saved to {run_cfg.run_path}/metasmoothness.json")


def run_metasmoothness(run_cfg: MetasmoothnessConfig):
    """Evaluate empirical metasmoothness for the given training config.

    Mirrors ``run_magic``'s data pipeline (doc_ids, shuffle seed) so the
    three trainings are exactly the ones a MAGIC run would perform.
    """
    os.makedirs(run_cfg.run_path, exist_ok=True)

    train_ds, train_n = setup_data_pipeline(run_cfg)
    train_ds = attach_doc_ids_if_missing(train_ds)
    train_ds = shuffled_epochs(train_ds, run_cfg.seed, max(1, run_cfg.num_epochs))

    launch_distributed_run(
        "metasmoothness",
        metasmoothness_worker,
        [train_ds, train_n, run_cfg],
        run_cfg.distributed,
    )
