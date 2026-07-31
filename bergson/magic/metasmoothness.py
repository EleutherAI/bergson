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

from ..config.config import MetasmoothnessConfig
from ..distributed import launch_distributed_run
from ..utils.utils import get_device, get_device_index
from ..utils.worker_utils import setup_data_pipeline
from .cli import attach_doc_ids_if_missing
from .data_stream import DataStream, pad_dataset_to_batch_size
from .trainer import prepare_trainer


def metasmoothness_score(
    theta0: torch.Tensor, theta_h: torch.Tensor, theta_2h: torch.Tensor
) -> float:
    """Movement-weighted sign agreement of consecutive finite differences.

    ``sign(theta_2h - theta_h) . diag(d / |d|_1) . sign(theta_h - theta0)``
    with ``d = |theta_2h - theta0|`` (Def. 2 of arXiv 2503.13751; the ``/h``
    factors cancel inside ``sign``).
    """
    d = (theta_2h - theta0).abs()
    total = d.sum()
    if total == 0:
        return 1.0
    s1 = torch.sign(theta_h - theta0)
    s2 = torch.sign(theta_2h - theta_h)
    return float((d / total * s1 * s2).sum())


def metasmoothness_group_scores(
    theta0: torch.Tensor,
    theta_h: torch.Tensor,
    theta_2h: torch.Tensor,
    group_ids: torch.Tensor,
    group_names: list[str],
) -> dict:
    """Per-optimizer-path breakdown of the metasmoothness score.

    Muon routes 2D parameters through Newton-Schulz and 1D parameters through
    an AdamW fallback; ``eps_root`` only enters the AdamW branch. Since the
    global score weights coordinates by their L1 movement, the 2D group (99.9%
    of GPT-2's parameters) can dominate it and mask a non-smooth 1D group.

    The decomposition is exact: ``score == sum(share_g * score_g)``.
    """
    d = (theta_2h - theta0).abs()
    total = d.sum()
    contrib = d * torch.sign(theta_h - theta0) * torch.sign(theta_2h - theta_h)

    out = {}
    for gid, name in enumerate(group_names):
        m = group_ids == gid
        dg = d[m].sum()
        out[name] = {
            "score": float(contrib[m].sum() / dg) if dg > 0 else 1.0,
            "movement_share": float(dg / total) if total > 0 else 0.0,
            "movement_l1": float(dg),
            "numel": int(m.sum()),
        }
    return out


def metasmoothness_worker(
    global_rank: int,
    rank: int,
    world_size: int,
    train_dataset,
    num_train_docs: int,
    run_cfg: MetasmoothnessConfig,
):
    torch.cuda.set_device(get_device_index(rank))

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(get_device(rank)),
            rank=global_rank,
            world_size=world_size,
        )

    assert not run_cfg.fsdp, "metasmoothness does not support FSDP parameters"
    assert not getattr(run_cfg, "per_token", False)

    if run_cfg.num_epochs > 1:
        train_dataset = train_dataset.repeat(run_cfg.num_epochs)
    assert run_cfg.batch_size % world_size == 0

    train_dataset, num_train_docs, pad_count, weight_pad_count = (
        pad_dataset_to_batch_size(
            train_dataset, run_cfg.batch_size, num_train_docs, "Train", global_rank
        )
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
    if pad_count:
        v[-weight_pad_count:] = 0.0

    thetas: list[torch.Tensor] = []
    theta_init: torch.Tensor | None = None
    group_ids: torch.Tensor | None = None
    group_names: list[str] = []
    for k in range(3):
        weights = 1.0 + run_cfg.fd_step * k * v
        if pad_count:
            weights[-weight_pad_count:] = 0.0
        stream.weights.data.copy_(weights.to(stream.weights.device))

        torch.manual_seed(run_cfg.seed)
        torch.cuda.manual_seed_all(run_cfg.seed)

        trainer, fwd_state, model = prepare_trainer(run_cfg, rank, schedule)
        fwd_state.detach_()
        if k == 0 and global_rank == 0:
            theta_init = torch.cat(
                [p.detach().float().cpu().flatten() for p in fwd_state.params.values()]
            )
            if run_cfg.save_models:
                os.makedirs(run_cfg.run_path, exist_ok=True)
                torch.save(
                    {n: p.detach().float().cpu() for n, p in fwd_state.params.items()},
                    os.path.join(run_cfg.run_path, "theta_init.pt"),
                )
        fwd_state = trainer.train(fwd_state, stream, inplace=True, fsdp=run_cfg.fsdp)

        if global_rank == 0:
            theta = torch.cat(
                [p.detach().float().cpu().flatten() for p in fwd_state.params.values()]
            )
            thetas.append(theta)
            if group_ids is None and run_cfg.optimizer == "muon":
                # Muon's own split: ndim==2 -> Newton-Schulz, else AdamW fallback
                # (the only branch eps_root enters). See optim.py muon().
                group_names = ["muon_2d", "adamw_1d"]
                group_ids = torch.cat(
                    [
                        torch.full(
                            (p.numel(),), 0 if p.ndim == 2 else 1, dtype=torch.uint8
                        )
                        for p in fwd_state.params.values()
                    ]
                )
            if run_cfg.save_models and k == 0:
                # Only the unperturbed run: the k>0 models are finite-difference
                # probes, not models anyone wants to reuse.
                path = os.path.join(run_cfg.run_path, "theta_final.pt")
                torch.save(
                    {n: p.detach().float().cpu() for n, p in fwd_state.params.items()},
                    path,
                )
                print(f"[metasmoothness] saved {path}")
            print(f"[metasmoothness] finished training {k + 1}/3 (w = 1 + {k}*h*v)")
        del trainer, fwd_state, model

    if global_rank == 0:
        score = metasmoothness_score(*thetas)
        movement = float((thetas[2] - thetas[0]).abs().sum())
        result = {
            "score": score,
            "fd_step": run_cfg.fd_step,
            "direction_seed": run_cfg.direction_seed,
            "total_movement_l1": movement,
        }
        if theta_init is not None:
            # Relative update norms of the unperturbed run, matching the ΔL1/ΔL2
            # convention in LDS_RESULTS (‖θ_final − θ_init‖ / ‖θ_init‖).
            delta = thetas[0] - theta_init
            result["delta_l1"] = float(delta.abs().sum() / theta_init.abs().sum())
            result["delta_l2"] = float(delta.norm() / theta_init.norm())
            print(
                f"[metasmoothness] deltaL1 = {result['delta_l1']:.4g}  "
                f"deltaL2 = {result['delta_l2']:.4g}"
            )
        print(f"[metasmoothness] score = {score:.4f} (h={run_cfg.fd_step})")
        if group_ids is not None:
            groups = metasmoothness_group_scores(
                thetas[0], thetas[1], thetas[2], group_ids, group_names
            )
            result["groups"] = groups
            for name, g in groups.items():
                print(
                    f"[metasmoothness]   {name}: score = {g['score']:.4f}  "
                    f"share = {g['movement_share']:.4f}  numel = {g['numel']:,}"
                )
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
    train_ds = train_ds.shuffle(seed=run_cfg.seed)

    launch_distributed_run(
        "metasmoothness",
        metasmoothness_worker,
        [train_ds, train_n, run_cfg],
        run_cfg.distributed,
    )
