"""Fill in SOURCE hyperparameters from a bergson training run.

SOURCE was built around HF Trainer runs: it infers per-step learning rates from
``log_history.json`` / ``trainer_state.json`` and reads steps out of
``checkpoint-<N>`` directory names. A bergson run has the same information in
its own form -- ``config.yaml`` plus the checkpoint schedule -- so this module
maps one onto the other.

Everything here is a *fallback*. An explicitly configured field always wins, so
checkpoints produced by any other trainer keep working exactly as before by
setting ``lr_list``/``step_size_list``/``momentum`` by hand and leaving
``trainer_run`` empty.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable

import yaml

from ..config.config import ApproxUnrollingConfig, TrainingConfig
from ..config.config_io import CONFIG_FILENAME
from ..utils.logger import get_logger

LR_HISTORY_FILENAME = "log_history.json"
"""Per-step learning rates, in HF Trainer's ``log_history`` shape.

Written next to a bergson run's checkpoints so
:func:`~bergson.approx_unrolling.approx_unrolling_math.compute_lr_times_steps_per_segment`
picks it up through the path it already checks first, with no bergson-specific
branch in the LR math.
"""

logger = get_logger(__name__)


def write_lr_history(
    save_dir: str | Path, schedule: Callable[[int], float], num_steps: int
) -> Path:
    """Record the run's realized per-step learning rates beside its checkpoints.

    ``schedule`` is the step -> lr callable the optimizer was built with, so
    this is the exact schedule that ran rather than a reconstruction of it.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    history = [
        {"step": step, "learning_rate": float(schedule(step))}
        for step in range(num_steps)
    ]
    path = save_dir / LR_HISTORY_FILENAME
    with open(path, "w") as f:
        json.dump(history, f)
    return path


def load_training_config(trainer_run: str | Path) -> TrainingConfig:
    """Load the ``TrainingConfig`` a bergson run was launched with.

    ``save_run_config`` writes ``config.yaml`` as ``{command_name: {...}}``, so
    the single value is the run's config regardless of which command wrote it.
    """
    path = Path(trainer_run) / CONFIG_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; trainer_run must point at a bergson run "
            "directory (the one containing config.yaml and checkpoints/)."
        )

    with open(path) as f:
        loaded = yaml.safe_load(f)

    # One-step configs are a list of {command: payload}; take the first payload.
    if isinstance(loaded, list):
        if not loaded:
            raise ValueError(f"{path} is empty")
        loaded = loaded[0]
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError(f"{path} is not a bergson run config")

    payload: Any = next(iter(loaded.values()))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a bergson run config")

    return TrainingConfig.from_dict(payload, drop_extra_fields=True)


def derive_momentum(training_cfg: TrainingConfig) -> float:
    """Heavy-ball momentum beta that a bergson run actually trained with.

    bergson's SGD passes ``adam_beta1`` as ``torchopt.sgd``'s momentum (see
    ``prepare_trainer``), so the SOURCE-relevant beta is ``adam_beta1`` for SGD
    and ``0.0`` for AdamW, whose own preconditioner already accounts for its
    first moment.
    """
    match training_cfg.optimizer:
        case "sgd":
            return float(training_cfg.adam_beta1)
        case "adamw":
            return 0.0
        case other:
            # Muon routes 2D params through Newton-Schulz, which the unrolling
            # derivation does not cover; don't invent a scaling for it.
            logger.warning(
                "Cannot derive a SOURCE momentum for optimizer %r; using 0.0. "
                "Set ApproxUnrollingConfig.momentum explicitly if that is wrong.",
                other,
            )
            return 0.0


def discover_checkpoints(trainer_run: str | Path) -> list[str]:
    """The run's saved checkpoints, in training order.

    Prefers exported HF-format ``checkpoint-<N>`` directories (what SOURCE can
    actually load) and falls back to the trainer's native ``step_<i>.ckpt``, so
    the error surfaced to the caller names the export step rather than an empty
    list.
    """
    root = Path(trainer_run)
    exported = sorted(
        (p for p in root.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if exported:
        return [str(p) for p in exported]

    ckpt_dir = root / "checkpoints"
    native = sorted(
        (p for p in ckpt_dir.glob("step_*.ckpt") if p.is_dir()),
        key=lambda p: int(p.name.removesuffix(".ckpt").removeprefix("step_")),
    )
    if native:
        raise FileNotFoundError(
            f"{trainer_run} has {len(native)} native checkpoints under "
            f"{ckpt_dir} but no exported checkpoint-<N> directories. SOURCE "
            "loads models with from_pretrained, so export them first with "
            "bergson.utils.trainer_export.export_checkpoints(run_path, out_dir)."
        )

    raise FileNotFoundError(
        f"No checkpoints found under {trainer_run}. Train with a save_mode "
        "that writes them, then export with "
        "bergson.utils.trainer_export.export_checkpoints."
    )


def resolve(cfg: ApproxUnrollingConfig) -> ApproxUnrollingConfig:
    """Return ``cfg`` with unset fields filled in from ``cfg.trainer_run``.

    A no-op when ``trainer_run`` is empty, so configs for other trainers pass
    through untouched. Fields already set are never overwritten -- this only
    ever supplies a value the caller did not give.
    """
    if not cfg.trainer_run:
        # Nothing to derive from; only normalize the momentum sentinel.
        if cfg.momentum is None:
            cfg.momentum = 0.0
        return cfg

    training_cfg = load_training_config(cfg.trainer_run)
    filled: list[str] = []

    if not cfg.checkpoints:
        cfg.checkpoints = discover_checkpoints(cfg.trainer_run)
        filled.append(f"checkpoints ({len(cfg.checkpoints)})")

    if cfg.model_path is None:
        cfg.model_path = training_cfg.model
        filled.append(f"model_path={cfg.model_path!r}")

    if cfg.momentum is None:
        cfg.momentum = derive_momentum(training_cfg)
        filled.append(f"momentum={cfg.momentum}")

    if filled:
        logger.info(
            "Filled from trainer_run %s: %s", cfg.trainer_run, ", ".join(filled)
        )

    return cfg


def lr_history_path(cfg: ApproxUnrollingConfig) -> Path | None:
    """Where a bergson run's LR history lives, if this config names one."""
    if not cfg.trainer_run:
        return None
    for candidate in (
        Path(cfg.trainer_run) / LR_HISTORY_FILENAME,
        Path(cfg.trainer_run) / "checkpoints" / LR_HISTORY_FILENAME,
    ):
        if candidate.is_file():
            return candidate
    return None


def _checkpoint_dir_step(path: str | os.PathLike) -> int | None:
    """Step index named by a checkpoint dir, across the layouts we emit/accept."""
    name = Path(path).name
    if name.startswith("checkpoint-") and name.removeprefix("checkpoint-").isdigit():
        return int(name.removeprefix("checkpoint-"))
    stem = name.removesuffix(".ckpt")
    if stem.startswith("step_") and stem.removeprefix("step_").isdigit():
        return int(stem.removeprefix("step_"))
    if name.isdigit():
        return int(name)
    return None
