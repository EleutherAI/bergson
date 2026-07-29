"""Fill in SOURCE hyperparameters from a bergson run's ``config.yaml``.

SOURCE was built for HF Trainer runs. Everything here is a fallback: explicit
config fields always win, so other trainers are unaffected.
"""

import json
from pathlib import Path
from typing import Any, Callable

import yaml

from ..config.config import ApproxUnrollingConfig, TrainingConfig
from ..config.config_io import CONFIG_FILENAME
from ..utils.logger import get_logger

EXPORT_DIRNAME = "exported"
"""Where export_checkpoints puts ``checkpoint-<N>`` dirs by default, and so the
first place discovery looks."""

LR_HISTORY_FILENAME = "log_history.json"
"""Per-step LRs in HF's ``log_history`` shape, written beside a run's
checkpoints -- the path the LR math already checks first."""

logger = get_logger(__name__)


def write_lr_history(
    save_dir: str | Path, schedule: Callable[[int], float], num_steps: int
) -> Path:
    """Record per-step LRs beside the checkpoints, from the ``schedule`` the
    optimizer was built with, so it is exact rather than reconstructed."""
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
    """Load the ``TrainingConfig`` a run was launched with. ``save_run_config``
    writes ``{command_name: {...}}``, so the single value is the config."""
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
    """Momentum beta a run trained with. bergson's SGD passes ``adam_beta1`` to
    ``torchopt.sgd``; AdamW's own preconditioner handles its first moment."""
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
    """Saved checkpoints in training order. Prefers exported ``checkpoint-<N>``
    dirs; finding only native ``step_<i>.ckpt`` raises, naming the export."""
    root = Path(trainer_run)
    # The export's default destination first, then the run root for an export
    # that was pointed there explicitly.
    for base in (root / EXPORT_DIRNAME, root):
        exported = sorted(
            (p for p in base.glob("checkpoint-*") if p.is_dir()),
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


def infer_trainer_run(checkpoints: list[str]) -> str:
    """The bergson run a checkpoint came from, or "" if it did not come from one.

    Only the two layouts we emit are considered -- ``<run>/exported/checkpoint-N``
    and ``<run>/checkpoint-N`` -- so an unrelated config.yaml further up the tree
    is never mistaken for the run that produced these checkpoints. Checkpoints
    from other trainers simply find nothing.
    """
    if not checkpoints:
        return ""
    first = Path(checkpoints[0])
    for candidate in (first.parent.parent, first.parent):
        if (candidate / CONFIG_FILENAME).is_file():
            return str(candidate)
    return ""


def resolve(cfg: ApproxUnrollingConfig) -> ApproxUnrollingConfig:
    """Fill unset fields from ``cfg.trainer_run``. A no-op when it is empty;
    never overwrites a field the caller set."""
    trainer_run = cfg.trainer_run or infer_trainer_run(cfg.checkpoints)
    if not trainer_run:
        # Not a bergson run; only normalize the momentum sentinel.
        if cfg.momentum is None:
            cfg.momentum = 0.0
        return cfg
    cfg.trainer_run = trainer_run

    training_cfg = load_training_config(trainer_run)
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
