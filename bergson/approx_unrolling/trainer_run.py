"""Fill in unset SOURCE configuration from a bergson run's ``config.yaml``
if present."""

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
    """Load the ``TrainingConfig`` a run was launched with.

    ``save_run_config`` writes ``{steps: [{command_name: {...}}], metadata: ...}``,
    so the training config is the first step's single value."""
    path = Path(trainer_run) / CONFIG_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; trainer_run must point at a bergson run "
            "directory (the one containing config.yaml and checkpoints/)."
        )

    with open(path) as f:
        loaded: Any = yaml.safe_load(f)

    if isinstance(loaded, dict) and "steps" in loaded:
        loaded = loaded["steps"]
    # Steps are a list of {command: payload}; take the first payload.
    if isinstance(loaded, list):
        if not loaded:
            raise ValueError(f"{path} is empty")
        loaded = loaded[0]
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError(f"{path} is not a bergson run config")

    payload: Any = next(iter(loaded.values()))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a bergson run config")

    try:
        return TrainingConfig.from_dict(payload, drop_extra_fields=True)
    except Exception as e:
        # An attribution run's config.yaml has the same shape, so reaching here
        # is expected; callers that guess at a run dir catch ValueError.
        raise ValueError(f"{path} is not a bergson training config: {e}") from e


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


def _ensure_exported(checkpoints: list[str]) -> list[str]:
    """Convert any raw DCP ``step_<n>.ckpt`` paths to the HF ``checkpoint-<n>``
    dirs SOURCE loads with from_pretrained, exporting on demand.

    A trainer checkpoint lives at ``<run>/checkpoints/step_<n>.ckpt`` and exports
    to ``<run>/exported/checkpoint-<n>``. Already-exported steps are reused, so
    this is idempotent across re-runs. Non-DCP paths (HF ids, exported dirs) pass
    through untouched.
    """
    # Import here: trainer_export imports this module, so a top-level import
    # would be circular.
    from ..utils.trainer_export import export_checkpoints

    resolved = list(checkpoints)
    # Group the DCP paths by their run so each run exports in a single pass
    # (export_checkpoints rebuilds the model once per call).
    todo: dict[Path, list[tuple[int, int]]] = {}
    for i, c in enumerate(checkpoints):
        p = Path(c)
        if not p.name.endswith(".ckpt"):
            continue
        step = int(p.name.removesuffix(".ckpt").removeprefix("step_"))
        run = p.parent.parent  # <run>/checkpoints/step_<n>.ckpt -> <run>
        dst = run / EXPORT_DIRNAME / f"checkpoint-{step}"
        resolved[i] = str(dst)
        if not dst.exists():
            todo.setdefault(run, []).append((i, step))

    for run, items in todo.items():
        steps = sorted({s for _, s in items})
        logger.info("Auto-exporting %s from %s", steps, run)
        export_checkpoints(run, steps=steps, overwrite=False)

    return resolved


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
    cfg.checkpoints = _ensure_exported(cfg.checkpoints)

    trainer_run = infer_trainer_run(cfg.checkpoints)
    if not trainer_run:
        # Not a bergson run; only normalize the momentum sentinel.
        if cfg.momentum is None:
            cfg.momentum = 0.0
        return cfg

    try:
        training_cfg = load_training_config(trainer_run)
    except ValueError as e:
        # A directory with a config.yaml is not necessarily a training run: an
        # attribution run writes one next to the checkpoints it produced. Infer
        # nothing from it rather than failing the pipeline.
        logger.warning("Ignoring %s as a trainer run: %s", trainer_run, e)
        if cfg.momentum is None:
            cfg.momentum = 0.0
        return cfg

    filled: list[str] = []

    if cfg.model_path is None:
        cfg.model_path = training_cfg.model
        filled.append(f"model_path={cfg.model_path!r}")

    if cfg.momentum is None:
        cfg.momentum = derive_momentum(training_cfg)
        filled.append(f"momentum={cfg.momentum}")

    if filled:
        logger.info("Filled from run %s: %s", trainer_run, ", ".join(filled))

    return cfg


def lr_history_path(checkpoints: list[str]) -> Path | None:
    """The LR history of the bergson run these checkpoints came from, if any."""
    run = infer_trainer_run(checkpoints)
    if not run:
        return None
    for candidate in (
        Path(run) / LR_HISTORY_FILENAME,
        Path(run) / "checkpoints" / LR_HISTORY_FILENAME,
    ):
        if candidate.is_file():
            return candidate
    return None
