"""Fill in unset SOURCE configuration from a bergson run's ``config.yaml``
if present."""

from pathlib import Path
from typing import Any

import yaml

from ..config.config import ApproxUnrollingConfig, TrainingConfig
from ..config.config_io import CONFIG_FILENAME
from ..magic.trainer import LR_HISTORY_FILENAME
from ..utils.logger import get_logger

logger = get_logger(__name__)


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
    """Fill unset fields from the training run the checkpoints came from;
    never overwrites a field the caller set."""
    # Local import: trainer_export imports load_training_config from here.
    from ..utils.trainer_export import ensure_exported

    cfg.checkpoints = ensure_exported(cfg.checkpoints)

    trainer_run = infer_trainer_run(cfg.checkpoints)
    if not trainer_run:
        # Not a bergson run; only normalize the momentum sentinel.
        if cfg.momentum is None:
            cfg.momentum = 0.0
        return cfg

    try:
        training_cfg = load_training_config(trainer_run)
    except ValueError as e:
        # trainer_run may be a config.yaml for something other than a
        # training run - infer nothing.
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
