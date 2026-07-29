"""Export a bergson training run's checkpoints to HF-format model directories.

The trainer saves distributed checkpoints as ``<run>/checkpoints/step_<i>.ckpt``
(torch.distributed.checkpoint), which SOURCE cannot read -- it loads each
checkpoint with ``from_pretrained``. This turns a run's DCP checkpoints into
``<out>/checkpoint-<i>/`` directories that ``from_pretrained`` accepts, carrying
each step's optimizer state and the run's LR history across so the SOURCE
pipeline finds everything where it expects it.

Runs offline against a finished run rather than during training, so a run only
pays the extra disk for the trajectories actually being attributed.
"""

import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..approx_unrolling.trainer_run import (
    LR_HISTORY_FILENAME,
    load_training_config,
)
from ..config.config import TrainingConfig
from ..magic.trainer import prepare_trainer
from .logger import get_logger
from .optimizer_placement import OPTIMIZER_STATE_FILE

logger = get_logger(__name__)


def sorted_dcp_checkpoints(checkpoints_dir: str | Path) -> list[tuple[int, Path]]:
    """``(step, path)`` for each ``step_<i>.ckpt``, ordered by step."""
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.is_dir():
        raise NotADirectoryError(f"{checkpoints_dir} is not a directory")

    found = []
    for entry in checkpoints_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("step_"):
            stem = entry.name.removesuffix(".ckpt").removeprefix("step_")
            if stem.isdigit():
                found.append((int(stem), entry))

    return sorted(found, key=lambda pair: pair[0])


def export_checkpoints(
    run_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    training_cfg: TrainingConfig | None = None,
    steps: list[int] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Export ``<run_path>/checkpoints/step_<i>.ckpt`` to ``checkpoint-<i>/`` dirs.

    ``training_cfg`` defaults to the run's own ``config.yaml``, so a finished run
    needs no arguments beyond its path. Pass one explicitly to export a run whose
    config lives elsewhere.

    ``steps`` restricts the export to those step indices -- useful because SOURCE
    needs only ``segments * per_segment`` checkpoints, and each export is a full
    model copy.

    Returns the exported directories in step order.
    """
    run_path = Path(run_path)
    out_dir = Path(out_dir) if out_dir is not None else run_path / "exported"
    cfg = training_cfg if training_cfg is not None else load_training_config(run_path)

    available = sorted_dcp_checkpoints(run_path / "checkpoints")
    if not available:
        raise FileNotFoundError(
            f"No step_<i>.ckpt directories under {run_path / 'checkpoints'}; "
            "train with a save_mode that writes checkpoints."
        )

    if steps is not None:
        by_step = dict(available)
        missing = [s for s in steps if s not in by_step]
        if missing:
            raise FileNotFoundError(
                f"Requested steps {missing} were not saved; available: "
                f"{[s for s, _ in available]}"
            )
        selected = [(s, by_step[s]) for s in sorted(steps)]
    else:
        selected = available

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [out_dir / f"checkpoint-{s}" for s, _ in selected]
    clashes = [p for p in existing if p.exists()]
    if clashes and not overwrite:
        raise FileExistsError(
            f"{[str(c) for c in clashes]} already exist; pass overwrite=True."
        )

    # One model/state is built and reloaded per checkpoint, rather than one per
    # step: the state's tensors are loaded in place by TrainerState.load.
    trainer, state, model = prepare_trainer(cfg, rank=0, schedule=lambda step: 0.0)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)

    exported: list[Path] = []
    for step, ckpt in selected:
        state.load(str(ckpt))

        dst = out_dir / f"checkpoint-{step}"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

        with state.activate(model), torch.no_grad():
            model.save_pretrained(str(dst), safe_serialization=True)
        tokenizer.save_pretrained(str(dst))

        # SOURCE's Adam variant reads <checkpoint>/optimizer.pt; carry the
        # trainer's sibling file in under the name it looks for.
        sibling = ckpt.parent / f"step_{step}.optimizer.pt"
        if sibling.is_file():
            shutil.copy2(sibling, dst / OPTIMIZER_STATE_FILE)

        exported.append(dst)
        logger.info("Exported %s -> %s", ckpt.name, dst)

    # The LR history sits beside the checkpoints; SOURCE looks for it beside the
    # *exported* ones, since that is the parent of cfg.checkpoints[0].
    history = run_path / "checkpoints" / LR_HISTORY_FILENAME
    if history.is_file():
        shutil.copy2(history, out_dir / LR_HISTORY_FILENAME)

    del trainer, state, model
    return exported
