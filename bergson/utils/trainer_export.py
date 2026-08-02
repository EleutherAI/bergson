"""Export a run's DCP ``step_<i>.ckpt`` checkpoints to HF-compatible
``checkpoint-<i>/`` model dirs."""

import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..approx_unrolling.train_cfg_io import load_training_config
from ..config.config import TrainingConfig
from ..magic.trainer import prepare_trainer
from .logger import get_logger

EXPORT_DIRNAME = "exported"
"""Where export_checkpoints puts ``checkpoint-<N>`` dirs by default."""

OPTIMIZER_STATE_FILE = "optimizer.pt"
"""Second moments, written inside the checkpoint dir they belong to -- both by
the trainer and here, so the file travels with its checkpoint and never has to
be matched back to one."""

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
    """Export ``step_<i>.ckpt`` to ``checkpoint-<i>/`` HF dirs, in step order.

    ``training_cfg`` defaults to the run's ``config.yaml``. ``steps`` limits which
    checkpoints are exported.
    """
    run_path = Path(run_path)
    out_dir = Path(out_dir) if out_dir is not None else run_path / EXPORT_DIRNAME
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

    # One model and state is built and reloaded per checkpoint.
    # The state's tensors are loaded in place by TrainerState.load.
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

        state_file = ckpt / OPTIMIZER_STATE_FILE
        if state_file.is_file():
            shutil.copy2(state_file, dst / OPTIMIZER_STATE_FILE)

        exported.append(dst)
        logger.info("Exported %s -> %s", ckpt.name, dst)

    del trainer, state, model
    return exported


def ensure_exported(checkpoints: list[str]) -> list[str]:
    """Convert any raw DCP ``step_<n>.ckpt`` paths to the HF ``checkpoint-<n>``
    dirs SOURCE loads with from_pretrained, exporting on demand.

    A trainer checkpoint lives at ``<run>/checkpoints/step_<n>.ckpt`` and exports
    to ``<run>/exported/checkpoint-<n>``. Already-exported steps are reused.
    """
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
