"""Place trainer-written optimizer states where attribution methods look for them.

The trainer writes each snapshot's second moments as a *sibling* file,
``<save_dir>/step_<i>.optimizer.pt``, next to that step's ``step_<i>.ckpt``
(see :meth:`bergson.magic.trainer.Trainer.train`). Both attribution consumers
instead expect ``optimizer.pt`` *inside* a directory:

- SOURCE / approximate unrolling reads ``<checkpoint>/optimizer.pt`` for every
  checkpoint when ``ApproxUnrollingConfig.use_adam_preconditioner`` is set.
- TrackStar (and any ``build``) resolves ``PreprocessConfig.optimizer_state``
  via :func:`bergson.utils.load_from_optimizer.load_optimizer`, which accepts a
  directory and loads ``optimizer.pt`` from inside it.

Exporting a run to HF-format checkpoint dirs does not carry the sibling files
across, which is what previously forced a manual copy. :func:`place_optimizer_states`
does that placement, and satisfies both consumers with one call since they agree
on the ``<dir>/optimizer.pt`` layout.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Literal, Sequence

OPTIMIZER_STATE_FILE = "optimizer.pt"
"""Filename both SOURCE and TrackStar look for inside a checkpoint directory."""

_STEP_OPTIMIZER_RE = re.compile(r"^step_(\d+)\.optimizer\.pt$")
_STEP_DIR_RE = re.compile(r"step_(\d+)")

PlacementMode = Literal["copy", "symlink", "hardlink", "move"]


def sorted_optimizer_states(save_dir: str | Path) -> list[tuple[int, Path]]:
    """Return ``(step, path)`` for each ``step_<i>.optimizer.pt``, sorted by step.

    Mirrors :func:`bergson.data.sorted_checkpoints`, which does the same for the
    ``step_<i>.ckpt`` directories these files sit beside.
    """
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        raise NotADirectoryError(f"{save_dir} is not a directory")

    found = []
    for name in os.listdir(save_dir):
        match = _STEP_OPTIMIZER_RE.match(name)
        if match:
            found.append((int(match.group(1)), save_dir / name))

    return sorted(found, key=lambda pair: pair[0])


def _step_of(path: Path) -> int | None:
    """Step index named by a checkpoint directory, or None if it names none."""
    match = _STEP_DIR_RE.search(path.name)
    return int(match.group(1)) if match else None


def _place(src: Path, dst: Path, mode: PlacementMode) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    match mode:
        case "copy":
            shutil.copy2(src, dst)
        case "move":
            shutil.move(str(src), str(dst))
        case "symlink":
            # Relative, so an exported run stays valid if the tree is moved.
            dst.symlink_to(os.path.relpath(src, dst.parent))
        case "hardlink":
            os.link(src, dst)
        case other:
            raise ValueError(f"Unsupported placement mode: {other}")


def place_optimizer_states(
    save_dir: str | Path,
    checkpoints: Sequence[str | Path],
    *,
    mode: PlacementMode = "symlink",
    overwrite: bool = False,
) -> dict[Path, Path]:
    """Put an ``optimizer.pt`` inside each checkpoint dir, from ``save_dir``.

    ``save_dir`` is the trainer's output directory, holding the
    ``step_<i>.optimizer.pt`` files written when
    ``TrainingConfig.save_optimizer_state`` is set. ``checkpoints`` are the
    destination directories -- typically the exported HF checkpoints listed in
    ``ApproxUnrollingConfig.checkpoints``.

    States are matched to destinations by step index when the destination
    directory names one (``step_12`` / ``step_12.ckpt``); otherwise they are
    matched positionally in step order, which requires the counts to be equal.
    Mixing the two is an error, since a partial name match usually means the
    caller passed a directory list that does not correspond to this run.

    ``mode`` defaults to ``"symlink"``: these files are large (second moments
    are one scalar per weight) and duplicating them per checkpoint is wasteful.
    Use ``"copy"`` when the destinations must be self-contained, e.g. before
    uploading them somewhere.

    Returns ``{destination optimizer.pt: source file}``. Raises rather than
    silently skipping, so a wrong directory list fails loudly instead of
    producing a preconditioner from the wrong steps.
    """
    states = sorted_optimizer_states(save_dir)
    if not states:
        raise FileNotFoundError(
            f"No step_<i>.optimizer.pt files in {save_dir}. Train with "
            "TrainingConfig.save_optimizer_state=True to write them."
        )

    dsts = [Path(c) for c in checkpoints]
    if not dsts:
        raise ValueError("checkpoints is empty; nothing to place")

    missing = [d for d in dsts if not d.is_dir()]
    if missing:
        raise NotADirectoryError(
            f"Checkpoint directories do not exist: {[str(m) for m in missing]}"
        )

    named = [_step_of(d) for d in dsts]
    if all(step is not None for step in named):
        steps = [step for step in named if step is not None]
        by_step = dict(states)
        unknown = [step for step in steps if step not in by_step]
        if unknown:
            raise FileNotFoundError(
                f"No step_<i>.optimizer.pt in {save_dir} for steps {unknown}; "
                f"available steps: {sorted(by_step)}"
            )
        pairs = [(by_step[step], dst) for step, dst in zip(steps, dsts)]
    elif any(step is not None for step in named):
        raise ValueError(
            "Some checkpoint directories name a step and some do not, so they "
            "cannot be matched reliably; pass directories that all name their "
            "step, or none that do (for positional matching)."
        )
    else:
        if len(states) != len(dsts):
            raise ValueError(
                f"Cannot match positionally: {len(states)} optimizer states in "
                f"{save_dir} but {len(dsts)} checkpoint directories. Name the "
                "destination dirs after their steps to match by step instead."
            )
        pairs = [(src, dst) for (_, src), dst in zip(states, dsts)]

    if not overwrite:
        existing = [dst / OPTIMIZER_STATE_FILE for _, dst in pairs]
        clashes = [p for p in existing if p.exists() or p.is_symlink()]
        if clashes:
            raise FileExistsError(
                f"{[str(c) for c in clashes]} already exist; pass overwrite=True "
                "to replace them."
            )

    placed: dict[Path, Path] = {}
    for src, dst in pairs:
        target = dst / OPTIMIZER_STATE_FILE
        _place(src, target, mode)
        placed[target] = src

    return placed


def place_final_optimizer_state(
    run_path: str | Path,
    destination: str | Path,
    *,
    mode: PlacementMode = "symlink",
    overwrite: bool = False,
) -> Path:
    """Put the run's final ``optimizer.pt`` inside ``destination``.

    ``TrainingConfig.save_optimizer_state`` writes the end-of-training state to
    ``<run_path>/optimizer.pt``. This places it inside an exported model
    directory so ``PreprocessConfig.optimizer_state`` can point at that
    directory -- the TrackStar gradient-normalization path.

    Returns the path written.
    """
    src = Path(run_path) / OPTIMIZER_STATE_FILE
    if not src.is_file():
        raise FileNotFoundError(
            f"{src} not found. Train with TrainingConfig.save_optimizer_state=True "
            "to write it."
        )

    dst_dir = Path(destination)
    if not dst_dir.is_dir():
        raise NotADirectoryError(f"{dst_dir} is not a directory")

    target = dst_dir / OPTIMIZER_STATE_FILE
    if target.resolve() == src.resolve():
        return target
    if not overwrite and (target.exists() or target.is_symlink()):
        raise FileExistsError(f"{target} already exists; pass overwrite=True.")

    _place(src, target, mode)
    return target
