"""Completion state for pipeline steps that write to a ``.part`` directory.

Steps write their outputs to ``<out>.part`` and rename it to ``<out>`` once
they finish. Therefore the final name can be mapped to state as follows:

- ``<out>`` exists                    -> the step finished
- ``<out>.part`` exists, ``<out>`` not -> it was interrupted
- neither                             -> it has not run

That lets a resumed run skip finished steps and restart interrupted ones,
instead of skipping anything that left a directory behind.
"""

import shutil
from pathlib import Path
from typing import Literal

StepState = Literal["complete", "partial", "missing"]


def partial_path(path: Path | str) -> Path:
    """The ``.part`` directory a step writes to before promoting it."""
    path = Path(path)
    return path.with_name(path.name + ".part")


def _superseded_path(path: Path) -> Path:
    """Where an old output is parked while its replacement is promoted."""
    return path.with_name(path.name + ".superseded")


def step_state(path: Path | str) -> StepState:
    """Whether the step writing to `path` finished, was interrupted, or is new."""
    path = Path(path)
    if path.exists():
        return "complete"
    if partial_path(path).exists():
        return "partial"
    return "missing"


def prepare_step(path: Path | str, *, resume: bool) -> bool:
    """Decide whether to run the step writing to `path`, clearing stale output.

    Returns True when the caller should run the step. A completed output is
    left in place even for a rerun: :func:`promote_step` replaces it atomically
    once the rerun finishes, so a crash mid-rerun keeps the old output. Only an
    interrupted ``.part`` is removed here.
    """
    path = Path(path)

    if resume and step_state(path) == "complete":
        return False

    part = partial_path(path)
    if part.exists():
        shutil.rmtree(part)
    superseded = _superseded_path(path)
    if superseded.exists():
        shutil.rmtree(superseded)
    return True


def promote_step(path: Path | str) -> None:
    """Rename `path`'s ``.part`` directory to its final name.

    Call once, from a single rank, after every writer has flushed.
    """
    path = Path(path)
    part = partial_path(path)
    if not part.exists():
        raise FileNotFoundError(f"No partial output to promote at {part}")

    if not path.exists():
        part.rename(path)
        return

    # Park the old output under .superseded first so it is only deleted once
    # the replacement is fully in place. A crash between the two renames leaves
    # the old output under .superseded and the new under .part; the next run's
    # prepare_step clears both and reruns, so no completed output is destroyed
    # before its replacement exists.
    superseded = _superseded_path(path)
    if superseded.exists():
        shutil.rmtree(superseded)
    path.rename(superseded)
    part.rename(path)
    shutil.rmtree(superseded)
