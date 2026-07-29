"""Placing trainer-written optimizer states where SOURCE and TrackStar read them.

The trainer writes ``step_<i>.optimizer.pt`` as a sibling of ``step_<i>.ckpt``;
both consumers want ``optimizer.pt`` *inside* a directory. These tests pin the
matching rules, since a wrong match silently builds a preconditioner from the
wrong step rather than failing.
"""

import os
import shutil
from pathlib import Path

import pytest
import torch

from bergson.utils.load_from_optimizer import load_optimizer
from bergson.utils.optimizer_placement import (
    OPTIMIZER_STATE_FILE,
    place_final_optimizer_state,
    place_optimizer_states,
    sorted_optimizer_states,
)


def _write_state(path: Path, step: int) -> None:
    """A minimal optimizer.pt whose payload identifies its step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": {0: {"exp_avg_sq": torch.full((2, 3), float(step))}},
            "param_groups": [{"betas": (0.9, 0.999), "eps": 1e-8, "lr": 1e-4}],
        },
        path,
    )


@pytest.fixture
def save_dir(tmp_path) -> Path:
    """A trainer output dir with sibling states at steps 0, 4 and 9."""
    d = tmp_path / "run"
    for step in (0, 4, 9):
        _write_state(d / f"step_{step}.optimizer.pt", step)
        (d / f"step_{step}.ckpt").mkdir(parents=True, exist_ok=True)
    return d


def _exported(root: Path, names) -> list[Path]:
    dirs = []
    for name in names:
        p = root / name
        p.mkdir(parents=True, exist_ok=True)
        dirs.append(p)
    return dirs


def test_sorted_optimizer_states_orders_numerically(save_dir):
    """Steps must sort numerically, not lexically (step_10 < step_9 as strings)."""
    _write_state(save_dir / "step_10.optimizer.pt", 10)
    assert [s for s, _ in sorted_optimizer_states(save_dir)] == [0, 4, 9, 10]


def test_sorted_optimizer_states_ignores_other_files(save_dir):
    """The final optimizer.pt and the .ckpt dirs are not per-step states."""
    _write_state(save_dir / OPTIMIZER_STATE_FILE, -1)
    assert [s for s, _ in sorted_optimizer_states(save_dir)] == [0, 4, 9]


def test_places_by_step_name(save_dir, tmp_path):
    """Directories naming a step get that step's state, regardless of order."""
    dsts = _exported(tmp_path / "hf", ["step_9", "step_0", "step_4"])
    placed = place_optimizer_states(save_dir, dsts, mode="copy")

    assert len(placed) == 3
    for dst, expected in zip(dsts, (9, 0, 4)):
        blob = load_optimizer(str(dst))
        assert blob["state"][0]["exp_avg_sq"][0, 0].item() == expected


def test_places_positionally_when_unnamed(save_dir, tmp_path):
    """Unnamed dirs are matched in step order, so seg 0 gets the earliest step."""
    dsts = _exported(tmp_path / "hf", ["a", "b", "c"])
    place_optimizer_states(save_dir, dsts, mode="copy")

    for dst, expected in zip(dsts, (0, 4, 9)):
        blob = load_optimizer(str(dst))
        assert blob["state"][0]["exp_avg_sq"][0, 0].item() == expected


def test_positional_requires_equal_counts(save_dir, tmp_path):
    """A short/long dir list means the caller passed the wrong run; don't guess."""
    dsts = _exported(tmp_path / "hf", ["a", "b"])
    with pytest.raises(ValueError, match="Cannot match positionally"):
        place_optimizer_states(save_dir, dsts, mode="copy")


def test_mixed_naming_is_rejected(save_dir, tmp_path):
    """Half-named lists can't be matched reliably, so refuse rather than guess."""
    dsts = _exported(tmp_path / "hf", ["step_0", "b", "c"])
    with pytest.raises(ValueError, match="name a step and some do not"):
        place_optimizer_states(save_dir, dsts, mode="copy")


def test_unknown_step_is_rejected(save_dir, tmp_path):
    """A named step with no matching state is an error, not a skip."""
    dsts = _exported(tmp_path / "hf", ["step_0", "step_7"])
    with pytest.raises(FileNotFoundError, match=r"steps \[7\]"):
        place_optimizer_states(save_dir, dsts, mode="copy")


def test_refuses_to_clobber_without_overwrite(save_dir, tmp_path):
    """Existing optimizer.pt files are protected; nothing is written on refusal."""
    dsts = _exported(tmp_path / "hf", ["step_0", "step_4", "step_9"])
    (dsts[1] / OPTIMIZER_STATE_FILE).write_text("do not clobber")

    with pytest.raises(FileExistsError):
        place_optimizer_states(save_dir, dsts, mode="copy")

    assert (dsts[1] / OPTIMIZER_STATE_FILE).read_text() == "do not clobber"
    # The check precedes all writes, so the untouched dirs stay untouched.
    assert not (dsts[0] / OPTIMIZER_STATE_FILE).exists()

    place_optimizer_states(save_dir, dsts, mode="copy", overwrite=True)
    assert load_optimizer(str(dsts[1]))["state"][0]["exp_avg_sq"][0, 0].item() == 4


def test_symlink_mode_is_relative_and_loadable(save_dir, tmp_path):
    """Default mode links rather than duplicating a state per checkpoint."""
    dsts = _exported(tmp_path / "hf", ["step_0", "step_4", "step_9"])
    place_optimizer_states(save_dir, dsts)

    link = dsts[0] / OPTIMIZER_STATE_FILE
    assert link.is_symlink()
    assert not os.path.isabs(os.readlink(link)), "link must be relative"
    assert load_optimizer(str(dsts[0]))["state"][0]["exp_avg_sq"][0, 0].item() == 0


def test_symlinks_survive_relocation(save_dir, tmp_path):
    """Relative links keep resolving when run and checkpoints move together."""
    dsts = _exported(tmp_path / "hf", ["step_0", "step_4", "step_9"])
    place_optimizer_states(save_dir, dsts)

    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copytree(tmp_path / "hf", moved / "hf", symlinks=True)
    shutil.copytree(save_dir, moved / save_dir.name, symlinks=True)

    relocated = moved / "hf" / "step_0" / OPTIMIZER_STATE_FILE
    assert relocated.is_symlink()
    assert relocated.resolve().is_file(), "relative link broke after relocation"
    assert (
        load_optimizer(str(moved / "hf" / "step_0"))["state"][0]["exp_avg_sq"][
            0, 0
        ].item()
        == 0
    )


def test_hardlink_and_move_modes(save_dir, tmp_path):
    dsts = _exported(tmp_path / "hf", ["step_0", "step_4", "step_9"])
    place_optimizer_states(save_dir, dsts, mode="hardlink")
    assert load_optimizer(str(dsts[2]))["state"][0]["exp_avg_sq"][0, 0].item() == 9

    dsts2 = _exported(tmp_path / "hf2", ["step_0", "step_4", "step_9"])
    place_optimizer_states(save_dir, dsts2, mode="move")
    assert not (save_dir / "step_9.optimizer.pt").exists()
    assert load_optimizer(str(dsts2[2]))["state"][0]["exp_avg_sq"][0, 0].item() == 9


def test_missing_states_names_the_config_flag(tmp_path):
    """The error should say how to produce the files, not just that they're absent."""
    empty = tmp_path / "empty"
    empty.mkdir()
    dsts = _exported(tmp_path / "hf", ["step_0"])
    with pytest.raises(FileNotFoundError, match="save_optimizer_state"):
        place_optimizer_states(empty, dsts)


def test_missing_destination_is_rejected(save_dir, tmp_path):
    with pytest.raises(NotADirectoryError):
        place_optimizer_states(save_dir, [tmp_path / "nope"])


def test_place_final_state_for_trackstar(save_dir, tmp_path):
    """TrackStar points optimizer_state at a dir; the final state must land there."""
    _write_state(save_dir / OPTIMIZER_STATE_FILE, 99)
    exported = _exported(tmp_path / "model", ["final"])[0]

    target = place_final_optimizer_state(save_dir, exported, mode="copy")

    assert target == exported / OPTIMIZER_STATE_FILE
    # load_optimizer resolves a directory, which is what PreprocessConfig takes.
    assert load_optimizer(str(exported))["state"][0]["exp_avg_sq"][0, 0].item() == 99


def test_place_final_state_is_idempotent_in_place(save_dir):
    """Pointing it at its own directory is a no-op, not a self-clobber."""
    _write_state(save_dir / OPTIMIZER_STATE_FILE, 99)
    target = place_final_optimizer_state(save_dir, save_dir)
    assert target.is_file()
    assert load_optimizer(str(save_dir))["state"][0]["exp_avg_sq"][0, 0].item() == 99


def test_place_final_state_missing_names_the_flag(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    dst = _exported(tmp_path / "model", ["final"])[0]
    with pytest.raises(FileNotFoundError, match="save_optimizer_state"):
        place_final_optimizer_state(empty, dst)
