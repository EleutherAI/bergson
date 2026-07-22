"""Completion state for steps that write to a ``.part`` directory."""

import pytest

from bergson.utils.step_state import (
    partial_path,
    prepare_step,
    promote_step,
    step_state,
)


def _write(path, name="out.bin"):
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text("x")


def test_states(tmp_path):
    out = tmp_path / "step"
    assert step_state(out) == "missing"

    _write(partial_path(out))
    assert step_state(out) == "partial"

    promote_step(out)
    assert step_state(out) == "complete"
    assert not partial_path(out).exists()


def test_resume_skips_only_completed_steps(tmp_path):
    out = tmp_path / "step"

    # Nothing yet -> run it.
    assert prepare_step(out, resume=True) is True

    # Interrupted -> run it again, and clear the stale partial first.
    _write(partial_path(out), "half.bin")
    assert prepare_step(out, resume=True) is True
    assert not partial_path(out).exists()

    # Completed -> skip.
    _write(partial_path(out))
    promote_step(out)
    assert prepare_step(out, resume=True) is False
    assert (out / "out.bin").exists()


def test_overwrite_reruns_but_keeps_output_until_promote(tmp_path):
    out = tmp_path / "step"
    _write(partial_path(out), "old.bin")
    promote_step(out)

    # A rerun must run, but the completed output survives until the rerun's
    # own promote_step replaces it -- a crash in between keeps the old data.
    assert prepare_step(out, resume=False) is True
    assert (out / "old.bin").exists(), "completed output must not be deleted early"


def test_prepare_step_clears_a_stale_partial(tmp_path):
    out = tmp_path / "step"
    _write(partial_path(out), "half.bin")
    _write(partial_path(out.with_suffix(".other")))  # unrelated, must be left

    assert prepare_step(out, resume=False) is True
    assert not partial_path(out).exists()


def test_promote_replaces_an_existing_output(tmp_path):
    out = tmp_path / "step"
    _write(partial_path(out), "first.bin")
    promote_step(out)

    _write(partial_path(out), "second.bin")
    promote_step(out)

    assert (out / "second.bin").exists()
    assert not (out / "first.bin").exists()


def test_promote_without_a_partial_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        promote_step(tmp_path / "nope")


def test_promote_keeps_old_output_until_replacement_is_in_place(tmp_path, monkeypatch):
    """A crash mid-swap must not destroy the old completed output."""
    import bergson.utils.step_state as ss

    out = tmp_path / "step"
    _write(partial_path(out), "old.bin")
    promote_step(out)  # out now holds old.bin

    # New replacement staged.
    _write(partial_path(out), "new.bin")

    # Simulate a crash right after the old output is parked aside, before the
    # new one is renamed into place.
    real_rename = ss.Path.rename
    calls = {"n": 0}

    def boom(self, target):
        calls["n"] += 1
        result = real_rename(self, target)
        if calls["n"] == 1:  # after parking old under .superseded
            raise RuntimeError("crash mid-promote")
        return result

    monkeypatch.setattr(ss.Path, "rename", boom)
    with pytest.raises(RuntimeError):
        promote_step(out)
    monkeypatch.setattr(ss.Path, "rename", real_rename)

    # Old output survives (parked), new output survives (still .part) -> no loss.
    superseded = out.with_name(out.name + ".superseded")
    assert (superseded / "old.bin").exists()
    assert (partial_path(out) / "new.bin").exists()

    # Next run cleans both and reruns from scratch.
    assert prepare_step(out, resume=True) is True
    assert not superseded.exists()
    assert not partial_path(out).exists()
