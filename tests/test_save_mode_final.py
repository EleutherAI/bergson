"""Tests for save_mode='final': post-training state only, no trajectory."""

import pytest

from bergson.magic.trainer import next_save_index


def _schedule(n, save_mode, save_interval=0):
    """Every step the training loop would save at (it saves before step i<n)."""
    saves, cur = [], 0
    while cur < n:
        saves.append(cur)
        cur = next_save_index(cur, n, save_mode, save_interval)
    return saves


def test_final_schedules_nothing_before_the_end():
    # next_save_index jumps straight to n, so the loop (which only tests i<n)
    # never fires. Step 0 in particular must NOT be written.
    assert next_save_index(0, 100, "final") == 100


def test_final_is_constant_space_in_n():
    for n in (10, 100, 10_000):
        assert next_save_index(0, n, "final") == n


@pytest.mark.parametrize("mode", ["all", "sqrt", "log"])
def test_trajectory_modes_still_save_during_training(mode):
    sched = _schedule(64, mode)
    assert sched[0] == 0
    assert len(sched) > 1, f"{mode} should save more than the initial state"


def test_final_saves_strictly_fewer_than_every_trajectory_mode():
    n = 64
    final = 1  # just step_n, written after the loop
    for mode in ("all", "sqrt", "log"):
        assert len(_schedule(n, mode)) > final


def test_interval_unchanged():
    assert next_save_index(0, 100, "interval", 25) == 25
    with pytest.raises(ValueError):
        next_save_index(0, 100, "interval", 0)


def test_final_ignores_save_interval():
    # 'final' must not require or consult save_interval, unlike 'interval'.
    assert next_save_index(0, 50, "final", 0) == 50
    assert next_save_index(0, 50, "final", 7) == 50
