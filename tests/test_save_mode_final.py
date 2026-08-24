"""Tests for save_mode='final': post-training state only, no trajectory."""

from bergson.magic.trainer import next_save_index


def test_final_schedules_nothing_before_the_end():
    # The training loop only saves at steps i < n, so returning n suppresses
    # every in-loop save regardless of n or save_interval.
    for n in (10, 100, 10_000):
        assert next_save_index(0, n, "final") == n
    assert next_save_index(0, 50, "final", 7) == 50

    for mode in ("all", "sqrt", "log"):
        assert next_save_index(0, 64, mode) < 64
