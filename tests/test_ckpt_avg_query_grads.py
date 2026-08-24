"""Tests for query-gradient averaging over the last k trajectory checkpoints."""

import os

import pytest

from bergson.magic.cli import _last_k_checkpoint_paths, compute_query_gradients_averaged


def _make_ckpts(root, steps):
    for n in steps:
        os.makedirs(os.path.join(root, f"step_{n}.ckpt"))
    # Non-checkpoint entries in the same directory must be ignored.
    open(os.path.join(root, "log_history.json"), "w").close()
    os.makedirs(os.path.join(root, "not_a_step"))


def test_orders_numerically_not_lexically(tmp_path):
    _make_ckpts(tmp_path, [2, 10, 40, 125])
    got = [os.path.basename(p) for p in _last_k_checkpoint_paths(str(tmp_path), 3)]
    # Lexical ordering would put step_40 last and drop step_125.
    assert got == ["step_10.ckpt", "step_40.ckpt", "step_125.ckpt"]


def test_returns_oldest_first(tmp_path):
    _make_ckpts(tmp_path, [1, 2, 3])
    got = [os.path.basename(p) for p in _last_k_checkpoint_paths(str(tmp_path), 3)]
    assert got == ["step_1.ckpt", "step_2.ckpt", "step_3.ckpt"]


def test_missing_directory_is_empty(tmp_path):
    assert _last_k_checkpoint_paths(str(tmp_path / "nope"), 4) == []


def test_k_greater_than_available_returns_all(tmp_path):
    _make_ckpts(tmp_path, [5, 6])
    assert len(_last_k_checkpoint_paths(str(tmp_path), 99)) == 2


def test_k_of_one_delegates_without_touching_checkpoints():
    """k<=1 must not read the checkpoint dir at all, so the default path is unchanged."""
    sentinel_grads = {"w": object()}
    calls = []

    def fake_compute(fwd_state, model, query_stream, method, fsdp, grad_accum_steps):
        calls.append((method, fsdp, grad_accum_steps))
        return sentinel_grads, 1.25

    import bergson.magic.cli as cli

    orig = cli.compute_query_gradients
    cli.compute_query_gradients = fake_compute
    try:
        grads, loss = compute_query_gradients_averaged(
            None, None, None, "/definitely/does/not/exist", 1, "mean", False, 3
        )
    finally:
        cli.compute_query_gradients = orig

    assert grads is sentinel_grads and loss == 1.25
    assert calls == [("mean", False, 3)]


def test_too_few_checkpoints_raises_with_actionable_message(tmp_path):
    _make_ckpts(tmp_path, [1, 2])
    with pytest.raises(RuntimeError, match="needs 4 saved checkpoints"):
        compute_query_gradients_averaged(None, None, None, str(tmp_path), 4)
