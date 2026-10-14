"""Leave-k-out retrains checkpoint and resume mid-run."""

import os
from types import SimpleNamespace

import pytest
import torch

from bergson.data import sorted_checkpoints
from bergson.validate import drop_retrain_checkpoints, run_retrain
from tests.test_magic import _fresh_trainer, _multi_step_stream


class _CrashAt:
    """A stream that raises when batch ``crash_at`` is requested."""

    def __init__(self, stream, crash_at: int):
        self.stream, self.crash_at = stream, crash_at

    def __len__(self):
        return len(self.stream)

    def __getitem__(self, i):
        if i == self.crash_at:
            raise RuntimeError("simulated crash")
        return self.stream[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def _cfg(run_path, interval: int):
    return SimpleNamespace(
        run_path=str(run_path),
        retrain_save_interval=interval,
        resume=True,
        fsdp=False,
        max_grad_norm=None,
        grad_accum_steps=1,
    )


def _params(state) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state.params.items()}


def test_keep_last_leaves_only_newest_checkpoint(tmp_path):
    trainer, state, _ = _fresh_trainer()
    trainer.train(
        state,
        _multi_step_stream(10),
        inplace=True,
        save_dir=str(tmp_path),
        save_mode="interval",
        save_interval=3,
        keep_last=1,
    )
    assert [idx for idx, _ in sorted_checkpoints(str(tmp_path))] == [9]


def test_retrain_resumes_to_uninterrupted_result(tmp_path):
    n, crash_at = 10, 7

    trainer, state, _ = _fresh_trainer()
    state = run_retrain(
        trainer, state, _multi_step_stream(n), _cfg(tmp_path / "a", 0), "subset_0"
    )
    expected = _params(state)

    cfg = _cfg(tmp_path / "b", 3)
    trainer, state, _ = _fresh_trainer()
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_retrain(
            trainer, state, _CrashAt(_multi_step_stream(n), crash_at), cfg, "subset_0"
        )
    ckpt_dir = os.path.join(cfg.run_path, "retrain_checkpoints", "subset_0")
    assert [idx for idx, _ in sorted_checkpoints(ckpt_dir)] == [6]

    trainer, state, _ = _fresh_trainer()
    state = run_retrain(trainer, state, _multi_step_stream(n), cfg, "subset_0")
    for k, v in _params(state).items():
        torch.testing.assert_close(v, expected[k])

    drop_retrain_checkpoints(cfg, "subset_0")
    assert not os.path.exists(ckpt_dir)
