from unittest.mock import MagicMock, patch

import pytest
import torch
import torchopt

from bergson.magic.data_stream import DataStream
from bergson.magic.trainer import Trainer
from bergson.utils.logging import wandb_log_fn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_log_fn_called_each_step(model, dataset):
    """log_fn is called once per training step with (step_idx, loss)."""
    model = model.to("cuda:0")
    opt = torchopt.adamw(1e-4)
    trainer, state = Trainer.initialize(model, opt)

    stream = DataStream(
        dataset,
        batch_size=2,
        device="cuda:0",
    )
    num_steps = len(stream)

    log = MagicMock()
    trainer.train(state, stream, inplace=True, log_fn=log)

    assert log.call_count == num_steps
    for i, call in enumerate(log.call_args_list):
        step, loss = call.args
        assert step == i
        assert isinstance(loss, float)


def test_wandb_log_fn_calls_wandb():
    """wandb_log_fn initializes wandb and logs correctly."""
    mock_wandb = MagicMock()
    mock_wandb.run = None
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        log = wandb_log_fn("test-project", config={"lr": 1e-4})

        mock_wandb.init.assert_called_once_with(
            project="test-project", config={"lr": 1e-4}
        )

        log(5, 0.123)
        mock_wandb.log.assert_called_once_with({"train/loss": 0.123}, step=5)


def test_wandb_log_fn_reuses_existing_run():
    """wandb_log_fn doesn't call init if a run already exists."""
    mock_wandb = MagicMock()
    mock_wandb.run = MagicMock()  # pretend a run exists
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        log = wandb_log_fn("test-project")

        mock_wandb.init.assert_not_called()

        log(0, 1.5)
        mock_wandb.log.assert_called_once_with({"train/loss": 1.5}, step=0)
