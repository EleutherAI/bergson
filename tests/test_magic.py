"""MAGIC integration test: forward + backward through 2 training steps."""

import tempfile

import pytest
import torch
import torchopt
from torchopt.pytree import tree_iter

from bergson.distributed import grad_tree
from bergson.magic import BackwardState, DataStream, Trainer
from bergson.utils.math import weighted_causal_lm_ce


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_magic_two_steps(model, dataset):
    device = "cuda"

    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)
    model.to(device)

    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)

    train_stream = DataStream(
        dataset,
        batch_size=len(dataset),
        device=device,
    )
    assert len(train_stream) == 1

    with tempfile.TemporaryDirectory() as ckpt_dir:
        fwd_state = trainer.train(
            fwd_state,
            train_stream,
            inplace=True,
            save_dir=ckpt_dir,
        )

        # Compute query gradients on the training batch
        with fwd_state.activate(model) as params:
            batch = train_stream[0]
            del batch["example_weight"]
            loss = model(**batch).loss
            query_grads = {
                k: g.detach().clone()
                for k, g in grad_tree(loss, params).items()
            }

            opt_grads = [
                torch.zeros_like(buf)
                for buf in tree_iter(fwd_state.opt_state)
                if isinstance(buf, torch.Tensor) and buf.is_floating_point()
            ]
            bwd_state = BackwardState(
                query_grads,
                opt_grads,
                torch.zeros_like(train_stream.weights),
            )

        # Backward pass through training
        train_stream.requires_grad = True
        bwd_state = trainer.backward(
            ckpt_dir,
            train_stream,
            bwd_state,
            fwd_state,
            inplace=True,
            cleanup=True,
        )

    scores = bwd_state.weight_grads.detach().cpu()
    assert scores.shape == (len(dataset),)
    assert scores.abs().sum() > 0, "Attribution scores are all zero"
