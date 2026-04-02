"""MAGIC integration test: forward + backward through 2 training steps."""

import tempfile

import pytest
import torch
import torchopt
from torchopt.pytree import tree_iter
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.distributed import grad_tree
from bergson.magic import BackwardState, DataStream, Trainer
from bergson.utils.math import weighted_causal_lm_ce

MODEL_CONFIGS = [
    "trl-internal-testing/tiny-Phi3ForCausalLM",
    "EleutherAI/pythia-14m",
]


@pytest.mark.parametrize("model_name", MODEL_CONFIGS)
def test_magic_two_steps(model_name, dataset):
    device = "cpu"

    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )

    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)

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
                k: g.detach().clone() for k, g in grad_tree(loss, params).items()
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


def test_magic_resume(dataset):
    """Resume from a checkpoint mid-training and verify identical final state."""
    device = "cpu"

    torch.manual_seed(42)
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)

    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)

    # batch_size=1 gives us 2 batches so resume has something to skip
    train_stream = DataStream(dataset, batch_size=1, device=device)
    assert len(train_stream) == 2

    with tempfile.TemporaryDirectory() as ckpt_dir:
        # Full training run (inplace=False to keep fwd_state intact)
        final_state = trainer.train(
            fwd_state,
            train_stream,
            inplace=False,
            save_dir=ckpt_dir,
            save_mode="all",
        )

        # Resume from checkpoints with the same initial state
        resumed_state = trainer.train(
            fwd_state,
            train_stream,
            inplace=False,
            save_dir=ckpt_dir,
            save_mode="all",
            resume=True,
        )

        for k in final_state.params:
            torch.testing.assert_close(resumed_state.params[k], final_state.params[k])
