"""Tests for global-norm gradient clipping (``max_grad_norm``) in the Bergson trainer.

Clipping must (a) match the standard clip-by-global-norm formula, (b) be a true
no-op when disabled or when the threshold doesn't bite, and (c) stay inside the
autograd graph so MAGIC influence scores remain correct — the same failure class
as the historical non-differentiable all-reduce bug.
"""

import tempfile

import pytest
import torch
import torchopt
from datasets import Dataset
from torchopt.pytree import tree_iter
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.distributed import grad_tree
from bergson.magic import BackwardState, DataStream, Trainer
from bergson.utils.math import weighted_causal_lm_ce

MODEL_NAME = "EleutherAI/pythia-14m"


@pytest.fixture
def clip_dataset():
    # Four single-doc rows. With batch_size=2 each step's gradient is a weighted
    # sum of two examples, so clipping couples (rather than cancels) the per-doc
    # weights — necessary for a non-degenerate influence/finite-difference check.
    rows = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
    ]
    return Dataset.from_dict(
        {"input_ids": rows, "labels": rows, "attention_mask": [[1] * 5] * 4}
    )


def _build_model():
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_config(
        config, dtype=torch.float32, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)
    return model


def test_clip_scales_update_by_coef(clip_dataset):
    """A biting ``max_grad_norm`` scales an SGD update by exactly the clip coef.

    With plain SGD the update is ``-lr * grad``, so clipping by ``coef`` must scale
    the whole update by ``coef`` (the optimizer is linear in the gradient here).
    """
    lr = 0.05

    def run(max_grad_norm):
        model = _build_model()
        trainer, state = Trainer.initialize(model, torchopt.sgd(lr))
        stream = DataStream(clip_dataset, batch_size=len(clip_dataset), device="cpu")
        new = trainer.step(
            state, stream[0], inplace=False, trace=False, max_grad_norm=max_grad_norm
        )
        return {k: (new.params[k] - state.params[k]).detach() for k in state.params}

    full = run(None)
    # ||grad|| recovered from the unclipped update: delta = -lr * grad.
    sq = torch.stack([(d / lr).pow(2).sum() for d in full.values()]).sum()
    global_norm = sq.sqrt()
    max_grad_norm = float(global_norm) * 0.2  # well below the norm, so it bites
    coef = min(1.0, max_grad_norm / (float(global_norm) + 1e-6))
    assert coef < 1.0, "test misconfigured: clip does not bite"

    clipped = run(max_grad_norm)
    for k in full:
        torch.testing.assert_close(clipped[k], coef * full[k], atol=1e-6, rtol=1e-5)


def test_no_clip_is_bit_identical(clip_dataset):
    """``None`` and a non-biting threshold reproduce the unclipped trajectory."""

    def run(max_grad_norm):
        model = _build_model()
        opt = torchopt.adamw(1e-3, betas=(0.95, 0.975), eps_root=1e-2)
        trainer, state = Trainer.initialize(model, opt)
        stream = DataStream(clip_dataset, batch_size=2, device="cpu")
        with tempfile.TemporaryDirectory() as ckpt_dir:
            return trainer.train(
                state,
                stream,
                inplace=True,
                save_dir=ckpt_dir,
                max_grad_norm=max_grad_norm,
            )

    baseline = run(None)
    huge = run(1e30)  # far above any real grad norm, so coef == 1 every step
    for k in baseline.params:
        assert torch.equal(baseline.params[k], huge.params[k])


def _query_batch(dataset):
    qs = DataStream(dataset, batch_size=len(dataset), device="cpu")
    batch = qs[0]
    del batch["example_weight"]
    return batch


def _magic_scores(dataset, max_grad_norm, weights=None):
    """Train + MAGIC backward, returning per-doc influence scores for the query batch.

    Mirrors the train→query-grad→backward flow exercised by ``test_magic``.
    """
    query = _query_batch(dataset)
    model = _build_model()
    opt = torchopt.sgd(0.05)
    trainer, fwd_state = Trainer.initialize(model, opt)
    stream = DataStream(dataset, batch_size=2, device="cpu")
    if weights is not None:
        stream.weights.data.copy_(weights)

    with tempfile.TemporaryDirectory() as ckpt_dir:
        fwd_state = trainer.train(
            fwd_state,
            stream,
            inplace=True,
            save_dir=ckpt_dir,
            max_grad_norm=max_grad_norm,
        )
        with fwd_state.activate(model) as params:
            loss = model(**query).loss
            query_grads = {
                k: g.detach().clone() for k, g in grad_tree(loss, params).items()
            }
            opt_grads = [
                torch.zeros_like(buf)
                for buf in tree_iter(fwd_state.opt_state)
                if isinstance(buf, torch.Tensor) and buf.is_floating_point()
            ]
            bwd_state = BackwardState(
                query_grads, opt_grads, torch.zeros_like(stream.weights)
            )
        stream.requires_grad = True
        bwd_state = trainer.backward(
            ckpt_dir,
            stream,
            bwd_state,
            fwd_state,
            inplace=True,
            cleanup=True,
            max_grad_norm=max_grad_norm,
        )
    return bwd_state.weight_grads.detach().cpu()


def _final_query_loss(dataset, max_grad_norm, weights):
    query = _query_batch(dataset)
    model = _build_model()
    opt = torchopt.sgd(0.05)
    trainer, state = Trainer.initialize(model, opt)
    stream = DataStream(dataset, batch_size=2, device="cpu")
    stream.weights.data.copy_(weights)
    with tempfile.TemporaryDirectory() as ckpt_dir:
        state = trainer.train(
            state, stream, inplace=True, save_dir=ckpt_dir, max_grad_norm=max_grad_norm
        )
        with state.activate(model), torch.no_grad():
            return model(**query).loss.item()


def test_clipped_scores_finite_and_change_trajectory(clip_dataset):
    """A biting clip yields finite, nonzero scores differing from the unclipped run."""
    unclipped = _magic_scores(clip_dataset, None)
    clipped = _magic_scores(clip_dataset, 2.0)

    assert torch.isfinite(clipped).all()
    assert clipped.abs().sum() > 0, "clipped scores are all zero"
    assert not torch.allclose(clipped, unclipped), "clip had no effect"


def test_clip_is_differentiable_finite_difference(clip_dataset):
    """MAGIC influence through the clip matches a finite-difference estimate.

    This is the core correctness check: if the clip left the autograd graph (e.g. an
    in-place update under ``trace=True``), the scores would silently disagree with the
    finite-difference gradient of the query loss w.r.t. the per-doc weights.
    """
    max_grad_norm = 2.0  # bites on every step for this model/data/lr
    scores = _magic_scores(clip_dataset, max_grad_norm)

    w0 = torch.ones(len(clip_dataset))
    eps = 1e-2
    for i in range(len(clip_dataset)):
        wp = w0.clone()
        wp[i] += eps
        wm = w0.clone()
        wm[i] -= eps
        fd = (
            _final_query_loss(clip_dataset, max_grad_norm, wp)
            - _final_query_loss(clip_dataset, max_grad_norm, wm)
        ) / (2 * eps)
        torch.testing.assert_close(scores[i], torch.tensor(fd), atol=1e-4, rtol=0.05)
