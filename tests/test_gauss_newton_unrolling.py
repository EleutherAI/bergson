"""Unrolled Gauss-Newton: the backward through training with Gauss-Newton curvature."""

import tempfile

import torch
import torchopt
from torchopt.pytree import tree_iter
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.distributed import grad_tree
from bergson.magic import BackwardState, DataStream, Trainer
from bergson.utils.math import weighted_causal_lm_ce

MODEL = "EleutherAI/pythia-14m"


def _trainer(head_only: bool):
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float64, attn_implementation="eager"
    ).eval()
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(not head_only)
    if head_only:
        model.get_output_embeddings().requires_grad_(True)
    optimizer = torchopt.adamw(1e-3, betas=(0.9, 0.99), eps_root=1e-8)
    trainer, state = Trainer.initialize(model, optimizer)
    return trainer, state, model


def _stream(n_steps: int, batch_size: int = 2) -> DataStream:
    from datasets import Dataset

    torch.manual_seed(1)
    rows = torch.randint(0, 1000, (n_steps * batch_size, 6)).tolist()
    ds = Dataset.from_dict({"input_ids": rows, "labels": rows})
    return DataStream(ds, batch_size=batch_size)


def _scores(curvature: str, head_only: bool) -> torch.Tensor:
    trainer, state, model = _trainer(head_only)
    stream = _stream(4)
    with tempfile.TemporaryDirectory() as ckpt_dir:
        state = trainer.train(
            state, stream, inplace=True, save_dir=ckpt_dir, save_mode="log"
        )
        with state.activate(model) as params:
            batch = stream[0]
            del batch["example_weight"]
            query = {
                k: g.detach().clone()
                for k, g in grad_tree(model(**batch).loss, params).items()
            }
        opt = [
            torch.zeros_like(t)
            for t in tree_iter(state.opt_state)
            if isinstance(t, torch.Tensor) and t.is_floating_point()
        ]
        stream.requires_grad = True
        bwd = trainer.backward(
            ckpt_dir,
            stream,
            BackwardState(query, opt, torch.zeros_like(stream.weights)),
            state,
            inplace=True,
            save_mode="log",
            curvature=curvature,
        )
    return bwd.weight_grads.detach()


def test_gauss_newton_matches_hessian_when_logits_are_linear():
    """Training only the output head makes the logits linear in the trained
    weights, where the Gauss-Newton matrix is the loss Hessian."""
    exact = _scores("hessian", head_only=True)
    assert exact.abs().sum() > 0
    torch.testing.assert_close(_scores("gauss_newton", head_only=True), exact)


def test_gauss_newton_step_keeps_first_order_weight_cotangent():
    """One step's data-weight cotangent is ``⟨∇ℓ_d, z⟩`` under either curvature,
    while the parameter cotangents differ through the curvature term."""
    trainer, state, _ = _trainer(head_only=False)
    stream = _stream(1)
    stream.requires_grad = True
    state.detach_()
    state.requires_grad = True
    torch.manual_seed(2)
    param_cot = {k: torch.randn_like(v) for k, v in state.params.items()}
    opt_cot = [
        torch.zeros_like(t)
        for t in tree_iter(state.opt_state)
        if isinstance(t, torch.Tensor) and t.is_floating_point()
    ]
    out = {}
    for curvature in ("hessian", "gauss_newton"):
        bwd = BackwardState(param_cot, opt_cot, torch.zeros_like(stream.weights))
        out[curvature] = trainer.metagrad_step(
            state, stream[0], bwd, stream.weights, curvature=curvature
        )
    torch.testing.assert_close(
        out["gauss_newton"].weight_grads, out["hessian"].weight_grads
    )
    assert any(
        not torch.allclose(out["gauss_newton"].param_grads[k], g)
        for k, g in out["hessian"].param_grads.items()
    )
