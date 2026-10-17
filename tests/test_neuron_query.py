"""Neuron queries: the query loss is a neuron's negative mean activation."""

import tempfile

import pytest
import torch
import torchopt
from datasets import Dataset
from torchopt.pytree import tree_iter
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.config import QuerySetConfig
from bergson.magic import BackwardState, DataStream, Trainer
from bergson.magic.cli import compute_query_gradients
from bergson.utils.math import weighted_causal_lm_ce
from bergson.utils.neuron import neuron_activations, neuron_query
from bergson.validate import mean_query_loss, per_doc_query_losses

MODEL = "EleutherAI/pythia-14m"
NEURON = "gpt_neox.layers.3.mlp.act:7"


def _model(dtype=torch.float32):
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=dtype, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    return model


def _docs(num_docs: int, offset: int = 0) -> Dataset:
    rows = [
        [(7 * (d + offset) + 3 * t) % 97 + 1 for t in range(6)] for d in range(num_docs)
    ]
    return Dataset.from_dict({"input_ids": rows, "labels": rows})


def _acts(model, input_ids: torch.Tensor) -> torch.Tensor:
    with neuron_activations(model, NEURON) as acts, torch.no_grad():
        model(input_ids=input_ids)
    return acts[0]


def test_neuron_spec_is_checked():
    with pytest.raises(ValueError, match="<module>:<index>"):
        QuerySetConfig(neuron="gpt_neox.layers.3.mlp.act")
    QuerySetConfig(neuron=NEURON)


def test_neuron_query_loss_is_negative_mean_activation():
    model = _model()
    loss_function = model.loss_function
    x = torch.tensor(_docs(3)["input_ids"])
    labels = x.clone()
    labels[1, 3:] = -100

    with neuron_query(model, NEURON), torch.no_grad():
        loss = model(input_ids=x, labels=labels).loss

    mask = labels[:, 1:] != -100
    expected = -_acts(model, x)[:, :-1][mask].mean()
    torch.testing.assert_close(loss, expected)
    assert model.loss_function is loss_function


def test_validation_losses_use_the_neuron():
    model = _model()
    docs = _docs(4)
    stream = DataStream(docs, batch_size=2, weight_shape=(4,))
    expected = -_acts(model, torch.tensor(docs["input_ids"]))[:, :-1]

    per_doc = per_doc_query_losses(model, stream, 4, neuron=NEURON)
    mean = mean_query_loss(model, stream, neuron=NEURON)

    torch.testing.assert_close(per_doc, expected.mean(dim=1))
    torch.testing.assert_close(mean, expected.mean())


def _train(weights: torch.Tensor, ckpt_dir: str):
    model = _model(torch.float64)
    model.requires_grad_(True)
    trainer, fwd_state = Trainer.initialize(
        model, torchopt.adamw(1e-2, betas=(0.9, 0.999), eps_root=1e-8)
    )
    stream = DataStream(_docs(4), batch_size=2, weight_shape=(4,))
    stream.weights.data.copy_(weights)
    fwd_state = trainer.train(fwd_state, stream, inplace=True, save_dir=ckpt_dir)
    return trainer, fwd_state, model, stream


def test_magic_neuron_scores_match_finite_differences():
    """Each document's score is the derivative of the final model's neuron
    query loss with respect to the document's weight."""
    query = DataStream(_docs(2, offset=10), batch_size=2, weight_shape=(2,))

    def query_loss(weights: torch.Tensor) -> float:
        with tempfile.TemporaryDirectory() as ckpt_dir:
            _, fwd_state, model, _ = _train(weights, ckpt_dir)
            with fwd_state.activate(model):
                return mean_query_loss(model, query, neuron=NEURON).item()

    with tempfile.TemporaryDirectory() as ckpt_dir:
        trainer, fwd_state, model, stream = _train(torch.ones(4), ckpt_dir)
        query_grads, _ = compute_query_gradients(
            fwd_state, model, query, "mean", neuron=NEURON
        )
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
            ckpt_dir, stream, bwd_state, fwd_state, inplace=True, cleanup=True
        )
    scores = bwd_state.weight_grads.detach()

    eps = 1e-4
    finite_differences = torch.zeros(4, dtype=torch.float64)
    for i in range(4):
        up, down = torch.ones(4), torch.ones(4)
        up[i] += eps
        down[i] -= eps
        finite_differences[i] = (query_loss(up) - query_loss(down)) / (2 * eps)

    assert finite_differences.abs().max() > 0
    torch.testing.assert_close(
        scores.double(), finite_differences, rtol=1e-3, atol=1e-8
    )
