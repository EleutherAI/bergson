from contextlib import contextmanager
from typing import Generator

from torch import Tensor, nn


def parse_neuron(neuron: str) -> tuple[str, int]:
    """Split a ``<module>:<index>`` spec into the module name and index."""
    module, sep, index = neuron.rpartition(":")
    if not sep or not module or not index.isdigit():
        raise ValueError(f"Expected a neuron as '<module>:<index>', got {neuron!r}")
    return module, int(index)


@contextmanager
def neuron_activations(
    model: nn.Module, neuron: str
) -> Generator[list[Tensor], None, None]:
    """Record the neuron's ``[batch, seq]`` activations from each forward."""
    module, index = parse_neuron(neuron)
    acts: list[Tensor] = []

    def hook(_module, _inputs, output):
        acts.append(output[..., index])

    handle = model.get_submodule(module).register_forward_hook(hook)
    try:
        yield acts
    finally:
        handle.remove()


def neuron_token_loss(acts: Tensor, labels: Tensor) -> Tensor:
    """Negative activation at each position that predicts a label token."""
    mask = labels[:, 1:] != -100
    return -acts[:, :-1] * mask


@contextmanager
def neuron_query(model: nn.Module, neuron: str) -> Generator[None, None, None]:
    """Make ``model(**batch).loss`` the neuron's negative mean activation over
    the positions that predict a label token. No-op for an empty spec."""
    if not neuron:
        yield
        return

    loss_function = model.loss_function
    with neuron_activations(model, neuron) as acts:

        def neuron_loss(logits: Tensor, labels: Tensor, **kwargs) -> Tensor:
            mask = labels[:, 1:] != -100
            loss = neuron_token_loss(acts.pop(), labels).sum()
            return loss / mask.sum().clamp_min(1)

        model.loss_function = neuron_loss  # type: ignore[assignment]
        try:
            yield
        finally:
            model.loss_function = loss_function
