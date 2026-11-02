"""Keep the activations saved for the backward pass on CPU."""

from contextlib import contextmanager

import torch


@contextmanager
def save_activations_on_cpu():
    """Like ``torch.autograd.graph.save_on_cpu``, but keeps the strides of
    attention biases.

    Memory-efficient SDPA checks the strides of its bias again in the backward
    pass. A plain device copy compacts a sliced, padded bias and loses its
    alignment, and copies a broadcast bias once per head, so both are restored
    with their original strides.
    """

    def pack(tensor):
        padded_bias = (
            tensor.ndim == 4
            and tensor.shape[-2] == tensor.shape[-1]
            and tensor.stride(-1) == 1
            and tensor.stride(-2) > tensor.shape[-1]
        )
        broadcast = any(
            stride == 0 and size > 1
            for size, stride in zip(tensor.shape, tensor.stride())
        )
        strides = tensor.stride() if padded_bias or broadcast else None
        source = tensor
        if strides is not None:
            # Store one copy of each broadcast axis, then restore the broadcast
            # view without writing into overlapping storage
            for axis, stride in enumerate(strides):
                if stride == 0 and source.shape[axis] > 1:
                    source = source.narrow(axis, 0, 1)
        return tensor.device, source.to("cpu"), strides, tensor.shape

    def unpack(saved):
        device, cpu, strides, shape = saved
        if strides is None:
            return cpu.to(device)
        restored = torch.empty_strided(
            cpu.shape, strides, device=device, dtype=cpu.dtype
        )
        restored.copy_(cpu)
        return restored.as_strided(shape, strides)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield
