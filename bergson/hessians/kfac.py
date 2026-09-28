import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file
from torch import Tensor

from bergson.collector.collector import HookCollectorBase
from bergson.hessians.sharded_computation import (
    ShardedMul,
    assign_module_owners,
    gather_batch_shapes,
    gather_to_owner,
    owned_to_row_shards,
)
from bergson.utils.utils import assert_type


def _input_key(x: Tensor) -> tuple:
    return (x.device, x.dtype, x.data_ptr(), x.shape, x.stride(), x._version)


def find_shared_inputs(
    model: nn.Module,
    target_info: dict[str, tuple[torch.device, torch.Size, bool]],
) -> dict[str, str]:
    """Map each module of ``target_info`` that reads the same input tensor as
    an earlier one, such as ``k_proj`` and ``v_proj`` after ``q_proj``, to the
    earliest, found by running ``model`` on a two-token input.

    Modules called more than once, with their own positions (MoE experts), or
    whose bias settings differ, are left out.
    """
    root = getattr(model, "base_model", model)
    inputs: dict[str, list[Tensor]] = defaultdict(list)
    own_positions = set()

    def record(name: str, module: nn.Module, inp: tuple, out):
        inputs[name].append(inp[0])
        if getattr(module, "_positions", None) is not None:
            own_positions.add(name)

    handles = [
        root.get_submodule(name).register_forward_hook(partial(record, name))
        for name in target_info
    ]
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            model(torch.zeros(1, 2, dtype=torch.long, device=device))
    finally:
        for handle in handles:
            handle.remove()

    # ``inputs`` holds every tensor until here, so no two can share an address.
    first_reader: dict[tuple, str] = {}
    shared = {}
    for name, xs in inputs.items():
        if len(xs) != 1 or name in own_positions:
            continue
        key = (*_input_key(xs[0]), target_info[name][2])
        if key in first_reader:
            shared[name] = first_reader[key]
        else:
            first_reader[key] = name
    return shared


def copy_shared(tensors: dict[str, Tensor], shared: dict[str, str]):
    """``tensors`` with every name in ``shared`` copied to the CPU, since
    safetensors refuses to save tensors that share memory."""
    return {
        name: t.to("cpu", copy=True) if name in shared else t
        for name, t in tensors.items()
    }


class SharedInputCheck:
    """Checks that each module in ``shared`` reads the same tensor as the
    module it maps to, in every batch."""

    def __init__(self, shared: dict[str, str]):
        self.shared = shared
        self.readers = Counter(shared.values())
        # Each input, its key when first read, and the number of modules yet to
        # read it. The input is held so its address can't be reused meanwhile.
        self.inputs: dict[str, tuple[Tensor, tuple, int]] = {}

    def __call__(self, name: str, x: Tensor) -> bool:
        """Whether ``name`` reads another module's input."""
        source = self.shared.get(name)
        if source is None:
            if name in self.readers:
                self.inputs[name] = (x, _input_key(x), self.readers[name])
            return False

        if source not in self.inputs or self.inputs[source][1] != _input_key(x):
            raise RuntimeError(
                f"{name} was expected to read the same input as {source} but "
                "didn't. Did the model's train/eval mode change?"
            )
        source_x, key, remaining = self.inputs.pop(source)
        if remaining > 1:
            self.inputs[source] = (source_x, key, remaining - 1)
        return True

    def clear(self):
        self.inputs.clear()


@dataclass(kw_only=True)
class CovarianceCollector(HookCollectorBase):
    """
    Collects activation and gradient covariances for EKFAC.

    Computes:
        A_cov = sum over batches of (X^T @ X)  for activations
        S_cov = sum over batches of (G^T @ G)  for gradients

    where X is input activations [N*S, I] and G is output gradients [N*S, O].

    Modules in ``shared_inputs`` share the activation covariance of the module
    they map to, which alone accumulates it.

    Distributed, each module's covariances belong to one rank, which receives
    every rank's positions for that module; teardown saves the usual row shards.
    """

    dtype: torch.dtype
    path: str

    shared_inputs: dict[str, str] = field(default_factory=dict)
    """Maps modules to the earlier module that reads the same input, as found
    by :func:`find_shared_inputs`."""

    def setup(self) -> None:
        """Initialize covariance storage dictionaries."""
        self.shared_inputs = {
            name: source
            for name, source in self.shared_inputs.items()
            if name in self.target_info and source in self.target_info
        }
        self.check_shared_input = SharedInputCheck(self.shared_inputs)
        self.A_cov_dict = {}
        self.S_cov_dict = {}
        self.shard_computer = ShardedMul()
        self.A_shapes, self.S_shapes = {}, {}
        for name, (_, (out_dim, in_dim), collect_bias) in self.target_info.items():
            self.A_shapes[name] = (in_dim + collect_bias,) * 2
            self.S_shapes[name] = (out_dim, out_dim)

        # Each module's owning rank; None in a single process.
        self.owners: dict[str, int] | None = None
        owned = list(self.target_info)
        if dist.is_initialized():
            self.owners = assign_module_owners(
                self.target_info, self.world_size, self.shared_inputs
            )
            self._rows = 0  # set per batch by with_batch
            owned = [name for name in owned if self.owners[name] == self.rank]

        device = self.shard_computer.device
        for name in owned:
            self.S_cov_dict[name] = torch.zeros(
                self.S_shapes[name], device=device, dtype=self.dtype
            )
            if name not in self.shared_inputs:
                self.A_cov_dict[name] = torch.zeros(
                    self.A_shapes[name], device=device, dtype=self.dtype
                )
        for name in owned:
            if name in self.shared_inputs:
                self.A_cov_dict[name] = self.A_cov_dict[self.shared_inputs[name]]

    def with_batch(self, collection_mask: Tensor | None = None):
        super().with_batch(collection_mask)
        self.check_shared_input.clear()
        if self.owners is not None and collection_mask is not None:
            # Every rank pads its positions to the batch's largest count.
            counts = gather_batch_shapes(
                int(collection_mask.sum()), device=collection_mask.device
            )
            self._rows = max(count for (count,) in counts)
        return self

    def forward_hook(self, module: nn.Module, a: Tensor) -> None:
        """Compute activation covariance: A^T @ A."""
        name = assert_type(str, module._name)
        if self.check_shared_input(name, a):
            return
        mask = self.collection_mask(module)
        assert mask is not None, "Collection mask not set for forward hook."

        # a: [N, S, I], collection mask: [N, S] -> select gradient-carrying positions
        a_bi = a[mask]  # [num_valid, I]

        # Augment with a ones column so A matches the [O, I+1] gradient layout
        # produced when the bias gradient is collected.
        if module._collect_bias:
            a_bi = torch.cat(
                [a_bi, a_bi.new_ones(a_bi.shape[0], 1)], dim=1
            )  # [num_valid, I+1]

        self._accumulate(self.A_cov_dict, name, a_bi)

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        """Compute gradient covariance: G^T @ G."""
        name = assert_type(str, module._name)
        mask = self.collection_mask(module)

        # g: [N, S, O], mask: [N, S] -> select gradient-carrying positions
        g_bo = g[mask]  # [num_valid, O]

        self._accumulate(self.S_cov_dict, name, g_bo)

    def _accumulate(self, covariances: dict[str, Tensor], name: str, x: Tensor):
        """Add ``X^T @ X`` to ``name``'s covariance, where ``X`` stacks every
        rank's ``x``, in the accumulation dtype."""
        if self.owners is None:
            x = x.to(self.dtype)
            covariances[name].addmm_(x.mT, x)
            return

        # Sent in the model's dtype and cast on the owner, which moves less data.
        stacked = gather_to_owner(x, self._rows, self.owners[name])
        if stacked is not None:
            # Padding rows are zero, so they add nothing.
            stacked = stacked.to(self.dtype)
            covariances[name].addmm_(stacked.mT, stacked)

    def process_batch(self, indices: list[int], **kwargs) -> None:
        """No per-batch processing needed for covariance collection."""
        pass

    def teardown(self) -> None:
        """Save covariance matrices to disk."""
        activation_path = os.path.join(self.path, "activation_sharded")
        gradient_path = os.path.join(self.path, "gradient_sharded")

        os.makedirs(activation_path, exist_ok=True)
        os.makedirs(gradient_path, exist_ok=True)
        self.logger.info(
            f"Saving sharded covariance matrices to {activation_path} "
            f"and {gradient_path}"
        )
        for covariances, shapes, path, shared in (
            (self.A_cov_dict, self.A_shapes, activation_path, self.shared_inputs),
            (self.S_cov_dict, self.S_shapes, gradient_path, {}),
        ):
            if self.owners is not None:
                shards = owned_to_row_shards(
                    covariances,
                    shapes,
                    self.owners,
                    self.dtype,
                    self.shard_computer.device,
                    shared,
                )
            else:
                shards = covariances
            save_file(
                copy_shared(shards, shared),
                os.path.join(path, f"shard_{self.rank}.safetensors"),
            )
            covariances.clear()
        self.check_shared_input.clear()
