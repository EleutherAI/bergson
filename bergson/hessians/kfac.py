import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from bergson.collector.collector import HookCollectorBase
from bergson.hessians.sharded_computation import (
    BackgroundSaves,
    OwnerGathers,
    ShardedMul,
    assign_module_owners,
    gather_batch_shapes,
    local_modules,
    owned_to_row_shards,
    save_shards,
)
from bergson.utils.utils import assert_type


def add_gram(out: Tensor, x: Tensor) -> None:
    """Add ``x^T @ x`` to ``out``.

    On the GPU, bf16 and fp16 rows are multiplied as they are and summed in
    ``out``'s fp32: every product of two of their values is exact in fp32, so
    this sums the same products as casting ``x`` to fp32 first, at half-precision
    speed and without the copy."""
    if (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and out.dtype == torch.float32
    ):
        torch.addmm(out, x.mT, x, out_dtype=torch.float32, out=out)
        return
    x = x.to(out.dtype)
    out.addmm_(x.mT, x)


@dataclass(kw_only=True)
class CovarianceCollector(HookCollectorBase):
    """
    Collects activation and gradient covariances for EKFAC.

    Computes:
        A_cov = sum over batches of (X^T @ X)  for activations
        S_cov = sum over batches of (G^T @ G)  for gradients

    where X is input activations [N*S, I] and G is output gradients [N*S, O].

    Distributed, each module's covariances belong to one rank. Every rank keeps
    the covariances in ``local``, as ``(module, "A" or "S")``, summing its own
    positions and sending the sums to their owners at teardown; owners receive
    every rank's positions for the other covariances each step. Teardown saves
    the usual row shards.
    Teardown leaves the covariances this rank owns, in full, on its device in
    ``A_cov_dict`` and ``S_cov_dict`` for the eigendecomposition.

    Modules called on the same input tensor in a batch, as LoRA's q, k and v
    adapters are, have the same activation covariance. The first batch records
    each such module in ``aliases`` with the first module that took the input,
    whose covariance alone is summed and sent; teardown copies it to the others.

    ``num_positions`` counts the positions each module's covariances summed,
    across all ranks after teardown. It is ``total_processed`` for a dense
    module, but only the routed positions for an MoE expert, and is what the
    eigendecomposition divides the sums by to get per-position means.
    """

    dtype: torch.dtype
    path: str
    local_bytes: float = 0
    """Bytes of covariances each rank may keep, smallest first. Each is chosen
    on its own, so a module's small factor can stay local while its large one
    goes to its owner."""
    saves: BackgroundSaves | None = None
    """Thread the shards are saved on; without one, teardown saves them
    before it returns."""

    def setup(self) -> None:
        """Initialize covariance storage dictionaries."""
        self.A_cov_dict = {}
        self.S_cov_dict = {}
        self.num_positions: dict[str, int] = dict.fromkeys(self.target_info, 0)
        self.shard_computer = ShardedMul()
        self.A_shapes: dict[str, tuple[int, ...]] = {}
        self.S_shapes: dict[str, tuple[int, ...]] = {}
        self.aliases: dict[str, str] = {}
        """Each module whose input an earlier module took, and that module."""
        self._inputs: dict[tuple, tuple[str, Tensor | None, Tensor]] = {}
        self._batches = 0
        for name, (_, (out_dim, in_dim), collect_bias) in self.target_info.items():
            self.A_shapes[name] = (in_dim + collect_bias,) * 2
            self.S_shapes[name] = (out_dim, out_dim)

        # Each module's owning rank; None in a single process.
        self.owners: dict[str, int] | None = None
        if not dist.is_initialized():
            self.shard_computer._init_covariance_dict(
                activation_covariance_dict=self.A_cov_dict,
                gradient_covariance_dict=self.S_cov_dict,
                dtype=self.dtype,
                target_info=self.target_info,
            )
            return

        self.owners = assign_module_owners(self.target_info, self.world_size)
        self._rows = 0  # set per batch by with_batch
        self.gathers = OwnerGathers()
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        self.local = local_modules(
            {
                (name, kind): math.prod(shapes[name]) * itemsize
                for name in self.owners
                for kind, shapes in (("A", self.A_shapes), ("S", self.S_shapes))
            },
            self.local_bytes,
        )
        device = self.shard_computer.device
        for name, owner in self.owners.items():
            for kind, covariances, shapes in self._kinds():
                if owner == self.rank or (name, kind) in self.local:
                    covariances[name] = torch.zeros(
                        shapes[name], device=device, dtype=self.dtype
                    )

    def _kinds(self):
        """Each kind of covariance with its storage and shapes."""
        return (
            ("A", self.A_cov_dict, self.A_shapes),
            ("S", self.S_cov_dict, self.S_shapes),
        )

    def with_batch(
        self,
        collection_mask: Tensor | None = None,
        spans: tuple[Tensor, Tensor] | None = None,
    ):
        super().with_batch(collection_mask, spans)
        self._batches += 1
        self._inputs.clear()
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
        mask = self.collection_mask(module)
        assert mask is not None, "Collection mask not set for forward hook."

        if self._takes_earlier_input(name, module, a, mask):
            return

        # a: [N, S, I], collection mask: [N, S] -> select gradient-carrying positions
        a_bi = a[mask]  # [num_valid, I]
        self.num_positions[name] += a_bi.shape[0]

        # Augment with a ones column so A matches the [O, I+1] gradient layout
        # produced when the bias gradient is collected.
        if module._collect_bias:
            a_bi = torch.cat(
                [a_bi, a_bi.new_ones(a_bi.shape[0], 1)], dim=1
            )  # [num_valid, I+1]

        self._accumulate("A", name, a_bi)

    def _takes_earlier_input(
        self, name: str, module: nn.Module, a: Tensor, mask: Tensor
    ) -> bool:
        """Whether ``name``'s activation covariance is an earlier module's, which
        took the same input with the same mask this batch."""
        # The version counter changes if the input is modified in place.
        key = (a.data_ptr(), a.shape, a.stride(), a._version, module._collect_bias)
        earlier, earlier_mask, _ = self._inputs.get(key, (None, None, None))
        if earlier_mask is not mask:
            earlier = None
        if self._batches == 1 and earlier is not None:
            self.aliases[name] = earlier
            self.A_cov_dict.pop(name, None)
            return True
        if name in self.aliases:
            assert earlier == self.aliases[name], (
                f"{name} took {self.aliases[name]}'s input in the first batch but "
                "not in this one"
            )
            return True
        if self._batches == 1 or name in self.aliases.values():
            # Holding the input until the backward pass keeps its memory from
            # going to another tensor, which could then match its key.
            self._inputs[key] = (name, mask, a)
        return False

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        """Compute gradient covariance: G^T @ G."""
        name = assert_type(str, module._name)
        mask = self.collection_mask(module)
        self._inputs.clear()

        # g: [N, S, O], mask: [N, S] -> select gradient-carrying positions
        g_bo = g[mask]  # [num_valid, O]

        self._accumulate("S", name, g_bo)

    def _accumulate(self, kind: str, name: str, x: Tensor):
        """Add ``X^T @ X`` to ``name``'s covariance of ``kind``, where ``X`` stacks
        every rank's ``x``, in the accumulation dtype."""
        covariances = self.A_cov_dict if kind == "A" else self.S_cov_dict
        if self.owners is None or (name, kind) in self.local:
            add_gram(covariances[name], x)
            return

        def add(stacked: Tensor):
            # Padding rows are zero, so they add nothing.
            add_gram(covariances[name], stacked)

        # Sent in the model's dtype and cast on the owner, which moves less data.
        self.gathers.send(x, self._rows, self.owners[name], add)

    def _copy_aliases(self) -> None:
        """Give each alias's owner its earlier module's activation covariance."""
        for name, earlier in sorted(self.aliases.items()):
            self.num_positions[name] = self.num_positions[earlier]
            if self.owners is None:
                self.A_cov_dict[name] = self.A_cov_dict[earlier].clone()
                continue
            owner, sender = self.owners[name], self.owners[earlier]
            if self.rank == owner == sender:
                self.A_cov_dict[name] = self.A_cov_dict[earlier].clone()
            elif self.rank == sender:
                dist.send(self.A_cov_dict[earlier], owner)
            elif self.rank == owner:
                self.A_cov_dict[name] = torch.empty(
                    self.A_shapes[name],
                    device=self.shard_computer.device,
                    dtype=self.dtype,
                )
                dist.recv(self.A_cov_dict[name], sender)

    def process_batch(self, indices: list[int], **kwargs) -> None:
        """No per-batch processing needed for covariance collection."""
        pass

    def teardown(self) -> None:
        """Save covariance matrices to disk."""
        if self.owners is not None:
            self.gathers.flush()
            for name, kind in sorted(self.local):
                if kind == "A" and name in self.aliases:
                    continue
                owner = self.owners[name]
                covariances = self.A_cov_dict if kind == "A" else self.S_cov_dict
                dist.reduce(covariances[name], dst=owner)
                if owner != self.rank:
                    del covariances[name]
        self._copy_aliases()
        activation_path = os.path.join(self.path, "activation_sharded")
        gradient_path = os.path.join(self.path, "gradient_sharded")

        os.makedirs(activation_path, exist_ok=True)
        os.makedirs(gradient_path, exist_ok=True)
        self.logger.info(
            f"Saving sharded covariance matrices to {activation_path} "
            f"and {gradient_path}"
        )
        for covariances, shapes, path in (
            (self.A_cov_dict, self.A_shapes, activation_path),
            (self.S_cov_dict, self.S_shapes, gradient_path),
        ):
            if self.owners is not None:
                shards = owned_to_row_shards(
                    covariances,
                    shapes,
                    self.owners,
                    self.dtype,
                    self.shard_computer.device,
                )
            else:
                shards = covariances
            save_shards(
                shards, os.path.join(path, f"shard_{self.rank}.safetensors"), self.saves
            )
            del shards

        if dist.is_initialized():
            counts = torch.tensor(
                list(self.num_positions.values()),
                device=self.shard_computer.device,
            )
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            self.num_positions = dict(zip(self.num_positions, counts.tolist()))
