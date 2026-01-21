import math
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import Dataset, Value
from jaxtyping import Float
from torch import Tensor

from bergson.collector.collector import HookCollectorBase
from bergson.config import IndexConfig, ReduceConfig
from bergson.data import Builder
from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    LayerAdapter,
)
from bergson.process_preconditioners import process_preconditioners
from bergson.score.scorer import Scorer
from bergson.utils.utils import assert_type, get_gradient_dtype


@dataclass(kw_only=True)
class GradientCollectorWithDistributedPreconditioners(HookCollectorBase):
    """
    Collects per-sample gradients from model layers and writes them to disk.
    Preconditioners are distributed across nodes, and data from each node is
    distributed to each preconditioner at every step. This enables the computation
    of preconditioners that are too large to fit on a single device.

    - For each forward/backward hook, we compute the the gradient or a low-rank
    approximation via random projections, if cfg.projection_dim is set.
    - Supports also normalization via Adam or Adafactor normalizers.
    - Uses Builder for index construction and gradient saving.
    - Also supports Scorer for on-the-fly scoring of gradients.
    """

    data: Dataset
    """The dataset being processed."""

    cfg: IndexConfig
    """Configuration for gradient index."""

    mod_grads: dict = field(default_factory=dict)
    """Temporary storage for gradients during a batch, keyed by module name."""

    reduce_cfg: ReduceConfig | None = None
    """Configuration for in-run gradient reduction."""

    builder: Builder | None = None
    """Handles writing gradients to disk. Created in setup() if save_index is True."""

    scorer: Scorer | None = None
    """Optional scorer for computing scores instead of building an index."""

    def setup(self) -> None:
        """
        Initialize collector state.

        Sets up a Builder for gradient storage if not using a Scorer.
        """
        assert isinstance(
            self.model.device, torch.device
        ), "Model device is not set correctly"
        if self.cfg.include_bias and self.processor.normalizers is not None:
            raise NotImplementedError(
                "Bias with normalizers not supported yet, "
                "consider disabling bias inclusion for now."
            )

        self.owned_modules: set[str] = set()
        self.module_to_rank: dict[str, int] = {}

        self.save_dtype = get_gradient_dtype(self.model)
        self.lo = torch.finfo(self.save_dtype).min
        self.hi = torch.finfo(self.save_dtype).max

        self.per_doc_losses = torch.full(
            (len(self.data),),
            device=self.model.device,
            dtype=torch.float32,
            fill_value=0.0,
        )

        # Compute whether we need to save the index
        self.save_index = self.scorer is None and not self.cfg.skip_index

        if self.save_index:
            grad_sizes = {name: math.prod(s) for name, s in self.shapes().items()}
            self.builder = Builder(
                self.cfg.partial_run_path,
                self.data,
                grad_sizes,
                self.save_dtype,
                self.reduce_cfg,
            )
        else:
            self.builder = None

        if dist.is_initialized():
            rank = dist.get_rank()
            num_devices = dist.get_world_size()
            available_modules = list(self.shapes().keys())

            num_modules = len(available_modules)
            base, remainder = divmod(num_modules, num_devices)

            assert base > 0, "Each rank must own at least one module"

            start_idx = rank * base + min(rank, remainder)
            end_idx = start_idx + base + (1 if rank < remainder else 0)
            self.owned_modules = set(available_modules[start_idx:end_idx])

            for i, module_name in enumerate(available_modules):
                # Inverse of the start_idx formula
                self.module_to_rank[module_name] = (
                    min(i // (base + 1), remainder - 1)
                    if i < remainder * (base + 1)
                    else remainder + (i - remainder * (base + 1)) // base
                )

            print(f"Rank {rank} owns {len(self.owned_modules)} modules")
        else:
            self.owned_modules = set(self.shapes().keys())

    def forward_hook(self, module: nn.Module, a: Float[Tensor, "N S I"]) -> None:
        """
        Cache activations for gradient computation with normalizer preprocessing
        and compress via random projection if configured.
        Stores result in module._inputs for use in backward_hook.
        """
        p = self.processor.projection_dim
        name = assert_type(str, module._name)
        i = getattr(module, LayerAdapter.in_attr(module))
        normalizer = self.processor.normalizers.get(name)

        if isinstance(normalizer, AdamNormalizer):
            module._inputs = a
            return
        if isinstance(normalizer, AdafactorNormalizer):
            a_factor = normalizer.col.add(1e-30)
            a_factor = a_factor.rsqrt()
            a = a * a_factor.type_as(a)  # [N, S, I] * [I] → [N, S, I]

        if module._has_bias:
            # Append ones to activation for bias term
            ones = torch.ones(a.size(0), a.size(1), 1, device=a.device, dtype=a.dtype)
            a = torch.cat([a, ones], dim=-1)
            i = i + 1
            setattr(module, LayerAdapter.in_attr(module), i)
        if p is not None:
            a_projection = self.projection(name, p, i, "right", a.device, a.dtype).T
            a = a @ a_projection  # type: ignore
        # set module._inputs to a
        module._inputs = a

    @HookCollectorBase.split_attention_heads
    def backward_hook(self, module: nn.Module, g: Float[Tensor, "N S O"]):
        """
        Compute per-sample gradient and store in mod_grads.

        Computes gradient as outer product g.T @ a (again with optional projection and
        normalization).
        """
        a = module._inputs  # [N, S, I/q]

        assert isinstance(a, torch.Tensor), "Activation cache missing for module"
        name = assert_type(str, module._name)
        p = self.processor.projection_dim
        i = getattr(module, LayerAdapter.in_attr(module))
        o = getattr(module, LayerAdapter.out_attr(module))
        normalizer = self.processor.normalizers.get(name)

        if isinstance(normalizer, AdamNormalizer):
            full_gradient = g.mT @ a  # [N, O, S] @ [N, S, I] → [N, O, I]
            P = normalizer.normalize_(full_gradient)
            if p is not None:
                g_projection = self.projection(name, p, o, "left", g.device, g.dtype)
                a_projection = self.projection(name, p, i, "right", g.device, g.dtype).T
                P = g_projection @ P @ a_projection
        else:
            if isinstance(normalizer, AdafactorNormalizer):
                g_factor = normalizer.row.add(1e-30)
                g_factor = g_factor.mean().sqrt() * g_factor.rsqrt()
                g = g * g_factor.type_as(g)  # [N, S, O] * [O] → [N, S, O]

            if p is not None:
                g_projection = self.projection(name, p, o, "left", g.device, g.dtype)
                g = g @ g_projection.T  # [N, S, p]

            P = g.mT @ a  # [N, O/p, S] @ [N, S, I/q] → [N, O/p, I/q]

        P = P.flatten(1).clamp_(self.lo, self.hi)

        # Keep gradients in original dtype for preconditioner computation
        self.mod_grads[name] = P

        if self.cfg.skip_preconditioners:
            if self.save_index:
                # Asynchronously move the gradient to CPU and convert to the final dtype
                self.mod_grads[name] = P.to(
                    device="cpu", dtype=self.save_dtype, non_blocking=True
                )
            else:
                self.mod_grads[name] = P.to(dtype=self.save_dtype)

        del module._inputs

    def process_batch(self, indices: list[int], **kwargs):
        """Process collected gradients for a batch and update losses."""
        losses = kwargs.get("losses")
        assert losses is not None, "losses must be provided in kwargs"

        # Send gradients to owning ranks and compute outer products there
        if not self.cfg.skip_preconditioners:
            exchange_preconditioner_gradients(
                self.mod_grads,
                self.processor.preconditioners,
                self.module_to_rank,
                self.owned_modules,
                self.rank,
            )

            # Convert mod_grads to the right dtype for save_index logic
            if self.save_index:
                for name in self.mod_grads:
                    self.mod_grads[name] = self.mod_grads[name].to(
                        device="cpu", dtype=self.save_dtype, non_blocking=True
                    )
            else:
                for name in self.mod_grads:
                    self.mod_grads[name] = self.mod_grads[name].to(
                        dtype=self.save_dtype
                    )

        if self.builder:
            self.builder(indices, self.mod_grads)
        if self.scorer:
            self.scorer(indices, self.mod_grads)
        self.mod_grads.clear()
        self.per_doc_losses[indices] = losses.detach().type_as(self.per_doc_losses)

    def teardown(self):
        """
        Finalize gradient collection, save results and flush/reduce the Builder.
        """
        assert isinstance(
            self.cfg, IndexConfig
        ), "cfg is required for GradientCollector"  # pleasing type checker
        if dist.is_initialized():
            dist.reduce(self.per_doc_losses, dst=0)

        grad_sizes = {name: math.prod(s) for name, s in self.shapes().items()}
        if self.processor.preconditioners:
            process_preconditioners(
                self.processor,
                self.processor.preconditioners,
                len(self.data),
                grad_sizes,
                self.rank,
            )

        if self.rank == 0:
            if self.cfg.drop_columns:
                self.data = self.data.remove_columns(["input_ids"])

            self.data = self.data.add_column(
                "loss",
                self.per_doc_losses.cpu().numpy(),
                feature=Value("float32"),
                new_fingerprint="loss",
            )

            self.data.save_to_disk(self.cfg.partial_run_path / "data.hf")

            self.processor.save(self.cfg.partial_run_path)

        # Flush and reduce builder if it exists
        if self.builder is not None:
            self.builder.flush()
            self.builder.dist_reduce()


def exchange_preconditioner_gradients(
    mod_grads: dict[str, torch.Tensor],
    preconditioners: dict[str, torch.Tensor],
    module_to_rank: dict[str, int],
    owned_modules: set[str],
    rank: int,
):
    """
    Send gradients to the ranks that own their preconditioners, and accumulate
    outer products on the owning ranks.
    Each rank sends gradients for modules it doesn't own to the owning ranks,
    and receives gradients for modules it owns to compute outer products.
    """
    # Process current rank data for owned modules
    for name, g in mod_grads.items():
        if name not in owned_modules:
            continue

        g = g.float()
        if name in preconditioners:
            preconditioners[name].addmm_(g.mT, g)
        else:
            preconditioners[name] = g.mT @ g

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    device = next(iter(mod_grads.values())).device

    module_names = list(mod_grads.keys())
    module_numel = {n: int(mod_grads[n].numel()) for n in module_names}

    current_rank_chunk = torch.empty(0, device=device, dtype=torch.float32)

    # Flatten batch dimension: all to all works on contiguous 1-D tensors
    send_chunks = [
        (
            current_rank_chunk
            if dest == rank
            else torch.cat(
                [
                    mod_grads[name].flatten()
                    for name in module_names
                    if module_to_rank[name] == dest
                ]
            )
        )
        for dest in range(world_size)
    ]

    # --- collective exchange of gradient sizes in order of mod_grads ---
    send_sizes = torch.tensor(
        [t.numel() for t in send_chunks], device=device, dtype=torch.int64
    )
    recv_sizes = torch.empty_like(send_sizes)

    dist.all_to_all_single(recv_sizes, send_sizes)

    # --- collective exchange of gradient in order of mod_grads ---
    send_buf = torch.cat(send_chunks)
    recv_buf = torch.empty(
        int(recv_sizes.sum().item()), device=device, dtype=torch.float32
    )

    dist.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=recv_sizes.tolist(),
        input_split_sizes=send_sizes.tolist(),
    )

    # Unpack gradients in src-rank order
    # Within each src partition, modules are in fixed order.
    offset = 0
    for src_rank in range(world_size):
        part_len = int(recv_sizes[src_rank].item())
        part = recv_buf[offset : offset + part_len]
        offset += part_len

        if part_len == 0 or src_rank == rank:
            continue

        p = 0
        for name in owned_modules:
            n = module_numel[name]
            flat = part[p : p + n]
            p += n

            feature_dim = mod_grads[name].shape[-1]
            g = flat.to(device, non_blocking=True).view(-1, feature_dim).float()

            if name in preconditioners:
                preconditioners[name].addmm_(g.mT, g)
            else:
                preconditioners[name] = g.mT @ g
