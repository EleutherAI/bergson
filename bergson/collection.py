import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, Value
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from bergson.config import AttentionConfig, IndexConfig, ReduceConfig
from bergson.data import create_index, pad_and_tensor
from bergson.gradients import GradientCollector, GradientProcessor
from bergson.peft import set_peft_enabled
from bergson.score.scorer import Scorer


def collect_gradients(
    model: PreTrainedModel,
    data: Dataset,
    processor: GradientProcessor,
    cfg: IndexConfig,
    *,
    batches: list[list[int]] | None = None,
    target_modules: set[str] | None = None,
    attention_cfgs: dict[str, AttentionConfig] | None = None,
    scorer: Scorer | None = None,
    reduce_cfg: ReduceConfig | None = None,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    score = scorer is not None
    save_index = not score and not cfg.skip_index

    # Batch size of one by default
    if batches is None:
        batches = [[idx] for idx in range(len(data))]

    print(
        f"Rank {rank} has {len(batches)} batches and thinks world "
        f"size is {dist.get_world_size()}."
    )

    # Mutable state for the GradientCollector callback
    mod_grads = {}
    preconditioners = processor.preconditioners

    # TODO: Handle this more elegantly
    dtype = torch.float32 if model.dtype == torch.float32 else torch.float16
    lo = torch.finfo(dtype).min
    hi = torch.finfo(dtype).max

    owned_modules: set[str] = set()
    module_to_rank: dict[str, int] = {}

    def callback(name: str, g: torch.Tensor):
        g = g.flatten(1).clamp_(lo, hi)
        # Keep gradients in original dtype for preconditioner computation
        mod_grads[name] = g
        if cfg.skip_preconditioners:
            if save_index:
                mod_grads[name] = g.to(dtype=dtype, device="cpu", non_blocking=True)
            else:
                mod_grads[name] = g.to(dtype=dtype)

    collector = GradientCollector(
        model.base_model,
        callback,
        processor,
        target_modules=target_modules,
        attention_cfgs=attention_cfgs or {},
    )

    # Determine which modules this rank owns for preconditioner computation
    if dist.is_initialized():
        num_devices = dist.get_world_size()
        # This list is sorted.
        available_modules = list(collector.shapes().keys())

        num_modules = len(available_modules)
        base, remainder = divmod(num_modules, num_devices)

        assert base > 0, "Each rank must own at least one module"

        start_idx = rank * base + min(rank, remainder)
        end_idx = start_idx + base + (1 if rank < remainder else 0)
        owned_modules = set(available_modules[start_idx:end_idx])

        for i, module_name in enumerate(available_modules):
            # Inverse of the start_idx formula
            module_to_rank[module_name] = (
                min(i // (base + 1), remainder - 1)
                if i < remainder * (base + 1)
                else remainder + (i - remainder * (base + 1)) // base
            )

        print(f"Rank {rank} owns {len(owned_modules)} modules")
    else:
        owned_modules = set(collector.shapes().keys())

    # Allocate space ahead of time for the gradients
    grad_sizes = {name: math.prod(s) for name, s in collector.shapes().items()}
    builder = (
        Builder(cfg.partial_run_path, data, grad_sizes, dtype, reduce_cfg)
        if save_index
        else None
    )

    per_doc_losses = torch.full(
        (len(data),),
        device=model.device,
        dtype=dtype,
        fill_value=0.0,
    )

    # rank != 0
    for indices in tqdm(batches, disable=False, desc="Building index"):
        batch = data[indices]
        x, y = pad_and_tensor(
            batch["input_ids"],  # type: ignore
            labels=batch.get("labels"),  # type: ignore
            device=model.device,
        )
        masks = y[:, 1:] != -100
        denoms = masks.sum(dim=1, dtype=dtype) if cfg.loss_reduction == "mean" else 1.0

        if cfg.loss_fn == "kl":
            with torch.inference_mode():
                set_peft_enabled(model, False)
                ref_lps = torch.log_softmax(model(x).logits[:, :-1], dim=-1)
                set_peft_enabled(model, True)

            with collector:
                ft_lps = torch.log_softmax(model(x).logits[:, :-1], dim=-1)

                # Compute average KL across all unmasked tokens
                kls = torch.sum(ft_lps.exp() * (ft_lps - ref_lps), dim=-1)
                losses = torch.sum(kls * masks, dim=-1) / denoms
                if "advantage" in batch:
                    losses *= torch.tensor(batch["advantage"], device=losses.device)

                losses.mean().backward()
        else:
            with collector:
                logits = model(x).logits[:, :-1]

                losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y[:, 1:].flatten(),
                    reduction="none",
                ).reshape_as(y[:, 1:])
                losses = losses.sum(1) / denoms
                if "advantage" in batch:
                    losses *= torch.tensor(batch["advantage"], device=losses.device)

                losses.mean().backward()

        model.zero_grad()

        # Send gradients to owning ranks and compute outer products there
        if not cfg.skip_preconditioners:
            exchange_preconditioner_gradients(
                mod_grads, preconditioners, module_to_rank, owned_modules, rank
            )

            # Convert mod_grads to the right dtype for save_index logic
            if save_index:
                for name in mod_grads:
                    mod_grads[name] = mod_grads[name].to(
                        device="cpu", dtype=dtype, non_blocking=True
                    )
            else:
                for name in mod_grads:
                    mod_grads[name] = mod_grads[name].to(dtype=dtype)

        if builder is not None:
            builder(indices, mod_grads)

        if score:
            scorer(indices, mod_grads)

        mod_grads.clear()
        per_doc_losses[indices] = losses.detach().type_as(per_doc_losses)

    if not cfg.skip_preconditioners:
        process_preconditioners(processor, preconditioners, len(data), grad_sizes, rank)

    if dist.is_initialized():
        dist.reduce(per_doc_losses, dst=0)

    if rank == 0:
        if cfg.drop_columns:
            data = data.remove_columns(["input_ids"])

        data = data.add_column(
            "loss",
            per_doc_losses.cpu().numpy(),
            feature=Value("float16" if dtype == torch.float16 else "float32"),
            new_fingerprint="loss",
        )

        data.save_to_disk(cfg.partial_run_path / "data.hf")

        processor.save(cfg.partial_run_path)

    # Make sure the gradients are written to disk
    if builder is not None:
        builder.flush()
        builder.dist_reduce()


class Builder:
    num_items: int

    grad_buffer: np.memmap

    reduce_cfg: ReduceConfig | None

    def __init__(
        self,
        path: Path,
        data: Dataset,
        grad_sizes: dict[str, int],
        dtype: torch.dtype,
        reduce_cfg: ReduceConfig | None = None,
    ):
        self.grad_sizes = grad_sizes
        self.num_items = len(data)
        self.reduce_cfg = reduce_cfg

        if reduce_cfg is not None:
            num_grads = 1
            self.in_memory_grad_buffer = torch.zeros(
                (num_grads, sum(self.grad_sizes.values())), dtype=torch.float32
            )
            np_dtype = np.float32
        else:
            num_grads = self.num_items
            self.in_memory_grad_buffer = None
            # TODO: Handle this more elegantly
            np_dtype = np.float32 if dtype == torch.float32 else np.float16

        self.grad_buffer = create_index(
            path,
            num_grads=num_grads,
            grad_sizes=self.grad_sizes,
            dtype=np_dtype,
            with_structure=False,
        )

    def reduce(self, indices: list[int], mod_grads: dict[str, torch.Tensor]):
        assert self.reduce_cfg is not None and self.in_memory_grad_buffer is not None

        if self.reduce_cfg.unit_normalize:
            ssqs = torch.zeros(len(indices))
            for mod_grad in mod_grads.values():
                ssqs += mod_grad.pow(2).sum(dim=-1)
            norms = ssqs.sqrt()
        else:
            norms = torch.ones(len(indices))

        offset = 0
        for module_name in self.grad_sizes.keys():
            mod_grads[module_name] /= norms.unsqueeze(1)

            grads = mod_grads[module_name].sum(dim=0).to(torch.float32)
            self.in_memory_grad_buffer[
                0, offset : offset + mod_grads[module_name].shape[1]
            ] += grads
            offset += mod_grads[module_name].shape[1]

    def __call__(self, indices: list[int], mod_grads: dict[str, torch.Tensor]):
        torch.cuda.synchronize()

        if self.reduce_cfg is not None:
            self.reduce(indices, mod_grads)
        else:
            # It turns out that it's very important for efficiency to write the
            # gradients sequentially instead of first concatenating them, then
            # writing to one vector
            offset = 0
            for module_name in self.grad_sizes.keys():
                self.grad_buffer[
                    indices, offset : offset + mod_grads[module_name].shape[1]
                ] = mod_grads[module_name].numpy()
                offset += mod_grads[module_name].shape[1]

    def flush(self):
        self.grad_buffer.flush()

    def dist_reduce(self):
        if self.reduce_cfg is None:
            return

        assert self.in_memory_grad_buffer is not None

        self.in_memory_grad_buffer = self.in_memory_grad_buffer.cuda()

        if dist.is_initialized():
            dist.reduce(self.in_memory_grad_buffer, dst=0, op=dist.ReduceOp.SUM)

        if self.reduce_cfg.method == "mean":
            self.in_memory_grad_buffer /= self.num_items

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            self.grad_buffer[:] = (
                self.in_memory_grad_buffer.cpu().numpy().astype(self.grad_buffer.dtype)
            )

        self.in_memory_grad_buffer = self.in_memory_grad_buffer.cpu()


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


def process_preconditioners(
    processor: GradientProcessor,
    preconditioners: dict[str, torch.Tensor],
    len_data: int,
    grad_sizes: dict[str, int],
    rank: int,
):
    """
    Aggregate preconditioners across ranks and compute their eigen decomposition
    distributed across all ranks.
    """
    preconditioners_eigen = {}

    device = next(iter(preconditioners.values())).device
    dtype = next(iter(preconditioners.values())).dtype

    if rank == 0:
        print("Saving preconditioners...")

    for name, prec in preconditioners.items():
        preconditioners[name] = (prec / len_data).cpu()

    if rank == 0:
        print("Computing preconditioner eigen decompositions...")

    for name in preconditioners.keys():
        prec = preconditioners[name].to(dtype=torch.float64, device=device)
        eigvals, eigvecs = torch.linalg.eigh(prec)
        preconditioners_eigen[name] = (
            eigvals.to(dtype=dtype).contiguous().cpu(),
            eigvecs.to(dtype=dtype).contiguous().cpu(),
        )

    if rank == 0:
        print("Gathering preconditioners...")

    cpu_group = dist.new_group(backend="gloo")

    for name, grad_size in grad_sizes.items():
        if name in preconditioners:
            local_prec = preconditioners[name]
            del preconditioners[name]
        else:
            local_prec = torch.zeros([grad_size, grad_size], dtype=dtype, device="cpu")

        dist.reduce(local_prec, dst=0, op=dist.ReduceOp.SUM, group=cpu_group)

        if rank == 0:
            preconditioners[name] = local_prec

    if rank == 0:
        processor.preconditioners = preconditioners

        print("Gathering eigen decompositions...")

    for name, grad_size in grad_sizes.items():
        prec_size = torch.Size([grad_size, grad_size])
        if name not in preconditioners_eigen:
            eigval = torch.zeros(prec_size[0], dtype=dtype)
            eigvec = torch.zeros(prec_size, dtype=dtype)
        else:
            eigval, eigvec = preconditioners_eigen[name]

        dist.reduce(eigval, dst=0, op=dist.ReduceOp.SUM, group=cpu_group)
        dist.reduce(eigvec, dst=0, op=dist.ReduceOp.SUM, group=cpu_group)

        if rank == 0:
            preconditioners_eigen[name] = (eigval, eigvec)

    if rank == 0:
        processor.preconditioners_eigen = preconditioners_eigen
