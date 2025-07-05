import math
import random
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, Value
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from .data import create_index, pad_and_tensor
from .gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientCollector,
    GradientProcessor,
    Normalizer,
)


def collect_gradients(
    model: PreTrainedModel,
    data: Dataset,
    processor: GradientProcessor,
    path: str,
    *,
    batches: list[list[int]] | None = None,
    skip_preconditioners: bool = False,
    target_modules: set[str] | None = None,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Batch size of one by default
    if batches is None:
        batches = [[idx] for idx in range(len(data))]

    # Mutable state for the GradientCollector callback
    mod_grads = {}
    preconditioners = {}

    # TODO: Handle this more elegantly
    lo = torch.finfo(torch.float16).min
    hi = torch.finfo(torch.float16).max

    def callback(name: str, g: torch.Tensor):
        g = g.flatten(1).clamp_(lo, hi)

        # Asynchronously move the gradient to CPU and convert to fp16
        mod_grads[name] = g.to(device="cpu", dtype=torch.float16, non_blocking=True)

        # Compute the outer product of the flattened gradient
        if not skip_preconditioners:
            g = g.float()
            preconditioner = preconditioners.get(name, None)
            if preconditioner is None:
                preconditioners[name] = g.mT @ g
            else:
                preconditioner.addmm_(g.mT, g)

    collector = GradientCollector(
        model.base_model,
        callback,
        processor,
        target_modules=target_modules,
    )
    
    # Allocate space ahead of time for the gradients
    grad_sizes = {name: math.prod(s) for name, s in collector.shapes().items()}
    
    # Allocate structured space ahead of time for the gradients
    grad_buffer = create_index(
        path,
        num_grads=len(data),
        grad_sizes=grad_sizes,
        dtype=np.float16
    )
    
    per_doc_losses = torch.full(
        (len(data),),
        device=model.device,
        dtype=torch.float16,
        fill_value=0.0,
    )

    for indices in tqdm(batches, disable=rank != 0, desc="Building index"):
        batch = data[indices]
        x, y = pad_and_tensor(
            batch["input_ids"],  # type: ignore
            labels=batch.get("labels"),  # type: ignore
            device=model.device,
        )

        with collector:
            logits = model(x).logits

            losses = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                y[:, 1:].flatten(),
                reduction="none",
            ).reshape_as(y[:, 1:])

            masks = y[:, 1:] != -100
            denoms = masks.sum(dim=1, dtype=logits.dtype)
            losses = losses.sum(1).div(denoms)
            losses.mean().backward()

            model.zero_grad()

        # It turns out that it's very important for efficiency to write the gradients
        # sequentially instead of first concatenating them and then writing to one vector.
        for layer_name in mod_grads.keys():
            if layer_name in mod_grads:
                grad_buffer[layer_name][indices] = mod_grads[layer_name].numpy()

        mod_grads.clear()
        per_doc_losses[indices] = losses.detach().type_as(per_doc_losses)

    chols = {}
    for name, prec in preconditioners.items():
        if dist.is_initialized():
            dist.all_reduce(prec)

        L, info = torch.linalg.cholesky_ex(prec / len(data))
        if info.any() and rank == 0:
            print(f"Warning: {name} has a singular second moment matrix.")

        chols[name] = L

    processor.preconditioners = chols
    if dist.is_initialized():
        dist.reduce(per_doc_losses, dst=0)

    if rank == 0:
        data = data.add_column(
            "loss",
            per_doc_losses.cpu().numpy(),
            feature=Value("float16"),
            new_fingerprint="loss",
        )
        data.save_to_disk(path + "/data.hf")

        processor.save(path)

    # Make sure the gradients are written to disk
    grad_buffer.flush()


def fit_normalizers(
    model: PreTrainedModel,
    data: Dataset,
    batches: list[list[int]],
    *,
    kind: Literal["adafactor", "adam"] = "adafactor",
    target_modules: set[str] | None = None,
) -> dict[str, Normalizer]:
    """
    Estimate the second moments of the model's gradients using a subset of the dataset.
    """
    normalizers: dict[str, Normalizer] = {}
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Just to make the pbar more accurate
    rng = random.Random(0)
    rng.shuffle(batches)

    def adafactor_update(name: str, g: torch.Tensor):
        # We follow the tensor2tensor implementation of Adafactor, which
        # takes the mean rather than summing over the rows and columns.
        # row: mean over columns, shape [O]
        sq = g.float().square_().sum(0)
        row_acc = sq.mean(dim=1)
        # col: mean over rows,    shape [I]
        col_acc = sq.mean(dim=0)

        if (normalizer := normalizers.get(name)) is None:
            # initialize accumulators at zero
            normalizers[name] = normalizer = AdafactorNormalizer(
                torch.zeros_like(row_acc),
                torch.zeros_like(col_acc),
            )
        else:
            assert isinstance(normalizer, AdafactorNormalizer)

        # in‐place accumulate
        normalizer.row.add_(row_acc)
        normalizer.col.add_(col_acc)

    def adam_update(name: str, g: torch.Tensor):
        sq = g.square_().float().sum(0)

        # initialize accumulators at zero
        if (normalizer := normalizers.get(name)) is None:
            normalizers[name] = normalizer = AdamNormalizer(torch.zeros_like(sq))
        else:
            assert isinstance(normalizer, AdamNormalizer)

        # in‐place accumulate
        normalizer.avg_sq.add_(sq)

    callback = adafactor_update if kind == "adafactor" else adam_update

    for indices in tqdm(batches, disable=rank != 0, desc="Estimating normalizers"):
        batch = data[indices]

        with GradientCollector(
            model.base_model,
            closure=callback,
            target_modules=target_modules,
        ):
            x, y = pad_and_tensor(
                batch["input_ids"],  # type: ignore
                labels=batch.get("labels", None),  # type: ignore
                device=model.device,
            )
            model(x, labels=y).loss.backward()
            model.zero_grad()

    # Divide by the number of documents processed and average across all ranks
    for normalizer in normalizers.values():
        if isinstance(normalizer, AdamNormalizer):
            normalizer.avg_sq.div_(len(data))

            if dist.is_initialized():
                dist.all_reduce(normalizer.avg_sq, op=dist.ReduceOp.AVG)

        elif isinstance(normalizer, AdafactorNormalizer):
            normalizer.row.div_(len(data))
            normalizer.col.div_(len(data))

            if dist.is_initialized():
                dist.all_reduce(normalizer.row, op=dist.ReduceOp.AVG)
                dist.all_reduce(normalizer.col, op=dist.ReduceOp.AVG)

    return normalizers