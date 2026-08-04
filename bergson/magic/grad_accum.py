"""Micro-batched gradient accumulation for the MAGIC trainer.

Splitting a batch into micro-batches and summing their rescaled gradients
reproduces the full-batch gradient (up to float associativity) while bounding
peak activation memory to one micro-batch. Differentiating through such a
step can't simply trace the accumulation loop — every micro-graph would stay
alive for the outer VJP — so :func:`microbatch_step_vjp` decomposes the step
and frees each micro-graph as it goes.

Faithful replay requires rewinding both the CPU and CUDA RNGs (CUDA dropout
draws from the CUDA generator); the snapshot helpers here are shared with the
trainer for that reason.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch import nn

from ..distributed import grad_tree
from .swap import swap_parameters

if TYPE_CHECKING:
    from .trainer import TrainerState


def maybe_get_cuda_rng_state() -> torch.Tensor:
    """Get the CUDA RNG state if CUDA is initialized, otherwise return zeros."""
    if torch.cuda.is_initialized():
        return torch.cuda.random.get_rng_state()

    # This corresponds to a manual seed of 0
    return torch.zeros(16, dtype=torch.uint8)


def maybe_set_cuda_rng_state(state: torch.Tensor) -> None:
    """Restore a CUDA RNG state; no-op when CUDA isn't initialized.

    CUDA-side stochastic ops (dropout) draw from this generator rather than
    the CPU one, so replaying a step means rewinding both.
    """
    if torch.cuda.is_initialized():
        torch.cuda.random.set_rng_state(state)


def rng_snapshot() -> tuple[torch.Tensor, torch.Tensor]:
    """Capture both generators, for rewinding with :func:`rng_restore`."""
    return torch.random.get_rng_state(), maybe_get_cuda_rng_state()


def rng_restore(snapshot: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Rewind both generators to a :func:`rng_snapshot`."""
    cpu, cuda = snapshot
    torch.random.set_rng_state(cpu)
    maybe_set_cuda_rng_state(cuda)


def loss_denom(batch: dict) -> float:
    """The denominator ``weighted_causal_lm_ce``'s "mean" reduction uses.

    The loss divides by the number of valid label tokens — ``shift_loss_mask``
    when present, otherwise non-ignored shifted labels — clamped to at least 1,
    so an all-empty batch has loss 0 rather than NaN. Micro-batch accumulation
    must rescale by this to stay exact.
    """
    mask = batch.get("shift_loss_mask")
    if mask is not None:
        return max(float(mask.sum()), 1.0)
    labels = batch.get("labels")
    if labels is not None:
        return max(float((labels[:, 1:] != -100).sum()), 1.0)
    return float(batch["input_ids"].shape[1] - 1)


def split_batch(inputs: dict, n: int) -> list[dict]:
    """Split a collated batch dict into up to ``n`` micro-batches along dim 0.
    Tensors whose leading dim equals the batch size are sliced; everything
    else is shared by reference."""
    tensors = [
        v for v in inputs.values() if isinstance(v, torch.Tensor) and v.ndim >= 1
    ]
    batch_size = tensors[0].shape[0]
    n = max(1, min(n, batch_size))
    sizes = [batch_size // n + (1 if i < batch_size % n else 0) for i in range(n)]
    micro_batches: list[dict] = []
    offset = 0
    for size in sizes:
        micro_batch = {}
        for key, value in inputs.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 1
                and value.shape[0] == batch_size
            ):
                micro_batch[key] = value[offset : offset + size]
            else:
                micro_batch[key] = value
        micro_batches.append(micro_batch)
        offset += size
    return micro_batches


def accumulate_grads(
    model,
    params,
    inputs: dict,
    grad_accum_steps: int,
    *,
    create_graph: bool,
    rng_snapshots: list | None = None,
) -> tuple[dict, float]:
    """Sum normalized per-micro-batch gradients into the full-batch gradient.

    Each micro-batch's loss is normalized by its own token count; scaling by
    ``denom_i / D`` (D = full-batch token count) makes the sum identical to the
    full-batch gradient up to float associativity. Returns ``(grads, loss)``.

    The caller seeds the RNG; the micro-batch forwards draw from it in order.
    Pass ``rng_snapshots`` to record each micro-batch's pre-forward RNG state
    so a later pass can replay the same draws (see :func:`microbatch_step_vjp`).
    """
    total_denom = loss_denom(inputs)
    grads: dict | None = None
    last_loss = 0.0
    for micro_batch in split_batch(inputs, grad_accum_steps):
        if rng_snapshots is not None:
            rng_snapshots.append(rng_snapshot())
        outputs = model(**micro_batch)

        # Two output types are supported: HuggingFace (a dict/dataclass with a
        # "loss" field) and "raw loss" (a scalar loss Tensor).
        micro_loss = outputs.loss if hasattr(outputs, "loss") else outputs
        assert isinstance(micro_loss, torch.Tensor), "Loss must be a Tensor"
        coef = loss_denom(micro_batch) / total_denom
        last_loss += float(micro_loss.detach()) * coef
        micro_grads = grad_tree(micro_loss * coef, params, create_graph=create_graph)
        if grads is None:
            grads = dict(micro_grads)
        else:
            for key in grads:
                if micro_grads[key] is not None:
                    grads[key] = (
                        grads[key] + micro_grads[key]
                        if grads[key] is not None
                        else micro_grads[key]
                    )
    assert grads is not None
    return grads, last_loss


def microbatch_step_vjp(
    model: nn.Module,
    apply_update: Callable[..., "TrainerState"],
    fwd_state: "TrainerState",
    inputs: dict[str, Any],
    param_grads: dict[str, torch.Tensor],
    opt_grads: list[torch.Tensor],
    data_weights: torch.Tensor,
    *,
    fsdp: bool = False,
    max_grad_norm: float | None = None,
    grad_accum_steps: int = 1,
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """VJP through one training step, one micro-batch graph at a time.

    Equivalent to differentiating a traced step, with peak memory bounded to
    one micro-batch. Exploits ``grads = Σ_i c_i · grad_tree(L_i)`` in two
    stages, driven by the next state's cotangents (``param_grads``,
    ``opt_grads``):

      A. VJP through only the all-reduce/clip/update graph (``apply_update``),
         yielding a cotangent on the combined gradient plus the direct-path
         cotangents on the incoming params/opt-state.
      B. Per micro-batch, recompute its gradient graph, VJP it against the
         combined-gradient cotangent, and free it before the next micro-batch.

    Both stages run the model and so must rewind the RNG.

    Returns ``(param_cotangents, opt_cotangents, weight_cotangents)`` for the
    incoming state and this batch's ``data_weights``.
    """
    state_params = fwd_state.params
    buffers = fwd_state.buffers
    flat_inputs = fwd_state.differentiable_tensors()
    param_keys = list(param_grads.keys())
    param_index = {key: i for i, key in enumerate(param_keys)}
    num_params = len(param_keys)
    num_inputs = len(flat_inputs)

    # Stage 0: combined gradient values (no graph) to drive the update.
    # Rewind as in Trainer.step and record where each micro-batch started.
    torch.random.set_rng_state(fwd_state.cpu_rng_state)
    maybe_set_cuda_rng_state(fwd_state.cuda_rng_state)
    rng_snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    with swap_parameters(model, state_params, buffers, preserve_graph=False) as params:
        grads_detached, _ = accumulate_grads(
            model,
            params,
            inputs,
            grad_accum_steps,
            create_graph=False,
            rng_snapshots=rng_snapshots,
        )
    grad_variables = {
        key: value.detach().clone().requires_grad_(True)
        for key, value in grads_detached.items()
    }

    # Stage A: VJP through the update only -> a cotangent on the combined
    # gradient, plus the direct-path cotangents on the incoming
    # params/opt-state.
    updated_state = apply_update(
        fwd_state,
        grad_variables,
        inplace=False,
        trace=True,
        fsdp=fsdp,
        max_grad_norm=max_grad_norm,
    )
    stage_a_results = list(
        torch.autograd.grad(
            updated_state.differentiable_tensors(),
            flat_inputs + list(grad_variables.values()),
            grad_outputs=list(param_grads.values()) + opt_grads,
            allow_unused=True,
        )
    )
    direct_cotangents = stage_a_results[:num_inputs]
    grad_cotangent = {
        key: (cot if cot is not None else torch.zeros_like(grad_variables[key]))
        for key, cot in zip(grad_variables.keys(), stage_a_results[num_inputs:])
    }

    def _or_zero(cotangent, reference):
        return cotangent if cotangent is not None else torch.zeros_like(reference)

    param_cotangents = [
        _or_zero(direct_cotangents[i], flat_inputs[i]).clone()
        for i in range(num_params)
    ]
    opt_cotangents = [
        _or_zero(direct_cotangents[i], flat_inputs[i])
        for i in range(num_params, num_inputs)
    ]

    # Free Stage A's update graph and gradient copies before Stage B's
    # double backward; keeping them alive raises the peak by several GiB.
    del grads_detached, grad_variables, updated_state, stage_a_results
    del direct_cotangents

    # ``example_weight`` is one indexing of ``data_weights`` shared by all
    # micro-batch slices, so accumulate cotangents on it and map them back
    # to ``data_weights`` once after the loop.
    example_weight = inputs.get("example_weight")
    example_weight_cotangent = (
        torch.zeros_like(example_weight) if example_weight is not None else None
    )

    # Stage B: per micro-batch through-gradient VJP, one graph at a time.
    total_denom = loss_denom(inputs)
    micro_batches = split_batch(inputs, grad_accum_steps)
    assert len(micro_batches) == len(rng_snapshots)
    for micro_batch, snapshot in zip(micro_batches, rng_snapshots):
        with swap_parameters(
            model, state_params, buffers, preserve_graph=True
        ) as params:
            rng_restore(snapshot)
            outputs = model(**micro_batch)
            micro_loss = outputs.loss if hasattr(outputs, "loss") else outputs
            coef = loss_denom(micro_batch) / total_denom
            micro_grads = grad_tree(micro_loss * coef, params, create_graph=True)
            grad_keys = list(micro_grads.keys())
            targets = [params[key] for key in grad_keys]
            if example_weight is not None:
                targets = targets + [example_weight]
            contributions = torch.autograd.grad(
                [micro_grads[key] for key in grad_keys],
                targets,
                grad_outputs=[grad_cotangent[key] for key in grad_keys],
                allow_unused=True,
            )
        for i, key in enumerate(grad_keys):
            if contributions[i] is not None:
                param_cotangents[param_index[key]] = (
                    param_cotangents[param_index[key]] + contributions[i]
                )
        if example_weight_cotangent is not None and contributions[-1] is not None:
            example_weight_cotangent = example_weight_cotangent + contributions[-1]
        del micro_grads, contributions, outputs, micro_loss

    weight_cotangent = torch.zeros_like(data_weights)
    if (
        example_weight is not None
        and example_weight_cotangent is not None
        and float(example_weight_cotangent.abs().sum()) > 0
    ):
        if example_weight is data_weights:
            weight_cotangent = weight_cotangent + example_weight_cotangent
        else:
            (data_weights_grad,) = torch.autograd.grad(
                example_weight,
                data_weights,
                grad_outputs=example_weight_cotangent,
                allow_unused=True,
            )
            if data_weights_grad is not None:
                weight_cotangent = weight_cotangent + data_weights_grad

    if dist.is_initialized() and not fsdp:
        # Same 1/world_size correction as the single-shot path in
        # `Trainer.backward`: a document only contributes to 1/world_size of
        # the averaged gradient the all-reduce produces.
        weight_cotangent = weight_cotangent / dist.get_world_size()

    param_cotangent_dict = {
        key: param_cotangents[i] for i, key in enumerate(param_keys)
    }
    return param_cotangent_dict, opt_cotangents, weight_cotangent
