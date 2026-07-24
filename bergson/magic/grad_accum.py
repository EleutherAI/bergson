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
    """The loss denominator ``weighted_causal_lm_ce`` uses for ``batch``.

    Weighted CE normalizes by ``shift_loss_mask.sum()`` — the number of valid
    label tokens — falling back to ``T-1`` when no mask is present.
    Micro-batch accumulation must rescale by this to stay exact.
    """
    mask = batch.get("shift_loss_mask")
    if mask is not None:
        return float(mask.sum())
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
    micro, off = [], 0
    for s in sizes:
        mb = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == batch_size:
                mb[k] = v[off : off + s]
            else:
                mb[k] = v
        micro.append(mb)
        off += s
    return micro


def nonempty_microbatches(inputs: dict, n: int) -> list[dict]:
    """:func:`split_batch`, dropping micro-batches with no valid label tokens."""
    return [mb for mb in split_batch(inputs, n) if loss_denom(mb) > 0]


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
    assert total_denom > 0, "Batch has no valid label tokens"
    grads: dict | None = None
    last_loss = 0.0
    # Skip micro-batches with no valid tokens: they contribute zero gradient,
    # and their loss is 0/0. Filtering is deterministic, so the VJP replay
    # (which filters identically) stays aligned with the recorded snapshots.
    for mb in nonempty_microbatches(inputs, grad_accum_steps):
        if rng_snapshots is not None:
            rng_snapshots.append(rng_snapshot())
        outputs = model(**mb)

        # Two output types are supported: HuggingFace (a dict/dataclass with a
        # "loss" field) and "raw loss" (a scalar loss Tensor).
        loss_i = outputs.loss if hasattr(outputs, "loss") else outputs
        assert isinstance(loss_i, torch.Tensor), "Loss must be a Tensor"
        coef = loss_denom(mb) / total_denom
        last_loss += float(loss_i.detach()) * coef
        g_i = grad_tree(loss_i * coef, params, create_graph=create_graph)
        if grads is None:
            grads = {k: v for k, v in g_i.items()}
        else:
            for k in grads:
                if g_i[k] is not None:
                    grads[k] = grads[k] + g_i[k] if grads[k] is not None else g_i[k]
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
         yielding a cotangent ``g_bar`` on the combined gradient plus the
         direct-path cotangents on the incoming params/opt-state.
      B. Per micro-batch, recompute its gradient graph, VJP it against
         ``g_bar``, and free it before the next micro-batch.

    Both stages run the model, so both rewind the RNG the way ``Trainer.step``
    does — otherwise the stages see different dropout masks and the result is
    silently wrong.

    Returns ``(param_cotangents, opt_cotangents, weight_cotangents)`` for the
    incoming state and this batch's ``data_weights``.
    """
    params = fwd_state.params
    buffers = fwd_state.buffers
    flat_i = fwd_state.differentiable_tensors()
    p_keys = list(param_grads.keys())
    p_index = {k: i for i, k in enumerate(p_keys)}
    p_grads = list(param_grads.values())
    n_p = len(p_keys)
    n_i = len(flat_i)

    # Stage 0: combined gradient values (no graph) to drive the update.
    # Rewind as in Trainer.step and record where each micro-batch started.
    torch.random.set_rng_state(fwd_state.cpu_rng_state)
    maybe_set_cuda_rng_state(fwd_state.cuda_rng_state)
    rng_snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    with swap_parameters(model, params, buffers, preserve_graph=False) as ps:
        grads_detached, _ = accumulate_grads(
            model,
            ps,
            inputs,
            grad_accum_steps,
            create_graph=False,
            rng_snapshots=rng_snapshots,
        )
    grads_var = {
        k: v.detach().clone().requires_grad_(True) for k, v in grads_detached.items()
    }

    # Stage A: VJP through the update only -> g_bar (on combined grad) and
    # the direct-path cotangents on the incoming params/opt-state.
    state_fa = apply_update(
        fwd_state,
        grads_var,
        inplace=False,
        trace=True,
        fsdp=fsdp,
        max_grad_norm=max_grad_norm,
    )
    flat_fa = state_fa.differentiable_tensors()
    res_a = list(
        torch.autograd.grad(
            flat_fa,
            flat_i + list(grads_var.values()),
            grad_outputs=p_grads + opt_grads,
            allow_unused=True,
        )
    )
    direct_i, g_bar_list = res_a[:n_i], res_a[n_i:]
    g_bar = {
        k: (g if g is not None else torch.zeros_like(grads_var[k]))
        for k, g in zip(grads_var.keys(), g_bar_list)
    }

    def _or_zero(c, ref):
        return c if c is not None else torch.zeros_like(ref)

    param_cot = [_or_zero(direct_i[i], flat_i[i]).clone() for i in range(n_p)]
    opt_cot = [_or_zero(direct_i[i], flat_i[i]) for i in range(n_p, n_i)]

    # ``example_weight`` is one indexing of ``data_weights`` shared by all
    # micro-batch slices, so accumulate cotangents on it and map them back
    # to ``data_weights`` once after the loop.
    ew = inputs.get("example_weight")
    ew_cot = torch.zeros_like(ew) if ew is not None else None

    # Stage B: per micro-batch through-gradient VJP, one graph at a time.
    total_denom = loss_denom(inputs)
    micro = nonempty_microbatches(inputs, grad_accum_steps)
    assert len(micro) == len(rng_snapshots)
    for mb, rng in zip(micro, rng_snapshots):
        with swap_parameters(model, params, buffers, preserve_graph=True) as ps:
            rng_restore(rng)
            outputs = model(**mb)
            loss_i = outputs.loss if hasattr(outputs, "loss") else outputs
            coef = loss_denom(mb) / total_denom
            g_i = grad_tree(loss_i * coef, ps, create_graph=True)
            keys = list(g_i.keys())
            targets = [ps[k] for k in keys]
            if ew is not None:
                targets = targets + [ew]
            contrib = torch.autograd.grad(
                [g_i[k] for k in keys],
                targets,
                grad_outputs=[g_bar[k] for k in keys],
                allow_unused=True,
            )
        for j, k in enumerate(keys):
            if contrib[j] is not None:
                param_cot[p_index[k]] = param_cot[p_index[k]] + contrib[j]
        if ew_cot is not None and contrib[-1] is not None:
            ew_cot = ew_cot + contrib[-1]
        del g_i, contrib

    weight_cot = torch.zeros_like(data_weights)
    if ew is not None and ew_cot is not None and float(ew_cot.abs().sum()) > 0:
        if ew is data_weights:
            weight_cot = weight_cot + ew_cot
        else:
            (dw,) = torch.autograd.grad(
                ew, data_weights, grad_outputs=ew_cot, allow_unused=True
            )
            if dw is not None:
                weight_cot = weight_cot + dw

    if dist.is_initialized() and not fsdp:
        # Same 1/world_size correction as the single-shot path in
        # `Trainer.backward`: a document only contributes to 1/world_size of
        # the averaged gradient the all-reduce produces.
        weight_cot = weight_cot / dist.get_world_size()

    out_param_cot = {k: param_cot[i] for i, k in enumerate(p_keys)}
    return out_param_cot, opt_cot, weight_cot
