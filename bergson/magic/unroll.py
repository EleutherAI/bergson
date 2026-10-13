"""Unrolled Gauss-Newton attribution through a MAGIC training trajectory.

Walking the AdamW run backward, each step ``t`` carries the query loss's
cotangents on the linear-layer weights (``alpha``) and on Adam's first and
second moments (``gam``, ``nu``)::

    gam <- b1 gam - c_t P_t alpha            nu <- b2 nu + R_t alpha
    z = (1 - b1) gam + 2 (1 - b2) g_t nu     score_d += <z, dg_t/dw_d>
    alpha <- (1 - lr_t wd) alpha + G_t z

where ``P_t`` is the AdamW preconditioner, ``R_t`` its derivative in the second
moment, ``g_t`` the batch gradient and ``G_t`` the Gauss-Newton matrix of the
batch loss. Each step's weights and moments are replayed from the run's
checkpoints with :meth:`Trainer.step`, data-parallel across ranks; each rank
then carries its own queries through the whole batch.
"""

import os
import time
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.func import functional_call, jvp, vjp, vmap

from ..data import sorted_checkpoints
from .data_stream import DataStream
from .grad_accum import loss_denom
from .trainer import Trainer, TrainerState


def unrolled_names(
    model: nn.Module, params: dict[str, Tensor], parameters: str = "linear"
) -> list[str]:
    """Parameters carried through the backward: every trainable one for
    ``parameters="all"``; for ``"linear"`` the linear-layer weights, excluding
    those shared with an embedding."""
    if parameters == "all":
        return list(params)
    embedding_ptrs = {
        params[f"{name}.weight"].data_ptr()
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Embedding) and f"{name}.weight" in params
    }
    names = []
    for name, mod in model.named_modules():
        key = f"{name}.weight"
        linear = isinstance(mod, nn.Linear) or type(mod).__name__ == "Conv1D"
        if linear and key in params and params[key].data_ptr() not in embedding_ptrs:
            names.append(key)
    return names


def tied_groups(params: dict[str, Tensor], names: list[str]) -> list[list[str]]:
    """``names`` grouped by shared storage. Each name keeps its own moments and
    gradient; the group shares one weight cotangent."""
    groups: dict[int, list[str]] = {}
    for k in names:
        groups.setdefault(params[k].data_ptr(), []).append(k)
    return list(groups.values())


def adam_moments(
    state: TrainerState, names: list[str]
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """First and second AdamW moments of ``names``."""
    adam = next(s for s in state.opt_state if hasattr(s, "nu"))
    order = sorted(state.params)
    mu, nu = dict(zip(order, adam.mu)), dict(zip(order, adam.nu))
    return {k: mu[k] for k in names}, {k: nu[k] for k in names}


def _to_host(t: Tensor) -> Tensor:
    """Pinned CPU copy of ``t`` when it lives on a GPU."""
    if t.device.type != "cuda":
        return t.detach().to("cpu", copy=True)
    out = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    out.copy_(t.detach())
    return out


def _snapshot(state: TrainerState, names: list[str]):
    """Pinned CPU copy of every trainable weight and of the moments of ``names``."""
    copies: dict[tuple, Tensor] = {}
    params = {}
    for k, v in state.params.items():
        key = (v.data_ptr(), tuple(v.shape))
        if key not in copies:
            copies[key] = _to_host(v)
        params[k] = copies[key]
    mu, nu = adam_moments(state, names)
    return (
        params,
        {k: _to_host(m) for k, m in mu.items()},
        {k: _to_host(v) for k, v in nu.items()},
    )


def _gauss_newton_and_scores(
    model: nn.Module,
    params: dict[str, Tensor],
    buffers: dict[str, Tensor],
    batch: dict,
    names: list[str],
    z: list[Tensor],
    weights: Tensor,
    scores: Tensor,
    *,
    micro_batch_size: int,
    bf16: bool,
    curvature: str = "gauss_newton",
    timings: dict[str, float] | None = None,
) -> list[Tensor]:
    """``H_t z`` over ``batch``; adds ``<z, dg_t/dw>`` to ``scores``.

    ``curvature="gauss_newton"`` applies ``J^T (d^2 loss / d logits^2) J``;
    ``"gradient_outer"`` applies the mean over documents of the outer product
    of their mean-token loss gradients."""
    nq = z[0].shape[0]
    num_docs = len(batch["input_ids"])

    def clock() -> float:
        if timings is None:
            return 0.0
        if ids.device.type == "cuda":
            torch.cuda.synchronize(ids.device)
        return time.monotonic()

    if timings is not None:
        timings.update(gn=0.0, scores=0.0)
    primals = tuple(params[k] for k in names)
    out = [torch.zeros_like(v) for v in z]
    denom = loss_denom(batch)
    ids, labels, ew = batch["input_ids"], batch["labels"], batch["example_weight"]

    for i in range(0, len(ids), micro_batch_size):
        t_start = clock()
        x = ids[i : i + micro_batch_size]
        y = labels[i : i + micro_batch_size, 1:]
        w = ew[i : i + micro_batch_size]
        w = w[:, :-1] if w.ndim == 2 else w[:, None]

        def logits_of(*ws):
            tensors = {**params, **buffers, **dict(zip(names, ws))}
            kwargs = {"input_ids": x, "use_cache": False}
            outputs = functional_call(model, tensors, (), kwargs, tie_weights=False)
            return outputs.logits[:, :-1]

        with torch.autocast(x.device.type, torch.bfloat16, enabled=bf16):
            jv = vmap(lambda *t: jvp(logits_of, primals, t)[1])(*z)
            vjp_out = vjp(logits_of, *primals)
            logits, vjp_fn = vjp_out[0], vjp_out[1]

        t_gn = clock()
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        p = torch.softmax(logits.to(dtype), -1)
        jv = jv.to(dtype)
        pj = (p * jv).sum(-1)
        valid = (y != -100).to(dtype)

        # Per-token <grad of the loss, z>; the weights' graph maps it onto documents
        jy = jv.gather(-1, y.clamp_min(0)[None, ..., None].expand(nq, -1, -1, 1))
        dots = valid * (pj - jy.squeeze(-1)) / denom
        t_scores = clock()
        for q in range(nq):
            (g,) = torch.autograd.grad((w * dots[q]).sum(), weights, retain_graph=True)
            scores[..., q] += g
        t_vjp = clock()

        if curvature == "gradient_outer":
            # Each document's gradient, weighted by its inner product with z
            onehot = torch.zeros_like(p).scatter_(-1, y.clamp_min(0)[..., None], 1.0)
            tokens = valid.sum(-1)
            coef = denom**2 / (num_docs * tokens.clamp_min(1) ** 2)
            coef = coef * (w.detach() * dots.detach()).sum(-1)
            scale = (coef[..., None] * valid / denom)[..., None]
            u = (p - onehot)[None] * scale
            del jv, p, pj, onehot
        else:
            # Gauss-Newton cotangent at the logits: the softmax Hessian times J z
            scale = (w.detach() * valid / denom)[None, ..., None]
            u = jv.sub_(pj[..., None]).mul_(p).mul_(scale)
            del jv, p, pj
        h = vmap(vjp_fn)(u.to(logits.dtype))
        for o, hh in zip(out, h):
            o += hh.to(o.dtype)
        del u, h, vjp_fn, logits
        if timings is not None:
            t_end = clock()
            timings["gn"] += (t_gn - t_start) + (t_end - t_vjp)
            timings["scores"] += t_vjp - t_scores

    return out


def unroll_scores(
    trainer: Trainer,
    ckpt_dir: str,
    data: DataStream,
    state: TrainerState,
    queries: list[dict[str, Tensor]],
    *,
    lr: Callable[[int], float] | float,
    betas: tuple[float, float],
    eps: float,
    eps_root: float,
    weight_decay: float,
    grad_accum_steps: int = 1,
    micro_batch_size: int = 4,
    tf32: bool = True,
    bf16: bool = False,
    curvature: str = "gauss_newton",
    parameters: str = "linear",
    debug: bool = False,
) -> Tensor:
    """Unrolled scores, ``[*data.weights.shape, len(queries)]``.

    ``queries`` are this rank's query gradients; every rank replays the
    trajectory together and scores its own queries against every document.
    ``state`` must be the final state of the run whose checkpoints are in
    ``ckpt_dir``; it is walked back through the trajectory and restored.
    Scores are loss-signed like MAGIC's. ``debug`` prints every rank's time
    per phase of each step.
    """
    model = trainer.model
    b1, b2 = betas
    rank = dist.get_rank() if dist.is_initialized() else 0
    main = rank == 0
    device = data.weights.device

    def clock() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.monotonic()

    # launch_distributed_run swallows the other ranks' stdout, so debug
    # timings also go to one file per rank beside the checkpoints.
    log = None
    if debug:
        log_path = os.path.join(os.path.dirname(ckpt_dir), f"unroll_rank{rank}.log")
        log = open(log_path, "a")

    def report(msg: str) -> None:
        print(msg, flush=True)
        if log is not None:
            log.write(msg + "\n")
            log.flush()

    num_steps = len(data)
    names = unrolled_names(model, state.params, parameters)
    groups = tied_groups(state.params, names)
    group_of = {k: g for g, group in enumerate(groups) for k in group}

    final = state.to("cpu").detach_()
    mu, nu = adam_moments(state, names)
    post = (
        {k: v.detach().clone() for k, v in mu.items()},
        {k: v.detach().clone() for k, v in nu.items()},
    )
    dtype = torch.float32
    if queries:
        dtype = torch.promote_types(queries[0][names[0]].dtype, torch.float32)
    # One weight cotangent per shared tensor, one moment pair per name
    alpha = []
    for group in groups:
        shape = (len(queries), *state.params[group[0]].shape)
        a = torch.zeros(shape, device=device, dtype=dtype)
        for k in group:
            for i, q in enumerate(queries):
                a[i] += q[k].to(device, dtype)
        alpha.append(a)
    timings = {"gn": 0.0, "scores": 0.0}
    gam = [torch.zeros_like(alpha[group_of[k]]) for k in names]
    nu_adj = [torch.zeros_like(alpha[group_of[k]]) for k in names]
    scores = torch.zeros(*data.weights.shape, len(queries), device=device)

    ckpts = [(i, p) for i, p in sorted_checkpoints(ckpt_dir) if i < num_steps]
    if not ckpts or ckpts[0][0] != 0:
        raise ValueError(f"{ckpt_dir} has no step_0.ckpt to replay from.")
    starts = [i for i, _ in ckpts] + [num_steps]
    if any(b <= a for a, b in zip(starts, starts[1:])):
        raise ValueError(f"checkpoint steps in {ckpt_dir} are not increasing.")
    start_time = time.monotonic()

    for seg in reversed(range(len(ckpts))):
        first, path = ckpts[seg]
        last = starts[seg + 1]

        # Replay the segment, keeping each step's weights and moments
        t_replay = clock()
        data.requires_grad = False
        state.detach_()
        state.load(path)
        state.batch_index = first
        replay, snaps = state, []
        for t in range(first, last):
            snaps.append(_snapshot(replay, names))
            if t < last - 1:
                replay = trainer.step(
                    replay, data[t], inplace=True, grad_accum_steps=grad_accum_steps
                )
        del replay
        if debug:
            report(f"[unroll r{rank}] replay {first}-{last}: {clock() - t_replay:.1f}s")

        for t in reversed(range(first, last)):
            t0 = clock()
            params, mu_t, nu_t = snaps.pop()
            if not queries:
                post = (mu_t, nu_t)
                continue
            step_lr = float(lr(t)) if callable(lr) else float(lr)
            count = t + 1
            coef = step_lr / (1 - b1**count)
            post_mu, post_nu = post

            z = []
            with torch.no_grad():
                for j, k in enumerate(names):
                    a = alpha[group_of[k]]
                    m_post = post_mu[k].to(device, non_blocking=True).double()
                    vh = post_nu[k].to(device, non_blocking=True).double()
                    vh = vh / (1 - b2**count)
                    vh = vh + eps_root
                    precond = 1 / (vh.sqrt() + eps)
                    gam[j].mul_(b1).add_((a * precond).to(dtype), alpha=-coef)
                    zj = (1 - b1) * gam[j]

                    r = coef * m_post * 0.5 * vh.rsqrt() * precond**2 / (1 - b2**count)
                    nu_adj[j] = (b2 * nu_adj[j] + r * a).to(dtype)
                    mu_prev = mu_t[k].to(device, non_blocking=True).double()
                    g = (m_post - b1 * mu_prev) / (1 - b1)
                    z.append(zj + (2 * (1 - b2) * g * nu_adj[j]).to(dtype))

            t1 = clock()
            data.requires_grad = True
            batch = data.full_batch(t)
            t_batch = clock()
            step_params = {
                k: v.to(device, non_blocking=True) for k, v in params.items()
            }
            t_h2d = clock()
            if main and t == num_steps - 1:
                print(f"[unroll] loss denominator {loss_denom(batch):g}", flush=True)
            tf32_before = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = tf32
            try:
                gz = _gauss_newton_and_scores(
                    model,
                    step_params,
                    state.buffers,
                    batch,
                    names,
                    z,
                    data.weights,
                    scores,
                    micro_batch_size=micro_batch_size,
                    bf16=bf16,
                    curvature=curvature,
                    timings=timings if debug else None,
                )
            finally:
                torch.backends.cuda.matmul.allow_tf32 = tf32_before
            del batch
            t2 = clock()

            with torch.no_grad():
                for g, group in enumerate(groups):
                    alpha[g].mul_(1 - len(group) * step_lr * weight_decay)
                    for k in group:
                        alpha[g].add_(gz[names.index(k)])
            post = (mu_t, nu_t)
            del z, gz, params, step_params
            if debug:
                t3 = clock()
                timed = timings["gn"] + timings["scores"]
                report(
                    f"[unroll r{rank}] step {t}: z {t1 - t0:.1f}s  "
                    f"batch {t_batch - t1:.1f}s  h2d {t_h2d - t_batch:.1f}s  "
                    f"gauss-newton {timings['gn']:.1f}s  "
                    f"scores {timings['scores']:.1f}s  "
                    f"other {t2 - t_h2d - timed:.1f}s  alpha {t3 - t2:.1f}s  "
                    f"total {t3 - t0:.1f}s"
                )

            if main:
                norm = torch.stack([a.norm() for a in alpha]).norm().item()
                elapsed = time.monotonic() - start_time
                print(
                    f"[unroll] step {t}: |alpha| {norm:.4e}  {elapsed:.0f}s", flush=True
                )

    if log is not None:
        log.close()
    data.requires_grad = False
    state.detach_()
    state.copy_(final.to(device))
    return scores.detach()
