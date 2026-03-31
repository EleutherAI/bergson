import math
from typing import NamedTuple

import torch
from torch import Tensor
from torchopt.alias.utils import (
    _get_use_chain_flat,
    scale_by_neg_lr,
)
from torchopt.combine import chain
from torchopt.typing import (
    GradientTransformation,
    OptState,
    Params,
    ScalarOrSchedule,
    Updates,
)


class MuonState(NamedTuple):
    traces: list  # per-param Nesterov momentum buffers; None for 1D params
    exp_avgs: list  # per-param AdamW first moments; None for 2D params
    exp_avg_sqs: list  # per-param AdamW second moments; None for 2D params
    step: torch.Tensor  # scalar step count for AdamW bias correction


def muon(
    lr: ScalarOrSchedule,
    momentum: float = 0.95,
    dampening: float = 0.0,
    weight_decay: float = 0.1,
    adamw_betas: tuple[float | None, float] = (None, 0.999),
    adamw_eps: float = 1e-8,
    adamw_eps_root: float = 0.0,
    *,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    """Create a functional version of the canonical Muon optimizer.

    2D parameters are updated via Nesterov momentum + Newton-Schulz orthogonalization.
    1D parameters (e.g. biases, scalar gains) are updated via AdamW.
    Parameters with more than 2 dimensions raise a ValueError.
    """
    if not (callable(lr) or lr >= 0.0):
        raise ValueError(f"Invalid learning rate: {lr}")
    if not momentum >= 0.0:
        raise ValueError(f"Invalid momentum value: {momentum}")
    if not weight_decay >= 0.0:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")
    if momentum <= 0.0 or dampening != 0.0:
        raise ValueError("Nesterov momentum requires a momentum and zero dampening")

    chain_fn = chain
    scale_by_neg_lr_fn = scale_by_neg_lr
    beta1, beta2 = adamw_betas
    if beta1 is None:
        beta1 = momentum

    def _to_list(x: Params) -> list:
        return list(x.values()) if hasattr(x, "values") else list(x)

    def zpns_init_fn(params: Params) -> OptState:
        traces, exp_avgs, exp_avg_sqs = [], [], []
        for param in _to_list(params):
            if param.ndim > 2:
                raise ValueError(
                    f"Muon does not support parameters with more than 2 dimensions, "
                    f"got shape {param.shape}"
                )
            elif param.ndim == 2:
                traces.append(
                    torch.zeros_like(param, requires_grad=moment_requires_grad)
                )
                exp_avgs.append(None)
                exp_avg_sqs.append(None)
            else:  # 1D — will be updated via AdamW
                traces.append(None)
                exp_avgs.append(
                    torch.zeros_like(param, requires_grad=moment_requires_grad)
                )
                exp_avg_sqs.append(
                    torch.zeros_like(param, requires_grad=moment_requires_grad)
                )
        return MuonState(
            traces=traces,
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
            step=torch.tensor(0, dtype=torch.long),
        )

    first_call = True

    # Modeled after the TorchOpt `trace` implementation, but with the Muon update logic
    # in place of the standard momentum update. 1D params are routed to AdamW instead.
    def zpns_update_fn(
        updates: Params,
        state: OptState,
        *,
        params: Params | None = None,
        inplace: bool = False,
    ) -> tuple[Updates, OptState]:
        assert params is not None, "Muon needs params for WD and LR adjustment"
        nonlocal first_call

        param_list = _to_list(params)
        update_list = _to_list(updates)
        step = state.step + 1

        new_traces, new_exp_avgs, new_exp_avg_sqs, new_update_list = [], [], [], []

        for update, param, trace, exp_avg, exp_avg_sq in zip(
            update_list, param_list, state.traces, state.exp_avgs, state.exp_avg_sqs
        ):
            if update is None:
                new_traces.append(trace)
                new_exp_avgs.append(exp_avg)
                new_exp_avg_sqs.append(exp_avg_sq)
                new_update_list.append(None)
                continue

            if param.ndim > 2:
                raise ValueError(
                    f"Muon does not support parameters with more than 2 dimensions, "
                    f"got shape {param.shape}"
                )
            elif param.ndim == 2:
                # --- Muon path: Nesterov momentum + Newton-Schulz ---
                # Apply orthogonalization and per-param lr adjustment ratio.
                # Weight decay is added here (decoupled, matching torch.optim.Muon):
                # wd*param is appended after NS so it is not mixed into the momentum
                # buffer or orthogonalization. Base lr factor is applied downstream by
                # scale_by_neg_lr.
                if inplace:
                    new_trace = (
                        trace.add_(update)
                        if first_call
                        else trace.mul_(momentum).add_(update)
                    )
                    ns_input = update.add_(new_trace, alpha=momentum)
                else:
                    new_trace = (
                        trace.add(update)
                        if first_call
                        else trace.mul(momentum).add(update)
                    )
                    ns_input = update.add(new_trace, alpha=momentum)

                ns = _zeropower_via_newtonschulz(ns_input, inplace=inplace)
                ratio = _adjust_lr(1.0, "match_rms_adamw", param.shape)
                ns_update = ns.to(update.dtype) * ratio
                if weight_decay != 0.0:
                    ns_update = ns_update + weight_decay * param.detach()

                new_traces.append(new_trace)
                new_exp_avgs.append(None)
                new_exp_avg_sqs.append(None)
                new_update_list.append(ns_update)
            else:
                # --- AdamW path for 1D params (biases, norms, etc.) ---
                # Weight decay applied decoupled (after moment normalization), matching
                # the same convention used for 2D params above.
                if inplace:
                    new_exp_avg = exp_avg.mul_(beta1).add_(update, alpha=1 - beta1)
                    new_exp_avg_sq = exp_avg_sq.mul_(beta2).addcmul_(
                        update, update, value=1 - beta2
                    )
                else:
                    new_exp_avg = exp_avg.mul(beta1).add(update, alpha=1 - beta1)
                    new_exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(
                        update, update, value=1 - beta2
                    )

                bias_correction1 = 1 - beta1 ** step.item()
                bias_correction2 = 1 - beta2 ** step.item()
                adam_update = (new_exp_avg / bias_correction1) / (
                    (new_exp_avg_sq / bias_correction2 + adamw_eps_root).sqrt()
                    + adamw_eps
                )
                if weight_decay != 0.0:
                    adam_update = adam_update + weight_decay * param.detach()

                new_traces.append(None)
                new_exp_avgs.append(new_exp_avg)
                new_exp_avg_sqs.append(new_exp_avg_sq)
                new_update_list.append(adam_update)

        if hasattr(updates, "keys"):
            new_updates = dict(zip(updates.keys(), new_update_list))
        else:
            new_updates = new_update_list

        first_call = False
        return new_updates, MuonState(
            traces=new_traces,
            exp_avgs=new_exp_avgs,
            exp_avg_sqs=new_exp_avg_sqs,
            step=step,
        )

    if _get_use_chain_flat():  # default behavior
        chain_fn = chain_fn.flat  # type: ignore[attr-defined]
        scale_by_neg_lr_fn = scale_by_neg_lr_fn.flat  # type: ignore[attr-defined]

    return chain_fn(
        GradientTransformation(zpns_init_fn, zpns_update_fn),
        scale_by_neg_lr_fn(lr),
    )


# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
# github permlink: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L16
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _zeropower_via_newtonschulz(grad: Tensor, inplace: bool = False) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We
    opt to use a quintic iteration whose coefficients are selected to maximize the
    slope at zero. For the purpose of minimizing steps, it turns out to be empirically
    effective to keep increasing the slope at zero even beyond the point where the
    iteration no longer converges all the way to one everywhere on the interval. This
    iteration therefore does not produce UV^T but rather something like US'V^T where
    S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Implementation reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
    with suggestions by @jxbz, @leloykun, and @YouJiacheng.
    """
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    a, b, c = DEFAULT_A, DEFAULT_B, DEFAULT_C
    ortho_grad = grad
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    # Ensure spectral norm is at most 1
    denom = ortho_grad.norm().clamp(min=EPS)
    if inplace:
        ortho_grad.div_(denom)
    else:
        ortho_grad = ortho_grad / denom

    # Perform the NS iterations
    for _ in range(DEFAULT_NS_STEPS):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """Default learning rate adjustment used by Muon."""
    A, B = param_shape[:2]

    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio
