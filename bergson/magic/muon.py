# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Muon optimizer."""

import math

import torch
from torch import Tensor
from torchopt.alias.utils import (
    _get_use_chain_flat,
    flip_sign_and_add_weight_decay,
    scale_by_neg_lr,
)
from torchopt.combine import chain
from torchopt.pytree import tree_map, tree_map_
from torchopt.transform.trace import TraceState
from torchopt.typing import (
    GradientTransformation,
    OptState,
    Params,
    ScalarOrSchedule,
    Updates,
)


def muon(
    lr: ScalarOrSchedule,
    momentum: float = 0.95,
    dampening: float = 0.0,
    weight_decay: float = 0.1,
    *,
    moment_requires_grad: bool = False,
    maximize: bool = False,
) -> GradientTransformation:
    """Create a functional version of the canonical Muon optimizer."""
    # pylint: disable=unneeded-not
    if not (callable(lr) or lr >= 0.0):  # pragma: no cover
        raise ValueError(f"Invalid learning rate: {lr}")
    if not momentum >= 0.0:  # pragma: no cover
        raise ValueError(f"Invalid momentum value: {momentum}")
    if not weight_decay >= 0.0:  # pragma: no cover
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")
    if momentum <= 0.0 or dampening != 0.0:  # pragma: no cover
        raise ValueError("Nesterov momentum requires a momentum and zero dampening")
    # pylint: enable=unneeded-not

    chain_fn = chain
    flip_sign_add_wd_fn = flip_sign_and_add_weight_decay
    scale_by_neg_lr_fn = scale_by_neg_lr

    def zpns_init_fn(params: Params) -> OptState:
        return TraceState(
            trace=tree_map(
                lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad),
                params,
            ),
        )

    first_call = True

    # Modeled after the TorchOpt `trace` implementation, but with the Muon update logic
    # in place of the standard momentum update.
    def zpns_update_fn(
        updates: Params,
        state: OptState,
        *,
        params: Params,
        inplace: bool = True,
    ) -> tuple[Updates, OptState]:
        nonlocal first_call

        if inplace:

            def f1(t: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor | None:
                if g is None:
                    return t
                if first_call:
                    return t.add_(g)
                return t.mul_(momentum).add_(g)

            def f2(t: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor | None:
                return g.add_(t, alpha=momentum) if g is not None else g

            new_trace = tree_map(f1, state.trace, updates)
            tree_map_(f2, new_trace, updates)
        else:

            def f1(t: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor | None:
                if g is None:
                    return t
                if first_call:
                    return t.add(g)
                return t.mul(momentum).add(g)

            def f2(t: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor | None:
                return g.add(t, alpha=momentum) if g is not None else g

            new_trace = tree_map(f1, state.trace, updates)
            updates = tree_map(f2, new_trace, updates)

        # Apply Newton-Schulz orthogonalization and per-param lr adjustment ratio.
        # Weight decay is added here (decoupled, matching torch.optim.Muon): wd*param is
        # appended after NS so it is not mixed into the momentum buffer or
        # orthogonalization. Base lr factor is applied downstream by scale_by_neg_lr
        param_list = (
            list(params.values()) if hasattr(params, "values") else list(params)
        )
        update_list = (
            list(updates.values()) if hasattr(updates, "values") else list(updates)
        )

        ns_update_list = []
        for update, param in zip(update_list, param_list):
            if update is None:
                ns_update_list.append(None)
                continue
            if update.ndim != 2:
                raise ValueError(
                    f"Muon requires 2D parameter gradients, got shape {update.shape}"
                )
            ns = _zeropower_via_newtonschulz(
                update, (DEFAULT_A, DEFAULT_B, DEFAULT_C), DEFAULT_NS_STEPS, EPS
            )
            ratio = _adjust_lr(1.0, "match_rms_adamw", param.shape)
            ns_update = ns.to(update.dtype) * ratio
            if weight_decay != 0.0:
                ns_update = ns_update + weight_decay * param.detach()
            ns_update_list.append(ns_update)

        if hasattr(updates, "keys"):
            updates = dict(zip(updates.keys(), ns_update_list))
        else:
            updates = ns_update_list

        first_call = False
        return updates, TraceState(trace=new_trace)

    if _get_use_chain_flat():  # default behavior
        chain_fn = chain_fn.flat  # type: ignore[attr-defined]
        flip_sign_add_wd_fn = flip_sign_add_wd_fn.flat  # type: ignore[attr-defined]
        scale_by_neg_lr_fn = scale_by_neg_lr_fn.flat  # type: ignore[attr-defined]

    return chain_fn(
        # weight_decay=0.0 here: WD handled inside zpns_update_fn (decoupled, after NS)
        # This transform is kept only to handle the sign flip when maximize=True.
        flip_sign_add_wd_fn(weight_decay=0.0, maximize=maximize),
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


def _zeropower_via_newtonschulz(
    grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float
) -> Tensor:
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
    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )

    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    # Ensure spectral norm is at most 1
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    # Perform the NS iterations
    for _ in range(ns_steps):
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
        # pyrefly: ignore [no-matching-overload]
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


def _single_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    for i, param in enumerate(params):
        grad = grads[i]
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = muon_momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf

        update = _zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)

        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)

        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)
