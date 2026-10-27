"""Small, model independent helpers for Direct Preference Optimization.

The functions in this module deliberately operate on already materialized log
probabilities (or logits) so they can also be used by attribution code without
coupling it to a trainer implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def sequence_log_probs(
    logits: Tensor,
    labels: Tensor,
    reduction: str = "mean",
    token_chunk_size: int = 128,
) -> Tensor:
    """Return the supervised causal log probability for each sequence.

    ``logits`` and ``labels`` are unshifted; the first label is therefore
    excluded from the causal objective.  Chunking is over tokens to avoid
    constructing a full log-softmax tensor.  Rows with no supervised target
    tokens are rejected since their mean is undefined (and a sum would hide a
    malformed example).
    """
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be [N,S,V] and labels must be [N,S]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must agree in their first two dimensions")
    if logits.shape[1] < 2:
        raise ValueError("sequence length must be at least 2")
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    if not isinstance(token_chunk_size, int) or isinstance(token_chunk_size, bool) or token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be a positive integer")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("labels must have an integer dtype")

    targets = labels[:, 1:]
    counts = (targets != -100).sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("each sequence must contain at least one supervised token")

    # CE naturally preserves float64. For lower precision, explicitly retain
    # at least float32 in the per-sequence accumulator.
    acc_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    totals = torch.zeros(logits.shape[0], device=logits.device, dtype=acc_dtype)
    shifted_logits = logits[:, :-1]
    for start in range(0, shifted_logits.shape[1], token_chunk_size):
        end = min(start + token_chunk_size, shifted_logits.shape[1])
        chunk_logits = shifted_logits[:, start:end].reshape(-1, logits.shape[-1])
        chunk_targets = targets[:, start:end].reshape(-1)
        # Sum CE by sequence, while retaining ignore_index semantics.
        losses = F.cross_entropy(
            chunk_logits, chunk_targets, reduction="none", ignore_index=-100
        ).reshape(logits.shape[0], end - start)
        totals += losses.to(acc_dtype).sum(dim=1)
    if reduction == "mean":
        totals = totals / counts.to(acc_dtype)
    return -totals


def _validate_pair_logs(policy_logps: Tensor, reference_logps: Tensor) -> int:
    if policy_logps.ndim != 1 or reference_logps.ndim != 1:
        raise ValueError("policy_logps and reference_logps must be one-dimensional")
    if policy_logps.shape != reference_logps.shape:
        raise ValueError("policy_logps and reference_logps must have equal shapes")
    if policy_logps.numel() == 0 or policy_logps.numel() % 2:
        raise ValueError("log-probability vectors must have a nonzero even length")
    return policy_logps.numel() // 2


def _validate_beta(beta: float) -> None:
    if not isinstance(beta, (float, int)) or isinstance(beta, bool) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")


def dpo_margin(policy_logps: Tensor, reference_logps: Tensor, beta: float) -> Tensor:
    """Compute beta times the chosen-vs-rejected relative log-prob margin."""
    pairs = _validate_pair_logs(policy_logps, reference_logps)
    _validate_beta(beta)
    chosen = policy_logps[:pairs] - reference_logps[:pairs]
    rejected = policy_logps[pairs:] - reference_logps[pairs:]
    return beta * (chosen - rejected)


def dpo_losses(policy_logps: Tensor, reference_logps: Tensor, beta: float) -> Tensor:
    """Per-pair DPO logistic losses."""
    return -F.logsigmoid(dpo_margin(policy_logps, reference_logps, beta))


def dpo_ggn_surrogate(policy_logps: Tensor, reference_logps: Tensor, beta: float) -> Tensor:
    """Scalar whose gradient outer product gives logistic preference curvature."""
    margin = dpo_margin(policy_logps, reference_logps, beta)
    weight = torch.sqrt(torch.sigmoid(margin) * torch.sigmoid(-margin)).detach()
    return margin * weight


def combine_pair_gradients(gradients: Tensor) -> Tensor:
    """Combine chosen and rejected gradient contributions pairwise.

    The input layout is all chosen rows followed by all rejected rows.  Signs
    already encoded by the caller are retained; this function only sums rows.
    """
    if gradients.ndim < 1 or gradients.shape[0] == 0 or gradients.shape[0] % 2:
        raise ValueError("gradients must have a nonzero even leading dimension")
    pairs = gradients.shape[0] // 2
    return gradients[:pairs] + gradients[pairs:]
