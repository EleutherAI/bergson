import math

import torch
import torch.nn.functional as F
from torch import Tensor


def weighted_causal_lm_ce(
    logits: Tensor,
    labels: Tensor,
    *,
    example_weight: Tensor | None = None,
    ignore_index: int = -100,
    vocab_size: int | None = None,
) -> Tensor:
    """
    HuggingFace-compatible causal LM loss with per-example weighting.

    Args:
    logits         : [B, T, V] float tensor of prediction scores
    labels         : [B, T] long tensor of target token ids, or ignore_index
    example_weight : [B] float tensor of per-example weights
    ignore_index   : int, label value to ignore in loss computation
    vocab_size     : optional int, vocabulary size (for validation)
    """
    assert logits.ndim == 3 and labels.ndim == 2
    B, T, V = logits.shape
    assert labels.shape == (B, T)
    if example_weight is not None:
        assert example_weight.shape == (B,)

    # HuggingFace always passes a vocab_size kwarg
    if vocab_size is not None:
        assert V == vocab_size, f"Expected vocab size {vocab_size}, got {V}"

    # Shift for causal LM
    shift_logits = logits[:, :-1, :].float().contiguous()  # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()  # [B, T-1]

    # Per-token loss (fused), no reduction
    tok_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view(
        B, T - 1
    )  # [B, T-1]

    # Implicitly assume the weights are all ones
    if example_weight is None:
        return tok_loss.mean()

    w = example_weight.to(tok_loss.dtype).view(B, 1)  # [B,1]
    return (tok_loss * w).mean()


def weighted_ce(
    labels: Tensor,
    logits: Tensor,
    cfg=None,
    *,
    example_weight: Tensor | None = None,
) -> Tensor:
    """
    HuggingFace-compatible cross-entropy loss with per-example weighting.

    Args:
    labels         : [B] long tensor of target ids, or ignore_index
    logits         : [B, V] float tensor of prediction scores
    example_weight : [B] float tensor of per-example weights
    """
    assert logits.ndim == 2 and labels.ndim == 1
    B, V = logits.shape
    assert labels.shape == (B,)
    if example_weight is not None:
        assert example_weight.shape == (B,)

    # Per-token loss (fused), no reduction
    tok_loss = F.cross_entropy(
        logits,
        labels,
        reduction="none",
    )  # [B,]

    # Implicitly assume the weights are all ones
    if example_weight is None:
        return tok_loss.mean()

    w = example_weight.to(tok_loss.dtype)  # [B,]
    return (tok_loss * w).mean()


def compute_damped_inverse(
    H: Tensor,
    damping_factor: float = 0.1,
    dtype: torch.dtype = torch.float64,
    regularizer: Tensor | None = None,
) -> Tensor:
    """Compute H^(-1) with damping for numerical stability.

    Uses eigendecomposition to compute the inverse of a positive semi-definite
    matrix with adaptive damping based on the matrix's mean absolute value.

    Args:
        H: Positive semi-definite matrix to invert.
        damping_factor: Multiplier for the damping term (default: 0.1).
        dtype: Dtype for intermediate computation (default: float64 for stability).
        regularizer: Optional matrix to use as regularizer instead of identity.
            If provided, computes inv(H + damping_factor * regularizer).
            If None (default), uses scaled identity:
            inv(H + damping_factor * |H|_mean * I).

    Returns:
        The damped inverse H^(-1) in the original dtype of H.
    """
    original_dtype = H.dtype
    H = H.to(dtype=dtype)
    if regularizer is not None:
        regularizer = regularizer.to(dtype=dtype, device=H.device)
        H = H + damping_factor * regularizer
    else:
        damping_val = damping_factor * H.abs().mean()
        H = H + damping_val * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    eigval, eigvec = torch.linalg.eigh(H)
    return (eigvec * (1.0 / eigval) @ eigvec.mT).to(original_dtype)


def trace(matrices: Tensor) -> Tensor:
    """Version of `torch.trace` that works for batches of matrices."""
    diag = torch.linalg.diagonal(matrices)
    return diag.sum(dim=-1, keepdim=True).unsqueeze(-1)


def reshape_to_nearest_square(a: torch.Tensor) -> torch.Tensor:
    """
    Reshape a 2-D (or any-D) tensor into the *most square* 2-D shape
    that preserves the total number of elements.

    Returns
    -------
    out   : reshaped tensor (view when possible)
    shape : tuple (rows, cols) that was chosen
    """
    n = math.prod(a.shape[-2:])
    if n == 0:
        raise ValueError("empty tensor")

    # search divisors closest to sqrt(n)
    root = math.isqrt(n)
    cols, rows = None, None
    for d in range(root, 0, -1):
        if n % d == 0:
            rows = d
            cols = n // d
            break

    if rows is None or cols is None:
        raise ValueError("could not find a valid shape for the tensor")

    return a.reshape(*a.shape[:-2], rows, cols)
