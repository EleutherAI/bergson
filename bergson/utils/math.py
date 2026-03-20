import math
from typing import Mapping

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


def reshape_to_nearest_square(a: Tensor) -> Tensor:
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


@torch.compile
def damped_psd_power(
    H: Tensor,
    power: float,
    damping_factor: float = 0.1,
    dtype: torch.dtype = torch.float64,
    regularizer: Tensor | None = None,
) -> Tensor:
    """Compute a damped power of p.s.d. matrix `H` via eigendecomposition.

    Adds adaptive damping before computing the power to improve numerical stability.

    Args:
        H: Positive semi-definite matrix.
        power: Exponent to apply to eigenvalues (e.g. -0.5 for rsqrt, -1 for inverse).
        damping_factor: Multiplier for the damping term (default: 0.1). Set to
            0 to disable damping.
        dtype: Dtype for intermediate computation (default: float64 for stability).
        regularizer: Optional matrix to use as regularizer instead of identity.
            If provided, computes (H + damping_factor * regularizer)^power.
            If None (default), uses scaled identity:
            (H + damping_factor * |H|_mean * I)^power.

    Returns:
        The damped power of H in the original dtype.
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
    return (eigvec * eigval.pow(power) @ eigvec.mH).to(original_dtype)


def trace(matrices: Tensor) -> Tensor:
    """Version of `torch.trace` that works for batches of matrices."""
    diag = torch.linalg.diagonal(matrices)
    return diag.sum(dim=-1, keepdim=True).unsqueeze(-1)


def compute_lambda(
    query_eigen: Mapping[str, tuple[Tensor, Tensor]],
    index_eigen: Mapping[str, tuple[Tensor, Tensor]],
    target_components: int = 1000,
) -> float:
    """Compute the mixing coefficient λ for TrackStar preconditioner mixing.

    Given eigendecompositions of query (R_eval) and index (R_train)
    preconditioners, finds λ such that the sorted singular-value curves
    of ``λ·R_eval`` and ``(1-λ)·R_train`` intersect at the
    ``target_components``-th component.  This downweights the top
    ``target_components`` high-magnitude gradient directions that are
    common across evaluation examples (e.g. task template components).

    Concretely, all eigenvalues from every module are pooled and sorted
    independently for R_eval and R_train.  Then λ is chosen so that at
    the ``target_components``-th position::

        λ · σ_eval[k]  =  (1-λ) · σ_train[k]

    Solving gives ``λ = σ_train[k] / (σ_eval[k] + σ_train[k])``.

    Following §A.1.3 of *Scalable Influence and Fact Tracing for Large
    Language Model Pretraining* (Chang et al., 2024).

    Args:
        query_eigen: Per-module eigendecompositions of the query (eval)
            preconditioner.  Maps module name → (eigenvalues, eigenvectors).
        index_eigen: Per-module eigendecompositions of the index (train)
            preconditioner.  Maps module name → (eigenvalues, eigenvectors).
        target_components: Number of gradient components to downweight.
            ~1000 out of ~65K is typical (T-REx → λ≈0.90, C4 → λ≈0.99).

    Returns:
        The mixing coefficient λ ∈ [0, 1].
    """
    query_eigvals_list: list[Tensor] = []
    index_eigvals_list: list[Tensor] = []

    for name in query_eigen:
        if name not in index_eigen:
            continue

        q_eigvals, _ = query_eigen[name]
        i_eigvals, _ = index_eigen[name]

        query_eigvals_list.append(q_eigvals.to(dtype=torch.float64).clamp(min=0))
        index_eigvals_list.append(i_eigvals.to(dtype=torch.float64).clamp(min=0))

    if not query_eigvals_list:
        return 0.99  # Fallback to the default if no common modules

    all_query = torch.cat(query_eigvals_list)
    all_index = torch.cat(index_eigvals_list)
    total = len(all_query)

    if target_components <= 0:
        return 1.0
    if target_components > total:
        target_components = total

    # Pool and sort all eigenvalues (= singular values for PSD matrices)
    # independently for query and index preconditioners.
    sorted_query = torch.sort(all_query, descending=True).values
    sorted_index = torch.sort(all_index, descending=True).values

    # At the target_components-th position (0-indexed: k = target - 1),
    # set λ·σ_eval[k] = (1-λ)·σ_train[k] and solve for λ.
    k = target_components - 1
    sigma_eval = sorted_query[k].item()
    sigma_train = sorted_index[k].item()

    denom = sigma_eval + sigma_train
    if denom == 0:
        return 0.99

    lam = sigma_train / denom
    return max(0.0, min(1.0, lam))
