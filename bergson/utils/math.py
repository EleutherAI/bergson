import math
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

_VALID_CE_REDUCTIONS = frozenset({"mean", "sum", "sum_of_means"})


def weighted_causal_lm_ce(
    logits: Tensor,
    labels: Tensor,
    *,
    example_weight: Tensor | None = None,
    ignore_index: int = -100,
    shift_loss_mask: Tensor | None = None,
    vocab_size: int | None = None,
    reduction: str = "mean",
    **kwargs,  # Ignored
) -> Tensor:
    """
    HuggingFace-compatible causal LM loss with per-example weighting.

    Args:
    logits          : [B, T, V] float tensor of prediction scores
    labels          : [B, T] long tensor of target token ids, or ignore_index
    example_weight  : [B] or [B, T] float tensor of weights
    ignore_index    : int, label value to ignore in loss computation
    shift_loss_mask : optional [B, T] bool tensor;
                      mask[i] = labels[i+1] != ignore_index
    vocab_size      : optional int, vocabulary size (for validation)
    reduction       : one of
                      "mean": mean over all valid tokens in the batch
                          (standard).
                      "sum": weighted sum over every valid token in the
                          batch, with no division.
                      "sum_of_means": mean over each sample's tokens,
                          summed over the batch.
    """
    assert reduction in _VALID_CE_REDUCTIONS, f"Unknown reduction {reduction!r}"

    assert logits.ndim == 3 and labels.ndim == 2
    B, T, V = logits.shape
    assert labels.shape == (B, T)
    if example_weight is not None:
        assert example_weight.shape == (B,) or example_weight.shape == (B, T)

    # HuggingFace always passes a vocab_size kwarg
    if vocab_size is not None:
        assert V == vocab_size, f"Expected vocab size {vocab_size}, got {V}"

    # Shift for causal LM
    shift_logits = logits[:, :-1, :].float().contiguous()  # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()  # [B, T-1]

    needs_per_token = example_weight is not None or reduction in (
        "sum",
        "sum_of_means",
    )

    # Per-token loss (fused), no reduction
    tok_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none" if needs_per_token else "mean",
        ignore_index=ignore_index,
    )

    # Implicitly assume the weights are all ones
    if not needs_per_token:
        return tok_loss

    tok_loss = tok_loss.view(B, T - 1)  # [B, T-1]

    # Per token weights
    if example_weight is None:
        w = tok_loss.new_ones(B, 1)  # [B,1]
    elif example_weight.shape == (B, T):
        w = example_weight[:, :-1].to(tok_loss.dtype)  # [B, T-1]
    else:
        w = example_weight.to(tok_loss.dtype).view(B, 1)  # [B,1]

    if reduction == "sum":
        # Weighted sum over every valid token.
        return (tok_loss * w).sum()

    if reduction == "sum_of_means":
        # Per-sample token-mean, summed over the batch (no /B).
        row_valid = (
            shift_loss_mask[:, :-1].to(tok_loss.dtype)
            if shift_loss_mask is not None
            else (shift_labels != ignore_index).to(tok_loss.dtype)
        )
        row_counts = row_valid.sum(dim=1).clamp_min(1.0)  # [B]
        row_loss = (tok_loss * w).sum(dim=1) / row_counts  # [B]
        return row_loss.sum()

    # Mean over every valid token in the batch, matching the unweighted branch's
    # F.cross_entropy(reduction="mean"). The count runs over the whole [B, T-1]
    # block.
    # The mask's last column is always False, so no [:, :-1] slice is needed.
    denom = (
        shift_loss_mask.sum()
        if shift_loss_mask is not None
        else (shift_labels != ignore_index).sum()
    )
    return (tok_loss * w).sum() / denom.clamp_min(1)


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


def trace(matrices: Tensor) -> Tensor:
    """Version of `torch.trace` that works for batches of matrices."""
    diag = torch.linalg.diagonal(matrices)
    return diag.sum(dim=-1, keepdim=True).unsqueeze(-1)


def compute_lambda(
    query_eigen: Mapping[str, tuple[Tensor, Tensor]],
    index_eigen: Mapping[str, tuple[Tensor, Tensor]],
    target_components: int = 1000,
) -> float:
    """Compute the mixing coefficient λ for TrackStar hessian mixing.

    Given eigendecompositions of query (R_eval) and index (R_train)
    hessians, finds λ such that the sorted singular-value curves
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
            hessian.  Maps module name → (eigenvalues, eigenvectors).
        index_eigen: Per-module eigendecompositions of the index (train)
            hessian.  Maps module name → (eigenvalues, eigenvectors).
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
    # independently for query and index hessians.
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
