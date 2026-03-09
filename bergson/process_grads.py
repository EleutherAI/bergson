import json
import warnings
from pathlib import Path
from typing import Literal

import torch

from bergson.gradients import GradientProcessor
from bergson.utils.math import compute_lambda, damped_psd_power


def normalize_grad(
    grad_dict: dict[str, torch.Tensor],
    unit_normalize: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Preprocess a single gradient. Optionally unit-normalizes
    across all columns, moves to device."""
    final_dtype = next(iter(grad_dict.values())).dtype
    grads = {
        name: g.to(device=device, dtype=torch.float32) for name, g in grad_dict.items()
    }

    if unit_normalize:
        norm = torch.sqrt(torch.stack([g.pow(2).sum() for g in grads.values()]).sum())
        if norm > 0:
            grads = {k: v / norm for k, v in grads.items()}
        else:
            warnings.warn("Gradient norm is zero, skipping normalization")

    return {k: v.to(final_dtype) for k, v in grads.items()}


def normalize_flat_grad(
    grad: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Unit-normalize a single gradient tensor."""
    final_dtype = grad.dtype
    grad = grad.to(device=device, dtype=torch.float32)
    norm = grad.norm()
    if norm > 0:
        grad /= norm
    else:
        warnings.warn("Gradient norm is zero, skipping normalization")
    return grad.to(final_dtype)


def mix_preconditioners(
    query_path: str | Path,
    index_path: str | Path,
    output_path: str | Path,
    target_downweight_components: int = 1000,
) -> Path:
    """Mix query and index preconditioners and save the result to disk.

    Computes ``H_mixed = coeff * H_query + (1 - coeff) * H_index`` for
    every module's raw H matrix, then persists a new
    :class:`~bergson.gradients.GradientProcessor` at *output_path*.

    A ``mix_config.json`` file is also written alongside for provenance.

    Parameters
    ----------
    query_path : str | Path
        Directory containing the query GradientProcessor.
    index_path : str | Path
        Directory containing the index GradientProcessor.
    output_path : str | Path
        Directory where the mixed GradientProcessor will be saved.
    target_downweight_components : int
        Number of gradient components to downweight via automatic lambda
        selection

    Returns
    -------
    Path
        The *output_path* as a :class:`pathlib.Path`.
    """
    query_path = Path(query_path)
    index_path = Path(index_path)
    output_path = Path(output_path)

    q_proc = GradientProcessor.load(query_path)
    i_proc = GradientProcessor.load(index_path)

    # Auto-compute mixing coefficient (§A.1.3 of Chang et al., 2024)
    mixing_coefficient = compute_lambda(
        query_eigen=q_proc.preconditioners_eigen,
        index_eigen=i_proc.preconditioners_eigen,
        target_components=target_downweight_components,
    )

    mixed_preconditioners = {
        k: q_proc.preconditioners[k] * mixing_coefficient
        + i_proc.preconditioners[k] * (1 - mixing_coefficient)
        for k in q_proc.preconditioners
    }

    # Build a new processor with the mixed preconditioners
    mixed_proc = GradientProcessor(
        normalizers=q_proc.normalizers,
        preconditioners=mixed_preconditioners,
        preconditioners_eigen={},
        projection_dim=q_proc.projection_dim,
        reshape_to_square=q_proc.reshape_to_square,
        projection_type=q_proc.projection_type,
        include_bias=q_proc.include_bias,
    )
    mixed_proc.save(output_path)

    # Save provenance metadata
    mix_config = {
        "query_path": str(query_path),
        "index_path": str(index_path),
        "mixing_coefficient": mixing_coefficient,
        "target_downweight_components": target_downweight_components,
    }
    with (output_path / "mix_config.json").open("w") as f:
        json.dump(mix_config, f, indent=2)

    return output_path


def get_trackstar_preconditioner(
    preconditioner_path: str | None,
    device: torch.device,
    power: float = -0.5,
    return_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Compute preconditioner matrices from a saved processor file.

    Parameters
    ----------
    preconditioner_path : str | None
        Directory containing the saved GradientProcessor.
    device : torch.device
        Device to load the preconditioner onto.
    power : float
        Matrix power to apply to each H matrix.

        * ``-0.5`` — H^(-1/2), used for split (two-sided) preconditioning
          where both query and index gradients are multiplied by H^(-1/2).
        * ``-1``   — H^(-1), used for one-sided preconditioning where only
          the query gradients are preconditioned.
    """
    if preconditioner_path is None:
        return {}

    # Load preconditioners on device one-by-one for memory efficiency
    preconditioners = GradientProcessor.load(
        Path(preconditioner_path),
        map_location="cpu",
    ).preconditioners

    final_dtype = return_dtype or next(iter(preconditioners.values())).dtype

    return {
        name: damped_psd_power(H.to(device=device), power=power).to(final_dtype)
        for name, H in preconditioners.items()
    }


def precondition_flat_grads(
    grads: torch.Tensor,
    h_inv: dict[str, torch.Tensor],
    ordered_modules: list[str],
    batch_size: int = 8192,
) -> torch.Tensor:
    """Precondition flat (concatenated) gradients in-place.

    Uses column offsets to avoid duplicating the full tensor and processes
    rows in batches to bound peak memory. Each small ``[batch, d]`` slice is
    moved to ``h_inv``'s device for the matmul and written back.
    """
    if not h_inv:
        return grads

    for start in range(0, grads.shape[0], batch_size):
        end = min(start + batch_size, grads.shape[0])
        col = 0
        for name in ordered_modules:
            h = h_inv[name]
            d = h.shape[0]
            grads[start:end, col : col + d] = (
                grads[start:end, col : col + d].to(device=h.device, dtype=h.dtype) @ h
            ).to(device=grads.device, dtype=grads.dtype)
            col += d

    return grads


def precondition_grad(
    grad: dict[str, torch.Tensor],
    h_inv: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Precondition a single example's gradients."""
    if not h_inv:
        return grad

    return {
        name: grad[name].to(device=h_inv[name].device, dtype=h_inv[name].dtype)
        @ h_inv[name]
        for name in grad.keys()
    }


def normalize_and_aggregate_grads(
    grad_dict: dict[str, torch.Tensor],
    grad_column_names: list[str],
    unit_normalize: bool,
    device: torch.device,
    aggregate_grads: Literal["mean", "sum", "none"] = "none",
    normalize_aggregated_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Preprocess the gradients. Returns a dictionary of preprocessed gradients
    with shape [N, grad_dim] or [1, grad_dim]. Preprocessing includes some
    combination of per-item unit normalization, aggregation, aggregated
    gradient normalization, and dtype conversion."""

    # Short-circuit if possible
    if aggregate_grads == "none" and not unit_normalize:
        return {name: grad_dict[name].to(device=device) for name in grad_column_names}

    grads = {
        name: grad_dict[name].to(device=device, dtype=torch.float32)
        for name in grad_column_names
    }

    # Per-item unit normalization
    if unit_normalize:
        norms = torch.cat(list(grads.values()), dim=1).norm(dim=1, keepdim=True)
        grads = {k: v / norms for k, v in grads.items()}

    # Aggregate across items
    if aggregate_grads == "mean":
        grads = {name: grads[name].mean(0, keepdim=True) for name in grad_column_names}
    elif aggregate_grads == "sum":
        grads = {name: grads[name].sum(0, keepdim=True) for name in grad_column_names}
    elif aggregate_grads != "none":
        raise ValueError(f"Invalid aggregate_grads: {aggregate_grads}")

    # Normalize the aggregated gradient
    if normalize_aggregated_grad:
        grad_norm = torch.cat(
            [grads[name].flatten() for name in grad_column_names], dim=0
        ).norm()
        for name in grad_column_names:
            grads[name] /= grad_norm

    return grads
