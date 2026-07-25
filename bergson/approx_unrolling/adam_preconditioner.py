"""Per-segment Adam/AdamW diagonal preconditioners for the preconditioned
SOURCE (approx-unrolling) variant.

:func:`build_segment_preconditioners` reads each checkpoint dir's
``optimizer.pt`` (written by ``TrainingConfig.save_optimizer_state``),
bias-corrects the second moments, averages them within each segment as the
K-FAC factors are, and writes

    P = 1 / (sqrt(v_hat_bar + eps_root) + eps)

to ``<run>/segment_<l>/preconditioner.safetensors``, oriented to bergson's
``[out, in]`` gradient grids.
"""

from glob import glob
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

from ..utils.load_from_optimizer import (
    get_unfactored_second_moment,
    load_optimizer,
    optimizer_param_index_to_name,
    orient_second_moment,
)

OPTIMIZER_STATE_FILE = "optimizer.pt"


def _module_grids(hessian_dir: str | Path) -> dict[str, tuple[int, int]]:
    """Module name -> full [out, in] grid shape, read from the factor shards.

    The activation eigenvectors are ``[c_i, I]`` (columns = full in-dim) and
    the gradient eigenvectors ``[c_o, O]``, so shard_0's column counts give
    the full grid shape regardless of world size.
    """
    grids: dict[str, tuple[int, int]] = {}
    a_dir = Path(hessian_dir) / "eigen_activation_sharded"
    g_dir = Path(hessian_dir) / "eigen_gradient_sharded"
    if not a_dir.is_dir() or not glob(str(g_dir / "shard_*.safetensors")):
        raise FileNotFoundError(f"No EKFAC factor shards under {hessian_dir}")
    with (
        safe_open(str(a_dir / "shard_0.safetensors"), framework="pt") as fa,
        safe_open(str(g_dir / "shard_0.safetensors"), framework="pt") as fg,
    ):
        for name in fa.keys():
            in_dim = fa.get_slice(name).get_shape()[1]
            out_dim = fg.get_slice(name).get_shape()[1]
            grids[name] = (out_dim, in_dim)
    return grids


def _match_params(
    grids: dict[str, tuple[int, int]], param_names: list[str]
) -> dict[str, str]:
    """Factor module name -> param name. Factor names are suffixes of the
    model's param names (e.g. ``h.0.attn.c_attn`` ->
    ``transformer.h.0.attn.c_attn.weight``)."""
    matches: dict[str, str] = {}
    for module in grids:
        suffix = f"{module}.weight"
        candidates = [n for n in param_names if n == suffix or n.endswith(f".{suffix}")]
        if len(candidates) != 1:
            raise KeyError(
                f"Module {module!r} matched {len(candidates)} params "
                f"({candidates}); expected exactly one *.{suffix}."
            )
        matches[module] = candidates[0]
    return matches


def _orient(p: torch.Tensor, grid: tuple[int, int], module: str, layer) -> torch.Tensor:
    """Orient a param-shaped grid to bergson's [out, in] layout, raising on a
    shape that fits neither orientation."""
    oriented = orient_second_moment(p, *grid, layer=layer)
    if oriented is None:
        raise ValueError(
            f"Second moments for {module!r} have shape {tuple(p.shape)}; "
            f"expected {grid} or its transpose."
        )
    return oriented


def _adam_hparams(optimizer_pt: dict) -> tuple[float, float, float]:
    """(beta2, eps, eps_root) from a standard ``optimizer.pt``'s param_groups.

    ``betas`` and ``eps`` are standard AdamW param-group keys (HF Trainer
    checkpoints carry them); ``eps_root`` is torchopt-specific and defaults
    to 0 (standard AdamW has no epsilon inside the sqrt)."""
    groups = optimizer_pt.get("param_groups", [])
    betas = groups[0].get("betas") if groups else None
    eps = groups[0].get("eps") if groups else None
    if betas is None or eps is None:
        raise ValueError(
            "optimizer.pt has no param_groups betas/eps; regenerate the "
            "snapshot exports with TrainingConfig.save_optimizer_state "
            "(older exports lack the hyperparameters needed to build the "
            "preconditioner)."
        )
    return float(betas[1]), float(eps), float(groups[0].get("eps_root", 0.0))


def _extract_v_hat(
    optimizer_pt: dict, index_to_name: dict[int, str], beta2: float
) -> dict[str, torch.Tensor]:
    """Bias-corrected second moments keyed by param name from a standard
    ``optimizer.pt`` dict carrying a per-entry ``step``."""

    v_hat: dict[str, torch.Tensor] = {}
    for idx, entry in optimizer_pt["state"].items():
        idx = int(idx)
        # Prefer the recorded param_name: positional indices written from an
        # FSDP-wrapped model do not match a fresh model's enumeration.
        name = entry.get("param_name") or index_to_name.get(idx)
        if name is None:
            continue
        if "step" not in entry:
            raise ValueError(
                f"optimizer.pt state entry {idx} has no step; regenerate the "
                "snapshot exports with TrainingConfig.save_optimizer_state."
            )
        step = int(
            entry["step"].item() if torch.is_tensor(entry["step"]) else entry["step"]
        )
        nu = get_unfactored_second_moment(entry).to(torch.float32)
        v_hat[name] = nu / (1.0 - beta2**step) if step > 0 else torch.zeros_like(nu)
    return v_hat


def build_segment_preconditioners(
    run_path: str | Path,
    method: str,
    checkpoints: list[str],
    segments: int,
) -> list[Path]:
    """Build ``<run>/segment_<l>/preconditioner.safetensors`` for every
    segment from the checkpoints' ``optimizer.pt`` files; see module docs.
    The Adam hyperparameters (betas/eps/eps_root) are read from the files.

    Returns the per-segment preconditioner paths.
    """
    base = Path(run_path)
    per_segment = len(checkpoints) // segments
    # For the optimizer.pt index mapping (used when entries carry no
    # param_name, e.g. HF Trainer checkpoints).
    config = AutoConfig.from_pretrained(checkpoints[0])
    reference_model = AutoModelForCausalLM.from_config(config)
    out_paths: list[Path] = []
    for seg in range(segments):
        seg_ckpts = checkpoints[seg * per_segment : (seg + 1) * per_segment]
        v_hat_sum: dict[str, torch.Tensor] = {}
        eps = eps_root = 0.0
        for ckpt in seg_ckpts:
            if not (Path(ckpt) / OPTIMIZER_STATE_FILE).exists():
                raise FileNotFoundError(
                    f"use_adam_preconditioner requires "
                    f"{Path(ckpt) / OPTIMIZER_STATE_FILE}; write it during "
                    "training with TrainingConfig.save_optimizer_state and "
                    "copy it into the exported checkpoint dir."
                )
            optimizer_pt = load_optimizer(str(ckpt))
            beta2, eps, eps_root = _adam_hparams(optimizer_pt)
            index_to_name = optimizer_param_index_to_name(optimizer_pt, reference_model)
            for name, v_hat in _extract_v_hat(
                optimizer_pt, index_to_name, beta2
            ).items():
                v_hat_sum[name] = v_hat_sum.get(name, 0) + v_hat

        grids = _module_grids(base / f"segment_{seg}" / method)
        matches = _match_params(grids, list(v_hat_sum))
        preconditioner = {}
        for module, param_name in matches.items():
            try:
                layer = reference_model.get_submodule(
                    param_name.removesuffix(".weight")
                )
            except AttributeError:
                layer = None
            v_hat_bar = v_hat_sum[param_name] / len(seg_ckpts)
            p = 1.0 / ((v_hat_bar + eps_root).sqrt() + eps)
            preconditioner[module] = _orient(p, grids[module], module, layer)

        out_path = base / f"segment_{seg}" / "preconditioner.safetensors"
        save_file(preconditioner, str(out_path))
        out_paths.append(out_path)
    return out_paths
