"""Preconditioners: apply a preconditioner (sometimes interpreted as an
inverse Hessian) to a set of gradients.

A :class:`Preconditioner` exposes one method, ``apply(grads) -> grads``, mapping a
``{module: [n, d]}`` gradient dict to its preconditioned counterpart. This class
has two implementations:

- :class:`DensePreconditioner` — the autocorrelation Gram, a dense per-module
  ``[d, d]`` matrix; ``apply`` is a matmul ``g @ H^p``.
- :class:`FactoredPreconditioner` — the factored EKFAC family (kfac / tkfac /
  shampoo), a Kronecker product stored as eigenvectors ``Q_A`` ``[I, I]`` /
  ``Q_G`` ``[O, O]`` and an eigenvalue grid ``λ`` ``[O, I]``; ``apply`` rotates
  each gradient into the eigenbasis, scales by the inverse eigenvalue function,
  and rotates back, never materializing the dense ``[O·I, O·I]`` Hessian.

:func:`load_preconditioner` builds one from a path, auto-detecting which
representation lives there.

The factored ``apply`` runs both in a single process (full factors) and
distributed (per-rank shards): :class:`~bergson.hessians.sharded_computation.ShardedMul`
provides the rotations and in-place scaling for both. The eigenvalue math lives
in :mod:`bergson.hessians.inversion`.
"""

import os
from glob import glob
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch import Tensor

from bergson.config import InversionConfig
from bergson.gradients import GradientProcessor
from bergson.hessians.inversion import eigenvalue_multiplier, invert_psd_matrix
from bergson.hessians.sharded_computation import ShardedMul
from bergson.utils.logger import get_logger


@runtime_checkable
class Preconditioner(Protocol):
    """Applies an inverse Hessian to a ``{module: [n, d]}`` gradient dict."""

    def apply(self, grads: dict[str, Tensor]) -> dict[str, Tensor]: ...


class DensePreconditioner:
    """Dense (autocorrelation) preconditioner: per-module ``H^p`` matrices."""

    def __init__(self, h_inv: dict[str, Tensor]):
        self.h_inv = h_inv

    @classmethod
    def from_processor(
        cls,
        processor: GradientProcessor,
        *,
        inversion_cfg: InversionConfig | None = None,
        power: float = -1.0,
        device: str | torch.device = "cpu",
    ) -> "DensePreconditioner":
        """Invert each dense per-module Gram in ``processor`` to ``H^power`` in fp32."""
        inversion_cfg = inversion_cfg or InversionConfig()
        h_inv = {
            name: invert_psd_matrix(
                H.to(device=device, dtype=torch.float32),
                inversion=inversion_cfg.inversion,
                damping_factor=inversion_cfg.damping_factor,
                power=power,
            )
            for name, H in processor.hessians.items()
        }
        return cls(h_inv)

    def apply(self, grads: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return ``{name: g @ H^p}``; modules absent from the Hessian pass
        through unchanged. Each result is returned on the gradient's device."""
        if not self.h_inv:
            return grads
        return {
            name: (
                (
                    g.to(device=self.h_inv[name].device, dtype=self.h_inv[name].dtype)
                    @ self.h_inv[name]
                ).to(g.device)
                if name in self.h_inv
                else g
            )
            for name, g in grads.items()
        }


class FactoredPreconditioner:
    """Factored (EKFAC) preconditioner applied via the eigenbasis rotation.

    Works single-process (full factors) and distributed (per-rank row-shards);
    :class:`ShardedMul` handles both. Construct via :meth:`from_path` (single
    process, concatenated factors) or :meth:`from_shards` (this rank's shard
    files, under an initialized process group).

    ``inversion_cfg`` and ``apply_fn`` are mutually exclusive: pass a standard
    eigenvalue inversion, or a custom eigenvalue function (approximate unrolling's
    learning-rate eigenfunctions, applied at ``power = -1``).
    """

    def __init__(
        self,
        eigen_a: dict[str, Tensor],
        eigen_g: dict[str, Tensor],
        lambdas: dict[str, Tensor],
        *,
        inversion_cfg: InversionConfig | None = None,
        apply_fn=None,
        power: float = -1.0,
        factor_eig_a: dict[str, Tensor] | None = None,
        factor_eig_g: dict[str, Tensor] | None = None,
    ):
        if inversion_cfg is not None and apply_fn is not None:
            raise ValueError("Pass either inversion_cfg or apply_fn, not both.")

        self.eigen_a = eigen_a
        self.eigen_g = eigen_g
        self.lambdas = lambdas
        self.factor_eig_a = factor_eig_a or {}
        self.factor_eig_g = factor_eig_g or {}
        self.inversion_cfg = inversion_cfg or InversionConfig()
        self.apply_fn = apply_fn
        self.power = power
        self.shard_computer = ShardedMul()
        # Per-step debug logs are off by default; consumer code may raise this
        # logger to DEBUG to surface the apply's progress trace.
        self.logger = get_logger("FactoredPreconditioner")

    @classmethod
    def from_path(
        cls,
        hessian_path: str | Path,
        *,
        inversion_cfg: InversionConfig | None = None,
        apply_fn=None,
        power: float = -1.0,
        ev_correction: bool = False,
        device: str | torch.device = "cpu",
    ) -> "FactoredPreconditioner":
        """Single-process: load the full factors (concatenating any shards)."""

        def load(sub):
            return _load_full(hessian_path, sub, device)

        def load_replicated(sub):
            # The eigenvalue grid λ is [O, I] and is row-sharded along O, so only
            # the O-indexed gradient factor (factor_eig_g, [O]) is split across
            # shards; the I-indexed activation factor (factor_eig_a, [I]) rides
            # the unsharded column dim and is written in full to every shard.
            # So load a single shard rather than concatenating across shards.
            return _load_shard(hessian_path, sub, 0, device)

        return cls._from_loaded(
            load, load_replicated, inversion_cfg, apply_fn, power, ev_correction
        )

    @classmethod
    def from_shards(
        cls,
        hessian_path: str | Path,
        *,
        rank: int,
        device: str | torch.device,
        inversion_cfg: InversionConfig | None = None,
        apply_fn=None,
        power: float = -1.0,
        ev_correction: bool = False,
    ) -> "FactoredPreconditioner":
        """Distributed: load this rank's shard of the factors."""

        def load(sub):
            return _load_shard(hessian_path, sub, rank, device)

        # Each rank's shard of factor_eig_a is already the full replicated [I].
        return cls._from_loaded(
            load, load, inversion_cfg, apply_fn, power, ev_correction
        )

    @classmethod
    def _from_loaded(
        cls, load, load_replicated, inversion_cfg, apply_fn, power, ev_correction
    ):
        factored_tikhonov = (
            inversion_cfg is not None and inversion_cfg.inversion == "factored_tikhonov"
        )
        if factored_tikhonov and ev_correction:
            raise ValueError(
                "factored_tikhonov inversion is incompatible with ev_correction: "
                "the corrected eigenvalues do not factorize as λ_G ⊗ λ_A. Use a "
                "different inversion (e.g. damped_inverse) or set ev_correction=False."
            )

        lambda_dir = (
            "eigenvalue_correction_sharded" if ev_correction else "eigenvalue_sharded"
        )
        factor_eig_a = factor_eig_g = None
        if factored_tikhonov:
            # factor_eig_a indexes the unsharded column dim I -> replicated, full
            # [I]; factor_eig_g indexes the sharded row dim O -> row-sharded.
            factor_eig_a = load_replicated("factor_eig_a")  # replicated, full [I]
            factor_eig_g = load("factor_eig_g")  # row-sharded along O
        return cls(
            load("eigen_activation_sharded"),
            load("eigen_gradient_sharded"),
            load(lambda_dir),
            inversion_cfg=inversion_cfg,
            apply_fn=apply_fn,
            power=power,
            factor_eig_a=factor_eig_a,
            factor_eig_g=factor_eig_g,
        )

    def _scale(self, name: str, g: Tensor) -> None:
        """In-place scale ``g`` (in the eigenbasis) by the inverse eigenvalues.

        ``g`` is the full ``[n, O, I]`` gradient; ``self.lambdas[name]`` is this
        rank's row-shard ``[c, I]`` of the eigenvalue grid (the full grid when not
        distributed). The inverse eigenvalues are computed from this rank's shard
        plus the globally-reduced means, then applied across shards in-place.
        """
        lam = self.lambdas[name]
        o, i = g.shape[1], lam.shape[1]
        if self.apply_fn is not None:
            inverse_eigvals = self.apply_fn(lam)
        else:
            mean = self.shard_computer.global_mean(lam, o * i)
            inversion = self.inversion_cfg.inversion
            if inversion == "factored_tikhonov":
                factor_a = self.factor_eig_a[name]
                factor_g = self.factor_eig_g[name]
                inverse_eigvals = eigenvalue_multiplier(
                    inversion,
                    lam,
                    mean,
                    self.inversion_cfg.damping_factor,
                    factor_a=factor_a,
                    factor_g=factor_g,
                    mean_a=factor_a.clamp_min(0).mean(),
                    mean_g=self.shard_computer.global_mean(factor_g.clamp_min(0), o),
                    power=self.power,
                )
            else:
                inverse_eigvals = eigenvalue_multiplier(
                    inversion,
                    lam,
                    mean,
                    self.inversion_cfg.damping_factor,
                    power=self.power,
                )
        self.shard_computer.scale_rows_in_place(g, inverse_eigvals)

    def apply(self, grads: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return ``grads`` with the factored inverse Hessian applied per module.

        Each ``grads[name]`` is a flat ``[n, O*I]`` block; it is reshaped to
        ``[n, O, I]``, rotated into the eigenbasis (``Q_G^T G Q_A``), scaled, and
        rotated back. Modules absent from the Hessian pass through unchanged.
        """
        out: dict[str, Tensor] = {}
        for name, flat in grads.items():
            if name not in self.eigen_a:
                out[name] = flat
                continue

            q_a = self.eigen_a[name]
            q_g = self.eigen_g[name]
            o, i = q_g.shape[1], q_a.shape[1]
            g = flat.to(q_a.device, torch.float32).view(-1, o, i)

            # Forward rotation: G' = Q_G^T @ G @ Q_A
            g = self.shard_computer._matmul(vector_nsa=g, matrix_cb=q_a)
            g = self.shard_computer._matmul(
                vector_nsa=g.transpose(-2, -1), matrix_cb=q_g
            ).transpose(-2, -1)
            self.logger.debug("%s: rotated into eigenbasis (Q_G^T G Q_A)", name)

            self._scale(name, g)
            self.logger.debug("%s: scaled by inverse eigenvalues", name)

            # Rotate back: Q_G @ G' @ Q_A^T
            g = self.shard_computer._transpose_matmul(
                vector_nsa=g.transpose(-2, -1), matrix_cb=q_g
            ).transpose(-2, -1)
            g = self.shard_computer._transpose_matmul(vector_nsa=g, matrix_cb=q_a)
            self.logger.debug("%s: rotated back (H^-1 G)", name)

            out[name] = g.reshape(flat.shape[0], -1).to(flat.dtype)
        return out


class DiagonalFactoredPreconditioner:
    """Applies ``apply_fn`` elementwise in parameter space, on the diagonal of
    ``P^1/2 H P^1/2`` for an optimizer preconditioner ``P`` — the Adam/AdamW
    approximate-unrolling variant (Bae et al., 2024, Appendices C and D.2).

    There is no eigenbasis rotation: the multiplier is ``apply_fn(p * d)``,
    with ``d = diag(H) = (Q_G ∘ Q_G) Λ (Q_A ∘ Q_A)^T`` from the EKFAC factors,
    times ``p`` when ``multiply_by_preconditioner`` is set (F_segment's
    ``P^1/2 F(M) P^1/2`` sandwich; F_backward's cancels).
    """

    def __init__(
        self,
        eigen_a: dict[str, Tensor],
        eigen_g: dict[str, Tensor],
        lambdas: dict[str, Tensor],
        preconditioner: dict[str, Tensor],
        *,
        apply_fn,
        multiply_by_preconditioner: bool = False,
    ):
        self.eigen_a = eigen_a
        self.eigen_g = eigen_g
        self.lambdas = lambdas
        self.preconditioner = preconditioner
        self.apply_fn = apply_fn
        self.multiply_by_preconditioner = multiply_by_preconditioner
        self.shard_computer = ShardedMul()
        self.logger = get_logger("DiagonalFactoredPreconditioner")
        self._multipliers: dict[str, Tensor] = {}

    @classmethod
    def from_shards(
        cls,
        hessian_path: str | Path,
        preconditioner_path: str | Path,
        *,
        rank: int,
        device: str | torch.device,
        apply_fn,
        multiply_by_preconditioner: bool = False,
        ev_correction: bool = False,
    ) -> "DiagonalFactoredPreconditioner":
        """Load this rank's factor shards plus its row-shard of the full
        parameter-space preconditioner grids saved at ``preconditioner_path``."""
        lambda_dir = (
            "eigenvalue_correction_sharded" if ev_correction else "eigenvalue_sharded"
        )
        eigen_a = _load_shard(hessian_path, "eigen_activation_sharded", rank, device)
        eigen_g = _load_shard(hessian_path, "eigen_gradient_sharded", rank, device)
        lambdas = _load_shard(hessian_path, lambda_dir, rank, device)

        full_grids = load_file(str(preconditioner_path), device=str(device))
        sharder = ShardedMul()
        preconditioner = {}
        for name, q_g in eigen_g.items():
            if name not in full_grids:
                raise KeyError(
                    f"Module {name!r} has EKFAC factors but no preconditioner "
                    f"grid in {preconditioner_path}; available: "
                    f"{sorted(full_grids.keys())}"
                )
            p = full_grids[name].to(torch.float32)
            o, i = q_g.shape[1], eigen_a[name].shape[1]
            if p.shape != (o, i):
                raise ValueError(
                    f"Preconditioner grid for {name!r} has shape "
                    f"{tuple(p.shape)}, expected [out, in] = ({o}, {i})."
                )
            start, end = sharder.shard_bounds(o)
            preconditioner[name] = p[start:end]
        return cls(
            eigen_a,
            eigen_g,
            lambdas,
            preconditioner,
            apply_fn=apply_fn,
            multiply_by_preconditioner=multiply_by_preconditioner,
        )

    def _diag_hessian_shard(self, name: str) -> Tensor:
        """This rank's ``[c_o, I]`` row-shard (rows = its block of the
        parameter out-dim) of ``diag(H) = (Q_G ∘ Q_G) Λ (Q_A ∘ Q_A)^T``.

        The factors are row-sharded: ``Q_A [c_i, I']`` / ``Q_G [c_o, O']`` on
        their parameter dims, ``Λ [c_o', I']`` on the eigen out-dim. Both
        contractions run over a sharded dim, so each is a broadcast loop in
        the style of :class:`ShardedMul`.
        """
        q_a = self.eigen_a[name]  # [c_i, I'] param rows, eigen cols
        q_g = self.eigen_g[name]  # [c_o, O'] param rows, eigen cols
        lam = self.lambdas[name]  # [c_o', I'] eigen rows, eigen cols
        sc = self.shard_computer
        n_eig_i = q_a.shape[1]
        n_eig_o = q_g.shape[1]

        if not sc.dist:
            # Single process: shards are the full factors.
            return (q_g**2) @ lam @ (q_a**2).T

        # x_shard[o', i] = sum_i' lam[o', i'] * q_a[i, i']^2 for this rank's
        # eigen-o' rows and ALL param-i columns (q_a's param rows are sharded).
        x_shard = torch.empty(lam.shape[0], n_eig_i, device=lam.device, dtype=lam.dtype)
        for rank_index in range(sc.world_size):
            start, end = sc.shard_bounds(n_eig_i, rank_index)
            if rank_index == sc.rank:
                shard = q_a
            else:
                shard = torch.empty(
                    (end - start, q_a.shape[1]), device=q_a.device, dtype=q_a.dtype
                )
            dist.broadcast(shard, src=rank_index)
            x_shard[:, start:end] = lam @ (shard**2).T
            if sc.rank != rank_index:
                del shard

        # d_shard[o, i] = sum_o' q_g[o, o']^2 * x[o', i]; x rows (eigen-o') are
        # sharded across ranks, q_g's eigen-o' columns are full.
        d_shard = torch.zeros(q_g.shape[0], n_eig_i, device=q_g.device, dtype=q_g.dtype)
        for rank_index in range(sc.world_size):
            start, end = sc.shard_bounds(n_eig_o, rank_index)
            if rank_index == sc.rank:
                shard = x_shard
            else:
                shard = torch.empty(
                    (end - start, n_eig_i), device=x_shard.device, dtype=x_shard.dtype
                )
            dist.broadcast(shard, src=rank_index)
            d_shard += (q_g[:, start:end] ** 2) @ shard
            if sc.rank != rank_index:
                del shard
        return d_shard

    def _multiplier(self, name: str) -> Tensor:
        """This rank's ``[c_o, I]`` row-shard of the elementwise multiplier,
        cached after the first batch (it is gradient-independent)."""
        if name not in self._multipliers:
            p = self.preconditioner[name]
            sigma = p * self._diag_hessian_shard(name)
            multiplier = self.apply_fn(sigma)
            if self.multiply_by_preconditioner:
                multiplier = multiplier * p
            self._multipliers[name] = multiplier
        return self._multipliers[name]

    def apply(self, grads: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return ``grads`` scaled elementwise in parameter space; modules
        absent from the factors pass through unchanged."""
        out: dict[str, Tensor] = {}
        for name, flat in grads.items():
            if name not in self.eigen_a:
                out[name] = flat
                continue
            o = self.eigen_g[name].shape[1]
            i = self.eigen_a[name].shape[1]
            # Copy rather than alias: the caller's gradients may be a read-only
            # mmap, and the scale below writes through. `.to` is a no-op when
            # they are already float32 on this device, as on a CPU-only run.
            g = flat.to(self.lambdas[name].device, torch.float32).clone().view(-1, o, i)
            self.shard_computer.scale_rows_in_place(g, self._multiplier(name))
            self.logger.debug("%s: scaled elementwise by f(p * diag(H))", name)
            out[name] = g.reshape(flat.shape[0], -1).to(flat.dtype)
        return out


def is_factored_hessian(hessian_path: str | Path) -> bool:
    """Whether ``hessian_path`` holds a Kronecker-factored (EKFAC) Hessian rather
    than a dense one — detected by the presence of the eigenvector shard directory.
    """
    return (Path(hessian_path) / "eigen_activation_sharded").is_dir()


def load_preconditioner(
    hessian_path: str | Path | None,
    *,
    inversion_cfg: InversionConfig | None = None,
    power: float | None = -1.0,
    ev_correction: bool = False,
    device: str | torch.device = "cpu",
) -> Preconditioner | None:
    """Build the preconditioner for the Hessian at ``hessian_path`` in fp32.

    Auto-detects factored (e.g. EKFAC) vs dense (a saved :class:`GradientProcessor`).
    Returns ``None`` when ``hessian_path`` is unset or ``power`` is ``None`` (no
    preconditioning).
    """
    if power is None or hessian_path is None:
        return None

    if is_factored_hessian(hessian_path):
        return FactoredPreconditioner.from_path(
            hessian_path,
            inversion_cfg=inversion_cfg,
            power=power,
            ev_correction=ev_correction,
            device=device,
        )

    # Dense: load the saved processor on CPU; from_processor moves each Gram to
    # device as it inverts it, so only one dense matrix is on the device at a time.
    processor = GradientProcessor.load(Path(hessian_path), map_location="cpu")
    return DensePreconditioner.from_processor(
        processor, inversion_cfg=inversion_cfg, power=power, device=device
    )


def _load_full(
    hessian_path: str | Path, subdir: str, device: str | torch.device
) -> dict[str, Tensor]:
    """Load every ``shard_*.safetensors`` under ``hessian_path/subdir`` onto
    ``device`` and concatenate each key along dim 0, reconstructing the full
    per-module tensors in fp32.

    The shards were written by row-splitting dim 0 with ``shard_bounds`` (rank 0
    takes the remainder), so concatenating in rank order is exact regardless of
    how many ranks fit the Hessian. With ``world_size == 1`` there is a single
    ``shard_0`` holding the full tensor.
    """
    shard_dir = Path(hessian_path) / subdir
    shards = sorted(
        glob(os.path.join(shard_dir, "shard_*.safetensors")),
        key=lambda p: int(Path(p).stem.split("_")[1]),
    )
    if not shards:
        raise FileNotFoundError(f"No shard_*.safetensors found in {shard_dir}")

    loaded = [load_file(s, device=str(device)) for s in shards]
    keys = loaded[0].keys()
    return {
        name: torch.cat([shard[name] for shard in loaded], dim=0).to(torch.float32)
        for name in keys
    }


def _load_shard(
    hessian_path: str | Path, subdir: str, rank: int, device: str | torch.device
) -> dict[str, Tensor]:
    """Load ``rank``'s shard of the factors under ``hessian_path/subdir``
    onto ``device`` and cast to fp32."""
    shard = load_file(
        os.path.join(str(hessian_path), subdir, f"shard_{rank}.safetensors"),
        device=str(device),
    )
    return {k: v.to(torch.float32) for k, v in shard.items()}
