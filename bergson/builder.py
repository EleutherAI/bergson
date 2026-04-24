from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset

from .collector.collector import create_projection_matrix
from .config import PreprocessConfig
from .data import compute_num_token_grads, create_index, create_token_index
from .preconditioners import (
    Preconditioner,
    _FactoredPreconditioner,
    load_preconditioner,
)
from .process_grads import normalize_flat_grad
from .utils.utils import convert_dtype_to_np, tensor_to_numpy

_EPS_SQ = torch.finfo(torch.float32).eps ** 2


def _preprocess(
    mod_grads: dict[str, torch.Tensor],
    grad_sizes,
    preconditioner: Preconditioner,
    do_normalize: bool,
    post_projection: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> torch.Tensor:
    """Precondition, optionally project, concatenate, and optionally
    unit-normalize gradients.

    ``post_projection`` is only used for the factored-preconditioner path:
    the preconditioner takes unprojected ``[N, O*I]`` inputs and emits
    unprojected outputs; we then apply a per-module double-sided random
    projection to compress each module's gradient to ``[N, p*p]`` before
    concatenation. For the autocorrelation path this stays ``None`` and
    the pipeline is unchanged (projection already happened in the
    collector)."""
    mod_grads = preconditioner.apply(mod_grads)

    if post_projection is not None:
        projected: dict[str, torch.Tensor] = {}
        for name, g in mod_grads.items():
            L, R = post_projection[name]  # L: [p, O], R: [p, I]
            O_dim = L.shape[1]
            I_dim = R.shape[1]
            g_shape = g.shape
            g = g.to(device=L.device, dtype=L.dtype).view(g_shape[0], O_dim, I_dim)
            # G' = L @ G @ R.T, shape [N, p, p] → flatten to [N, p*p]
            g = torch.einsum("p o, n o i -> n p i", L, g)
            g = torch.einsum("q i, n p i -> n p q", R, g)
            projected[name] = g.reshape(g_shape[0], -1)
        mod_grads = projected

    grads = torch.cat([mod_grads[m] for m in grad_sizes.keys()], dim=-1)

    if do_normalize:
        inv_norms = grads.pow(2).sum(dim=-1).clamp_min_(_EPS_SQ).rsqrt().unsqueeze(1)
        grads = grads * inv_norms

    return grads


class Builder:
    """Gradient index writer.

    Handles all combinations of storage (disk / in-memory) and
    granularity (per-sequence / per-token), with optional
    preconditioning and aggregation.

    Parameters
    ----------
    data : Dataset
        The dataset being indexed.
    grad_sizes : dict[str, int]
        Per-module gradient dimensions **as they arrive from the collector**
        (i.e. unprojected ``O*I`` when ``post_projection_dim`` is set,
        projected ``p*p`` otherwise).
    dtype : torch.dtype
        Torch dtype for the gradients.
    preprocess_cfg : PreprocessConfig
        Preconditioning, normalization, and aggregation settings.
    attribute_tokens : bool
        Per-token gradients instead of per-example.
    path : Path | None
        When given, write to a memory-mapped file on disk.
        When ``None``, store in a plain numpy array.
    post_projection_dim : int | None
        When set (only supported with a factored preconditioner on
        ``preconditioner_path``), the builder applies a per-module
        double-sided random projection to ``[p, p]`` per module after
        preconditioning. Matches the collector's ``double_sided_projection``
        machinery so the on-disk layout is identical to the
        autocorrelation path's projected layout.
    projection_type : {"normal", "rademacher"}
        Projection-matrix distribution (ignored when
        ``post_projection_dim`` is ``None``).
    """

    grad_buffer: np.ndarray

    def __init__(
        self,
        data: Dataset,
        grad_sizes: dict[str, int],
        dtype: torch.dtype,
        preprocess_cfg: PreprocessConfig,
        *,
        attribute_tokens: bool = False,
        path: Path | None = None,
        post_projection_dim: int | None = None,
        projection_type: Literal["normal", "rademacher"] = "rademacher",
    ):
        self.num_items = len(data)
        self.preprocess_cfg = preprocess_cfg

        # ── Device & precomputed preconditioner ──────────────────────────────────────
        device = torch.device("cuda", torch.cuda.current_device())
        self.preconditioner = load_preconditioner(
            preprocess_cfg.preconditioner_path,
            device=device,
            power=-0.5 if preprocess_cfg.unit_normalize else -1,
        )

        # ── Post-preconditioning projection (factored variants only) ───────
        self._post_projection: (
            dict[str, tuple[torch.Tensor, torch.Tensor]] | None
        ) = None
        if post_projection_dim is not None:
            if not isinstance(self.preconditioner, _FactoredPreconditioner):
                raise ValueError(
                    "post_projection_dim is only supported with a factored "
                    "preconditioner (EKFAC or KFAC)."
                )
            self._post_projection = {}
            for name, (O_dim, I_dim) in self.preconditioner._shapes.items():
                L = create_projection_matrix(
                    f"{name}/left",
                    post_projection_dim,
                    O_dim,
                    dtype=torch.float32,
                    device=device,
                    projection_type=projection_type,
                )
                R = create_projection_matrix(
                    f"{name}/right",
                    post_projection_dim,
                    I_dim,
                    dtype=torch.float32,
                    device=device,
                    projection_type=projection_type,
                )
                self._post_projection[name] = (L, R)

            # The on-disk layout uses the projected size, not the
            # unprojected ``grad_sizes`` we received from the collector.
            grad_sizes = {
                name: post_projection_dim * post_projection_dim
                for name in grad_sizes
            }

        self.grad_sizes = grad_sizes
        total_grad_dim = sum(grad_sizes.values())

        # ── Aggregation buffer (sequence-level only) ─────────────────────
        if preprocess_cfg.aggregation != "none":
            np_dtype = np.float32
            num_grads = 1
            self.in_memory_grad_buffer: torch.Tensor | None = torch.zeros(
                (1, total_grad_dim),
                dtype=torch.float32,
                device=device,
            )
        else:
            np_dtype = convert_dtype_to_np(dtype)
            num_grads = self.num_items
            self.in_memory_grad_buffer = None

        # ── Gradient buffer (disk or memory, sequence or token) ──────────
        if attribute_tokens:
            self.num_token_grads = compute_num_token_grads(data)
            if path is not None:
                self.grad_buffer, self.offsets = create_token_index(
                    path,
                    self.num_token_grads,
                    grad_sizes,
                    np_dtype,
                )
            else:
                self.offsets = np.zeros(
                    len(self.num_token_grads) + 1,
                    dtype=np.int64,
                )
                np.cumsum(self.num_token_grads, out=self.offsets[1:])
                total_tokens = int(self.offsets[-1])
                self.grad_buffer = np.zeros(
                    (total_tokens, total_grad_dim),
                    dtype=np_dtype,
                )
            self._scatter_flat = self._scatter_flat_tokens
        else:
            self.num_token_grads = None
            self.offsets = None
            if path is not None:
                self.grad_buffer = create_index(
                    path,
                    num_grads=num_grads,
                    grad_sizes=grad_sizes,
                    dtype=np_dtype,
                    with_structure=False,
                )
            else:
                self.grad_buffer = np.zeros(
                    (num_grads, total_grad_dim),
                    dtype=np_dtype,
                )
            self._scatter_flat = self._scatter_flat_sequences

    # ── __call__ ─────────────────────────────────────────────────────────

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ) -> None:
        grads = _preprocess(
            mod_grads,
            self.grad_sizes,
            self.preconditioner,
            self.preprocess_cfg.unit_normalize,
            post_projection=self._post_projection,
        )

        if self.preprocess_cfg.aggregation != "none":
            assert self.in_memory_grad_buffer is not None
            self.in_memory_grad_buffer[0] += grads.sum(dim=0).to(
                dtype=torch.float32, device=self.in_memory_grad_buffer.device
            )
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._scatter_flat(indices, grads)

    # ── Scatter strategies ───────────────────────────────────────────────

    def _scatter_flat_sequences(
        self,
        indices: list[int],
        grads: torch.Tensor,
    ) -> None:
        self.grad_buffer[indices] = tensor_to_numpy(grads.cpu())

    def _scatter_flat_tokens(
        self,
        indices: list[int],
        grads: torch.Tensor,
    ) -> None:
        assert self.num_token_grads is not None and self.offsets is not None
        per_example_lengths = self.num_token_grads[indices]
        g_np = tensor_to_numpy(grads.cpu())

        row = 0
        for idx, sl in zip(indices, per_example_lengths):
            buf_start = int(self.offsets[idx])
            buf_end = int(self.offsets[idx + 1])
            self.grad_buffer[buf_start:buf_end] = g_np[row : row + sl]
            row += sl

    # ── Lifecycle ────────────────────────────────────────────────────────

    def flush(self) -> None:
        if isinstance(self.grad_buffer, np.memmap):
            self.grad_buffer.flush()

    def teardown(self) -> None:
        self.flush()

        if self.preprocess_cfg.aggregation == "none":
            # Gather in-memory data from other ranks
            if dist.is_initialized() and not isinstance(self.grad_buffer, np.memmap):
                dist.all_reduce(
                    torch.from_numpy(self.grad_buffer),
                    op=dist.ReduceOp.SUM,
                )
            return

        assert self.in_memory_grad_buffer is not None

        if dist.is_initialized():
            dist.reduce(
                self.in_memory_grad_buffer,
                dst=0,
                op=dist.ReduceOp.SUM,
            )

        if self.preprocess_cfg.aggregation == "mean":
            self.in_memory_grad_buffer /= self.num_items

        if self.preprocess_cfg.normalize_aggregated_grad:
            self.in_memory_grad_buffer = normalize_flat_grad(
                self.in_memory_grad_buffer,
                self.in_memory_grad_buffer.device,
            )

        self.in_memory_grad_buffer = self.in_memory_grad_buffer.cpu()

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            self.grad_buffer[:] = tensor_to_numpy(self.in_memory_grad_buffer).astype(
                self.grad_buffer.dtype
            )
