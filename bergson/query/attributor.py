from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Literal

import torch
from torch import Tensor, nn

from bergson.collector.gradient_collectors import TraceCollector
from bergson.config import InversionConfig
from bergson.data import load_gradients
from bergson.gradients import GradientProcessor
from bergson.hessians.preconditioner import (
    DensePreconditioner,
    FactoredPreconditioner,
    load_preconditioner,
)
from bergson.query.faiss_index import FaissConfig, FaissIndex
from bergson.utils.utils import numpy_to_tensor


class TraceResult:
    """Result of a .trace() call."""

    def __init__(self):
        # Should be set by the Attributor after a search
        self._indices: Tensor | None = None
        self._scores: Tensor | None = None

    @property
    def indices(self) -> Tensor:
        """The indices of the top-k examples."""
        if self._indices is None:
            raise ValueError("No indices available. Exit the context manager first.")

        return self._indices

    @property
    def scores(self) -> Tensor:
        """The attribution scores of the top-k examples."""
        if self._scores is None:
            raise ValueError("No scores available. Exit the context manager first.")

        return self._scores


class Attributor:

    precondition: Literal["one-sided", "two-sided", "none"]

    def __init__(
        self,
        index_path: str | Path,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        unit_norm: bool = False,
        faiss_cfg: FaissConfig | None = None,
        inversion_cfg: InversionConfig | None = None,
        hessian_path: str | Path | None = None,
        ev_correction: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.unit_norm = unit_norm
        self.inversion_cfg = inversion_cfg or InversionConfig()

        self.faiss_index = None
        index_path = Path(index_path)

        # Load the gradient processor
        self.processor = GradientProcessor.load(index_path, map_location=device)

        # Set `hessian_path` to enable preconditioning. Two-sided
        # preconditioning (H^(-1/2) on index as well as query) is
        # applied if unit norm is enabled.
        if hessian_path is None:
            self.precondition = "none"
            self.preconditioner = None
        else:
            self.precondition = "two-sided" if unit_norm else "one-sided"
            self.preconditioner = load_preconditioner(
                hessian_path,
                inversion_cfg=self.inversion_cfg,
                power=-0.5 if unit_norm else -1.0,
                ev_correction=ev_correction,
                device=device,
            )

        # Load the gradients into a FAISS index
        if faiss_cfg:
            if isinstance(self.preconditioner, FactoredPreconditioner):
                raise NotImplementedError(
                    "FAISS preconditioning is not supported for factored "
                    "(EKFAC) Hessians; omit faiss or use the dense path."
                )
            h_inv = (
                self.preconditioner.h_inv
                if isinstance(self.preconditioner, DensePreconditioner)
                else {}
            )
            faiss_index_name = (
                f"faiss_{faiss_cfg.index_factory.replace(',', '_')}"
                f"{'_cosine' if unit_norm else ''}"
                f"{'_precondition' if self.precondition != 'none' else ''}"
            )
            faiss_path = index_path / faiss_index_name

            if not (faiss_path / "config.yaml").exists():
                FaissIndex.create_index(
                    index_path,
                    faiss_path,
                    faiss_cfg,
                    device,
                    unit_norm,
                    h_inv,
                )

            self.faiss_index = FaissIndex(
                faiss_path, device, mmap_index=faiss_cfg.mmap_index
            )
            self.N = self.faiss_index.ntotal
            self.ordered_modules = self.faiss_index.ordered_modules
            return

        # Load the gradients into memory
        mmap = load_gradients(index_path)
        assert mmap.dtype.names is not None
        # Copy gradients into device memory
        self.grads = {
            name: numpy_to_tensor(mmap[name]).to(device=device, dtype=dtype)
            for name in mmap.dtype.names
        }
        self.N = mmap[mmap.dtype.names[0]].shape[0]

        self.ordered_modules = mmap.dtype.names

        if unit_norm:
            if self.preconditioner is not None:
                # Split: apply H^(-1/2) to the index grads before
                # normalization (`self.precondition == "two-sided"`, so the
                # preconditioner holds the H^(-1/2) factors/matrices).
                self.grads = {
                    name: g.to(dtype=dtype)
                    for name, g in self.preconditioner.apply(self.grads).items()
                }

            norm_sq = sum(
                (
                    torch.square(self.grads[name]).sum(dim=1, keepdim=True)
                    for name in self.ordered_modules
                ),
                start=0.0,
            )
            norm = norm_sq**0.5

            for name in self.grads:
                # Divide by norm (may create NaN/inf if norm is zero)
                normalized = self.grads[name] / norm
                # Convert NaN/inf to 0 and warn if any were found
                if not torch.isfinite(normalized).all():
                    print(
                        f"Warning: NaN/inf values detected after normalization in "
                        f"{name}, converting to 0"
                    )
                self.grads[name] = torch.nan_to_num(
                    normalized, nan=0.0, posinf=0.0, neginf=0.0
                )

    def search(
        self,
        queries: dict[str, Tensor],
        k: int | None,
        modules: set[str] | None = None,
        reverse: bool = False,
    ):
        """
        Search for the `k` nearest examples in the index based on the query or queries.

        Args:
            queries: The query tensor of shape [..., d].
            k: The number of nearest examples to return for each query.
            module: The name of the module to search for. If `None`,
                all modules will be searched.
            reverse: If True, return the lowest influence examples instead of highest.

        Returns:
            A namedtuple containing the top `k` indices and inner products for each
            query. Both have shape [..., k].
        """
        q = {
            name: queries[name].to(self.device, self.dtype)
            for name in self.ordered_modules
        }

        # One- or two-sided preconditioning: apply H^(-1) or H^(-0.5) to query
        if self.preconditioner is not None:
            q = {
                name: g.to(self.dtype)
                for name, g in self.preconditioner.apply(q).items()
            }

        if self.unit_norm:
            norm = torch.cat(list(q.values()), dim=1).norm(dim=1, keepdim=True)

            for name in q:
                q[name] /= norm + torch.finfo(norm.dtype).eps

        if self.faiss_index:
            if modules:
                raise NotImplementedError(
                    "FAISS index does not implement module-specific search."
                )

            q = (
                torch.cat([q[name] for name in self.ordered_modules], dim=1)
                .cpu()
                .numpy()
            )

            distances, indices = self.faiss_index.search(q, k, reverse=reverse)

            return torch.from_numpy(distances), torch.from_numpy(indices)

        if modules:
            modules = set([name for name in self.ordered_modules if name in modules])
        else:
            modules = set(self.ordered_modules)

        k = min(k or self.N, self.N)

        scores = torch.stack(
            [q[name] @ self.grads[name].mT for name in modules], dim=-1
        ).sum(-1)

        return torch.topk(scores, k, largest=not reverse)  # type: ignore

    @contextmanager
    def trace(
        self,
        module: nn.Module,
        k: int | None,
        *,
        modules: set[str] | None = None,
        reverse: bool = False,
    ) -> Generator[TraceResult, None, None]:
        """
        Context manager to trace the gradients of a module and return the
        corresponding Attributor instance.

        Args:
            module: The module to trace.
            k: The number of nearest examples to return.
            modules: The modules to trace. If None, all modules will be traced.
            reverse: If True, return the lowest influence examples instead of highest.
        """

        result = TraceResult()

        collector = TraceCollector(
            model=module,
            processor=self.processor,
            target_modules=modules,
            device=self.device,
            dtype=self.dtype,
        )
        mod_grads = collector.mod_grads
        with collector:
            yield result

        if not mod_grads:
            raise ValueError("No grads collected. Did you forget to call backward?")

        queries = {
            name: torch.cat(mod_grads[name], dim=1)
            for name in self.ordered_modules
            if name in mod_grads
        }

        if any(q.isnan().any() for q in queries.values()):
            raise ValueError("NaN found in queries.")

        result._scores, result._indices = self.search(queries, k, modules, reverse)
