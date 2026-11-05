"""ASTRA (Wang et al. 2025, https://arxiv.org/abs/2507.14740): refine a
Kronecker-factored approximate Hessian's inverse-Hessian-vector products with
SGD preconditioned by that same inverse."""

import json
import math
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from torch import Tensor

from bergson.config import AstraConfig, IndexConfig, InversionConfig
from bergson.data import column_offsets, load_gradients, pad_and_tensor
from bergson.distributed import init_dist
from bergson.gradients import LayerAdapter
from bergson.hessians.preconditioner import (
    FactoredPreconditioner,
    factor_shapes,
    load_eigenvalues,
)
from bergson.utils.logger import get_logger
from bergson.utils.utils import get_device, numpy_to_tensor
from bergson.utils.worker_utils import setup_data_pipeline, setup_model_and_peft


@dataclass
class AstraPaths:
    query_path: str
    """Query gradients ``q``."""

    init_path: str
    """Solutions ``(P + D)^-1 q`` to start from."""

    hessian_path: str
    """Kronecker factors of the preconditioner ``P``."""

    run_path: str
    """Where the refined solutions are written, in ``init_path``'s layout."""


class GaussNewtonProduct:
    """Gauss-Newton Hessian-vector products of the training loss, over
    the modules and in the ``[O, I]`` gradient layout of the query index."""

    def __init__(
        self,
        model,
        data: Dataset,
        index_cfg: IndexConfig,
        names: list[str],
        loss_reduction: Literal["mean", "sum"] | None = None,
        micro_batch_size: int = 16,
    ):
        if index_cfg.loss_fn != "ce":
            raise ValueError("ASTRA supports loss_fn='ce' only.")

        self.model = model
        self.data = data
        self.names = names
        self.mean_reduction = (loss_reduction or index_cfg.loss_reduction) == "mean"
        self.micro_batch_size = micro_batch_size
        self.device = next(model.parameters()).device

        param_names = {id(p): n for n, p in model.named_parameters()}
        self.layers = {n: model.base_model.get_submodule(n) for n in names}
        self.weight_names = {
            n: param_names[id(layer.weight)] for n, layer in self.layers.items()
        }
        self.bias_names = {
            n: param_names[id(layer.bias)]
            for n, layer in self.layers.items()
            if index_cfg.include_bias and layer.bias is not None
        }
        for layer in self.layers.values():
            layer.requires_grad_(True)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def _to_params(self, name: str, v: Tensor) -> dict[str, Tensor]:
        """Split a flat ``[O, I]`` (``[O, I + 1]`` with bias) vector into the
        layer's parameter shapes."""
        layer = self.layers[name]
        weight = layer.weight
        o = getattr(layer, LayerAdapter.out_attr(layer))
        v = v.view(o, -1)
        out = {}
        if name in self.bias_names:
            out[self.bias_names[name]] = v[:, -1]
            v = v[:, :-1]
        if LayerAdapter.weight_transposed(layer):
            v = v.T
        out[self.weight_names[name]] = v.reshape(weight.shape)
        return out

    def _to_flat(self, name: str, grads: dict[str, Tensor]) -> Tensor:
        """Inverse of :meth:`_to_params`."""
        layer = self.layers[name]
        w = grads[self.weight_names[name]]
        if LayerAdapter.weight_transposed(layer):
            w = w.T
        w = w.reshape(w.shape[0], -1)
        if name in self.bias_names:
            w = torch.cat([w, grads[self.bias_names[name]][:, None]], dim=1)
        return w.flatten()

    def __call__(self, v: dict[str, Tensor], indices: list[int]) -> dict[str, Tensor]:
        """``H_B v`` on the documents ``indices``, a mean over documents like the
        fitted Hessians. Each rank of a process group takes every
        ``world_size``-th document, and the ranks sum their shares."""
        out = {name: torch.zeros_like(v[name]) for name in self.names}
        share = indices[self.rank :: self.world_size]
        # Every rank runs as many micro-batches as the largest share needs, as
        # FSDP's forwards and backwards gather parameters from every rank
        largest = math.ceil(len(indices) / self.world_size)
        for start in range(0, largest, self.micro_batch_size):
            chunk = share[start : start + self.micro_batch_size]
            scale = 1.0 if chunk else 0.0
            chunk = chunk or indices[:1]
            for name, h in self._product(v, chunk, len(indices), scale).items():
                out[name] += h.to(out[name].device)
        if self.world_size > 1:
            for h in out.values():
                dist.all_reduce(h)
        return out

    def _product(
        self, v: dict[str, Tensor], indices: list[int], batch_size: int, scale: float
    ) -> dict[str, Tensor]:
        """This chunk's share of the product over a batch of ``batch_size``,
        times ``scale``."""
        batch = self.data[indices]
        x, y, _, _ = pad_and_tensor(
            batch["input_ids"],
            labels=batch.get("labels"),
            device=self.device,
            sync_max_len=False,
        )
        tangents: dict[str, Tensor] = {}
        for name in self.names:
            tangents.update(self._to_params(name, v[name]))

        # The JVP's forward records no graph, which would also hold the
        # tangents' activations (and FSDP's hooks take no tangents in a forward
        # that records one); a second forward records the graph for the backward.
        with fwAD.dual_level(), torch.no_grad(), self._forward_params(tangents):
            jvp = fwAD.unpack_dual(self.model(x).logits[:, :-1]).tangent
            assert jvp is not None
        with self._forward_params() as params:
            logits = self.model(x).logits[:, :-1]

        # Cross-entropy's Hessian in the logits, diag(p) - p p^T, per position.
        mask = (y[:, 1:] != -100).to(logits.device, logits.dtype)
        weight = mask / batch_size * scale
        if self.mean_reduction:
            weight = weight / mask.sum(1, keepdim=True).clamp_min(1)
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            hjvp = probs * (jvp - (probs * jvp).sum(-1, keepdim=True))
            hjvp *= weight[..., None]
        del jvp, probs

        grads = torch.autograd.grad(logits, list(params.values()), hjvp)
        grads = dict(zip(params, grads))
        return {name: self._to_flat(name, grads) for name in self.names}

    @contextmanager
    def _forward_params(self, tangents: dict[str, Tensor] | None = None):
        """The layers' parameters as their forwards run with them, FSDP's
        gathered ones under FSDP, made dual tensors with ``tangents`` if given."""
        params: dict[str, Tensor] = {}
        attrs = {
            n: [("weight", self.weight_names[n])]
            + ([("bias", self.bias_names[n])] if n in self.bias_names else [])
            for n in self.names
        }

        def swap_in(name):
            def hook(layer, args):
                for attr, key in attrs[name]:
                    p = originals[name, attr] = layer._parameters[attr]
                    if tangents is not None:
                        p = fwAD.make_dual(p, tangents[key].to(p.device, p.dtype))
                    layer._parameters[attr] = params[key] = p

            return hook

        def swap_out(name):
            def hook(layer, args, output):
                for attr, _ in attrs[name]:
                    layer._parameters[attr] = originals.pop((name, attr))

            return hook

        originals: dict[tuple[str, str], Tensor] = {}
        handles = []
        for n, layer in self.layers.items():
            handles.append(layer.register_forward_pre_hook(swap_in(n)))
            handles.append(layer.register_forward_hook(swap_out(n)))
        try:
            yield params
        finally:
            for handle in handles:
                handle.remove()


class Astra:
    """Refines a subset of the queries' inverse-Hessian-vector products."""

    def __init__(
        self,
        paths: AstraPaths,
        index_cfg: IndexConfig,
        inversion_cfg: InversionConfig,
        astra_cfg: AstraConfig,
        ev_correction: bool,
        device: str,
    ):
        if inversion_cfg.inversion not in ("damped_inverse", "absolute_damped_inverse"):
            raise ValueError(
                "ASTRA needs inversion='damped_inverse' or 'absolute_damped_inverse'."
            )

        self.paths = paths
        self.cfg = astra_cfg
        self.logger = get_logger("Astra")
        self.device = device
        self.store_device = "cpu" if astra_cfg.factors_on_cpu else device

        # Ranks that refine every query together each hold and apply the
        # factors of a share of the modules, balanced by the factors' sizes
        self.shared = dist.is_initialized() and dist.get_world_size() > 1
        lambda_dir = (
            "eigenvalue_correction_sharded" if ev_correction else "eigenvalue_sharded"
        )
        shapes = factor_shapes(paths.hessian_path, lambda_dir)
        self.names = list(shapes)
        world_size = dist.get_world_size() if self.shared else 1
        loads = [0] * world_size
        self.owner = {}
        for n in sorted(self.names, key=lambda n: -sum(d * d for d in shapes[n])):
            self.owner[n] = loads.index(min(loads))
            loads[self.owner[n]] += sum(d * d for d in shapes[n])
        rank = dist.get_rank() if self.shared else 0
        self.owned = [n for n in self.names if self.owner[n] == rank]

        # Under device_map_auto each module's factors go to its layer's GPU
        self.preconditioner = FactoredPreconditioner.from_path(
            paths.hessian_path,
            inversion_cfg=inversion_cfg,
            ev_correction=ev_correction,
            device="cpu" if astra_cfg.device_map_auto else self.store_device,
            compute_device=self.device if astra_cfg.factors_on_cpu else None,
            modules=self.owned,
        )
        if inversion_cfg.inversion == "absolute_damped_inverse":
            self.damping = {n: inversion_cfg.damping_factor for n in self.names}
        else:
            lambdas = load_eigenvalues(paths.hessian_path, ev_correction)
            self.damping = {
                n: inversion_cfg.damping_factor * lam.to(device).mean()
                for n, lam in lambdas.items()
            }

        model_kwargs = {}
        if astra_cfg.device_map_auto and torch.cuda.device_count() > 1:
            model_dir = Path(index_cfg.model)
            if (model_dir / "adapter_config.json").exists():
                # A PEFT adapter: the base model holds nearly all the weights
                with open(model_dir / "adapter_config.json") as f:
                    model_dir = Path(json.load(f)["base_model_name_or_path"])
            weights = sorted(model_dir.glob("*.safetensors"))
            if weights:
                # "auto" fills GPUs in order; an even share per GPU spreads the
                # layers, and so their activations, over all of them.
                share = sum(f.stat().st_size for f in weights)
                share /= torch.cuda.device_count()
                model_kwargs["max_memory"] = {
                    i: int(1.2 * share) for i in range(torch.cuda.device_count())
                }
        model, _ = setup_model_and_peft(
            index_cfg,
            device_map_auto=astra_cfg.device_map_auto,
            attn_implementation="eager",
            **model_kwargs,
        )
        model.eval()
        data, _ = setup_data_pipeline(index_cfg)
        self.hvp = GaussNewtonProduct(
            model,
            data,
            index_cfg,
            self.names,
            astra_cfg.loss_reduction,
            astra_cfg.micro_batch_size,
        )
        if astra_cfg.device_map_auto and not astra_cfg.factors_on_cpu:
            self.preconditioner.to_devices(
                {n: self.hvp.layers[n].weight.device for n in self.owned}
            )
        self.num_docs = len(data)

    def _read_row(
        self, mmap: np.memmap, offsets, row: int, device: str | None = None
    ) -> dict[str, Tensor]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given NumPy array is not writable"
            )
            return {
                n: numpy_to_tensor(mmap[row, slice(*offsets[n])]).to(
                    device or self.device, torch.float32
                )
                for n in self.names
            }

    def refine(self, q: dict[str, Tensor], x: dict[str, Tensor], row: int):
        """Minimize ``x^T (H + D) x / 2 - x^T q`` from ``x``."""
        gen = torch.Generator().manual_seed(self.cfg.seed * 1_000_003 + row)
        if self.cfg.solver == "cg":
            indices = torch.randperm(self.num_docs, generator=gen)[
                : self.cfg.batch_size
            ].tolist()
            return self._cg(q, x, row, indices)
        q = {n: q[n].to(self.store_device) for n in self.names}
        buf = {n: torch.zeros_like(q[n]) for n in self.names}
        lr = self.cfg.lr
        for step in range(self.cfg.num_steps):
            if step > 0 and step % self.cfg.lr_decay_interval == 0:
                lr *= self.cfg.lr_decay
            indices = torch.randperm(self.num_docs, generator=gen)[
                : self.cfg.batch_size
            ].tolist()
            hx = self.hvp(x, indices)
            objective = torch.stack(
                [
                    (x[n] @ (hx[n] + self.damping[n] * x[n])) / 2
                    - x[n] @ q[n].to(self.device)
                    for n in self.names
                ]
            ).sum()
            self.logger.info(
                f"query {row} step {step}: objective {objective.item():.6g}"
            )
            residuals = (
                hx.pop(n) + self.damping[n] * x[n] - q[n].to(self.device)
                for n in self.names
            )
            for n, direction in zip(self.names, self._directions(residuals)):
                velocity = buf[n].to(self.device).mul_(self.cfg.momentum)
                velocity.add_(direction)
                x[n].sub_(velocity, alpha=lr)
                buf[n] = velocity.to(self.store_device)
        return x

    def _directions(self, residuals):
        """The preconditioned residuals, in the order of ``self.names``, from
        ``residuals`` in that order. Shared ranks each precondition their own
        modules, then each module's owner sends its direction to the rest."""
        if not self.shared:
            for n, r in zip(self.names, residuals):
                yield self.preconditioner.apply({n: r[None]})[n][0].to(r.device)
            return
        mine, shapes = {}, {}
        for n, r in zip(self.names, residuals):
            shapes[n] = r.shape
            if self.owner[n] == dist.get_rank():
                mine[n] = self.preconditioner.apply({n: r[None]})[n][0].to(r.device)
            del r
        for n in self.names:
            d = mine.pop(n, None)
            if d is None:
                d = torch.empty(shapes[n], device=self.device)
            dist.broadcast(d, src=self.owner[n])
            yield d

    def _damped_product(
        self, v: dict[str, Tensor], indices: list[int]
    ) -> dict[str, Tensor]:
        """``(H_B + D) v`` on the documents ``indices``."""
        hv = self.hvp(v, indices)
        return {n: hv[n] + self.damping[n] * v[n] for n in self.names}

    def _precondition(self, r: dict[str, Tensor]) -> dict[str, Tensor]:
        directions = self._directions(r[n] for n in self.names)
        return dict(zip(self.names, directions))

    def _dot(self, u: dict[str, Tensor], v: dict[str, Tensor]) -> float:
        return sum((u[n].double() @ v[n].double()).item() for n in self.names)

    def _cg(
        self, q: dict[str, Tensor], x: dict[str, Tensor], row: int, indices: list[int]
    ):
        """Preconditioned conjugate gradients on the documents ``indices``."""
        ax = self._damped_product(x, indices)
        r = {n: q[n] - ax[n] for n in self.names}
        z = self._precondition(r)
        p = {n: z[n].clone() for n in self.names}
        rz = self._dot(r, z)
        for step in range(self.cfg.num_steps):
            # With r = q - (H_B + D) x, the objective is -(q + r)^T x / 2.
            objective = -(self._dot(q, x) + self._dot(r, x)) / 2
            self.logger.info(f"query {row} step {step}: objective {objective:.6g}")
            ap = self._damped_product(p, indices)
            pap = self._dot(p, ap)
            if pap <= 0:
                self.logger.warning(f"query {row} step {step}: curvature {pap:.3g}")
                break
            alpha = rz / pap
            for n in self.names:
                x[n].add_(p[n], alpha=alpha)
                r[n].sub_(ap[n], alpha=alpha)
            z = self._precondition(r)
            rz, rz_old = self._dot(r, z), rz
            for n in self.names:
                p[n] = z[n] + (rz / rz_old) * p[n]
        return x

    def run(self, rows: range):
        query = load_gradients(self.paths.query_path)
        init = load_gradients(self.paths.init_path)
        with open(Path(self.paths.query_path) / "info.json") as f:
            query_offsets = column_offsets(json.load(f)["grad_sizes"])
        with open(Path(self.paths.init_path) / "info.json") as f:
            offsets = column_offsets(json.load(f)["grad_sizes"])
        out = np.memmap(
            Path(self.paths.run_path) / "gradients.bin",
            dtype=init.dtype,
            mode="r+",
            shape=init.shape,
        )
        for row in rows:
            # SGD reads the query where it keeps it, so no copy stays on the GPU
            q_device = self.store_device if self.cfg.solver == "sgd" else self.device
            q = self._read_row(query, query_offsets, row, q_device)
            x = self._read_row(init, offsets, row)
            x = self.refine(q, x, row)
            # Under FSDP every rank refines every row to the same solution
            if self.hvp.rank == 0:
                for n in self.names:
                    out[row, slice(*offsets[n])] = x[n].cpu().numpy()
                out.flush()


def astra_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    paths: AstraPaths,
    index_cfg: IndexConfig,
    inversion_cfg: InversionConfig,
    astra_cfg: AstraConfig,
    ev_correction: bool,
):
    """Refine this rank's share of the queries; each rank holds the full
    factors. Under FSDP or ``share_documents`` the ranks refine every query
    together, each taking a share of every batch's documents and holding the
    factors of a share of the modules."""
    # Under device_map_auto a rank's model spans its own GPUs, from cuda:0
    gpu = 0 if astra_cfg.device_map_auto else local_rank
    device = get_device(gpu)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    num_queries = load_gradients(paths.init_path).shape[0]
    if index_cfg.fsdp or astra_cfg.share_documents:
        init_dist(rank, gpu, world_size)
        rows = range(num_queries)
    else:
        per_rank = math.ceil(num_queries / world_size)
        rows = range(rank * per_rank, min((rank + 1) * per_rank, num_queries))
    Astra(paths, index_cfg, inversion_cfg, astra_cfg, ev_correction, device).run(rows)
