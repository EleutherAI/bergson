"""ASTRA (Wang et al. 2025, https://arxiv.org/abs/2507.14740): refine the
EK-FAC inverse-Hessian-vector products with EK-FAC-preconditioned SGD."""

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
from datasets import Dataset
from torch import Tensor
from torch.func import functional_call

from bergson.config import AstraConfig, IndexConfig, InversionConfig
from bergson.data import column_offsets, load_gradients, pad_and_tensor
from bergson.gradients import LayerAdapter
from bergson.hessians.preconditioner import FactoredPreconditioner
from bergson.utils.logger import get_logger
from bergson.utils.utils import get_device, numpy_to_tensor
from bergson.utils.worker_utils import setup_data_pipeline, setup_model_and_peft


@dataclass
class AstraPaths:
    query_path: str
    """Query gradients ``q``."""

    init_path: str
    """EK-FAC solutions ``(P + D)^-1 q`` to start from."""

    hessian_path: str
    """EK-FAC factors of the preconditioner ``P``."""

    run_path: str
    """Where the refined solutions are written, in ``init_path``'s layout."""


class GaussNewtonProduct:
    """Gauss-Newton Hessian-vector products of the summed training loss, over
    the modules and in the ``[O, I]`` gradient layout of the query index."""

    def __init__(self, model, data: Dataset, index_cfg: IndexConfig, names: list[str]):
        if index_cfg.loss_fn != "ce":
            raise ValueError("ASTRA supports loss_fn='ce' only.")

        self.model = model
        self.data = data
        self.names = names
        self.mean_reduction = index_cfg.loss_reduction == "mean"
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
        params = dict(model.named_parameters())
        self.params = {
            k: params[k]
            for k in [*self.weight_names.values(), *self.bias_names.values()]
        }

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
        """``H_B v`` on the documents ``indices``, scaled by ``len(data) /
        len(indices)`` so it estimates the Hessian of the whole training set."""
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

        with fwAD.dual_level():
            duals = {k: fwAD.make_dual(p, tangents[k]) for k, p in self.params.items()}
            out = functional_call(self.model, duals, (x,)).logits[:, :-1]
            logits, jvp = fwAD.unpack_dual(out)
            assert jvp is not None
            jvp = jvp.detach()

        # Cross-entropy's Hessian in the logits, diag(p) - p p^T, per position.
        mask = (y[:, 1:] != -100).to(logits.dtype)
        weight = mask * len(self.data) / len(indices)
        if self.mean_reduction:
            weight = weight / mask.sum(1, keepdim=True).clamp_min(1)
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            hjvp = probs * (jvp - (probs * jvp).sum(-1, keepdim=True))
            hjvp *= weight[..., None]
        del jvp, probs

        grads = torch.autograd.grad(logits, list(self.params.values()), hjvp)
        grads = dict(zip(self.params, grads))
        return {name: self._to_flat(name, grads) for name in self.names}


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
        if inversion_cfg.inversion != "damped_inverse":
            raise ValueError("ASTRA needs inversion='damped_inverse'.")

        self.paths = paths
        self.cfg = astra_cfg
        self.logger = get_logger("Astra")
        self.device = device

        self.preconditioner = FactoredPreconditioner.from_path(
            paths.hessian_path,
            inversion_cfg=inversion_cfg,
            ev_correction=ev_correction,
            device=self.device,
        )
        self.names = list(self.preconditioner.lambdas)
        self.damping = {
            n: inversion_cfg.damping_factor * lam.mean()
            for n, lam in self.preconditioner.lambdas.items()
        }

        model, _ = setup_model_and_peft(index_cfg, attn_implementation="eager")
        model.eval()
        data, _ = setup_data_pipeline(index_cfg)
        self.hvp = GaussNewtonProduct(model, data, index_cfg, self.names)
        self.num_docs = len(data)

    def _read_row(self, mmap: np.memmap, offsets, row: int) -> dict[str, Tensor]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given NumPy array is not writable"
            )
            return {
                n: numpy_to_tensor(mmap[row, slice(*offsets[n])]).to(
                    self.device, torch.float32
                )
                for n in self.names
            }

    def refine(self, q: dict[str, Tensor], x: dict[str, Tensor], row: int):
        """Momentum SGD from ``x`` on ``x^T (H + D) x / 2 - x^T q``."""
        gen = torch.Generator().manual_seed(self.cfg.seed * 1_000_003 + row)
        buf = {n: torch.zeros_like(x[n]) for n in self.names}
        lr = self.cfg.lr
        for step in range(self.cfg.num_steps):
            if step > 0 and step % self.cfg.lr_decay_interval == 0:
                lr *= self.cfg.lr_decay
            indices = torch.randperm(self.num_docs, generator=gen)[
                : self.cfg.batch_size
            ].tolist()
            hx = self.hvp(x, indices)
            residual = {
                n: (hx[n] + self.damping[n] * x[n] - q[n])[None] for n in self.names
            }
            if step % 50 == 0 or step == self.cfg.num_steps - 1:
                objective = torch.stack(
                    [
                        (x[n] @ (hx[n] + self.damping[n] * x[n])) / 2 - x[n] @ q[n]
                        for n in self.names
                    ]
                ).sum()
                self.logger.info(
                    f"query {row} step {step}: objective {objective.item():.6g}"
                )
            direction = self.preconditioner.apply(residual)
            for n in self.names:
                buf[n].mul_(self.cfg.momentum).add_(direction[n][0])
                x[n].sub_(buf[n], alpha=lr)
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
            q = self._read_row(query, query_offsets, row)
            x = self._read_row(init, offsets, row)
            x = self.refine(q, x, row)
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
    factors, so no process group is needed."""
    device = get_device(local_rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    num_queries = load_gradients(paths.init_path).shape[0]
    per_rank = math.ceil(num_queries / world_size)
    rows = range(rank * per_rank, min((rank + 1) * per_rank, num_queries))
    Astra(paths, index_cfg, inversion_cfg, astra_cfg, ev_correction, device).run(rows)
