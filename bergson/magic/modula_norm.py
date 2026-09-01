"""Modular norm normalization for GPT-2 weights.

Uses the modula package to compute per-layer target norms from the recursive
modular norm formula, then normalizes weights via spectral normalization
(Linear layers) or row normalization (Embedding layers) after each optimizer step.

Reference: https://github.com/jxbz/modula
"""

from dataclasses import dataclass

import torch
from modula.abstract import CompositeModule, TupleModule
from modula.atom import Embedding as ModulaEmbedding
from modula.atom import Linear as ModulaLinear
from modula.compound import GPT as ModulaGPT


def _extract_atoms_and_norms(
    module, target_norm: float = 1.0
) -> list[tuple[ModulaLinear | ModulaEmbedding, float]]:
    """Walk the modula architecture tree and extract (atom, target_norm) pairs.

    The target norm for each atom is computed recursively from the architecture's
    mass and sensitivity values, following modula's normalization formula.
    """
    if isinstance(module, (ModulaLinear, ModulaEmbedding)):
        return [(module, target_norm)]

    if isinstance(module, CompositeModule):
        m0, m1 = module.children
        if module.mass > 0:
            t0 = m0.mass / module.mass * target_norm / m1.sensitivity
            t1 = m1.mass / module.mass * target_norm
            return _extract_atoms_and_norms(m0, t0) + _extract_atoms_and_norms(m1, t1)
        return []

    if isinstance(module, TupleModule):
        if module.mass > 0:
            result: list[tuple[ModulaLinear | ModulaEmbedding, float]] = []
            for child in module.children:
                t = child.mass / module.mass * target_norm
                result.extend(_extract_atoms_and_norms(child, t))
            return result
        return []

    return []


@dataclass
class _EmbeddingEntry:
    name: str
    target_norm: float


@dataclass
class _LinearEntry:
    name: str
    target_norm: float
    transpose: bool  # True for Conv1D params stored as (in, out)
    u: torch.Tensor  # Power iteration vector, shape (in_features,)


@dataclass
class _CAttnEntry:
    """Entry for the combined c_attn weight that holds Q, K, V projections."""

    name: str
    target_norms: list[float]  # [Q, K, V]
    u_vectors: list[torch.Tensor]  # Power iteration vectors for Q, K, V
    n_embd: int


class ModulaNormalizer:
    """Normalizes GPT-2 weights in the modular norm after each optimizer step.

    Builds a modula architecture matching the GPT-2 config, computes the
    per-layer target norms from the recursive modular norm formula, and
    normalizes weights via spectral normalization (Linear) or row normalization
    (Embedding) after each optimizer step.

    Supports both in-place (no-grad) normalization for training and
    differentiable normalization for the MAGIC backward pass.
    """

    def __init__(self, model_config, device: str | torch.device):
        n_embd = model_config.n_embd
        n_layer = model_config.n_layer
        n_head = model_config.n_head
        vocab_size = model_config.vocab_size
        n_positions = model_config.n_positions

        self.n_embd = n_embd
        self.n_layer = n_layer

        # Build modula GPT architecture and extract target norms
        gpt = ModulaGPT(
            vocab_size=vocab_size,
            context=n_positions,
            num_heads=n_head,
            d_embed=n_embd,
            d_query=n_embd // n_head,
            d_value=n_embd // n_head,
            num_blocks=n_layer,
        )

        # Initialize to create power iteration vectors on the atoms
        gpt.initialize(device)

        atoms_and_norms = _extract_atoms_and_norms(gpt, target_norm=1.0)
        expected = 2 + n_layer * 6 + 1
        assert (
            len(atoms_and_norms) == expected
        ), f"Expected {expected} atoms, got {len(atoms_and_norms)}"

        # Build entries mapping modula atoms to HF parameter names
        self._entries: list[_EmbeddingEntry | _LinearEntry | _CAttnEntry] = []
        idx = 0

        # Token embedding
        _, tn = atoms_and_norms[idx]
        idx += 1
        self._entries.append(_EmbeddingEntry("transformer.wte.weight", tn))

        # Position embedding
        _, tn = atoms_and_norms[idx]
        idx += 1
        self._entries.append(_EmbeddingEntry("transformer.wpe.weight", tn))

        # Transformer blocks
        for i in range(n_layer):
            # Q, K, V from combined c_attn
            qkv_norms = []
            qkv_us = []
            for _ in range(3):
                atom, tn = atoms_and_norms[idx]
                idx += 1
                qkv_norms.append(tn)
                assert isinstance(atom, ModulaLinear)
                qkv_us.append(atom.u.to(device))
            self._entries.append(
                _CAttnEntry(
                    f"transformer.h.{i}.attn.c_attn.weight",
                    qkv_norms,
                    qkv_us,
                    n_embd,
                )
            )

            # Attention output projection (Conv1D: stored as (in, out))
            atom, tn = atoms_and_norms[idx]
            idx += 1
            assert isinstance(atom, ModulaLinear)
            self._entries.append(
                _LinearEntry(
                    f"transformer.h.{i}.attn.c_proj.weight",
                    tn,
                    transpose=True,
                    u=atom.u.to(device),
                )
            )

            # MLP up projection (Conv1D)
            atom, tn = atoms_and_norms[idx]
            idx += 1
            assert isinstance(atom, ModulaLinear)
            self._entries.append(
                _LinearEntry(
                    f"transformer.h.{i}.mlp.c_fc.weight",
                    tn,
                    transpose=True,
                    u=atom.u.to(device),
                )
            )

            # MLP down projection (Conv1D)
            atom, tn = atoms_and_norms[idx]
            idx += 1
            assert isinstance(atom, ModulaLinear)
            self._entries.append(
                _LinearEntry(
                    f"transformer.h.{i}.mlp.c_proj.weight",
                    tn,
                    transpose=True,
                    u=atom.u.to(device),
                )
            )

        # lm_head: skip — in GPT-2 it is tied to transformer.wte.weight,
        # which is already normalized as an Embedding above.  Keeping the
        # architecture identical between baseline and modula runs.
        idx += 1  # consume the atom but don't create an entry

    @torch.no_grad()
    def warmup(self, params: dict[str, torch.Tensor], n_steps: int = 10):
        """Run power iteration steps to converge u vectors without scaling weights.

        Should be called once on the initial weights before the first normalize().
        """
        for entry in self._entries:
            if entry.name not in params:
                continue

            weight = params[entry.name]

            if isinstance(entry, _LinearEntry):
                wt = weight.t() if entry.transpose else weight
                for _ in range(n_steps):
                    _power_iter_step(wt, entry.u)

            elif isinstance(entry, _CAttnEntry):
                d = entry.n_embd
                for j in range(3):
                    part = weight[:, j * d : (j + 1) * d]
                    wt = part.t()
                    for _ in range(n_steps):
                        _power_iter_step(wt, entry.u_vectors[j])

    def normalize(
        self, params: dict[str, torch.Tensor], trace: bool = False
    ) -> dict[str, torch.Tensor]:
        """Normalize weights in the modular norm.

        Args:
            params: Dict mapping parameter names to tensors.
            trace: If True, returns new dict with differentiable normalized tensors.
                   If False, normalizes in-place and returns the same dict.
        """
        if trace:
            return self._normalize_traced(params)

        self._normalize_inplace(params)
        return params

    @torch.no_grad()
    def _normalize_inplace(self, params: dict[str, torch.Tensor]):
        for entry in self._entries:
            if entry.name not in params:
                continue

            weight = params[entry.name]

            if isinstance(entry, _EmbeddingEntry):
                norms = weight.norm(dim=1, keepdim=True).clamp_(min=1e-8)
                weight.mul_(entry.target_norm / norms)

            elif isinstance(entry, _LinearEntry):
                wt = weight.t() if entry.transpose else weight
                sigma = _power_iter_step(wt, entry.u)
                weight.mul_(entry.target_norm / sigma)

            elif isinstance(entry, _CAttnEntry):
                d = entry.n_embd
                for j in range(3):
                    part = weight[:, j * d : (j + 1) * d]
                    wt = part.t()
                    sigma = _power_iter_step(wt, entry.u_vectors[j])
                    part.mul_(entry.target_norms[j] / sigma)

    def _normalize_traced(
        self, params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_params = dict(params)

        for entry in self._entries:
            if entry.name not in params:
                continue

            weight = params[entry.name]

            if isinstance(entry, _EmbeddingEntry):
                norms = weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
                new_params[entry.name] = weight * (entry.target_norm / norms)

            elif isinstance(entry, _LinearEntry):
                wt = weight.t() if entry.transpose else weight
                sigma = _spectral_norm_diff(wt, entry.u.detach())
                new_params[entry.name] = weight * (entry.target_norm / sigma)

            elif isinstance(entry, _CAttnEntry):
                d = entry.n_embd
                parts = []
                for j in range(3):
                    part = weight[:, j * d : (j + 1) * d]
                    wt = part.t()
                    sigma = _spectral_norm_diff(wt, entry.u_vectors[j].detach())
                    parts.append(part * (entry.target_norms[j] / sigma))
                new_params[entry.name] = torch.cat(parts, dim=1)

        return new_params


@torch.no_grad()
def _power_iter_step(weight: torch.Tensor, u: torch.Tensor) -> float:
    """One step of power iteration. Updates u in-place. Returns approx spectral norm."""
    v = torch.mv(weight, u)
    v /= v.norm()
    torch.mv(weight.t(), v, out=u)
    return u.norm().item()


def _spectral_norm_diff(weight: torch.Tensor, u_detached: torch.Tensor) -> torch.Tensor:
    """Differentiable spectral norm using a detached power iteration vector."""
    v = torch.mv(weight, u_detached)
    v = v / v.norm().clamp(min=1e-8)
    u = torch.mv(weight.t(), v)
    return u.norm().clamp(min=1e-8)
