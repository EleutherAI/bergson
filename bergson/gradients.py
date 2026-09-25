from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Mapping

import torch
import torch.nn as nn
import yaml
from torch import Tensor
from transformers.pytorch_utils import Conv1D as HFConv1D

from bergson.moe import ExpertLinear
from bergson.utils.logger import get_logger

logger = get_logger("gradients", level="INFO")

NORMALIZER_TYPES: dict[str, type["Normalizer"]] = {}


class Normalizer(ABC):
    """
    Base class for normalizers that can be used to scale gradients.
    """

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses in the NORMALIZER_TYPES dict."""
        super().__init_subclass__(**kwargs)
        NORMALIZER_TYPES[cls.__name__] = cls

    @staticmethod
    def from_state_dict(state_dict: dict[str, str | Tensor]) -> "Normalizer":
        """
        Create a normalizer instance from a state dictionary.
        The state dictionary should contain the class name and the tensors.
        """
        class_name = state_dict.pop("__class__")
        assert isinstance(class_name, str), "Expected '__class__' to be a string"

        if (cls := NORMALIZER_TYPES.get(class_name)) is None:
            raise ValueError(f"Unknown normalizer class: '{class_name}'")

        # Migration: avg_sq was renamed to weight_avg_sq
        if "avg_sq" in state_dict:
            state_dict["weight_avg_sq"] = state_dict.pop("avg_sq")

        return cls(**state_dict)

    @abstractmethod
    def normalize_weight(
        self,
        grad: Tensor,
        eps: float = 1e-8,
    ) -> Tensor:
        """
        Normalize weight gradients in-place.
        Adds a small epsilon to avoid division by zero.
        """

    @abstractmethod
    def normalize_bias(
        self,
        grad: Tensor,
        eps: float = 1e-8,
    ) -> Tensor:
        """
        Normalize bias gradients in-place.
        Adds a small epsilon to avoid division by zero.
        """

    def state_dict(self) -> dict[str, str | Tensor]:
        """
        Return the state of the normalizer as a dictionary of tensors.
        This is used for saving and loading the normalizer.
        """
        tensors = {k: v for k, v in self.__dict__.items() if isinstance(v, Tensor)}
        return {
            "__class__": self.__class__.__name__,
            **tensors,
        }


PROJECTION_SETTINGS = (
    "projection_dim",
    "projection_type",
    "projection_scale",
    "projection_seed",
    "projection_target",
    "include_bias",
)
"""Settings that must match between gradients projected separately.
``include_bias`` widens the right projection matrix of modules with a bias,
which changes all of its entries."""


@dataclass
class GradientProcessor:
    """Configuration for processing and compressing gradients."""

    normalizers: Mapping[str, Normalizer] = field(default_factory=dict)
    """
    Dictionary of normalizers for each matrix-valued parameter in the model. The keys
    should match the names of the parameters in the model. If a parameter does not have
    a normalizer, it will be skipped.
    """

    hessians: dict[str, Tensor] = field(default_factory=dict)
    """
    Dictionary of hessians for each matrix-valued parameter in the model.
    These are applied after the normalization and random projection steps.
    """

    hessians_eigen: Mapping[str, tuple[Tensor, Tensor]] = field(default_factory=dict)
    """
    Dictionary of eigen decompositions of hessians for each matrix-valued
    parameter in the model. Each value is a tuple of (eigenvalues, eigenvectors).
    These are used to efficiently apply inverse square-root of the hessians
    to the gradients."""

    projection_dim: int | None = None
    """Number of rows and columns to project the gradients to. If `None`, keep the
    original shape of the gradients."""

    reshape_to_square: bool = False
    """Whether to reshape the gradients into a nearly square matrix before projection.
    This is useful when the matrix-valued parameters are far from square, like in the
    case of LoRA adapters."""

    projection_type: Literal["normal", "rademacher"] = "rademacher"
    """
    Type of random projection to use for compressing gradients. Can be either "normal"
    for Gaussian projections or "rademacher" for Rademacher projections, which use a
    uniform distribution over {-1, 1}.
    """

    projection_target: Literal["per_module", "global"] = "per_module"

    projection_scale: Literal["jl", "row_norm"] = "jl"
    """Scaling of the random projection entries. See ``IndexConfig``."""
    """
    Projection target. ``per_module`` does a double-sided random projection of each
    module's gradient independently. ``global`` does an independent
    single-sided right projection of each module's flattened gradient then sums the
    results, producing one ``[proj_dim]`` vector per example.
    """

    include_bias: bool = False
    """Whether to include bias gradients when present on a module."""

    projection_seed: int | None = None
    """Seed of the random projection."""

    def __post_init__(self):
        # Configs use 0 for no projection.
        if self.projection_dim == 0:
            self.projection_dim = None
        if self.projection_dim is None:
            self.projection_target = "per_module"
        self._projection_matrices: dict[
            tuple[str, Literal["left", "right", "single"], torch.device], Tensor
        ] = {}

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        map_location: str | torch.device | None = None,
        skip_hessians: bool = False,
    ) -> "GradientProcessor":
        """
        Load the normalizers and hessians from a file.
        """
        path = Path(path)
        norm_path = path / "normalizers.pth"

        # Fall back to legacy "preconditioners*.pth" filenames if the new
        # "hessians*.pth" files don't exist on disk.
        # TODO Lucia Quirke: remove on the 28 October 2026
        hess_path = path / "hessians.pth"
        if not hess_path.exists():
            hess_path = path / "preconditioners.pth"
        hess_eigen_path = path / "hessians_eigen.pth"
        if not hess_eigen_path.exists():
            hess_eigen_path = path / "preconditioners_eigen.pth"

        cfg = cls._read_config(path)

        # Load normalizers
        norm_state = torch.load(
            norm_path,
            map_location=map_location,
            weights_only=True,
        )
        normalizers = {
            name: Normalizer.from_state_dict(state)
            for name, state in norm_state.items()
        }

        hessians, hessians_eigen = {}, {}
        if not skip_hessians:
            hessians = torch.load(
                hess_path,
                map_location=map_location,
                weights_only=True,
            )
            hessians_eigen = torch.load(
                hess_eigen_path,
                map_location=map_location,
                weights_only=True,
            )

        return cls(
            normalizers=normalizers,
            hessians=hessians,
            hessians_eigen=hessians_eigen,
            **cfg,
        )

    @classmethod
    def load_config(cls, path: Path | str) -> "GradientProcessor":
        """Load the processor saved at ``path`` without its normalizers or
        hessians."""
        return cls(**cls._read_config(Path(path)))

    @staticmethod
    def _read_config(path: Path) -> dict:
        with (path / "processor_config.yaml").open("r") as f:
            cfg = yaml.safe_load(f)

        # Backward compatibility
        if "projection_type" not in cfg:
            cfg["projection_type"] = "normal"
        if "include_bias" not in cfg:
            cfg["include_bias"] = False
        if "projection_scale" not in cfg:
            cfg["projection_scale"] = "row_norm"
        # Defensive: rename any legacy preconditioner* keys that may appear in
        # configs saved by older versions of this code.
        for legacy_key in list(cfg.keys()):
            if "preconditioner" in legacy_key or "precond" in legacy_key:
                new_key = (
                    legacy_key.replace("preconditioners", "hessians")
                    .replace("preconditioner", "hessian")
                    .replace("precond", "hess")
                )
                cfg[new_key] = cfg.pop(legacy_key)
        return cfg

    def check_projection_matches(self, other: "GradientProcessor", what: str) -> None:
        """Raise if ``other``, the processor ``what`` was built with, projected
        gradients differently from this one."""
        # Without a projection, the other settings don't do anything.
        names = PROJECTION_SETTINGS if self.projection_dim else ("projection_dim",)
        differences = [
            (name, getattr(other, name), getattr(self, name))
            for name in names
            if getattr(other, name) != getattr(self, name)
        ]
        if differences:
            listed = "; ".join(
                f"{name}={theirs!r}, not {ours!r}" for name, theirs, ours in differences
            )
            raise ValueError(
                f"{what} was projected with different settings than this run: "
                f"{listed}. Rebuild it with matching settings."
            )

    def check_saved_projection(self, path: Path | str, what: str) -> None:
        """``check_projection_matches`` against the processor saved with the
        gradients or hessians at ``path``."""
        if not (Path(path) / "processor_config.yaml").exists():
            logger.warning(
                f"{what} at {path} has no processor_config.yaml, so its projection "
                "settings can't be checked."
            )
            return
        self.check_projection_matches(
            GradientProcessor.load_config(path), f"{what} at {path}"
        )

    def save(self, path: Path):
        """
        Save the normalizers and hessians to a file.
        """
        path.mkdir(parents=True, exist_ok=True)

        cfg_path = path / "processor_config.yaml"
        norm_path = path / "normalizers.pth"
        hess_path = path / "hessians.pth"
        hess_eigen_path = path / "hessians_eigen.pth"

        # Save configuration separately
        cfg = asdict(self)
        del cfg["normalizers"]
        del cfg["hessians"]
        del cfg["hessians_eigen"]
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        # Save normalizers
        norm_state = {
            name: normalizer.state_dict()
            for name, normalizer in self.normalizers.items()
        }
        torch.save(norm_state, norm_path)
        torch.save(self.hessians, hess_path)
        torch.save(self.hessians_eigen, hess_eigen_path)


class LayerAdapter:
    supported_modules = (
        nn.Linear,
        HFConv1D,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        ExpertLinear,
    )

    @staticmethod
    def in_attr(layer: nn.Module) -> str:
        match layer:
            case nn.Linear() | ExpertLinear():
                return "in_features"
            case HFConv1D():
                return "nx"
            case nn.Conv1d() | nn.Conv2d() | nn.Conv3d():
                return "in_channels"
            case _:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

    @staticmethod
    def out_attr(layer: nn.Module) -> str:
        match layer:
            case nn.Linear() | ExpertLinear():
                return "out_features"
            case HFConv1D():
                return "nf"
            case nn.Conv1d() | nn.Conv2d() | nn.Conv3d():
                return "out_channels"
            case _:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

    @staticmethod
    def weight_transposed(layer: nn.Module) -> bool:
        """Whether the layer stores its weight ``[in, out]`` (HF Conv1D, and
        MoE experts under ``is_transposed``) rather than ``[out, in]``."""
        if isinstance(layer, ExpertLinear):
            return layer.transposed
        return isinstance(layer, HFConv1D)


@dataclass
class AdafactorNormalizer(Normalizer):
    """
    Row and column sums of second moments of gradients for a matrix-valued parameter.
    Weight normalization mutates gradient values in-place.

    Args:
        row: Row statistics [O]
        col: Column statistics [I]
        bias_avg_sq: Optional second moments for bias [O]
    """

    row: Tensor  # shape [O]
    col: Tensor  # shape [I]
    bias_avg_sq: Tensor | None = None  # shape [O]

    def __post_init__(self):
        assert self.row.ndim == 1, f"Expected 1D tensor for row, got {self.row.ndim}D"
        assert self.col.ndim == 1, f"Expected 1D tensor for col, got {self.col.ndim}D"
        if self.bias_avg_sq is not None:
            assert (
                self.bias_avg_sq.ndim == 1
            ), f"Expected 1D tensor for bias_avg_sq, got {self.bias_avg_sq.ndim}D"

    def normalize_weight(
        self,
        grad: Tensor,
        eps: float = 1e-30,
    ) -> Tensor:
        """
        Normalize the row and column sums by adding a small epsilon.

        Note: Our `eps` corresponds to epsilon_1 in the original Adafactor paper. They
        recommend 1e-30, but we use 1e-16 for extra numerical stability.
        """
        # We follow the Adafactor implementation in the tensor2tensor repo, which is
        # different from the paper and from the PyTorch implementation. First add eps
        # to ensure these second moments are sufficiently far from zero. Then we don't
        # need to worry about numerical stability anywhere else, and we don't need to
        # materialize the outer product at any point.
        r, c = self.row.add(eps), self.col.add(eps)

        # This is the denominator for V, the rank-one matrix of second moment estimates:
        # V = torch.outer(r, c) / denom
        # V_ij = r_i * c_j / denom
        # But we want to (implicitly) take the Hadamard product with the elementwise
        # reciprocal square root of V:
        # (V_ij)^{-1/2} = denom.sqrt() * r_i.rsqrt() * c_j.rsqrt()
        denom = r.mean()

        # Hadamard product with a rank-one matrix ab^T is the same as left-multiplying
        # by diag(a) and right-multiplying by diag(b). In this case we can represent
        # the elementwise reciprocal square root of V as ab^T where:
        # a = denom.sqrt() * r.rsqrt() and b = c.rsqrt()
        a = denom.sqrt() * r.rsqrt_()  # shape [O]
        b = c.rsqrt_()

        # Implicitly do the Hadamard product
        grad *= a[:, None]  # [O, I] * [O, 1] → [O, I]
        grad *= b[None, :]  # [O, I] * [1, I] → [O, I]

        return grad

    def normalize_bias(
        self,
        grad: Tensor,
        eps: float = 1e-8,
    ) -> Tensor:
        """Normalize the gradients by the square root of the second moments."""
        assert self.bias_avg_sq is not None

        # Adafactor-style epsilon is added inside the square root.
        # Differs slightly from the PyTorch implementation which uses clamp.
        return grad * self.bias_avg_sq.add(eps).rsqrt_()

    def to_adam(self) -> "AdamNormalizer":
        """
        Convert this Adafactor normalizer to an Adam normalizer by materializing the
        rank-one second moment matrix.

        Preserves bias_avg_sq if present.
        """
        # Compute the second moment matrix as a square matrix of shape [O, I]
        # NOTE: We don't add the epsilon here, since the AdamNormalizer is going to
        # add it outside the square root. This could cause infs though if there are
        # any exactly zero rows or columns, so we should be careful.
        weight_avg_sq = torch.outer(self.row, self.col) / self.row.mean()
        return AdamNormalizer(weight_avg_sq=weight_avg_sq, bias_avg_sq=self.bias_avg_sq)


@dataclass
class AdamNormalizer(Normalizer):
    """
    Contains the second moments of the gradients. Weight normalization mutates gradient
    values in-place.

    Args:
        weight_avg_sq: Second moments for weights [O, I]
        bias_avg_sq: Optional second moments for bias [O]
    """

    weight_avg_sq: Tensor
    bias_avg_sq: Tensor | None = None

    def normalize_weight(
        self,
        grad: Tensor,
        eps: float = 1e-8,
    ) -> Tensor:
        """Normalize the gradients by the square root of the second moments."""
        return grad.div_(self.weight_denominator(eps))

    def weight_denominator(self, eps: float = 1e-8) -> Tensor:
        """The [O, I] tensor ``normalize_weight`` divides the gradients by."""
        # Adam-style epsilon is added outside the square root
        return self.weight_avg_sq.sqrt().add_(eps)

    def normalize_bias(
        self,
        grad: Tensor,
        eps: float = 1e-8,
    ) -> Tensor:
        """Normalize the gradients by the square root of the second moments."""
        assert self.bias_avg_sq is not None
        denom = self.bias_avg_sq.sqrt()

        # Adam-style epsilon is added outside the square root
        return grad / (denom.add_(eps))

    def to_adafactor(self) -> AdafactorNormalizer:
        """
        Convert this Adam normalizer to an Adafactor normalizer, minimizing the
        I-divergence (generalized Kullback-Leibler divergence) between the original
        and the factored second moments.

        Preserves bias_avg_sq if present.
        """
        # We assume weight_avg_sq is a square matrix of shape [O, I]
        assert (
            self.weight_avg_sq.ndim == 2
        ), f"Expected 2D tensor for avg_sq, got {self.weight_avg_sq.ndim}D"

        # Compute row and column means
        return AdafactorNormalizer(
            row=self.weight_avg_sq.mean(dim=1),  # shape [O]
            col=self.weight_avg_sq.mean(dim=0),  # shape [I]
            bias_avg_sq=self.bias_avg_sq,
        )


@dataclass
class OuterProductGradients:
    """A module's gradients ``(g ⊗ a) ⊘ divisor + bias ⊗ bias_col``, kept as the
    vectors they are formed from.

    With ``g`` of shape [T, O] there is one gradient per token. With shape
    [N, S, O] there is one per example, summed over its S positions.
    """

    g: Tensor
    """Output gradients, [T, O] or [N, S, O]."""

    a: Tensor
    """Inputs, [T, W] or [N, S, W], zero in the bias column if there is one."""

    bias: Tensor | None = None
    """Bias gradients, [T, O] or [N, O]."""

    bias_col: Tensor | None = None
    """The [W] vector the bias gradients are paired with: the bias column's
    indicator, or its projection."""

    divisor: Tensor | None = None
    """[O, W] entry-wise divisor of ``g ⊗ a``, from Adam normalization."""

    @property
    def per_example(self) -> bool:
        """Whether each row sums an example's positions."""
        return self.g.ndim == 3

    def dot(self, q: Tensor) -> Tensor:
        """Dot products [rows, Q] of the flattened gradients with each row of
        ``q`` [Q, O * W].

        Per-token gradients are contracted with ``q`` one vector at a time,
        which builds a [T, Q, min(O, W)] tensor instead of the [T, O, W]
        gradients, unless that is larger.
        """
        o, w = self.g.shape[-1], self.a.shape[-1]
        dtype = q.dtype
        q = q.to(self.g.dtype).reshape(len(q), o, w)

        if self.per_example or len(q) >= max(o, w) or self.divisor is not None:
            return (self.materialize().flatten(1) @ q.flatten(1).T).to(dtype)

        # ⟨g ⊗ a, q⟩ = gᵀ q a
        if o <= w:
            # [T, W] @ [W, Q * O] → [T, Q, O]
            a_q = self.a @ q.reshape(-1, w).T
            part = torch.einsum("tqo,to->tq", a_q.view(len(self.a), -1, o), self.g)
        else:
            # [T, O] @ [Q, O, W] → [Q, T, W]
            part = torch.einsum("qtw,tw->tq", self.g @ q, self.a)
        if self.bias is not None:
            assert self.bias_col is not None
            part.add_(self.bias @ (q @ self.bias_col).T)
        return part.to(dtype)

    def sq_norm(self) -> Tensor:
        """Squared norms [T] of per-token gradients, in float32."""
        assert not self.per_example, "Form per-example gradients to take norms"
        if self.divisor is not None:
            return self.materialize().flatten(1).float().pow(2).sum(-1)

        g, a = self.g.float(), self.a.float()
        n = g.pow(2).sum(-1) * a.pow(2).sum(-1)  # ‖g ⊗ a‖ = ‖g‖·‖a‖

        if self.bias is not None:
            assert self.bias_col is not None
            # The bias term and its cross term with g ⊗ a
            bias, bias_col = self.bias.float(), self.bias_col.float()
            n += bias.pow(2).sum(-1) * bias_col.pow(2).sum()
            n += 2 * (g * bias).sum(-1) * (a * bias_col).sum(-1)

        # The cross term can round the total below zero
        return n.clamp_min_(0)

    def materialize(self) -> Tensor:
        """Form the gradients, [T, O, W] or [N, O, W]."""
        if self.g.ndim == 2:
            P = self.g.unsqueeze(-1) * self.a.unsqueeze(-2)
        else:
            P = self.g.mT @ self.a

        if self.divisor is not None:
            P.div_(self.divisor)
        if self.bias is not None:
            assert self.bias_col is not None
            P.addcmul_(self.bias.unsqueeze(-1), self.bias_col)
        return P
