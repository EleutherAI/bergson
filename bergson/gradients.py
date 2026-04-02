import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Mapping

import torch
import torch.nn as nn
from torch import Tensor
from transformers.pytorch_utils import Conv1D as HFConv1D

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


@dataclass
class GradientProcessor:
    """Configuration for processing and compressing gradients."""

    normalizers: Mapping[str, Normalizer] = field(default_factory=dict)
    """
    Dictionary of normalizers for each matrix-valued parameter in the model. The keys
    should match the names of the parameters in the model. If a parameter does not have
    a normalizer, it will be skipped.
    """

    preconditioners: dict[str, Tensor] = field(default_factory=dict)
    """
    Dictionary of preconditioners for each matrix-valued parameter in the model.
    These are applied after the normalization and random projection steps.
    """

    preconditioners_eigen: Mapping[str, tuple[Tensor, Tensor]] = field(
        default_factory=dict
    )
    """
    Dictionary of eigen decompositions of preconditioners for each matrix-valued
    parameter in the model. Each value is a tuple of (eigenvalues, eigenvectors).
    These are used to efficiently apply inverse square-root of the preconditioners
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

    include_bias: bool = False
    """Whether to include bias gradients when present on a module."""

    def __post_init__(self):
        self._projection_matrices: dict[
            tuple[str, Literal["left", "right"], torch.device], Tensor
        ] = {}

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        map_location: str | torch.device | None = None,
        skip_preconditioners: bool = False,
    ) -> "GradientProcessor":
        """
        Load the normalizers and preconditioners from a file.
        """
        path = Path(path)
        cfg_path = path / "processor_config.yaml"
        norm_path = path / "normalizers.pth"
        precond_path = path / "preconditioners.pth"
        precond_eigen_path = path / "preconditioners_eigen.pth"

        # Load configuration
        with cfg_path.open("r") as f:
            cfg = json.load(f)

        # Backward compatibility
        if "projection_type" not in cfg:
            cfg["projection_type"] = "normal"
        if "include_bias" not in cfg:
            cfg["include_bias"] = False

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

        preconditioners, preconditioners_eigen = {}, {}
        if not skip_preconditioners:
            preconditioners = torch.load(
                precond_path,
                map_location=map_location,
                weights_only=True,
            )
            preconditioners_eigen = torch.load(
                precond_eigen_path,
                map_location=map_location,
                weights_only=True,
            )

        return cls(
            normalizers=normalizers,
            preconditioners=preconditioners,
            preconditioners_eigen=preconditioners_eigen,
            **cfg,
        )

    def save(self, path: Path):
        """
        Save the normalizers and preconditioners to a file.
        """
        path.mkdir(parents=True, exist_ok=True)

        cfg_path = path / "processor_config.yaml"
        norm_path = path / "normalizers.pth"
        precond_path = path / "preconditioners.pth"
        precond_eigen_path = path / "preconditioners_eigen.pth"

        # Save configuration separately
        cfg = asdict(self)
        del cfg["normalizers"]
        del cfg["preconditioners"]
        del cfg["preconditioners_eigen"]
        with cfg_path.open("w") as f:
            json.dump(cfg, f, indent=2)

        # Save normalizers
        norm_state = {
            name: normalizer.state_dict()
            for name, normalizer in self.normalizers.items()
        }
        torch.save(norm_state, norm_path)
        torch.save(self.preconditioners, precond_path)
        torch.save(self.preconditioners_eigen, precond_eigen_path)


class LayerAdapter:
    supported_modules = (nn.Linear, HFConv1D, nn.Conv1d, nn.Conv2d, nn.Conv3d)

    @staticmethod
    def in_attr(layer: nn.Module) -> str:
        match layer:
            case nn.Linear():
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
            case nn.Linear():
                return "out_features"
            case HFConv1D():
                return "nf"
            case nn.Conv1d() | nn.Conv2d() | nn.Conv3d():
                return "out_channels"
            case _:
                raise ValueError(f"Unsupported layer type: {type(layer)}")


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

    @torch.compile
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
        grad *= a[:, None]  # [N, O] * [O] → [N, O]
        grad *= b[None, :]

        return grad

    @torch.compile
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

    @torch.compile
    def normalize_weight(
        self,
        grad: Tensor,
        eps: float = 1e-8,
    ) -> Tensor:
        """Normalize the gradients by the square root of the second moments."""
        # Adam-style epsilon is added outside the square root
        denom = self.weight_avg_sq.sqrt()
        return grad.div_(denom.add_(eps))

    @torch.compile
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
