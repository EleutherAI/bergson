"""Test EKFAC against ground truth via the recomposed Fisher matrices."""

import os

import torch
from jaxtyping import Float
from safetensors.torch import load_file
from torch import Tensor

from tests.ekfac_tests.test_utils import (
    load_sharded_covariances,
)


def test_ekfac_recomposition(
    ekfac_results_path: str,
    ground_truth_eigenvectors_path: str,
    ground_truth_eigenvalue_corrections_path: str,
    world_size: int,
) -> None:
    """Recomposed GNH ``(Q_G ⊗ Q_A) Λ (Q_G ⊗ Q_A)^T`` should match ground truth."""
    print("\nTesting recomposed EKFAC GNH...")

    eigens_a_gt = load_file(
        os.path.join(
            ground_truth_eigenvectors_path, "eigenvectors_activations.safetensors"
        )
    )
    eigens_g_gt = load_file(
        os.path.join(
            ground_truth_eigenvectors_path, "eigenvectors_gradients.safetensors"
        )
    )
    lambdas_gt = load_file(
        os.path.join(
            ground_truth_eigenvalue_corrections_path,
            "eigenvalue_corrections.safetensors",
        )
    )

    eigens_a_run = load_sharded_covariances(
        os.path.join(ekfac_results_path, "eigen_activation_sharded")
    )
    eigens_g_run = load_sharded_covariances(
        os.path.join(ekfac_results_path, "eigen_gradient_sharded")
    )
    lambdas_run = load_sharded_covariances(
        os.path.join(ekfac_results_path, "eigenvalue_correction_sharded")
    )

    device = lambdas_run[list(lambdas_run.keys())[0]].device
    total = torch.load(
        os.path.join(ekfac_results_path, "total_processed.pt"),
        map_location=device,
    )
    lambdas_run = {k: v / total for k, v in lambdas_run.items()}

    per_layer = compute_gnh_recomposition_errors(
        eigens_a_gt=eigens_a_gt,
        eigens_g_gt=eigens_g_gt,
        lambdas_gt=lambdas_gt,
        eigens_a_run=eigens_a_run,
        eigens_g_run=eigens_g_run,
        lambdas_run=lambdas_run,
    )

    tol = 0.05 if world_size > 1 else 0.002
    max_err = max(v.item() for v in per_layer.values())
    assert max_err < tol, (
        f"EKFAC GNH recomposition: max per-layer rel_error={max_err:.2e}, "
        f"expected < {tol}\n" + format_per_layer_gnh_errors(per_layer)
    )
    print(f"EKFAC GNH recomposition matches (max per-layer rel_error={max_err:.2e})")


def gnh_frobenius_inner_product(
    eigen_a_gt: Float[Tensor, "i i"],
    eigen_g_gt: Float[Tensor, "o o"],
    lam_gt: Float[Tensor, "o i"],
    eigen_a_run: Float[Tensor, "i i"],
    eigen_g_run: Float[Tensor, "o o"],
    lam_run: Float[Tensor, "o i"],
) -> Float[Tensor, ""]:
    """Return <F_gt, F_run>_F = tr(F_gt F_run).

    Let V = Q_G ⊗ Q_A, D = diag(vec Λ), we have:

        tr(F_gt F_run) = tr(V_gt D_gt V_gt^T V_run D_run V_run^T)
                       = tr(D_gt M D_run M^T)                     {cyclic property}
                         where M := V_gt^T V_run.

    Using the mixed-product identity `(A⊗B)(C⊗D) = AC ⊗ BD` we get:

        M = R_G ⊗ R_A
        where R_G := Q_G_gt^T Q_G_run
              R_A := Q_A_gt^T Q_A_run

    Since D_gt and D_run are diagonal,
    expanding the trace over the joint index (o, i) gives

        <F_gt, F_run>_F = Σ_{o,i,p,j} Λ_gt[o,i] * Λ_run[p,j]
                                      * R_G[o,p]² * R_A[i,j]²
    """
    R_G_sq = (eigen_g_gt.T @ eigen_g_run).pow(2)  # (O, O)
    R_A_sq = (eigen_a_gt.T @ eigen_a_run).pow(2)  # (I, I)

    intermediate = R_G_sq @ lam_run @ R_A_sq.T  # (O, I)
    return (lam_gt * intermediate).sum()


def frobenius_squared_norm(
    lam: Float[Tensor, "o i"],
) -> Float[Tensor, ""]:
    return lam.pow(2).sum()


def gnh_squared_diff_and_norm(
    eigen_a_gt: Float[Tensor, "i i"],
    eigen_g_gt: Float[Tensor, "o o"],
    lam_gt: Float[Tensor, "o i"],
    eigen_a_run: Float[Tensor, "i i"],
    eigen_g_run: Float[Tensor, "o o"],
    lam_run: Float[Tensor, "o i"],
) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
    """Return (||F_gt - F_run||_F^2, ||F_gt||_F^2)"""

    # The norm is orthogonally invariant
    sq_gt = frobenius_squared_norm(lam_gt)
    sq_run = frobenius_squared_norm(lam_run)

    inner = gnh_frobenius_inner_product(
        eigen_a_gt,
        eigen_g_gt,
        lam_gt,
        eigen_a_run,
        eigen_g_run,
        lam_run,
    )
    # ||A - B||_F^2 = ||A||_F^2 - 2 <A, B>_F + ||B||_F^2
    # Clamp the minimum to zero to account for numerical errors.
    sq_diff = (sq_gt - 2 * inner + sq_run).clamp_min(0.0)
    return sq_diff, sq_gt


def compute_gnh_recomposition_errors(
    eigens_a_gt: dict[str, Float[Tensor, "i i"]],
    eigens_g_gt: dict[str, Float[Tensor, "o o"]],
    lambdas_gt: dict[str, Float[Tensor, "o i"]],
    eigens_a_run: dict[str, Float[Tensor, "i i"]],
    eigens_g_run: dict[str, Float[Tensor, "o o"]],
    lambdas_run: dict[str, Float[Tensor, "o i"]],
) -> dict[str, Float[Tensor, ""]]:
    """Per-layer ||F_gt - F_run||_F / ||F_gt||_F for the recomposed GNH."""
    per_layer: dict[str, Tensor] = {}
    for k in sorted(lambdas_gt.keys()):
        sq_diff, sq_gt = gnh_squared_diff_and_norm(
            eigens_a_gt[k].float(),
            eigens_g_gt[k].float(),
            lambdas_gt[k].float(),
            eigens_a_run[k].float(),
            eigens_g_run[k].float(),
            lambdas_run[k].float(),
        )
        per_layer[k] = (sq_diff / sq_gt).sqrt()
    return per_layer


def format_per_layer_gnh_errors(per_layer_errors: dict[str, Tensor]) -> str:
    """Format per-layer recomposed-GNH relative errors for debug output."""
    lines = []
    for k in sorted(per_layer_errors.keys()):
        lines.append(f"  {k}: rel_error={per_layer_errors[k].item():.2e}")
    return "\n".join(lines)
