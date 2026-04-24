"""Diagnostic: verify Builder's post-projection matches manual projection.

For one training example and one module, extract:
  (a) The flattened per-module vector from train_reference (EKFAC'd, unprojected).
  (b) The flattened per-module vector from train_compressed (EKFAC'd + projected).

Reshape (a) to [O, I], manually apply ``L @ G @ R.T`` using the exact
``create_projection_matrix`` seeding the Builder uses, compare to (b).

If (a)_projected ≈ (b): projection math is correct; bug is elsewhere
(likely gradient value mismatch between the two build runs).
If (a)_projected ≠ (b): the Builder's projection implementation is
broken.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from bergson.collector.collector import create_projection_matrix
from bergson.data import load_gradients
from bergson.preconditioners import load_preconditioner, _FactoredPreconditioner

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/validate_compressed_ekfac")
P = 16  # PROJECTION_DIM from the validation script


def bf16_mmap_to_float32(arr: np.ndarray) -> np.ndarray:
    """Handle bf16 numpy arrays by round-tripping through torch."""
    if arr.dtype == np.float32:
        return arr
    t = torch.from_numpy(np.ascontiguousarray(arr).view(np.uint16)).view(torch.bfloat16)
    return t.float().numpy()


def main() -> None:
    train_ref = load_gradients(OUT / "train_reference")
    train_cmp = load_gradients(OUT / "train_compressed")

    ref_names = train_ref.dtype.names
    cmp_names = train_cmp.dtype.names
    print(f"reference modules : {len(ref_names)}")
    print(f"compressed modules: {len(cmp_names)}")
    print(f"names match       : {ref_names == cmp_names}")
    if ref_names != cmp_names:
        print("DIFF in module ordering:")
        for i, (a, b) in enumerate(zip(ref_names, cmp_names)):
            if a != b:
                print(f"  [{i}] {a!r} vs {b!r}")
        return

    # Load the EKFAC preconditioner to get per-module shapes (and to confirm
    # it's factored — which is what triggers the Builder's post-projection).
    precond = load_preconditioner(
        str(OUT / "hessian" / "kfac"),
        device=torch.device("cpu"),
        power=-0.5,  # matches default in Builder when unit_normalize=False … wait, power is -1 by default.
    )
    assert isinstance(precond, _FactoredPreconditioner), type(precond)

    # Inspect one module: the first one in the index.
    name = ref_names[0]
    O, I = precond._shapes[name]
    print(f"\nFocus module: {name!r}  (O={O}, I={I})")

    # (a) Reference vector for example 0, module `name`.
    ref_vec = bf16_mmap_to_float32(np.ascontiguousarray(train_ref[name][0]))
    print(f"  ref_vec.shape  = {ref_vec.shape}   expected {(O * I,)}")

    # (b) Compressed vector for example 0, module `name`.
    cmp_vec = bf16_mmap_to_float32(np.ascontiguousarray(train_cmp[name][0]))
    print(f"  cmp_vec.shape  = {cmp_vec.shape}   expected {(P * P,)}")

    # Manually project (a) with the exact matrices the Builder would have used.
    L = create_projection_matrix(
        f"{name}/left", P, O, torch.float32, torch.device("cpu"), "rademacher"
    ).numpy()
    R = create_projection_matrix(
        f"{name}/right", P, I, torch.float32, torch.device("cpu"), "rademacher"
    ).numpy()
    print(f"  L.shape = {L.shape}   R.shape = {R.shape}")

    G = ref_vec.reshape(O, I)
    # Two orderings to test — what my builder claims to do and its transpose-
    # partner, to see if I got the column-major reshape right.
    manual_v1 = (L @ G @ R.T).reshape(-1)             # vec([L G R^T])
    manual_v2 = (L @ G @ R.T).T.reshape(-1)           # vec([L G R^T]^T) = vec([R G^T L^T])

    def close(a, b, label):
        diff = np.abs(a - b).max()
        rel = diff / (np.abs(b).max() + 1e-12)
        return f"max|diff|={diff:.4g}  rel={rel:.4g}  corr={np.corrcoef(a, b)[0,1]:.4f}"

    print("\nManual vs on-disk compressed (module 0, example 0):")
    print(f"  v1 (vec L G R^T)    vs cmp_vec : {close(manual_v1, cmp_vec, 'v1')}")
    print(f"  v2 (vec (L G R^T)^T) vs cmp_vec: {close(manual_v2, cmp_vec, 'v2')}")

    # Also check: does train_compressed[name][i] look like a projection of
    # train_reference[name][i] for several different i?
    print("\nPer-example projection check (first 5 examples, correlation):")
    for i in range(5):
        ref_i = bf16_mmap_to_float32(np.ascontiguousarray(train_ref[name][i]))
        cmp_i = bf16_mmap_to_float32(np.ascontiguousarray(train_cmp[name][i]))
        manual_i = (L @ ref_i.reshape(O, I) @ R.T).reshape(-1)
        corr = np.corrcoef(manual_i, cmp_i)[0, 1]
        print(
            f"  ex{i}: corr(manual, cmp) = {corr:.4f},  "
            f"ref_norm={np.linalg.norm(ref_i):.3g}  "
            f"cmp_norm={np.linalg.norm(cmp_i):.3g}  "
            f"manual_norm={np.linalg.norm(manual_i):.3g}"
        )

    # ───────────────────────────────────────────────────────────────────
    # Per-module Spearman: for each module, compute compressed vs reference
    # score vector for query 0 against all 200 training examples. If
    # per-module Spearman is high but the summed Spearman is low, the sum
    # is noise-dominated. If it's low per-module, something deeper is off.
    # ───────────────────────────────────────────────────────────────────
    from scipy.stats import spearmanr

    query_ref = load_gradients(OUT / "query_reference")
    query_cmp = load_gradients(OUT / "query_compressed")

    print("\nPer-module score Spearman (query 0 vs N_TRAIN training examples):")
    print(f"{'module':>45}  {'O*I':>7}  {'p^2/(OI)':>9}  {'ρ(A,B)':>9}  {'||Q_ref||':>9}  {'⟨Q,T⟩ std':>10}")
    per_module_rhos = []
    for mname in ref_names:
        Om, Im = precond._shapes[mname]
        qr = bf16_mmap_to_float32(np.ascontiguousarray(query_ref[mname][0]))
        qc = bf16_mmap_to_float32(np.ascontiguousarray(query_cmp[mname][0]))
        tr = bf16_mmap_to_float32(np.ascontiguousarray(train_ref[mname][:]))
        tc = bf16_mmap_to_float32(np.ascontiguousarray(train_cmp[mname][:]))
        # Flatten train to [N, d_m]
        tr = tr.reshape(tr.shape[0], -1)
        tc = tc.reshape(tc.shape[0], -1)
        score_A = tc @ qc
        score_B = tr @ qr
        rho = spearmanr(score_A, score_B).statistic
        per_module_rhos.append(rho)
        q_norm = np.linalg.norm(qr)
        inner_std = score_B.std()
        print(
            f"{mname:>45}  {Om * Im:>7d}  {P * P / (Om * Im):>9.4g}  {rho:>9.3f}  {q_norm:>9.3g}  {inner_std:>10.3g}"
        )
    rhos = np.array(per_module_rhos)
    print(
        f"\n  Per-module Spearman stats: "
        f"mean={rhos.mean():.3f}  median={np.median(rhos):.3f}  "
        f"min={rhos.min():.3f}  max={rhos.max():.3f}"
    )


if __name__ == "__main__":
    main()
