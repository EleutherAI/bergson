"""Rigorous end-to-end validation of the compressed EKFAC pipeline.

Builds four indices from the same EKFAC factors fit on a small slice of
pile-10k:

* ``train_compressed``  — pile-10k[:N_TRAIN], projection_dim=PROJECTION_DIM
* ``train_reference``   — pile-10k[:N_TRAIN], projection_dim=0
* ``query_compressed``  — pile-10k[N_TRAIN:N_TRAIN+N_QUERY], projection_dim=PROJECTION_DIM
* ``query_reference``   — pile-10k[N_TRAIN:N_TRAIN+N_QUERY], projection_dim=0

The "reference" indices still apply EKFAC preconditioning per example
but skip the random projection, so they hold the full-dim
``vec(H^{-1/2} G)`` per example — the target that compressed scoring is
supposed to approximate under Johnson-Lindenstrauss.

For each query, compares top-K retrieval under the compressed dot
product against the reference dot product, reporting:

* ``recall@K`` for K in {5, 10, 20}
* Spearman correlation of the full per-query score vectors
* Qualitative top-3 neighbors side-by-side

If compression preserves neighbor structure (recall @ K well above
K / N_TRAIN, Spearman high), commits 1-3 function end-to-end:
EKFAC math correct, projection matrices consistent between index
and query sides, and the overall build-to-retrieve loop works. If not,
something is broken and commit 4's notebook is premature.

Usage:
    python scripts/validate_compressed_ekfac.py [out_root]

``out_root`` defaults to ``/tmp/validate_compressed_ekfac``. Existing
artifacts under ``out_root`` are reused — delete the directory to force
a rebuild.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr

from bergson.build import build
from bergson.config import DataConfig, HessianConfig, IndexConfig, PreprocessConfig
from bergson.data import load_gradients
from bergson.hessians.hessian_approximations import approximate_hessians
from bergson.utils.worker_utils import validate_run_path

# Defaults — override via CLI.
DEFAULT_MODEL = "EleutherAI/pythia-14m"
DEFAULT_DATASET = "NeelNanda/pile-10k"
DEFAULT_N_TRAIN = 200
DEFAULT_N_QUERY = 5
DEFAULT_PROJECTION_DIM = 64  # p²=4096 per module; needed because Kronecker-structured
# projection L·G·R^T requires p roughly proportional to
# sqrt(O·I) to preserve inner products.
DEFAULT_UNIT_NORMALIZE = True  # power=-0.5, i.e. split preconditioning so
#  <q_{H^-0.5}, t_{H^-0.5}> = <q, H^-1, t>.
DEFAULT_TOKEN_BATCH_SIZE = 1024
DEFAULT_PRECISION = "bf16"
DEFAULT_NPROC_PER_NODE = 1

# Module-level globals set by `main` so helper functions don't have to thread
# them through; cleaner for a one-file diagnostic script.
MODEL = DEFAULT_MODEL
DATASET = DEFAULT_DATASET
N_TRAIN = DEFAULT_N_TRAIN
N_QUERY = DEFAULT_N_QUERY
PROJECTION_DIM = DEFAULT_PROJECTION_DIM
UNIT_NORMALIZE = DEFAULT_UNIT_NORMALIZE
TOKEN_BATCH_SIZE = DEFAULT_TOKEN_BATCH_SIZE
PRECISION = DEFAULT_PRECISION
NPROC_PER_NODE = DEFAULT_NPROC_PER_NODE


def make_index_cfg(
    run_path: Path,
    split: str,
    projection_dim: int,
    nproc_per_node: int | None = None,
) -> IndexConfig:
    """Build an IndexConfig.

    ``nproc_per_node`` defaults to the script-wide ``NPROC_PER_NODE``. The
    query-side builds force this to 1 because bergson's `_allocate_batches_world`
    requires `total_batches % world_size == 0`, which fails for tiny
    query splits at high world sizes — and there's no real-workflow benefit
    to scattering a handful of held-out queries across many GPUs anyway.
    """
    cfg = IndexConfig(
        run_path=str(run_path),
        model=MODEL,
        precision=PRECISION,
        projection_dim=projection_dim,
        token_batch_size=TOKEN_BATCH_SIZE,
        skip_preconditioners=True,
        data=DataConfig(dataset=DATASET, split=split, truncation=True),
        debug=True,  # enables setup_reproducibility — essential for
        # comparing two independent builds bitwise.
    )
    cfg.distributed.nproc_per_node = (
        nproc_per_node if nproc_per_node is not None else NPROC_PER_NODE
    )
    return cfg


def flatten_structured(mmap) -> np.ndarray:
    """Concatenate every module's gradients into ``[N, total_dim]`` as float32.

    Bergson writes bf16 as ``ml_dtypes.bfloat16`` in the structured mmap;
    numpy can't cast that natively, so we round-trip via torch (same
    ``view(uint16) → view(bfloat16) → float()`` trick as
    :func:`bergson.utils.utils.numpy_to_tensor`)."""
    names = mmap.dtype.names
    parts: list[np.ndarray] = []
    for n in names:
        arr = np.ascontiguousarray(mmap[n]).reshape(len(mmap), -1)
        if arr.dtype == np.float32:
            parts.append(arr)
        else:
            # Assume bfloat16 on disk — reinterpret via torch for cast.
            t = torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16).float()
            parts.append(t.numpy())
    return np.concatenate(parts, axis=-1)


def run_build_if_missing(
    run_path: Path, cfg: IndexConfig, pre: PreprocessConfig, desc: str
) -> None:
    if run_path.exists() and (run_path / "info.json").exists():
        print(f"  [skip] {desc}: artifacts exist at {run_path}")
        return
    if run_path.exists():
        shutil.rmtree(run_path)
    print(f"  [run ] {desc} → {run_path}")
    validate_run_path(cfg)
    build(cfg, pre)


def main(out_root: Path) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    factor_path = out_root / "hessian"
    train_compressed = out_root / "train_compressed"
    train_reference = out_root / "train_reference"
    query_compressed = out_root / "query_compressed"
    query_reference = out_root / "query_reference"

    # ── 1: Fit EKFAC factors on training set ─────────────────────────────
    print("Step 1/5: Fitting EKFAC factors...")
    if not (factor_path / "kfac").exists():
        hessian_cfg = HessianConfig(method="kfac", ev_correction=True)
        h_cfg = make_index_cfg(factor_path, f"train[:{N_TRAIN}]", projection_dim=0)
        validate_run_path(h_cfg)
        approximate_hessians(h_cfg, hessian_cfg)
    else:
        print(f"  [skip] factors exist at {factor_path / 'kfac'}")
    ekfac_path = str(factor_path / "kfac")

    train_split = f"train[:{N_TRAIN}]"
    query_split = f"train[{N_TRAIN}:{N_TRAIN + N_QUERY}]"

    # ── 2+3: Train compressed + train reference ──────────────────────────
    print("Step 2/5: Compressed train index...")
    run_build_if_missing(
        train_compressed,
        make_index_cfg(train_compressed, train_split, projection_dim=PROJECTION_DIM),
        PreprocessConfig(preconditioner_path=ekfac_path, unit_normalize=UNIT_NORMALIZE),
        "compressed train",
    )
    print("Step 3/5: Reference (unprojected EKFAC) train index...")
    run_build_if_missing(
        train_reference,
        make_index_cfg(train_reference, train_split, projection_dim=0),
        PreprocessConfig(preconditioner_path=ekfac_path, unit_normalize=UNIT_NORMALIZE),
        "reference train",
    )

    # ── 4+5: Query compressed + query reference (always single-GPU) ─────
    # bergson requires `total_batches % world_size == 0`, which is brittle
    # for tiny query splits at high world sizes. Force nproc_per_node=1 here.
    print("Step 4/5: Compressed query index...")
    run_build_if_missing(
        query_compressed,
        make_index_cfg(
            query_compressed,
            query_split,
            projection_dim=PROJECTION_DIM,
            nproc_per_node=1,
        ),
        PreprocessConfig(preconditioner_path=ekfac_path, unit_normalize=UNIT_NORMALIZE),
        "compressed query",
    )
    print("Step 5/5: Reference (unprojected EKFAC) query index...")
    run_build_if_missing(
        query_reference,
        make_index_cfg(
            query_reference,
            query_split,
            projection_dim=0,
            nproc_per_node=1,
        ),
        PreprocessConfig(preconditioner_path=ekfac_path, unit_normalize=UNIT_NORMALIZE),
        "reference query",
    )

    # ── Load indices ─────────────────────────────────────────────────────
    print("\nLoading indices...")
    train_A = flatten_structured(load_gradients(train_compressed))
    train_B = flatten_structured(load_gradients(train_reference))
    query_A = flatten_structured(load_gradients(query_compressed))
    query_B = flatten_structured(load_gradients(query_reference))
    print(f"  train_compressed: {train_A.shape}    train_reference: {train_B.shape}")
    print(f"  query_compressed: {query_A.shape}    query_reference: {query_B.shape}")
    assert train_A.shape[1] == query_A.shape[1]
    assert train_B.shape[1] == query_B.shape[1]

    # ── Score + metrics ──────────────────────────────────────────────────
    print("\nComputing scores...")
    scores_A = query_A @ train_A.T  # [N_QUERY, N_TRAIN]
    scores_B = query_B @ train_B.T

    random_recall = {k: k / N_TRAIN for k in (5, 10, 20)}
    print("\nPer-query metrics (random-baseline recalls for reference):")
    print(
        f"  random@5={random_recall[5]:.0%}  "
        f"random@10={random_recall[10]:.0%}  "
        f"random@20={random_recall[20]:.0%}"
    )
    print(
        f"\n{'q':>3} {'recall@5':>10} {'recall@10':>10} {'recall@20':>10} {'spearman':>10}"
    )
    mean_recalls = {k: 0.0 for k in (5, 10, 20)}
    mean_rho = 0.0
    for q in range(N_QUERY):
        a = scores_A[q]
        b = scores_B[q]
        rec = {}
        for k in (5, 10, 20):
            top_A = set(np.argsort(a)[-k:].tolist())
            top_B = set(np.argsort(b)[-k:].tolist())
            rec[k] = len(top_A & top_B) / k
            mean_recalls[k] += rec[k] / N_QUERY
        rho = spearmanr(a, b).statistic
        mean_rho += rho / N_QUERY
        print(f"{q:>3d} {rec[5]:>10.2%} {rec[10]:>10.2%} {rec[20]:>10.2%} {rho:>10.3f}")
    print(
        f"{'mean':>3} {mean_recalls[5]:>10.2%} {mean_recalls[10]:>10.2%} {mean_recalls[20]:>10.2%} {mean_rho:>10.3f}"
    )

    # ── Qualitative ──────────────────────────────────────────────────────
    print("\nQualitative top-3 neighbors (first 3 queries):")
    ds = load_dataset(DATASET, split="train")
    for q in range(min(N_QUERY, 3)):
        q_text = ds[N_TRAIN + q]["text"][:100].replace("\n", " ")
        print(f"\n  Query {q}: {q_text!r}")
        print(f"  {'rank':>4}  {'compressed':>8}  {'reference':>8}  text")
        for k in range(3):
            a_idx = int(np.argsort(scores_A[q])[-(k + 1)])
            b_idx = int(np.argsort(scores_B[q])[-(k + 1)])
            a_text = ds[a_idx]["text"][:60].replace("\n", " ")
            b_text = ds[b_idx]["text"][:60].replace("\n", " ")
            print(f"  {k + 1:>4}  A[{a_idx:>3d}]  B[{b_idx:>3d}]")
            print(f"        A: {a_text!r}")
            print(f"        B: {b_text!r}")

    # ── Verdict ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    PASS_RECALL_10 = 0.40  # 4× random baseline @ K=10
    PASS_SPEARMAN = 0.30
    passed = mean_recalls[10] >= PASS_RECALL_10 and mean_rho >= PASS_SPEARMAN
    if passed:
        print(
            f"✓ PASS: mean recall@10 = {mean_recalls[10]:.1%} ≥ {PASS_RECALL_10:.0%}, "
            f"mean Spearman = {mean_rho:.3f} ≥ {PASS_SPEARMAN:.2f}"
        )
        print("  Compressed retrieval is consistent with reference retrieval.")
    else:
        print(
            f"✗ FAIL: mean recall@10 = {mean_recalls[10]:.1%} (need ≥ {PASS_RECALL_10:.0%}), "
            f"mean Spearman = {mean_rho:.3f} (need ≥ {PASS_SPEARMAN:.2f})"
        )
        print("  Compression is NOT preserving neighbor structure.")
    print("=" * 72)
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "out_root",
        nargs="?",
        default="/tmp/validate_compressed_ekfac",
        help="Where to materialize artifacts (default: %(default)s).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--n_train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--n_query", type=int, default=DEFAULT_N_QUERY)
    parser.add_argument("--projection_dim", type=int, default=DEFAULT_PROJECTION_DIM)
    parser.add_argument(
        "--unit_normalize",
        type=lambda s: s.lower() in ("1", "true", "yes"),
        default=DEFAULT_UNIT_NORMALIZE,
    )
    parser.add_argument(
        "--token_batch_size", type=int, default=DEFAULT_TOKEN_BATCH_SIZE
    )
    parser.add_argument("--precision", default=DEFAULT_PRECISION)
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=DEFAULT_NPROC_PER_NODE,
        help="Number of GPUs per node (passed to bergson DistributedConfig).",
    )
    args = parser.parse_args()

    MODEL = args.model
    DATASET = args.dataset
    N_TRAIN = args.n_train
    N_QUERY = args.n_query
    PROJECTION_DIM = args.projection_dim
    UNIT_NORMALIZE = args.unit_normalize
    TOKEN_BATCH_SIZE = args.token_batch_size
    PRECISION = args.precision
    NPROC_PER_NODE = args.nproc_per_node

    print(
        f"Config: model={MODEL}  n_train={N_TRAIN}  n_query={N_QUERY}  "
        f"projection_dim={PROJECTION_DIM}  unit_normalize={UNIT_NORMALIZE}  "
        f"nproc_per_node={NPROC_PER_NODE}  precision={PRECISION}\n"
    )
    sys.exit(main(Path(args.out_root)))
