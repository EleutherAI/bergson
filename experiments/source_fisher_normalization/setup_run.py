"""Seed a SOURCE run dir from an existing one, reusing everything the Fisher
normalization does not change.

Only the segment lambdas (step 4) and everything downstream depend on
``fisher_normalization``; the per-checkpoint covariances/lambdas, the segment
eigenvectors and the query gradients do not. Those are hardlinked from the
baseline run so a normalization sweep costs no extra Hessian passes.

Derived outputs are deliberately NOT linked: the pipeline truncates them in
place, which would corrupt the baseline through a shared inode.

    python experiments/source_fisher_normalization/setup_run.py \
        runs/lotus_source_q50_damp0 runs/lotus_source_q50_docnorm
"""

import argparse
import os
from pathlib import Path

# Per-segment inputs that are independent of the lambda normalization.
SEGMENT_REUSE = (
    "activation_sharded",
    "gradient_sharded",
    "eigen_activation_sharded",
    "eigen_gradient_sharded",
    "total_processed.pt",
)
CKPT_REUSE = (
    "activation_sharded",
    "gradient_sharded",
    "averaged_ev_correct_sharded",
    "total_processed.pt",
)


def link_tree(src: Path, dst: Path) -> int:
    """Hardlink every file under ``src`` into ``dst``. Returns the file count."""
    n = 0
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.link(src, dst)
        return 1
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        out = dst / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            os.link(path, out)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("target", type=Path)
    args = ap.parse_args()

    baseline, target = args.baseline, args.target
    if not baseline.exists():
        raise SystemExit(f"baseline run {baseline} not found")
    if target.exists():
        raise SystemExit(f"{target} already exists; remove it first")

    total = 0
    total += link_tree(baseline / "query", target / "query")

    for seg_dir in sorted(baseline.glob("segment_*")):
        seg = seg_dir.name
        # A Hessian method dir, as opposed to a derived dir (scores,
        # query_grad_*) that also happens to hold a total_processed.pt.
        method_dirs = [
            d for d in seg_dir.iterdir() if (d / "eigen_activation_sharded").is_dir()
        ]
        for method_dir in method_dirs:
            for name in SEGMENT_REUSE:
                src = method_dir / name
                if src.exists():
                    total += link_tree(src, target / seg / method_dir.name / name)

        for ckpt_dir in sorted(seg_dir.glob("ckpt_*")):
            for method_dir in ckpt_dir.iterdir():
                if not method_dir.is_dir():
                    continue
                # kfac.part holds the token count the backfill reads.
                names = (
                    CKPT_REUSE
                    if not method_dir.name.endswith(".part")
                    else ("total_processed.pt",)
                )
                for name in names:
                    src = method_dir / name
                    if src.exists():
                        total += link_tree(
                            src, target / seg / ckpt_dir.name / method_dir.name / name
                        )

    print(f"hardlinked {total} files from {baseline} -> {target}")
    print(
        "not linked (regenerated): eigenvalue_correction_sharded, query_grad_*, scores"
    )


if __name__ == "__main__":
    main()
