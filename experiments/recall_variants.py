"""Sweep attribution variants on a recall experiment (see examples/recall/).

For an existing recall run (trained hf_model + EK-FAC query index + fitted
KFAC factors), computes scores for:

- EK-FAC inversion strategies (damped_inverse / cauchy / pseudoinverse) at
  several damping/truncation strengths (apply + score per variant, reusing
  the fitted factors; transformed queries are deleted after scoring);
- cosine variants (index gradients unit-normalized at score time) for raw
  gradient dot product and for damped EK-FAC;
- uncorrected eigenvalues (ev_correction off) for damped EK-FAC.

Each variant writes ``<run_path>/variants/scores_<tag>``, so
``examples.recall.recall_eval <run_path>/variants`` evaluates them all.

Every CLI invocation is printed so it can be reproduced.

Example:
    python -m experiments.recall_variants runs/recall_asym
"""

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from simple_parsing import field as sp_field
from simple_parsing import parse


@dataclass
class VariantSweepConfig:
    """Config for the attribution-variant sweep."""

    run_path: str = sp_field(positional=True)
    """Recall run directory containing train/hf_model, data/, and ekfac/
    (query index + hessian/kfac factors)."""

    inversions: list[str] = field(
        default_factory=lambda: [
            "cauchy:0.01",
            "cauchy:0.1",
            "cauchy:1.0",
            "pseudoinverse:0.001",
            "pseudoinverse:0.01",
            "pseudoinverse:0.1",
        ]
    )
    """inversion:damping variants to apply and score."""

    cosine_damping: float = 0.1
    """Damping for the EK-FAC + cosine-index variant."""

    token_batch_size: int = 1024
    max_batch_size: int = 8


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def score_cmd(
    run_cfg: VariantSweepConfig, query_path: str, out_dir: str, cosine: bool
) -> list[str]:
    cmd = [
        sys.executable, "-m", "bergson", "score", out_dir,
        "--model", f"{run_cfg.run_path}/train/hf_model",
        "--dataset", f"{run_cfg.run_path}/data/train",
        "--split", "train",
        "--truncation", "true",
        "--projection_dim", "0",
        "--token_batch_size", str(run_cfg.token_batch_size),
        "--max_batch_size", str(run_cfg.max_batch_size),
        "--filter_modules", "lm_head",
        "--query_path", query_path,
        "--score_cfg.precision", "bf16",
        "--overwrite", "true",
    ]  # fmt: skip
    if cosine:
        cmd += ["--unit_normalize", "true"]
    return cmd


def apply_cmd(
    run_cfg: VariantSweepConfig,
    out_dir: str,
    inversion: str,
    damping: float,
    ev_correction: bool = True,
) -> list[str]:
    return [
        sys.executable, "-m", "bergson.hessians.apply_hessian",
        "--hessian_method_path", f"{run_cfg.run_path}/ekfac/hessian/kfac",
        "--gradient_path", f"{run_cfg.run_path}/ekfac/query",
        "--run_path", out_dir,
        "--ev_correction", str(ev_correction).lower(),
        "--inversion", inversion,
        "--lambda_damp_factor", str(damping),
    ]  # fmt: skip


def main():
    run_cfg = parse(VariantSweepConfig)
    variants_dir = Path(run_cfg.run_path) / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    raw_query = f"{run_cfg.run_path}/ekfac/query"

    # (tag, transformed-query maker or raw path, cosine)
    jobs: list[tuple[str, dict | None, bool]] = [
        ("dot_cosine", None, True),
        (
            f"damped_{run_cfg.cosine_damping:g}_cosine",
            dict(inversion="damped_inverse", damping=run_cfg.cosine_damping),
            True,
        ),
        (
            "damped_0.1_no_evcorr",
            dict(inversion="damped_inverse", damping=0.1, ev_correction=False),
            False,
        ),
    ]
    for spec in run_cfg.inversions:
        inversion, damping = spec.split(":")
        jobs.append(
            (
                f"{inversion}_{float(damping):g}",
                dict(inversion=inversion, damping=float(damping)),
                False,
            )
        )

    for tag, apply_spec, cosine in jobs:
        scores_dir = variants_dir / f"scores_{tag}"
        if (scores_dir / "scores.bin").exists():
            print(f"[{tag}] skip (scores exist)")
            continue

        if apply_spec is None:
            query_path = raw_query
        else:
            query_path = str(variants_dir / f"query_{tag}")
            run(
                apply_cmd(
                    run_cfg,
                    query_path,
                    apply_spec["inversion"],
                    apply_spec["damping"],
                    apply_spec.get("ev_correction", True),
                )
            )

        run(score_cmd(run_cfg, query_path, str(scores_dir), cosine))

        if apply_spec is not None:
            print(f"[{tag}] cleaning up {query_path}")
            shutil.rmtree(query_path, ignore_errors=True)

    print(
        f"\nDone. Evaluate with: python -m examples.recall.recall_eval "
        f"{variants_dir} --train_dataset {run_cfg.run_path}/data/train "
        f"--query_dataset {run_cfg.run_path}/data/query"
    )


if __name__ == "__main__":
    main()
