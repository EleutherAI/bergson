"""Export checkpoints -> EK-FAC IF -> SOURCE -> LDS for the WikiText-2 replication.

Assumes examples/replication/wikitext_gpt2_train.yaml has already run.
Ground truth is kronfluence's shipped masks/losses (see GROUND_TRUTH below).

    PYTHONPATH=$PWD python experiments/wikitext_replication/run_chain.py

Last run (2026-07-31, 8x GPU): EK-FAC IF 0.4451 (kronfluence reports 0.44),
SOURCE 0.4506. SOURCE > IF reproduces the ordering in the paper's Figure 6.
"""

import subprocess
import sys
from pathlib import Path

from bergson.utils.trainer_export import export_checkpoints
from bergson.validate import lds_from_precomputed_subsets

RUN = Path("runs/wikitext_repro")
GROUND_TRUTH = Path("/mnt/ssd-1/lucia/bergson-source-repro/runs/wikitext_kron")
STEPS = [291, 582, 873, 1164, 1455, 1746]


def run(cmd: str) -> None:
    print(f"\n=== RUN: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    masks, losses = GROUND_TRUTH / "masks.pt", GROUND_TRUTH / "losses.pt"
    for p in (masks, losses):
        if not p.exists():
            sys.exit(f"missing ground truth: {p}")

    exported = export_checkpoints(RUN / "train", steps=STEPS, overwrite=True)
    print(f"exported: {[str(p) for p in exported]}", flush=True)

    run(
        "python -m bergson examples/replication/wikitext_gpt2_ekfac.yaml"
        f" > {RUN}/if.log 2>&1"
    )
    print("EK-FAC IF: ", end="")
    lds_from_precomputed_subsets(
        str(RUN / "if_scores/scores"),
        str(masks),
        str(losses),
        summary_path=str(RUN / "if_lds.csv"),
    )

    run(
        "python -m bergson examples/replication/wikitext_gpt2_source.yaml"
        f" > {RUN}/source.log 2>&1"
    )
    print("SOURCE: ", end="")
    lds_from_precomputed_subsets(
        str(RUN / "source_scores/scores"),
        str(masks),
        str(losses),
        summary_path=str(RUN / "source_lds.csv"),
    )


if __name__ == "__main__":
    main()
