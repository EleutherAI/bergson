"""Build a MAGIC run over only the documents PAC labeling sends to the expert.

Usage:
    python examples/pac_labeling/subset_magic.py \
        --reference <full run>/config.yaml --expert-set <pac out>/expert_set_q3.npy \
        --queries 3 --run-path <dir> [--background 0] [--batch-size 256] \
        [--num-epochs 2] [--lr 1e-4] [--seed 42]

Writes ``<run-path>/train.hf`` (the selected rows of the reference run's
training set, in corpus order) with ``orig_ids.npy`` beside it, a query set
holding the requested queries, and ``<run-path>/magic.yaml`` cloned from the
reference MAGIC step with the dataset, run path and any overridden training
hyperparameters swapped in. Prints the launch command. ``--background`` adds
that many uniformly random documents from outside the expert set, so the
model also sees ordinary data.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from datasets import load_from_disk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--expert-set", type=Path, required=True)
    ap.add_argument("--queries", type=int, nargs="+", required=True)
    ap.add_argument("--run-path", type=Path, required=True)
    ap.add_argument("--background", type=int, default=0)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--num-epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--grad-accum-steps", type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nproc", type=int, default=2)
    args = ap.parse_args()

    ref = yaml.safe_load(args.reference.read_text())
    step = next(s["magic"] for s in ref["steps"] if "magic" in s)
    ids = np.sort(np.load(args.expert_set))
    train = load_from_disk(step["data"]["dataset"])
    if args.background:
        rng = np.random.default_rng(args.seed)
        rest = np.setdiff1d(np.arange(len(train)), ids)
        ids = np.sort(np.concatenate([ids, rng.choice(rest, args.background, False)]))

    args.run_path.mkdir(parents=True, exist_ok=True)
    keep = [c for c in ("input_ids", "length") if c in train.column_names]
    train.select(ids).select_columns(keep).save_to_disk(str(args.run_path / "train.hf"))
    np.save(args.run_path / "orig_ids.npy", ids)
    queries = load_from_disk(step["query"]["dataset"])
    queries.select(args.queries).save_to_disk(str(args.run_path / "query.hf"))
    np.save(args.run_path / "query_ids.npy", np.asarray(args.queries))

    step = dict(step)
    step["run_path"] = str(args.run_path / "magic")
    step["data"] = {**step["data"], "dataset": str(args.run_path / "train.hf")}
    step["query"] = {**step["query"], "dataset": str(args.run_path / "query.hf")}
    step["resume"] = False
    step["overwrite"] = False
    step["distributed"] = {**step["distributed"], "nproc_per_node": args.nproc}
    for key in ("batch_size", "num_epochs", "grad_accum_steps"):
        value = getattr(args, key)
        if value is not None:
            step[key] = value
    if args.lr is not None:
        step["lr_schedule"] = {**step["lr_schedule"], "lr": args.lr}
    cfg_path = args.run_path / "magic.yaml"
    cfg_path.write_text(yaml.safe_dump({"steps": [{"magic": step}]}, sort_keys=False))

    steps = len(ids) * step["num_epochs"] // step["batch_size"]
    print(f"{len(ids)} documents, {len(args.queries)} queries, ~{steps} steps")
    print(f"python -m bergson {cfg_path}")


if __name__ == "__main__":
    main()
