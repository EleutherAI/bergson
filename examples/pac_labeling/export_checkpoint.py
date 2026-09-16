"""Export a MAGIC trajectory checkpoint as a HuggingFace model directory.

Usage:
    python examples/pac_labeling/export_checkpoint.py \
        --reference <magic run>/config.yaml \
        --checkpoint <magic run>/checkpoints/step_250.ckpt --out <dir>

The checkpoint is a torch distributed checkpoint of the trainer state whose
parameter entries are keyed by the model's parameter names; only those are
read, and the model config is loaded from the reference MAGIC step.
"""

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ref = yaml.safe_load(args.reference.read_text())
    step = next(s["magic"] for s in ref["steps"] if "magic" in s)
    model_kwargs = dict(
        kv.split("=") for kv in step.get("model_kwargs", "").split(",") if kv
    )
    model_kwargs = {k: float(v) for k, v in model_kwargs.items()}
    model = AutoModelForCausalLM.from_pretrained(
        step["model"], dtype=torch.float32, **model_kwargs
    )
    state = {k: v.detach().clone() for k, v in model.named_parameters()}
    dcp.load(state, checkpoint_id=str(args.checkpoint))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any("lm_head" not in k for k in missing):
        raise RuntimeError(f"missing {missing} unexpected {unexpected}")
    model.tie_weights()
    model.save_pretrained(str(args.out))
    AutoTokenizer.from_pretrained(step["model"]).save_pretrained(str(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
