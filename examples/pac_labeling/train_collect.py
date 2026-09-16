"""Retrain a MAGIC run's recipe with the HuggingFace Trainer while collecting
every example's projected, optimizer-normalized gradient at the step it was
trained on.

Usage:
    python examples/pac_labeling/train_collect.py \
        --reference <magic run>/config.yaml --run-path <dir> [--projection-dim 32]

Reads the model, data and optimizer settings from the reference MAGIC step
(AdamW betas, epsilon, weight decay, polynomial schedule with warmup, batch
size, epochs, seed) and trains with ``GradientCollectorCallback``, which
stores ``lr * g / sqrt(v)`` per example and epoch under
``<run-path>/gradients/train/epoch_*``. The final model is saved to
``<run-path>/model`` and a ``build`` config for the query gradients with the
same projection to ``<run-path>/query_build.yaml``.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.pytorch_utils import Conv1D

from bergson import GradientProcessor
from bergson.collector.gradient_collectors import StreamingGradientCollector
from bergson.data import column_offsets
from bergson.huggingface import (
    GradientCollectorCallback,
    prepare_for_gradient_collection,
)


class CollectorCallback(GradientCollectorCallback):
    """The stock callback over every Linear/Conv1D module except ``lm_head``,
    the modules EK-FAC attributes through."""

    def on_train_begin(self, args, state, control, *, model, **kwargs):
        if not hasattr(args, "__gradient_collection_enabled__"):
            raise RuntimeError("call prepare_for_gradient_collection on the trainer")
        base = getattr(model, "base_model", model)
        names = [
            n
            for n, m in base.named_modules()
            if isinstance(m, (Conv1D, torch.nn.Linear)) and not n.endswith("lm_head")
        ]
        self.collector = StreamingGradientCollector(
            model=base,
            processor=GradientProcessor({}, projection_dim=self.projection_dim),
            target_modules=names,
            attention_cfgs=self.attention_cfgs,
            dtype=self.torch_dtype,
        )
        self.grad_sizes = {
            k: int(np.prod(s)) for k, s in self.collector.shapes().items()
        }
        self.grad_offsets = column_offsets(self.grad_sizes)
        self.collector.__enter__()
        self.fwd_handle = model.register_forward_pre_hook(
            self.on_forward_begin, with_kwargs=True
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--run-path", type=Path, required=True)
    ap.add_argument("--projection-dim", type=int, default=32)
    ap.add_argument("--per-device-batch-size", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    ref = yaml.safe_load(args.reference.read_text())
    step = next(s["magic"] for s in ref["steps"] if "magic" in s)
    sched = step["lr_schedule"]
    world = max(1, torch.cuda.device_count())
    accum = step["batch_size"] // (args.per_device_batch_size * world)

    model_kwargs = dict(
        kv.split("=") for kv in step.get("model_kwargs", "").split(",") if kv
    )
    model_kwargs = {k: float(v) for k, v in model_kwargs.items()}
    model = AutoModelForCausalLM.from_pretrained(
        step["model"], dtype=torch.float32, **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(step["model"])
    tokenizer.pad_token = tokenizer.eos_token

    train = load_from_disk(step["data"]["dataset"]).select_columns(["input_ids"])
    callback = CollectorCallback(
        args.run_path / "gradients",
        projection_dim=args.projection_dim,
        accumulate_grads=False,
        use_optimizer_state=True,
        scale_by_lr=True,
        track_order=True,
    )
    training_args = TrainingArguments(
        output_dir=str(args.run_path / "trainer"),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=accum,
        num_train_epochs=step["num_epochs"],
        max_steps=args.max_steps,
        learning_rate=sched["lr"],
        lr_scheduler_type="polynomial",
        lr_scheduler_kwargs={"lr_end": sched["lr_end"], "power": sched["power"]},
        warmup_ratio=sched["warmup_steps"],
        adam_beta1=step["adam_beta1"],
        adam_beta2=step["adam_beta2"],
        adam_epsilon=step["adam_eps"],
        weight_decay=step["weight_decay"],
        max_grad_norm=step.get("max_grad_norm") or 0.0,
        seed=step["seed"],
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[callback],
    )
    trainer = prepare_for_gradient_collection(trainer)
    print(f"steps per epoch {len(train) // step['batch_size']}, accum {accum}")
    trainer.train()
    trainer.save_model(str(args.run_path / "model"))
    tokenizer.save_pretrained(str(args.run_path / "model"))

    build = {
        "steps": [
            {
                "build": {
                    "index_cfg": {
                        "run_path": str(args.run_path / "query"),
                        "model": str(args.run_path / "model"),
                        "overwrite": True,
                        "projection_dim": args.projection_dim,
                        "token_batch_size": 1024,
                        "filter_modules": "lm_head",
                        "distributed": {"nproc_per_node": 1, "nnode": 1},
                        "data": {**step["query"], "chunk_length": 0},
                    },
                    "preprocess_cfg": {"unit_normalize": False, "aggregation": "none"},
                }
            }
        ]
    }
    (args.run_path / "query_build.yaml").write_text(yaml.safe_dump(build))
    print(f"python -m bergson {args.run_path / 'query_build.yaml'}")


if __name__ == "__main__":
    main()
