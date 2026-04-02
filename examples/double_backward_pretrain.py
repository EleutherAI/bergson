#!/usr/bin/env python3
"""MAGIC attribution on a pretrained model.

Trains from a random init checkpoint (Pythia step0), attributes eval loss
to training examples, and validates via leave-subset-out retraining.

Usage:
    python examples/magic_pretrain.py runs/magic_pretrain

    # Or via the CLI:
    bergson magic runs/magic_pretrain \
        --model EleutherAI/pythia-160m \
        --revision step0 \
        --data.dataset EleutherAI/SmolLM2-135M-10B \
        --query.dataset EleutherAI/SmolLM2-135M-10B \
        --query.split "train[:1]"
"""

from bergson.config import DataConfig, DistributedConfig, LRScheduleConfig
from bergson.magic import MagicConfig, run_magic


def main():
    run_cfg = MagicConfig(
        run_path="runs/magic_pretrain",
        model="EleutherAI/pythia-160m",
        revision="step0",
        data=DataConfig(
            dataset="EleutherAI/SmolLM2-135M-10B",
            split="train[:200]",
        ),
        query=DataConfig(
            dataset="EleutherAI/SmolLM2-135M-10B",
            split="train[:1]",
        ),
        lr_schedule=LRScheduleConfig(lr=1e-5, warmup_steps=10),
        batch_size=8,
        num_subsets=100,
        seed=42,
    )
    dist_cfg = DistributedConfig()
    run_magic(run_cfg, dist_cfg)


if __name__ == "__main__":
    main()
