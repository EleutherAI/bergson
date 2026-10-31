"""Time per-sample gradient collection with Adam and Adafactor normalizers.

Runs one forward/backward pass on a randomly initialized model with no hooks,
with GradientCollector hooks and no normalizer, and with each normalizer built
from the squared gradients of the hooked pass.
"""

import time
from argparse import ArgumentParser

import torch
from datasets import Dataset
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig
from bergson.gradients import AdafactorNormalizer, AdamNormalizer, GradientProcessor

# Disable cache size limit for this example
torch._dynamo.cache_size = 0  # type: ignore


def timed_fwd_bwd(model: nn.Module, inputs: dict[str, torch.Tensor]) -> float:
    """Run one forward/backward pass and return its wall-clock time in seconds."""
    torch.cuda.synchronize()
    start = time.monotonic()
    model(**inputs).loss.backward()
    torch.cuda.synchronize()
    return time.monotonic() - start


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size to use for the test",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-1b",
        help="Model to use for the test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=512,
        help="Length of the input sequence",
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=16,
        help="Per-module random projection dimension, 0 to keep full gradients",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_config(config)
    model.to("cuda:0")
    model.float()

    tokens = torch.arange(args.sequence_length, device=model.device)
    tokens = tokens.unsqueeze(0).expand(args.batch_size, -1)
    inputs = dict(input_ids=tokens, labels=tokens)

    # The collector needs a dataset row per example but writes nothing to disk
    data = Dataset.from_dict({"input_ids": tokens.tolist()})
    cfg = IndexConfig(
        run_path="runs/adafactor_timing",
        model=args.model,
        projection_dim=args.projection_dim,
    )

    print("Warming up the torch.compile cache...")
    model(**inputs).loss.backward()
    model.zero_grad()

    print("Running with no hook at all")
    print(f"Time taken: {timed_fwd_bwd(model, inputs):.3f} seconds")
    model.zero_grad()

    print("Running with no normalization")
    with GradientCollector(
        model=model,
        data=data,
        cfg=cfg,
        processor=GradientProcessor(projection_dim=args.projection_dim),
        skip_index=True,
    ) as collector:
        print(f"Time taken: {timed_fwd_bwd(model, inputs):.3f} seconds")
    collector.mod_grads.clear()

    # Build both normalizers from the squared gradients of the tracked modules
    adafactors: dict[str, AdafactorNormalizer] = {}
    adams: dict[str, AdamNormalizer] = {}
    for name in collector.target_info:
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Linear):
            continue

        g = layer.weight.grad
        assert g is not None

        adams[name] = AdamNormalizer(g.square())
        adafactors[name] = adams[name].to_adafactor()

    for normalizers in (adafactors, adams):
        model.zero_grad()

        print(
            f"Using {type(next(iter(normalizers.values()))).__name__} for normalization"
        )
        with GradientCollector(
            model=model,
            data=data,
            cfg=cfg,
            processor=GradientProcessor(
                normalizers=normalizers, projection_dim=args.projection_dim
            ),
            skip_index=True,
        ) as collector:
            print(f"Time taken: {timed_fwd_bwd(model, inputs):.3f} seconds")
        collector.mod_grads.clear()
