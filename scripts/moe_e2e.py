"""End-to-end check for MoE fused-parameter expert and router tracking.

Runs the real collection pipeline — ``setup_model_and_peft`` -> ``allocate_batches``
-> ``CollectorComputer`` -> ``GradientCollector`` — over a Hugging Face MoE model
and dataset, then repeats it for the EK-FAC covariance collector.

``bergson build`` itself calls ``torch.cuda.set_device`` unconditionally and
``Builder`` allocates on ``torch.cuda.current_device()``, so the CLI cannot run
without a GPU (the repo's own build/score e2e tests are GPU-gated for the same
reason). This script exercises everything below that boundary, which is where the
MoE changes live, and so runs on CPU.

    python scripts/moe_e2e.py --model trl-internal-testing/tiny-GptOssForCausalLM
"""

import argparse
import shutil
import time
from pathlib import Path

import torch

from bergson.collector.collector import CollectorComputer
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig, PreprocessConfig
from bergson.data import allocate_batches
from bergson.gradients import LayerAdapter
from bergson.hessians.kfac import CovarianceCollector
from bergson.moe import ExpertLinear, is_bare_linear
from bergson.utils.worker_utils import create_processor, setup_data_pipeline


def summarize(collector) -> str:
    experts = sum(
        1
        for name in collector.target_info
        if isinstance(collector.model.get_submodule(name), ExpertLinear)
    )
    routers = sum(
        1
        for name in collector.target_info
        if is_bare_linear(collector.model.get_submodule(name))
    )
    return (
        f"{len(collector.target_info)} tracked modules "
        f"({experts} expert projections, {routers} routers)"
    )


class CheckedCollector(GradientCollector):
    """A collector that checks Builder's invariant on every batch.

    ``process_batch`` clears ``mod_grads``, so the module set can only be
    inspected while a batch is still in hand: every module in ``shapes()`` must
    have contributed a finite gradient, including experts that received no
    tokens.
    """

    def __post_init__(self):
        super().__post_init__()
        self.seen = {"batches": 0}

    def process_batch(self, indices, **kwargs):
        expected = set(self.shapes())
        assert set(self.mod_grads) == expected, (
            f"missing {sorted(expected - set(self.mod_grads))}, "
            f"unexpected {sorted(set(self.mod_grads) - expected)}"
        )
        for name, grad in self.mod_grads.items():
            assert grad.shape[0] == len(indices), name
            assert torch.isfinite(grad).all(), name
        self.seen["batches"] += 1
        super().process_batch(indices, **kwargs)


def build_index_cfg(args, run_path: str, **overrides) -> IndexConfig:
    cfg = IndexConfig(
        run_path=run_path,
        model=args.model,
        projection_dim=args.projection_dim,
        include_bias=True,
        token_batch_size=args.token_batch_size,
        **overrides,
    )
    cfg.data.dataset = args.dataset
    cfg.data.split = args.split
    cfg.data.truncation = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="trl-internal-testing/tiny-GptOssForCausalLM"
    )
    parser.add_argument("--dataset", default="NeelNanda/pile-10k")
    parser.add_argument("--split", default="train[:512]")
    parser.add_argument("--token_batch_size", type=int, default=512)
    parser.add_argument("--projection_dim", type=int, default=8)
    parser.add_argument("--min_seconds", type=float, default=200.0)
    args = parser.parse_args()

    root = Path("runs/moe_e2e")
    shutil.rmtree(root, ignore_errors=True)

    from bergson.utils.worker_utils import setup_model_and_peft

    cfg = build_index_cfg(args, str(root / "index"))
    model, target_modules = setup_model_and_peft(cfg)
    ds, _ = setup_data_pipeline(cfg)
    batches = allocate_batches(ds["length"][:], cfg.token_batch_size)
    print(f"{len(ds)} documents in {len(batches)} batches")

    # Tracked-module counts with and without the opt-out.
    counts = {}
    for track in (True, False):
        cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
        collector = GradientCollector(
            model=model.base_model,  # type: ignore[arg-type]
            cfg=cfg,
            processor=create_processor(model, cfg, target_modules),
            data=ds,
            skip_index=True,
            preprocess_cfg=PreprocessConfig(),
            track_moe_experts=track,
        )
        counts[track] = len(collector.target_info)
        print(f"track_moe_experts={track}: {summarize(collector)}")
    assert counts[True] > counts[False], "expansion tracked nothing extra"

    # Full collection passes, repeated until min_seconds so a soak shows up.
    start = time.time()
    rounds = 0
    while time.time() - start < args.min_seconds:
        rounds += 1
        cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
        collector = CheckedCollector(
            model=model.base_model,  # type: ignore[arg-type]
            cfg=cfg,
            processor=create_processor(model, cfg, target_modules),
            data=ds,
            skip_index=True,
            preprocess_cfg=PreprocessConfig(),
        )
        seen = collector.seen
        CollectorComputer(
            model=model,  # type: ignore[arg-type]
            data=ds,
            collector=collector,
            batches=batches,
            cfg=cfg,
        ).run_with_collector_hooks(desc=f"gradients (round {rounds})")

        assert seen["batches"] == len(batches), "not every batch was collected"
        print(
            f"round {rounds}: {seen['batches']} batches, "
            f"{len(collector.shapes())} module gradients each, all finite"
        )

    # EK-FAC covariance, which reads each expert's own routed-row mask.
    cov_path = root / "ekfac"
    cov_path.mkdir(parents=True, exist_ok=True)
    cov_cfg = build_index_cfg(args, str(root / "ekfac_run"))
    cov_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    cov = CovarianceCollector(
        model=model.base_model,  # type: ignore[arg-type]
        processor=create_processor(model, cov_cfg, target_modules),
        dtype=torch.float32,
        path=str(cov_path),
    )
    CollectorComputer(
        model=model,  # type: ignore[arg-type]
        data=ds,
        collector=cov,
        batches=batches[:20],
        cfg=cov_cfg,
    ).run_with_collector_hooks(desc="EK-FAC covariance")

    for name in cov.target_info:
        layer = cov.model.get_submodule(name)
        i = getattr(layer, LayerAdapter.in_attr(layer))
        o = getattr(layer, LayerAdapter.out_attr(layer))
        expected_i = i + 1 if cov.target_info[name][2] else i
        assert cov.A_cov_dict[name].shape[-1] == expected_i, name
        assert cov.S_cov_dict[name].shape[-1] == o, name
    print(f"EK-FAC covariance factors correct for {len(cov.target_info)} modules")

    print(f"\nOK — {rounds} collection rounds in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
