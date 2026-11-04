import json

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

from bergson.approx_unrolling.approx_unrolling_math import average_checkpoints
from bergson.approx_unrolling.pipeline import approx_unrolling_pipeline
from bergson.config import (
    ApproxUnrollingConfig,
    AstraConfig,
    DataConfig,
    HessianConfig,
    IndexConfig,
    QuerySetConfig,
)
from bergson.data import load_gradients
from bergson.hessians.astra import GaussNewtonProduct

TOKENIZER = "EleutherAI/pythia-14m"


def _checkpoints(root, seeds):
    config = GPT2Config(
        vocab_size=16, n_positions=8, n_embd=4, n_layer=1, n_head=2, n_inner=8
    )
    paths = []
    for step, seed in enumerate(seeds, start=1):
        torch.manual_seed(seed)
        path = root / f"checkpoint-{10 * step}"
        GPT2LMHeadModel(config).save_pretrained(path)
        paths.append(str(path))
    return paths


def test_average_checkpoints(tmp_path):
    paths = _checkpoints(tmp_path, [0, 1, 2])
    average_checkpoints(paths, tmp_path / "averaged")

    models = [AutoModelForCausalLM.from_pretrained(p) for p in paths]
    averaged = AutoModelForCausalLM.from_pretrained(tmp_path / "averaged")
    for name, p in averaged.named_parameters():
        expected = sum(dict(m.named_parameters())[name] for m in models) / 3
        torch.testing.assert_close(p, expected)


def test_astra_source_rejects_the_adam_variant(tmp_path):
    with pytest.raises(ValueError, match="SGD variant"):
        approx_unrolling_pipeline(
            IndexConfig(run_path=str(tmp_path / "run"), model=TOKENIZER),
            HessianConfig(method="kfac", ev_correction=True),
            ApproxUnrollingConfig(
                checkpoints=["checkpoint-10"],
                segments=1,
                lr_list=[0.05],
                step_size_list=[10],
                use_adam_preconditioner=True,
                astra=AstraConfig(num_steps=1),
            ),
        )


def test_astra_source_solves_the_damped_segment_hessians(tmp_path):
    """Each segment's query solves ``(G + 1 / (lr K)) x = q`` with ``G`` the
    Gauss-Newton Hessian at the mean of the segment's checkpoint weights."""
    paths = _checkpoints(tmp_path, [0, 1, 2, 3])
    ids = torch.randint(16, (6, 8), generator=torch.Generator().manual_seed(0))
    data = Dataset.from_dict({"input_ids": ids.tolist()})
    data.save_to_disk(str(tmp_path / "data"))

    lrs, steps = [0.05, 1.0], [10, 10]
    run = tmp_path / "run"
    index_cfg = IndexConfig(
        run_path=str(run),
        model=paths[-1],
        tokenizer=TOKENIZER,
        data=DataConfig(dataset=str(tmp_path / "data")),
        precision="fp32",
        token_batch_size=64,
    )
    approx_unrolling_pipeline(
        index_cfg,
        HessianConfig(method="kfac", ev_correction=True),
        ApproxUnrollingConfig(
            checkpoints=paths,
            segments=2,
            lr_list=lrs,
            step_size_list=steps,
            momentum=0.0,
            query=QuerySetConfig(
                dataset=str(tmp_path / "data"), split="train[:2]", aggregation="none"
            ),
            # Full-batch CG reaches the exact solution
            astra=AstraConfig(num_steps=100, solver="cg", batch_size=len(data)),
        ),
    )
    assert (run / "scores").exists()

    with open(run / "query" / "info.json") as f:
        sizes = json.load(f)["grad_sizes"]
    names = list(sizes)
    dim = sum(sizes.values())
    docs = list(range(len(data)))
    queries = [run / "segment_0" / "query_grad_backward", run / "query"]

    def split(v):
        return dict(zip(names, v.split(list(sizes.values()))))

    for l in range(2):
        models = [
            AutoModelForCausalLM.from_pretrained(
                p, dtype=torch.float64, attn_implementation="eager"
            )
            for p in paths[2 * l : 2 * l + 2]
        ]
        model = models[0].eval().requires_grad_(False)
        for p0, p1 in zip(model.parameters(), models[1].parameters()):
            p0.copy_((p0 + p1) / 2)
        hvp = GaussNewtonProduct(model, data, index_cfg, names)
        h = torch.stack(
            [
                torch.cat(list(hvp(split(e), docs).values()))
                for e in torch.eye(dim).double()
            ]
        )
        q = torch.from_numpy(load_gradients(queries[l]).copy()).double()
        damping = 1 / (lrs[l] * steps[l])
        expected = torch.linalg.solve(h + damping * torch.eye(dim), q.T).T

        solved = load_gradients(run / f"segment_{l}" / "query_grad_segment")
        actual = torch.from_numpy(solved.copy()).double()
        assert (actual - expected).norm() / expected.norm() < 1e-3
