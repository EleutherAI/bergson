import numpy as np
import pytest
import torch
from datasets import Dataset
from torch.func import functional_call, jacrev
from transformers import GPT2Config, GPT2LMHeadModel

from bergson.config import (
    DataConfig,
    IndexConfig,
    LissaConfig,
    PreprocessConfig,
    QuerySetConfig,
    ScoreConfig,
)
from bergson.data import load_scores
from bergson.hessians.astra import GaussNewtonProduct
from bergson.hessians.lissa import lissa, lissa_pipeline


def test_lissa_reaches_the_damped_solution():
    """With every document in each batch, the series converges to
    ``(G + damping I)^-1 q`` for the explicit Gauss-Newton Hessian ``G``."""
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=11,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            attn_implementation="eager",
        )
    ).double()
    model.eval().requires_grad_(False)
    data = Dataset.from_dict({"input_ids": torch.randint(11, (6, 5)).tolist()})
    names = ["h.0.attn.c_proj", "h.0.mlp.c_fc"]
    product = GaussNewtonProduct(model, data, IndexConfig(run_path=""), names)
    sizes = {"h.0.attn.c_proj": 8 * 8, "h.0.mlp.c_fc": 32 * 8}

    def logits(flat):
        params, start = {}, 0
        for n in names:
            params.update(product._to_params(n, flat[start : start + sizes[n]]))
            start += sizes[n]
        x = torch.tensor(data["input_ids"])
        return functional_call(model, params, (x,)).logits[:, :-1]

    theta = torch.cat([product._to_flat(n, product.params) for n in names])
    jac = jacrev(logits)(theta).flatten(0, 1)
    probs = torch.softmax(logits(theta), -1).flatten(0, 1)
    out_hessian = torch.diag_embed(probs) - probs[:, :, None] * probs[:, None, :]
    gauss_newton = torch.einsum("tvp,tvw,twq->pq", jac, out_hessian, jac) / len(data)

    damping = 0.05
    damped = gauss_newton + damping * torch.eye(len(theta), dtype=torch.float64)
    q = {n: torch.randn(s, dtype=torch.float64) for n, s in sizes.items()}
    expected = torch.linalg.solve(damped, torch.cat(list(q.values())))

    cfg = LissaConfig(
        damping=damping,
        scale=torch.linalg.eigvalsh(damped.detach()).max().item(),
        depth=4000,
        repeats=2,
        batch_size=len(data),
    )
    actual = lissa(product, q, len(data), cfg, torch.Generator().manual_seed(0))
    torch.testing.assert_close(torch.cat([actual[n] for n in names]), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_lissa_pipeline_scores_every_query(tmp_path):
    """The pipeline writes one score column per query for every training row."""
    run = tmp_path / "run"
    lissa_pipeline(
        IndexConfig(
            run_path=str(run),
            model="EleutherAI/pythia-14m",
            data=DataConfig(
                dataset="NeelNanda/pile-10k", split="train[:8]", truncation=True
            ),
            token_batch_size=512,
            precision="fp32",
            filter_modules="embed_out",
        ),
        ScoreConfig(batch_size=64),
        PreprocessConfig(),
        LissaConfig(
            query=QuerySetConfig(
                dataset="NeelNanda/pile-10k",
                split="train[8:10]",
                truncation=True,
                aggregation="none",
            ),
            depth=4,
            repeats=1,
            batch_size=2,
        ),
    )
    scores = load_scores(run / "scores")[:]
    assert scores.shape == (8, 2)
    assert np.isfinite(scores).all()
