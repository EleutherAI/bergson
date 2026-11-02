from dataclasses import replace

import pytest
import torch
from datasets import Dataset
from torch import Tensor
from torch.func import functional_call, jacrev
from transformers import GPT2Config, GPT2LMHeadModel

from bergson.config import (
    AstraConfig,
    DataConfig,
    DistributedConfig,
    HessianConfig,
    HessianPipelineConfig,
    IndexConfig,
    PreprocessConfig,
    QuerySetConfig,
    ScoreConfig,
)
from bergson.data import load_gradients
from bergson.hessians.astra import Astra, GaussNewtonProduct
from bergson.hessians.pipeline import _run_astra, hessian_pipeline
from bergson.utils.logger import get_logger

requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA devices",
)


@pytest.mark.parametrize("loss_reduction", [None, "mean"])
def test_gauss_newton_product_matches_explicit_jacobian(loss_reduction):
    """``H v`` equals ``J^T (diag(p) - p p^T) J v`` built from the full Jacobian,
    in the query index's ``[O, I + 1]`` layout of HF Conv1D layers."""
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
    product = GaussNewtonProduct(
        model, data, IndexConfig(run_path="", include_bias=True), names, loss_reduction
    )
    sizes = {"h.0.attn.c_proj": 8 * 9, "h.0.mlp.c_fc": 32 * 9}
    v = {n: torch.randn(s, dtype=torch.float64) for n, s in sizes.items()}
    indices = [1, 4]

    def logits(flat):
        params, start = {}, 0
        for n in names:
            params.update(product._to_params(n, flat[start : start + sizes[n]]))
            start += sizes[n]
        x = torch.tensor(data[indices]["input_ids"])
        return functional_call(model, params, (x,)).logits[:, :-1]

    params = dict(model.named_parameters())
    theta = torch.cat([product._to_flat(n, params) for n in names])
    jac = jacrev(logits)(theta).flatten(0, 1)
    probs = torch.softmax(logits(theta), -1).flatten(0, 1)
    out_hessian = torch.diag_embed(probs) - probs[:, :, None] * probs[:, None, :]
    expected = torch.einsum("tvp,tvw,twq->pq", jac, out_hessian, jac) @ torch.cat(
        list(v.values())
    )
    expected /= len(indices) * (4 if loss_reduction == "mean" else 1)

    actual = product(v, indices)
    torch.testing.assert_close(torch.cat([actual[n] for n in names]), expected)

    product.micro_batch_size = 1
    chunked = product(v, indices)
    torch.testing.assert_close(torch.cat([chunked[n] for n in names]), expected)


class _Identity:
    def apply(self, r):
        return r


def test_cg_reaches_the_damped_solution():
    """On the full batch, ``cg`` converges to ``(H + D)^-1 q``."""
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
    sizes = {"h.0.attn.c_proj": 8 * 8, "h.0.mlp.c_fc": 32 * 8}
    hvp = GaussNewtonProduct(model, data, IndexConfig(run_path=""), names)
    docs = list(range(len(data)))

    def dense(v: Tensor) -> dict[str, Tensor]:
        out, start = {}, 0
        for n in names:
            out[n] = v[start : start + sizes[n]]
            start += sizes[n]
        return out

    dim = sum(sizes.values())
    h = torch.stack(
        [torch.cat(list(hvp(dense(e), docs).values())) for e in torch.eye(dim).double()]
    )
    damping = h.diagonal().mean().item()

    astra = Astra.__new__(Astra)
    astra.cfg = AstraConfig(num_steps=60, batch_size=len(data), solver="cg")
    astra.names, astra.hvp, astra.num_docs = names, hvp, len(data)
    astra.damping = {n: damping for n in names}
    astra.preconditioner = _Identity()
    astra.logger = get_logger("Astra")

    q = torch.randn(dim, dtype=torch.float64)
    expected = torch.linalg.solve(h + damping * torch.eye(dim), q)
    x = astra.refine(dense(q), dense(torch.zeros(dim, dtype=torch.float64)), 0)
    actual = torch.cat([x[n] for n in names])
    assert (actual - expected).norm() / expected.norm() < 1e-6


@pytest.fixture(scope="module")
def two_gpu_kfac_run(tmp_path_factory):
    """Queries and their K-FAC solutions for Pythia-14m, fitted on two GPUs."""
    root = tmp_path_factory.mktemp("astra_fsdp")
    ids = torch.randint(10, 1000, (32, 64), generator=torch.Generator().manual_seed(0))
    Dataset.from_dict({"input_ids": ids.tolist()}).save_to_disk(str(root / "data"))
    index_cfg = IndexConfig(
        run_path=str(root / "run"),
        model="EleutherAI/pythia-14m",
        data=DataConfig(dataset=str(root / "data")),
        token_batch_size=256,
        precision="fp32",
        distributed=DistributedConfig(nproc_per_node=2),
    )
    hessian_cfg = HessianConfig(method="kfac", ev_correction=True)
    hessian_pipeline(
        index_cfg,
        hessian_cfg,
        ScoreConfig(batch_size=256),
        PreprocessConfig(),
        HessianPipelineConfig(
            query=QuerySetConfig(
                dataset=str(root / "data"), split="train[:4]", aggregation="none"
            ),
        ),
    )
    return index_cfg, hessian_cfg


@requires_multi_gpu
@pytest.mark.parametrize("solver", ["sgd", "cg"])
def test_astra_under_fsdp_matches_unsharded(two_gpu_kfac_run, solver):
    """ASTRA over FSDP-sharded weights refines the queries to the solutions it
    reaches with each rank holding the whole model, with the ranks' shares of a
    5-document batch split unevenly."""
    index_cfg, hessian_cfg = two_gpu_kfac_run
    run = index_cfg.run_path
    pipeline_cfg = HessianPipelineConfig(
        astra=AstraConfig(
            num_steps=4, solver=solver, batch_size=5, micro_batch_size=2, lr=1e-4
        ),
    )
    solutions = {}
    for fsdp in (False, True):
        out = f"{run}/astra_{solver}_{fsdp}"
        _run_astra(
            replace(index_cfg, fsdp=fsdp),
            hessian_cfg,
            pipeline_cfg,
            f"{run}/query",
            f"{run}/kfac_query",
            out,
        )
        solutions[fsdp] = torch.from_numpy(load_gradients(out).astype("float32"))

    init = torch.from_numpy(load_gradients(f"{run}/kfac_query").astype("float32"))
    moved = (solutions[False] - init).norm()
    assert moved > 0
    # The ranks sum the documents in another order, which moves a product
    # about as much as changing its micro-batch size does
    assert (solutions[True] - solutions[False]).norm() < 1e-2 * moved


@requires_multi_gpu
@pytest.mark.parametrize("solver", ["sgd", "cg"])
@pytest.mark.parametrize("fsdp", [False, True])
def test_astra_factors_on_cpu_matches_gpu_factors(two_gpu_kfac_run, solver, fsdp):
    """Holding the factors in CPU memory doesn't change the solutions."""
    index_cfg, hessian_cfg = two_gpu_kfac_run
    run = index_cfg.run_path
    solutions = {}
    for on_cpu in (False, True):
        out = f"{run}/astra_on_cpu_{solver}_{fsdp}_{on_cpu}"
        _run_astra(
            replace(index_cfg, fsdp=fsdp),
            hessian_cfg,
            HessianPipelineConfig(
                astra=AstraConfig(
                    num_steps=4,
                    solver=solver,
                    batch_size=5,
                    lr=1e-4,
                    factors_on_cpu=on_cpu,
                ),
            ),
            f"{run}/query",
            f"{run}/kfac_query",
            out,
        )
        solutions[on_cpu] = torch.from_numpy(load_gradients(out).astype("float32"))

    init = torch.from_numpy(load_gradients(f"{run}/kfac_query").astype("float32"))
    moved = (solutions[False] - init).norm()
    assert moved > 0
    assert (solutions[True] - solutions[False]).norm() < 1e-5 * moved
