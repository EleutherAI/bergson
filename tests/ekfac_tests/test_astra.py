import pytest
import torch
from datasets import Dataset
from torch import Tensor
from torch.func import functional_call, jacrev
from transformers import GPT2Config, GPT2LMHeadModel

from bergson.config import AstraConfig, IndexConfig
from bergson.hessians.astra import Astra, GaussNewtonProduct
from bergson.utils.logger import get_logger


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

    theta = torch.cat(
        [
            product._to_flat(n, {k: product.params[k] for k in product.params})
            for n in names
        ]
    )
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
