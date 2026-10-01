import pytest
import torch
from datasets import Dataset
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


def test_scaled_start_has_the_lowest_objective_of_its_multiples():
    """No other multiple of the starting point has a lower objective."""
    h = torch.tensor([[2.0, 0.5], [0.5, 3.0]])
    astra = object.__new__(Astra)
    astra.names = ["layer"]
    astra.num_docs = 4
    astra.cfg = AstraConfig(batch_size=2, seed=0)
    astra.damping = {"layer": torch.tensor(0.25)}
    astra.logger = get_logger("test")
    astra.hvp = lambda v, indices: {"layer": h @ v["layer"]}

    q = {"layer": torch.tensor([1.0, -2.0])}
    x = {"layer": torch.tensor([3.0, -4.0])}
    start = x["layer"].clone()
    astra._scale_start(q, x, row=0)

    def objective(s):
        v = s * start
        return v @ (h @ v + astra.damping["layer"] * v) / 2 - v @ q["layer"]

    scale = (x["layer"] / start)[0]
    grid = torch.linspace(scale - 0.5, scale + 0.5, 101)
    assert objective(scale) <= min(objective(s) for s in grid)
