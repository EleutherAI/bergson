import torch
from datasets import Dataset
from torch.func import functional_call, jacrev
from transformers import GPT2Config, GPT2LMHeadModel

from bergson.config import IndexConfig
from bergson.hessians.astra import GaussNewtonProduct


def test_gauss_newton_product_matches_explicit_jacobian():
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
        model, data, IndexConfig(run_path="", include_bias=True), names
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
    expected *= len(data) / len(indices)

    actual = product(v, indices)
    torch.testing.assert_close(torch.cat([actual[n] for n in names]), expected)
