import torch
import torch.nn.functional as F

from bergson.magic.fsdp import _shardable_attention, _tp_kind
from bergson.magic.tensor_parallel import _FrozenLinear, _linear


def test_frozen_linear_matches_linear_to_second_order():
    torch.manual_seed(0)
    w = torch.randn(5, 7, dtype=torch.float32)
    x = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)

    def narrow(x):
        return _FrozenLinear.apply(x, w).tanh()

    def wide(x):
        return F.linear(x, w.double()).tanh()

    assert torch.allclose(narrow(x), wide(x))
    assert torch.autograd.gradcheck(narrow, (x,))
    assert torch.autograd.gradgradcheck(narrow, (x,))

    v = torch.randn(3, 5, dtype=torch.float64)
    hvps = []
    for fn in (narrow, wide):
        (g,) = torch.autograd.grad(fn(x), x, v, create_graph=True)
        (h,) = torch.autograd.grad(g.pow(2).sum(), x)
        hvps.append(h)
    assert torch.allclose(*hvps)


def test_linear_only_takes_the_narrow_path_for_frozen_weights():
    x = torch.randn(2, 4)
    w = torch.randn(3, 4, dtype=torch.bfloat16)
    b = torch.randn(3)
    assert torch.equal(_linear(x, w, b), F.linear(x, w.float(), b))
    same = torch.randn(3, 4)
    assert torch.equal(_linear(x, same, b), F.linear(x, same, b))


def test_tp_kind_routes_lora_halves_around_sharded_dims():
    attn = {"model.layers.0.self_attn"}
    kinds = {
        "model.layers.0.mlp.gate_proj.base_layer.weight": "col_local",
        "model.layers.0.mlp.gate_proj.lora_A.default.weight": "col",
        "model.layers.0.mlp.gate_proj.lora_B.default.weight": "col_local",
        "model.layers.0.mlp.down_proj.base_layer.weight": "row",
        "model.layers.0.mlp.down_proj.lora_A.default.weight": "row",
        "model.layers.0.mlp.down_proj.lora_B.default.weight": "col",
        "model.layers.0.self_attn.q_proj.weight": "col_local",
        "model.layers.0.self_attn.o_proj.lora_A.default.weight": "row",
        "model.layers.1.self_attn.q_proj.weight": "col",
        "model.lm_head.weight": "col",
    }
    for path, kind in kinds.items():
        assert _tp_kind(path, attn) == kind, path


class _Attention(torch.nn.Module):
    def __init__(self, heads, kv_heads, head_dim, q_norm=False):
        super().__init__()
        hidden = heads * head_dim
        self.head_dim = head_dim
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.k_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)
        if q_norm:
            self.q_norm = torch.nn.LayerNorm(hidden)


def test_attention_is_head_sharded_only_when_slices_are_whole_heads():
    model = torch.nn.ModuleDict(
        {
            "ok": _Attention(8, 4, 2),
            "ragged_kv": _Attention(8, 2, 2),
            "normed": _Attention(8, 4, 2, q_norm=True),
        }
    )
    assert _shardable_attention(model, 4) == {"ok"}
    assert _shardable_attention(model, 2) == {"ok", "ragged_kv"}
