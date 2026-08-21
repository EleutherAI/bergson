import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel

from bergson.models.modeling_gpt2_custom import GPT2CustomConfig, GPT2CustomLMHeadModel

SMALL = dict(n_embd=32, n_head=4, n_layer=2, n_positions=64, vocab_size=97)


def small_custom(arch_mod: str) -> GPT2CustomLMHeadModel:
    torch.manual_seed(0)
    return GPT2CustomLMHeadModel(GPT2CustomConfig(arch_mod=arch_mod, **SMALL)).eval()


def test_none_mod_is_bit_identical_to_stock():
    """arch_mod='none' must be exactly stock GPT-2: same parameter set, and a
    stock checkpoint's weights produce bit-identical logits."""
    torch.manual_seed(0)
    stock = GPT2LMHeadModel(GPT2Config(**SMALL)).eval()
    custom = small_custom("none")
    assert set(custom.state_dict()) == set(stock.state_dict())
    custom.load_state_dict(stock.state_dict())
    x = torch.randint(0, SMALL["vocab_size"], (2, 16))
    with torch.no_grad():
        assert torch.equal(custom(x).logits, stock(x).logits)


@pytest.mark.parametrize(
    "mod,param_frag,n_added",
    [
        # RMSNorm is weight-only: 2 layers x (q_norm + k_norm) = 4 tensors.
        ("qk_norm", ("q_norm", "k_norm"), 4),
        # LayerNorm has weight + bias: 2 layers x 2 = 4 tensors.
        ("preact_layernorm", ("preact_ln",), 4),
    ],
)
def test_mods_add_params_change_output_and_train(mod, param_frag, n_added):
    torch.manual_seed(0)
    stock = GPT2LMHeadModel(GPT2Config(**SMALL)).eval()
    custom = small_custom(mod)
    added = [n for n in custom.state_dict() if any(f in n for f in param_frag)]
    assert len(added) == n_added, added
    # Stock weights still load (norms keep their init).
    missing, unexpected = custom.load_state_dict(stock.state_dict(), strict=False)
    assert not unexpected and all(any(f in n for f in param_frag) for n in missing)
    x = torch.randint(0, SMALL["vocab_size"], (2, 16))
    with torch.no_grad():
        assert not torch.equal(custom(x).logits, stock(x).logits)
    # Gradients reach the new norms.
    custom.train()
    loss = custom(x, labels=x).loss
    loss.backward()
    grads = [p.grad for n, p in custom.named_parameters() if any(f in n for f in param_frag)]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("mod", ["none", "qk_norm", "preact_layernorm"])
def test_save_load_roundtrip_preserves_arch_and_function(tmp_path, mod):
    m = small_custom(mod)
    x = torch.randint(0, SMALL["vocab_size"], (2, 16))
    with torch.no_grad():
        before = m(x).logits
    m.save_pretrained(tmp_path)
    m2 = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert isinstance(m2, GPT2CustomLMHeadModel)
    assert m2.config.arch_mod == mod
    with torch.no_grad():
        assert torch.equal(m2(x).logits, before)
