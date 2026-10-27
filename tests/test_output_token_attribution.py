"""Output-loss JVPs must reconstruct the original CE and normalized DPO scores."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from bergson.config import IndexConfig, ScoreConfig
from bergson.score.output_influence import output_token_influence, query_directions
from bergson.score.output_model import chunked_attention, install_output_loss_head, register_output_attention


class CausalToy(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.embed = nn.Embedding(9, 4, dtype=dtype)
        self.proj = nn.Linear(4, 9, dtype=dtype)

    @property
    def base_model(self):
        return self

    def forward(self, x):
        # Earlier input positions affect later losses, distinguishing output
        # attribution from a layer-position gradient decomposition.
        hidden = self.embed(x).cumsum(dim=1).tanh()
        return SimpleNamespace(logits=self.proj(hidden))


def fixture(dtype, objective):
    torch.manual_seed(42)
    model = CausalToy(dtype)
    x = torch.randint(0, 9, (4, 7))
    y = x.clone()
    y[:, :2] = -100
    y[1, -2:] = -100
    y[3, -1:] = -100
    block = torch.randn(9*5, 2, dtype=dtype)
    directions = query_directions(model, {'proj': block}, {'proj': (torch.device('cpu'), model.proj.weight.shape, True)})
    cfg = IndexConfig(run_path='/unused', loss_fn=objective, loss_reduction='mean', dpo_beta=5)
    refs = torch.tensor([-2, -1.7, -2.1, -2.2], dtype=dtype)
    return model, x, y, block, directions, cfg, refs


def direct_terms(model, x, y):
    logits = model(x).logits[:, :-1]
    labels = y[:, 1:]
    mask = labels != -100
    ce = F.cross_entropy(logits.reshape(-1, 9), labels.reshape(-1), reduction='none').reshape_as(labels)
    return ce/mask.sum(dim=1, keepdim=True), mask


def dot_gradient(loss, model, column):
    weight, bias = torch.autograd.grad(loss, (model.proj.weight, model.proj.bias), retain_graph=True)
    return torch.cat((weight, bias[:, None]), dim=1).reshape(-1).dot(column)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('objective', ['ce', 'dpo'])
def test_token_and_pair_scores_match_direct_autograd(dtype, objective):
    model, x, y, block, directions, cfg, refs = fixture(dtype, objective)
    rates, losses = output_token_influence(model, x, y, directions, cfg, reference_logps=refs)
    terms, mask = direct_terms(model, x, y)
    factor = torch.ones(4, dtype=dtype)
    if objective == 'dpo':
        logps = -terms.sum(dim=1)
        margin = 5*((logps[:2]-refs[:2])-(logps[2:]-refs[2:]))
        expected_losses = F.softplus(-margin)
        f = (5*torch.sigmoid(-margin)).detach()
        factor = torch.cat((f, -f))
        pair_rates = rates[:2].sum(dim=1)+rates[2:].sum(dim=1)
    else:
        expected_losses = terms.sum(dim=1)
        pair_rates = rates.sum(dim=1)
    tolerance = dict(rtol=5e-5, atol=2e-5) if dtype == torch.float32 else dict(rtol=2e-10, atol=2e-10)
    assert rates.dtype == losses.dtype == dtype
    assert rates.shape == (4, 6, 2)
    assert torch.count_nonzero(rates[~mask]) == 0
    torch.testing.assert_close(losses, expected_losses, **tolerance)
    for column in range(2):
        for example, loss in enumerate(expected_losses):
            torch.testing.assert_close(pair_rates[example, column], dot_gradient(loss, model, block[:, column]), **tolerance)
        for sequence in range(4):
            for position in range(6):
                target = terms[sequence, position]*factor[sequence]
                torch.testing.assert_close(rates[sequence, position, column], dot_gradient(target, model, block[:, column]), **tolerance)


def test_fp64_query_cannot_be_silently_downcast():
    model, x, y, _, directions, cfg, refs = fixture(torch.float32, 'dpo')
    directions = [(block.double(), name, bias, transposed) for block, name, bias, transposed in directions]
    with pytest.raises(ValueError, match='preserve FP64'):
        output_token_influence(model, x, y, directions, cfg, reference_logps=refs)


def test_output_mode_config_roundtrip(tmp_path):
    cfg = ScoreConfig(token_influence='output')
    cfg.save_yaml(tmp_path/'score.yaml')
    assert ScoreConfig.load(tmp_path/'score.yaml').token_influence == 'output'


def test_dpo_rejects_missing_reference_and_mismatched_pair_weights():
    model, x, y, _, directions, cfg, refs = fixture(torch.float64, 'dpo')
    with pytest.raises(ValueError, match='frozen references'):
        output_token_influence(model, x, y, directions, cfg)
    with pytest.raises(ValueError, match='weights must match'):
        output_token_influence(model, x, y, directions, cfg, [1, 1, 2, 1], reference_logps=refs)


@pytest.mark.parametrize('window', [None, 3])
def test_chunked_attention_matches_causal_and_sliding_reference(window):
    torch.manual_seed(5)
    q = torch.randn(2, 4, 9, 8, dtype=torch.float64)
    k = torch.randn(2, 2, 9, 8, dtype=torch.float64)
    v = torch.randn_like(k)
    module = SimpleNamespace(_output_query_chunk=3)
    actual, _ = chunked_attention(module, q, k, v, scaling=8**-0.5, sliding_window=window)
    p = torch.arange(9)
    mask = p[None, :] <= p[:, None]
    if window is not None:
        mask &= p[None, :] > p[:, None]-window
    expected = F.scaled_dot_product_attention(q, k.repeat_interleave(2, dim=1),
                                             v.repeat_interleave(2, dim=1), attn_mask=mask).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


def test_real_olmo_lora_chunked_jvp_and_per_row_lengths():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Olmo3Config, Olmo3ForCausalLM
    from bergson.collector.collector import HookCollectorBase
    from bergson.utils.worker_utils import extract_peft_target_modules

    torch.manual_seed(17)
    register_output_attention()
    config = Olmo3Config(vocab_size=23, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         pad_token_id=0, eos_token_id=1, sliding_window=4,
                         layer_types=['full_attention', 'sliding_attention'])
    config._attn_implementation = 'bergson_output_jvp'
    model = get_peft_model(Olmo3ForCausalLM(config).double(), LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, lora_dropout=0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    model.eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if '.lora_B.' in name:
                parameter.normal_(std=.01)
    targets = extract_peft_target_modules(model)
    info = HookCollectorBase.discover_targets(model.base_model, targets)
    blocks = {name: torch.randn(shape.numel(), 1, dtype=torch.float64) for name, (_, shape, _) in info.items()}
    directions = query_directions(model, blocks, info)
    install_output_loss_head(model, chunk_rows=3)
    x = torch.randint(2, 23, (2, 9))
    y = x.clone()
    y[:, :3] = -100
    y[1, -2:] = -100
    cfg = IndexConfig(run_path='/unused', loss_fn='dpo', loss_reduction='mean', dpo_beta=5)
    refs = torch.tensor([-2., -2.1], dtype=torch.float64)
    rates, losses = output_token_influence(model, x, y, directions, cfg, reference_logps=refs, sequence_lengths=[9, 7])
    # Outside the scoped output request the head returns full ordinary logits.
    logits = model(x, use_cache=False).logits[:, :-1]
    ce = F.cross_entropy(logits.reshape(-1, 23), y[:, 1:].reshape(-1), reduction='none').reshape(2, 8)
    lp = -ce.sum(dim=1)/(y[:, 1:] != -100).sum(dim=1)
    loss = F.softplus(-5*((lp[0]-refs[0])-(lp[1]-refs[1])))
    named = dict(model.named_parameters())
    names = [name for _, name, _, _ in directions]
    gradients = torch.autograd.grad(loss, [named[name] for name in names])
    expected = sum((gradient.flatten()*block[:, 0]).sum() for gradient, (block, _, _, _) in zip(gradients, directions))
    torch.testing.assert_close(losses[0], loss, rtol=2e-10, atol=2e-10)
    torch.testing.assert_close(rates.sum(), expected, rtol=2e-9, atol=2e-9)
