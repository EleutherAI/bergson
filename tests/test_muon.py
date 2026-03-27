"""Tests for the bergson functional Muon optimizer."""

import pytest
import torch
import torchopt

from bergson.magic import muon


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("momentum", [0.95, 0.9])
def test_muon_matches_torch_optim(weight_decay, momentum):
    """Functional muon should match torch.optim.Muon to within bfloat16 precision."""
    torch.manual_seed(42)
    W_init = torch.randn(8, 4)
    X = torch.randn(16, 4)
    Y = torch.randn(16, 8)
    lr = 0.01

    # Reference: torch.optim.Muon
    W_ref = W_init.clone().requires_grad_(True)
    ref_opt = torch.optim.Muon(
        [W_ref],
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        adjust_lr_fn="match_rms_adamw",
    )

    # Ours: bergson functional muon
    params = {"W": W_init.clone().requires_grad_(True)}
    opt = muon(lr=lr, momentum=momentum, weight_decay=weight_decay)
    opt_state = opt.init(params)

    for _ in range(10):
        ref_opt.zero_grad()
        ((X @ W_ref.T) - Y).pow(2).mean().backward()
        ref_opt.step()

        ((X @ params["W"].T) - Y).pow(2).mean().backward()
        grads = {k: v.grad.clone() for k, v in params.items()}
        updates, opt_state = opt.update(grads, opt_state, params=params)
        params = torchopt.apply_updates(params, updates, inplace=False)
        params = {k: v.detach().requires_grad_(True) for k, v in params.items()}

    # bfloat16 NS introduces ~1e-4 numerical noise per step; atol=1e-3 is well within
    # bfloat16 precision while still catching any real algorithmic divergence.
    torch.testing.assert_close(
        params["W"], W_ref.data, atol=1e-3, rtol=0, msg="Mismatch vs torch.optim.Muon"
    )
