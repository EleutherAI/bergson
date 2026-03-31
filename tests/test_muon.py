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


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("eps_root", [0.0, 1e-8])
def test_muon_1d_matches_adamw(weight_decay, eps_root):
    """1D params should match torchopt.adamw to within float32 precision."""
    torch.manual_seed(0)
    b_init = torch.randn(8)
    X = torch.randn(16, 8)
    Y = torch.randn(16, 1)
    lr = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8

    # Reference: torchopt.adamw (decoupled weight decay, supports eps_root)
    b_ref = b_init.clone().requires_grad_(True)
    ref_opt = torchopt.adamw(
        lr=lr, betas=betas, eps=eps, eps_root=eps_root, weight_decay=weight_decay
    )
    ref_state = ref_opt.init({"b": b_ref})

    params = {"b": b_init.clone().requires_grad_(True)}
    opt = muon(
        lr=lr,
        weight_decay=weight_decay,
        adamw_betas=betas,
        adamw_eps=eps,
        adamw_eps_root=eps_root,
    )
    opt_state = opt.init(params)

    for _ in range(10):
        (X @ b_ref - Y.squeeze()).pow(2).mean().backward()
        ref_grads = {"b": b_ref.grad.clone()}
        ref_updates, ref_state = ref_opt.update(
            ref_grads, ref_state, params={"b": b_ref}
        )
        b_ref_new = torchopt.apply_updates({"b": b_ref}, ref_updates, inplace=False)
        b_ref = b_ref_new["b"].detach().requires_grad_(True)

        (X @ params["b"] - Y.squeeze()).pow(2).mean().backward()
        grads = {k: v.grad.clone() for k, v in params.items()}
        updates, opt_state = opt.update(grads, opt_state, params=params)
        params = torchopt.apply_updates(params, updates, inplace=False)
        params = {k: v.detach().requires_grad_(True) for k, v in params.items()}

    torch.testing.assert_close(
        params["b"],
        b_ref.data,
        atol=1e-5,
        rtol=0,
        msg="1D param mismatch vs torchopt.adamw",
    )


def test_muon_mixed_1d_2d():
    """Mixed 1D and 2D params should update without error."""
    torch.manual_seed(1)
    W = torch.randn(4, 4, requires_grad=True)
    b = torch.randn(4, requires_grad=True)

    params = {"W": W, "b": b}
    opt = muon(lr=0.01)
    opt_state = opt.init(params)

    grads = {k: torch.randn_like(v) for k, v in params.items()}
    updates, opt_state = opt.update(grads, opt_state, params=params)
    assert updates["W"].shape == W.shape
    assert updates["b"].shape == b.shape


def test_muon_rejects_3d_params():
    """Parameters with ndim > 2 should raise ValueError at init."""
    param_3d = torch.randn(2, 3, 4, requires_grad=True)
    opt = muon(lr=0.01)
    with pytest.raises(ValueError, match="more than 2 dimensions"):
        opt.init({"x": param_3d})
