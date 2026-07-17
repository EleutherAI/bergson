"""Tests for the SOURCE (approx-unrolling) optimizer variants: SGD heavy-ball
momentum lr scaling and the Adam/AdamW diagonal-preconditioner eigenfunction
path (Bae et al., 2024, Appendix C / D.2).
"""

import pytest
import torch
from safetensors.torch import save_file

from bergson.approx_unrolling.approx_unrolling_math import (
    compute_lr_times_steps_per_segment,
    f_backward,
    f_segment,
)
from bergson.config import ApproxUnrollingConfig
from bergson.hessians.preconditioner import DiagonalFactoredPreconditioner

MODULE = "lm_head"
OUT_DIM, IN_DIM = 6, 4
LR_TIMES_STEPS = 0.05


def test_momentum_scales_lr_times_steps():
    """SGDm terminal velocity: lr*K scaled by 1/(1-beta)."""
    base_cfg = ApproxUnrollingConfig(
        checkpoints=["a", "b"],
        segments=2,
        lr_list=[1e-3, 2e-3],
        step_size_list=[10, 20],
    )
    momentum_cfg = ApproxUnrollingConfig(
        checkpoints=["a", "b"],
        segments=2,
        lr_list=[1e-3, 2e-3],
        step_size_list=[10, 20],
        momentum=0.9,
    )
    base = compute_lr_times_steps_per_segment(base_cfg)
    scaled = compute_lr_times_steps_per_segment(momentum_cfg)
    assert scaled == pytest.approx([10 * b for b in base])


def test_momentum_out_of_range_raises():
    for bad in (1.0, -0.1):
        bad_cfg = ApproxUnrollingConfig(
            checkpoints=["a"],
            segments=1,
            lr_list=[1e-3],
            step_size_list=[10],
            momentum=bad,
        )
        with pytest.raises(ValueError, match="momentum"):
            compute_lr_times_steps_per_segment(bad_cfg)


def _random_factors():
    """Random EKFAC-style factors: orthogonal eigenvectors, non-negative
    eigenvalue grid."""
    torch.manual_seed(0)
    q_a, _ = torch.linalg.qr(torch.randn(IN_DIM, IN_DIM, dtype=torch.float64))
    q_g, _ = torch.linalg.qr(torch.randn(OUT_DIM, OUT_DIM, dtype=torch.float64))
    lam = torch.rand(OUT_DIM, IN_DIM, dtype=torch.float64)
    return q_a.float().contiguous(), q_g.float().contiguous(), lam.float()


def _write_factor_shards(tmp_path, q_a, q_g, lam, precond):
    """Write the single-shard on-disk layout DiagonalFactoredPreconditioner
    loads from, plus the preconditioner grid."""
    for sub, tensor in [
        ("eigen_activation_sharded", q_a),
        ("eigen_gradient_sharded", q_g),
        ("eigenvalue_correction_sharded", lam),
    ]:
        (tmp_path / sub).mkdir()
        save_file({MODULE: tensor}, str(tmp_path / sub / "shard_0.safetensors"))
    precond_path = tmp_path / "precond.safetensors"
    save_file({MODULE: precond}, str(precond_path))
    return precond_path


def _reference_diag_hessian(q_a, q_g, lam):
    """diag(H) via the dense Kronecker Hessian — independent of the
    implementation's factored formula.

    The [OUT, IN] grid flattens row-major to vec index o*IN + i, matching
    kron(q_g, q_a)'s row ordering, so H = kron(Q_G, Q_A) diag(vec Λ)
    kron(Q_G, Q_A)^T.
    """
    q_kron = torch.kron(q_g.double(), q_a.double())
    h = q_kron @ torch.diag(lam.double().flatten()) @ q_kron.T
    return torch.diagonal(h).reshape(OUT_DIM, IN_DIM).float()


@pytest.mark.parametrize("fn_kind", ["f_backward", "f_segment"])
def test_diagonal_preconditioner_matches_dense_reference(tmp_path, fn_kind):
    """The diagonal path multiplies gradients elementwise by
    f(p * diag(H)) — times p for f_segment's P^1/2 F(M) P^1/2 sandwich —
    with diag(H) recovered exactly from the EKFAC factors."""
    q_a, q_g, lam = _random_factors()
    precond = torch.rand(OUT_DIM, IN_DIM) + 0.5
    precond_path = _write_factor_shards(tmp_path, q_a, q_g, lam, precond)

    fn = {"f_backward": f_backward, "f_segment": f_segment}[fn_kind](LR_TIMES_STEPS)
    preconditioner = DiagonalFactoredPreconditioner.from_shards(
        tmp_path,
        precond_path,
        rank=0,
        device="cpu",
        apply_fn=fn,
        multiply_by_precond=fn_kind == "f_segment",
        ev_correction=True,
    )

    grads = {MODULE: torch.randn(3, OUT_DIM * IN_DIM)}
    out = preconditioner.apply({k: v.clone() for k, v in grads.items()})[MODULE]

    sigma = precond * _reference_diag_hessian(q_a, q_g, lam)
    if fn_kind == "f_backward":
        multiplier = torch.exp(-LR_TIMES_STEPS * sigma)
    else:
        multiplier = precond * (-torch.expm1(-LR_TIMES_STEPS * sigma)) / sigma
    expected = grads[MODULE].view(3, OUT_DIM, IN_DIM) * multiplier
    torch.testing.assert_close(
        out.view(3, OUT_DIM, IN_DIM), expected, rtol=1e-4, atol=1e-6
    )


def test_f_segment_zero_eigenvalue_limit(tmp_path):
    """With Λ = 0 the segment multiplier hits its lr*K limit, times the
    preconditioner: lr*K * p, with no NaN/inf from the 0/0."""
    q_a, q_g, _ = _random_factors()
    lam = torch.zeros(OUT_DIM, IN_DIM)
    precond = torch.rand(OUT_DIM, IN_DIM) + 0.5
    precond_path = _write_factor_shards(tmp_path, q_a, q_g, lam, precond)

    preconditioner = DiagonalFactoredPreconditioner.from_shards(
        tmp_path,
        precond_path,
        rank=0,
        device="cpu",
        apply_fn=f_segment(LR_TIMES_STEPS),
        multiply_by_precond=True,
        ev_correction=True,
    )
    grads = {MODULE: torch.ones(1, OUT_DIM * IN_DIM)}
    out = preconditioner.apply(grads)[MODULE].view(OUT_DIM, IN_DIM)
    torch.testing.assert_close(out, LR_TIMES_STEPS * precond)


def test_precond_shape_mismatch_raises(tmp_path):
    q_a, q_g, lam = _random_factors()
    bad_precond = torch.rand(OUT_DIM + 1, IN_DIM)
    precond_path = _write_factor_shards(tmp_path, q_a, q_g, lam, bad_precond)
    with pytest.raises(ValueError, match="shape"):
        DiagonalFactoredPreconditioner.from_shards(
            tmp_path,
            precond_path,
            rank=0,
            device="cpu",
            apply_fn=f_backward(LR_TIMES_STEPS),
            ev_correction=True,
        )


def test_precond_missing_module_raises(tmp_path):
    q_a, q_g, lam = _random_factors()
    for sub, tensor in [
        ("eigen_activation_sharded", q_a),
        ("eigen_gradient_sharded", q_g),
        ("eigenvalue_correction_sharded", lam),
    ]:
        (tmp_path / sub).mkdir()
        save_file({MODULE: tensor}, str(tmp_path / sub / "shard_0.safetensors"))
    precond_path = tmp_path / "precond.safetensors"
    save_file({"other_module": torch.rand(OUT_DIM, IN_DIM)}, str(precond_path))
    with pytest.raises(KeyError, match=MODULE):
        DiagonalFactoredPreconditioner.from_shards(
            tmp_path,
            precond_path,
            rank=0,
            device="cpu",
            apply_fn=f_backward(LR_TIMES_STEPS),
            ev_correction=True,
        )


def test_build_segment_preconditioners(tmp_path):
    """Per-segment P built from the checkpoints' optimizer.pt files:
    bias-corrected via the stored step/betas, index-mapped to param names,
    suffix-matched to the factor modules, oriented to [out, in] (transposed
    storage flipped), averaged within the segment, and eps-transformed."""
    from transformers import AutoModelForCausalLM, GPT2Config

    from bergson.approx_unrolling.adam_preconditioner import (
        build_segment_preconditioners,
    )

    torch.manual_seed(0)
    tiny_cfg = GPT2Config(n_layer=1, n_embd=4, n_head=2, n_positions=8, vocab_size=16)
    module, out_dim, in_dim = "h.0.attn.c_attn", 12, 4

    run = tmp_path / "run"
    seg_kfac = run / "segment_0" / "kfac"
    seg_kfac.mkdir(parents=True)
    for sub, cols in [
        ("eigen_activation_sharded", in_dim),
        ("eigen_gradient_sharded", out_dim),
        ("eigenvalue_correction_sharded", in_dim),
    ]:
        (seg_kfac / sub).mkdir()
        save_file(
            {module: torch.rand(2, cols)}, str(seg_kfac / sub / "shard_0.safetensors")
        )

    tiny_model = AutoModelForCausalLM.from_config(tiny_cfg)
    param_names = [n for n, _ in tiny_model.named_parameters()]
    idx = param_names.index(f"transformer.{module}.weight")

    # Two checkpoints with raw (uncorrected) moments stored transposed
    # ([in, out], like GPT-2 Conv1D params), at different steps.
    beta2, eps_root, eps = 0.975, 1e-6, 1e-8
    steps, nus = [5, 9], [torch.rand(in_dim, out_dim) for _ in range(2)]
    ckpts = []
    for step, nu in zip(steps, nus):
        ckpt = tmp_path / f"models_step_{step}"
        ckpt.mkdir()
        tiny_cfg.save_pretrained(ckpt)
        # One checkpoint keyed positionally (legacy fallback), the other under
        # a bogus index with param_name recorded (the FSDP-scrambled case).
        if step == steps[0]:
            entry = {idx: {"exp_avg_sq": nu, "step": torch.tensor(step)}}
        else:
            entry = {
                999: {
                    "exp_avg_sq": nu,
                    "step": torch.tensor(step),
                    "param_name": f"transformer.{module}.weight",
                }
            }
        torch.save(
            {
                "state": entry,
                "param_groups": [
                    {
                        "params": list(entry),
                        "betas": (0.9, beta2),
                        "eps": eps,
                        "eps_root": eps_root,
                    }
                ],
            },
            ckpt / "optimizer.pt",
        )
        ckpts.append(str(ckpt))

    paths = build_segment_preconditioners(
        run_path=run,
        method="kfac",
        checkpoints=ckpts,
        segments=1,
    )
    assert paths == [run / "segment_0" / "preconditioner.safetensors"]
    from safetensors.torch import load_file

    precond = load_file(str(paths[0]))[module]
    v_hats = [nu / (1 - beta2**step) for step, nu in zip(steps, nus)]
    v_bar = (v_hats[0] + v_hats[1]).T / 2
    expected = 1.0 / ((v_bar + eps_root).sqrt() + eps)
    assert precond.shape == (out_dim, in_dim)
    torch.testing.assert_close(precond, expected)


def test_optimizer_pt_snapshot_fields(tmp_path):
    """save_second_moments_as_optimizer_pt stores the standard step and betas
    fields when given, making snapshot exports self-describing for the SOURCE
    Adam variant's bias correction."""
    import torch.nn as nn
    import torchopt

    from bergson.utils.load_from_optimizer import save_second_moments_as_optimizer_pt

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blk = nn.Linear(3, 2, bias=False)
            self.head = nn.Linear(2, 4, bias=False)

    model = TinyModel()
    params = dict(model.named_parameters(remove_duplicate=False))
    opt_state = torchopt.adamw(1e-3, betas=(0.9, 0.975)).init(params)
    adam_state = next(s for s in opt_state if hasattr(s, "nu"))
    for nu, count in zip(adam_state.nu, adam_state.count):
        nu.copy_(torch.rand_like(nu))
        count.fill_(7)

    path = tmp_path / "step_7.optimizer.pt"
    n = save_second_moments_as_optimizer_pt(
        model, opt_state, path, step=7, betas=(0.9, 0.975), eps=1e-8, eps_root=1e-6
    )
    assert n == 2
    optimizer_pt = torch.load(path, weights_only=False)
    assert optimizer_pt["param_groups"][0]["betas"] == (0.9, 0.975)
    assert optimizer_pt["param_groups"][0]["eps"] == 1e-8
    assert optimizer_pt["param_groups"][0]["eps_root"] == 1e-6
    names = [n for n, _ in model.named_parameters()]
    # nu lists are in sorted(params) order; blk.weight sorts before head.weight.
    for idx, entry in optimizer_pt["state"].items():
        assert int(entry["step"].item()) == 7
        # param_name recorded per entry: FSDP-scrambled indices stay readable.
        assert entry["param_name"] == names[idx]
        nu_idx = sorted(params).index(names[idx])
        torch.testing.assert_close(entry["exp_avg_sq"], adam_state.nu[nu_idx])
