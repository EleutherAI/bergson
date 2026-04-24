"""Smoke test for the ``bergson compressed_ekfac`` CLI + orchestrator.

Mirrors :func:`tests.test_build.test_build_e2e` — runs the CLI end-to-end
on pythia-14m / pile-10k[:100] and asserts the on-disk layout is what
downstream code (and the two-stage retrieval notebook) expects.
"""

import subprocess
from pathlib import Path

import pytest
import torch

from bergson import GradientProcessor
from bergson.data import load_gradients
from bergson.preconditioners import _detect_variant


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compressed_ekfac_e2e(tmp_path: Path):
    run_path = tmp_path / "ce_run"

    result = subprocess.run(
        [
            "python",
            "-m",
            "bergson",
            "compressed_ekfac",
            str(run_path),
            "--model",
            "EleutherAI/pythia-14m",
            "--dataset",
            "NeelNanda/pile-10k",
            "--split",
            "train[:100]",
            "--truncation",
            # projection_dim=64 hits the Kronecker-JL floor for pythia-14m's
            # module sizes (max O*I ≈ 65k, sqrt ≈ 256 ⇒ p ≥ ~128 would be
            # ideal, p=64 is the smallest that gave PASS in
            # `scripts/validate_compressed_ekfac.py`). A smaller p would still
            # smoke-cleanly but would silently retrieve noise — see §18 of
            # COMPRESSED_EKFAC_PLAN.md.
            "--projection_dim",
            "64",
            "--token_batch_size",
            "1024",
            "--precision",
            "bf16",
            "--nproc_per_node",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"CLI exited non-zero:\nstdout:\n{result.stdout}\n\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Error" not in result.stderr, f"Error found in stderr:\n{result.stderr}"

    # Step 1 artifacts: EKFAC factor shards for the kfac method.
    hessian_method_path = run_path / "hessian" / "kfac"
    assert hessian_method_path.is_dir(), (
        f"Missing Hessian artifacts at {hessian_method_path}"
    )
    # The new detection must see this as EKFAC (ev_correction was forced on).
    assert _detect_variant(hessian_method_path) == "ekfac"

    # Step 2 artifacts: a compressed gradient index.
    index_path = run_path / "index"
    assert index_path.is_dir(), f"Missing compressed index at {index_path}"

    # Load the index and check we got gradients for a non-empty set of
    # modules, with the right projected dimension per module.
    index = load_gradients(index_path)
    module_names = index.dtype.names
    assert module_names, "Compressed index has no modules"

    # Each module's gradient row should have length == projection_dim**2,
    # since the builder applies a double-sided [p, p] random projection after
    # baking EKFAC in, and flattens to [p*p].
    for name in module_names:
        per_row_len = index[name].shape[-1]
        assert per_row_len == 64 * 64, (
            f"Module {name!r}: expected projection_dim**2=4096, got {per_row_len}"
        )

    # The GradientProcessor dump saved by build should exist. Because
    # skip_preconditioners=True was forced in the orchestrator, the
    # processor's autocorrelation preconditioners dict must be empty —
    # preconditioning came from the EKFAC factors, not this dump.
    processor = GradientProcessor.load(index_path)
    assert len(processor.preconditioners) == 0, (
        "compressed_ekfac should not fit an autocorrelation preconditioner "
        "at build time; EKFAC is baked in via preconditioner_path."
    )
