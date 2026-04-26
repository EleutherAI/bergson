"""Smoke test for the ``bergson compressed_ekfac`` CLI + orchestrator.

Mirrors :func:`tests.test_build.test_build_e2e` — runs the CLI end-to-end
on pythia-14m / pile-10k[:100] and asserts the on-disk layout is what
downstream code (and the two-stage retrieval notebook) expects.

Also covers the gap-fill cases from §19 of COMPRESSED_EKFAC_PLAN.md:
* ``test_compressed_ekfac_resume`` — second invocation with ``resume=True``
  skips already-built steps (§19.5).
* ``test_load_preconditioner_rejects_non_kfac_method`` — defensive gate
  on tkfac/shampoo factor layouts (§19.3).
"""

import json
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from bergson import GradientProcessor
from bergson.config import HessianConfig, IndexConfig, PreprocessConfig
from bergson.data import load_gradients
from bergson.preconditioners import _detect_variant, load_preconditioner


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compressed_ekfac_resume(tmp_path: Path):
    """Second invocation with ``resume=True`` skips both pipeline steps.

    Closes §19.5 of COMPRESSED_EKFAC_PLAN.md."""
    from bergson.config import DataConfig
    from bergson.hessians.compressed_ekfac import compressed_ekfac_pipeline

    run_path = tmp_path / "ce_resume"
    index_cfg = IndexConfig(
        run_path=str(run_path),
        model="EleutherAI/pythia-14m",
        precision="bf16",
        projection_dim=64,
        token_batch_size=1024,
        skip_preconditioners=True,
        data=DataConfig(
            dataset="NeelNanda/pile-10k", split="train[:50]", truncation=True,
        ),
        debug=True,  # determinism — second run must produce identical artifacts
    )
    index_cfg.distributed.nproc_per_node = 1
    hessian_cfg = HessianConfig(method="kfac", ev_correction=True)
    preprocess_cfg = PreprocessConfig(unit_normalize=True)

    # First run: produces hessian + index dirs.
    out = compressed_ekfac_pipeline(index_cfg, hessian_cfg, preprocess_cfg)
    assert out == run_path / "index"
    assert (run_path / "hessian" / "kfac").is_dir()
    assert (run_path / "index").is_dir()
    first_hessian_mtime = (run_path / "hessian" / "kfac").stat().st_mtime
    first_index_mtime = (run_path / "index").stat().st_mtime

    # Second run with resume=True: must NOT touch existing dirs.
    compressed_ekfac_pipeline(index_cfg, hessian_cfg, preprocess_cfg, resume=True)
    assert (run_path / "hessian" / "kfac").stat().st_mtime == first_hessian_mtime
    assert (run_path / "index").stat().st_mtime == first_index_mtime


def test_load_preconditioner_rejects_non_kfac_method(tmp_path: Path):
    """Factored-preconditioner directories with method!=kfac raise.

    Closes §19.3: tkfac/shampoo write the same on-disk layout as kfac, so
    detection alone can't distinguish them. Only kfac has been validated
    end-to-end against a reference. ``load_preconditioner`` reads
    ``hessian_config.yaml`` and refuses to load until each method is
    individually validated."""
    # Synthesize a minimal "tkfac" EKFAC artifact: two eigenvector dirs +
    # one eigenvalue-correction dir + a hessian_config.yaml claiming tkfac.
    fake_tensor = {"layers.0.dummy": torch.eye(4)}
    for sub in (
        "eigen_activation_sharded",
        "eigen_gradient_sharded",
        "eigenvalue_correction_sharded",
    ):
        (tmp_path / sub).mkdir(parents=True)
        save_file(fake_tensor, str(tmp_path / sub / "shard_0.safetensors"))
    (tmp_path / "hessian_config.yaml").write_text(
        json.dumps({"method": "tkfac", "ev_correction": True})
    )

    with pytest.raises(NotImplementedError, match="tkfac"):
        load_preconditioner(tmp_path, device=torch.device("cpu"), power=-1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compressed_ekfac_token_attribution(tmp_path: Path):
    """End-to-end ``compressed_ekfac`` with ``attribute_tokens=True``.

    Closes §19.2 of COMPRESSED_EKFAC_PLAN.md: the per-token path was unit-
    tested at ``EkfacPreconditioner.apply`` shape level (commit 2's
    ``test_ekfac_preconditioner_per_token_shape``) but never run end-to-end
    through the orchestrator + Builder's ``_scatter_flat_tokens`` path.
    This test exercises that chain on a tiny pile-10k slice."""
    import json as _json

    from bergson.config import DataConfig
    from bergson.data import load_token_gradients
    from bergson.hessians.compressed_ekfac import compressed_ekfac_pipeline

    run_path = tmp_path / "ce_tokens"
    p = 16  # tiny — pile-10k[:30] generates only a handful of total tokens
    index_cfg = IndexConfig(
        run_path=str(run_path),
        model="EleutherAI/pythia-14m",
        precision="bf16",
        projection_dim=p,
        token_batch_size=1024,
        skip_preconditioners=True,
        attribute_tokens=True,  # the trigger
        data=DataConfig(
            dataset="NeelNanda/pile-10k", split="train[:30]", truncation=True,
        ),
    )
    index_cfg.distributed.nproc_per_node = 1
    hessian_cfg = HessianConfig(method="kfac", ev_correction=True)
    preprocess_cfg = PreprocessConfig(unit_normalize=True)

    out = compressed_ekfac_pipeline(index_cfg, hessian_cfg, preprocess_cfg)

    # Layout sanity: info.json declares the per-token layout.
    with (out / "info.json").open() as f:
        info = _json.load(f)
    assert info["attribute_tokens"] is True, info
    assert info["total_tokens"] > 30, (
        f"Per-token index should have more rows than the 30 input docs, "
        f"got total_tokens={info['total_tokens']}"
    )
    assert info["total_grad_dim"] > 0

    # Reload and verify shape: [total_tokens, n_modules * p²].
    mmap, num_token_grads, offsets = load_token_gradients(out)
    assert len(num_token_grads) == 30
    assert int(offsets[-1]) == int(info["total_tokens"]) == mmap.shape[0]
    # Each module contributes p² to the per-row dim; pythia-14m has 24
    # tracked modules, so the row width should be 24 * p² = 24 * 256 = 6144.
    assert mmap.shape[1] == 24 * (p * p), mmap.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compressed_ekfac_rejects_include_bias(tmp_path: Path):
    """``include_bias=True`` + factored preconditioner raises in build_worker.

    Closes §19.4. The factored Q_A is sized [I, I] but with bias the
    gradient has an extra column [I+1]; safest default is to refuse rather
    than silently produce wrong output."""
    from bergson.config import DataConfig
    from bergson.hessians.compressed_ekfac import compressed_ekfac_pipeline

    run_path = tmp_path / "ce_bias"
    index_cfg = IndexConfig(
        run_path=str(run_path),
        model="EleutherAI/pythia-14m",
        precision="bf16",
        projection_dim=64,
        token_batch_size=1024,
        skip_preconditioners=True,
        include_bias=True,  # the trigger
        data=DataConfig(
            dataset="NeelNanda/pile-10k", split="train[:20]", truncation=True,
        ),
    )
    index_cfg.distributed.nproc_per_node = 1
    hessian_cfg = HessianConfig(method="kfac", ev_correction=True)
    preprocess_cfg = PreprocessConfig(unit_normalize=True)

    # The hessian-fit step (step 1) currently allows include_bias; the
    # factored-preconditioner guard fires in step 2 (the build worker).
    # Match against the marker text in the error so the test pins the
    # specific guard, not just any error.
    with pytest.raises(NotImplementedError, match="include_bias"):
        compressed_ekfac_pipeline(index_cfg, hessian_cfg, preprocess_cfg)
