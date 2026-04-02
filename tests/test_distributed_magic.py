"""Test that FSDP and DDP produce equivalent MAGIC attribution scores.

Runs the CLI's run_magic twice (once FSDP, once DDP) with a tiny model
and asserts the resulting scores match.

Requires at least 2 CUDA devices.
"""

import tempfile

import pytest
import torch

from bergson.config import DataConfig, DistributedConfig
from bergson.magic.cli import MagicConfig, run_magic


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA devices",
)
def test_fsdp_ddp_scores_match():
    """FSDP and DDP should produce equivalent attribution scores."""
    world_size = min(torch.cuda.device_count(), 4)

    data = DataConfig(
        dataset="Salesforce/wikitext",
        subset="wikitext-2-raw-v1",
        split="train[:1024]",
        chunk_length=32,
    )
    dist_cfg = DistributedConfig(nproc_per_node=world_size)

    with tempfile.TemporaryDirectory() as tmpdir:
        ddp_cfg = MagicConfig(
            run_path=f"{tmpdir}/ddp",
            model="trl-internal-testing/tiny-Phi3ForCausalLM",
            fsdp=False,
            data=data,
            query=data,
            batch_size=8,
            num_epochs=1,
            overwrite=True,
            num_subsets=2,
            distributed=dist_cfg,
        )
        fsdp_cfg = MagicConfig(
            run_path=f"{tmpdir}/fsdp",
            model="trl-internal-testing/tiny-Phi3ForCausalLM",
            fsdp=True,
            data=data,
            query=data,
            batch_size=8,
            num_epochs=1,
            overwrite=True,
            num_subsets=2,
            distributed=dist_cfg,
        )

        run_magic(ddp_cfg)
        run_magic(fsdp_cfg)

        ddp_scores = torch.load(f"{tmpdir}/ddp/scores.pt", weights_only=True)
        fsdp_scores = torch.load(f"{tmpdir}/fsdp/scores.pt", weights_only=True)

    assert (
        fsdp_scores.shape == ddp_scores.shape
    ), f"Shape mismatch: FSDP {fsdp_scores.shape} vs DDP {ddp_scores.shape}"

    atol = 1e-4
    rtol = 1e-3
    if not torch.allclose(fsdp_scores, ddp_scores, atol=atol, rtol=rtol):
        diff = (fsdp_scores - ddp_scores).abs()
        ratio = fsdp_scores.abs().mean() / ddp_scores.abs().mean()
        pytest.fail(
            f"FSDP and DDP scores differ.\n"
            f"  FSDP: mean|s|={fsdp_scores.abs().mean():.6f}, "
            f"sum={fsdp_scores.sum():.6f}\n"
            f"  DDP:  mean|s|={ddp_scores.abs().mean():.6f}, "
            f"sum={ddp_scores.sum():.6f}\n"
            f"  Ratio (FSDP/DDP): {ratio:.4f}\n"
            f"  Max abs diff: {diff.max():.6f}, "
            f"Mean abs diff: {diff.mean():.6f}"
        )


if __name__ == "__main__":
    test_fsdp_ddp_scores_match()
