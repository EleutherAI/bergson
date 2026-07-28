"""Test that FSDP and DDP produce equivalent MAGIC attribution scores.

Runs the CLI's run_magic (once per FSDP/DDP × clip/no-clip combination) with a
tiny model and asserts the resulting scores match. The two no-clip runs are
shared between tests via a module-scoped fixture, since they also serve as the
control for the gradient-clipping test.

Requires at least 2 CUDA devices.
"""

import pytest
import torch

from bergson.config import DataConfig, DistributedConfig, LRScheduleConfig
from bergson.magic.cli import MagicConfig, run_magic

# Both tests consume the module-scoped noclip_scores fixture, so they must run
# on the same xdist worker or each worker recomputes the two no-clip runs.
pytestmark = pytest.mark.xdist_group("distributed_magic")

# Config default lr (1e-5) gives scores ~1e-6, too small to separate a
# world-size gradient-sync bug from fp32 noise.
_LR_SCHEDULE = LRScheduleConfig(lr=8e-4)

# tiny-Phi3 grad norms on this data are ~0.35, so 0.2 clips on every step.
MAX_GRAD_NORM = 0.2

requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA devices",
)


def magic_cfg(run_path: str, *, fsdp: bool, clip: bool) -> MagicConfig:
    data = DataConfig(
        dataset="EleutherAI/bergson-wikitext-512-chunks",
        subset="wikitext-2-raw-v1",
        split="train[:512]",
        chunk_length=32,
    )
    return MagicConfig(
        run_path=run_path,
        model="trl-internal-testing/tiny-Phi3ForCausalLM",
        fsdp=fsdp,
        data=data,
        query=data,
        lr_schedule=_LR_SCHEDULE,
        batch_size=8,
        num_epochs=1,
        overwrite=True,
        num_subsets=2,
        max_grad_norm=MAX_GRAD_NORM if clip else None,
        distributed=DistributedConfig(nproc_per_node=min(torch.cuda.device_count(), 4)),
    )


@pytest.fixture(scope="module")
def noclip_scores(tmp_path_factory) -> dict[str, torch.Tensor]:
    """DDP and FSDP attribution scores without gradient clipping."""
    tmpdir = tmp_path_factory.mktemp("magic_noclip")
    scores = {}
    for mode, fsdp in [("ddp", False), ("fsdp", True)]:
        run_magic(magic_cfg(f"{tmpdir}/{mode}", fsdp=fsdp, clip=False))
        scores[mode] = torch.load(f"{tmpdir}/{mode}/scores.pt", weights_only=True)
    return scores


@requires_multi_gpu
def test_fsdp_ddp_scores_match(noclip_scores):
    """FSDP and DDP should produce equivalent attribution scores."""
    ddp_scores = noclip_scores["ddp"]
    fsdp_scores = noclip_scores["fsdp"]

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


@requires_multi_gpu
def test_fsdp_ddp_scores_match_with_grad_clipping(noclip_scores, tmp_path):
    """Gradient clipping is consistent across FSDP shards and DDP replicas.

    The global norm is reduced differently in each mode (FSDP sums partial squared
    norms across shards; DDP computes it from replicated grads), so matching scores
    confirm the cross-shard reduction is correct. The unclipped DDP run is the
    control that the chosen threshold actually clips.

    Scores here are tiny (~1e-6), so the checks are scale-free: clipping must change
    the scores by an O(1) fraction, while FSDP and DDP must agree to far tighter than
    that change — a broken cross-shard reduction would move scores by ~their own size.
    """
    ddp_noclip = noclip_scores["ddp"]

    run_magic(magic_cfg(f"{tmp_path}/ddp", fsdp=False, clip=True))
    run_magic(magic_cfg(f"{tmp_path}/fsdp", fsdp=True, clip=True))

    ddp_scores = torch.load(f"{tmp_path}/ddp/scores.pt", weights_only=True)
    fsdp_scores = torch.load(f"{tmp_path}/fsdp/scores.pt", weights_only=True)

    assert fsdp_scores.shape == ddp_scores.shape

    scale = ddp_noclip.abs().max()
    clip_effect = (ddp_scores - ddp_noclip).abs().max()
    fsdp_ddp_diff = (fsdp_scores - ddp_scores).abs().max()

    assert clip_effect > 0.1 * scale, (
        f"clip barely changed the scores ({clip_effect:.2e} vs scale {scale:.2e}); "
        "lower max_grad_norm so it bites"
    )
    assert fsdp_ddp_diff < 0.05 * clip_effect, (
        f"clipped FSDP and DDP scores differ too much: {fsdp_ddp_diff:.2e} "
        f"(clip effect {clip_effect:.2e}) — cross-shard norm reduction is likely wrong"
    )
