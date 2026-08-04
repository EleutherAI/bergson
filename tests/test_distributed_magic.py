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


def magic_cfg(
    run_path: str,
    *,
    fsdp: bool,
    clip: bool,
    grad_accum: int = 1,
    dropout: float = 0.0,
    lr: float | None = None,
) -> MagicConfig:
    data = DataConfig(
        dataset="Salesforce/wikitext",
        subset="wikitext-2-raw-v1",
        split="train[:512]",
        chunk_length=32,
    )
    # Single query doc: query_method="none" runs one backward per query.
    query = DataConfig(
        dataset="Salesforce/wikitext",
        subset="wikitext-2-raw-v1",
        split="train[9:10]",
        chunk_length=32,
    )
    return MagicConfig(
        run_path=run_path,
        model="trl-internal-testing/tiny-Phi3ForCausalLM",
        fsdp=fsdp,
        data=data,
        query=query,
        lr_schedule=LRScheduleConfig(lr=lr) if lr is not None else _LR_SCHEDULE,
        batch_size=8,
        num_epochs=1,
        overwrite=True,
        num_subsets=2,
        max_grad_norm=MAX_GRAD_NORM if clip else None,
        grad_accum_steps=grad_accum,
        train_mode=dropout > 0,
        model_kwargs=(
            f"attention_dropout={dropout},resid_pdrop={dropout}" if dropout else ""
        ),
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


@requires_multi_gpu
def test_grad_accum_matches_full_batch(noclip_scores, tmp_path):
    """grad_accum_steps > 1 must not change the trajectory or the metagradient.

    Micro-batch accumulation rescales each micro-loss by its token count, so the
    summed gradient equals the full-batch gradient up to float associativity;
    the replayed backward routes through the two-stage micro-VJP
    (Trainer.metagrad_step) instead of the single-shot traced step. Comparing
    against the full-batch no-clip runs checks both: DDP accum vs DDP full-batch
    verifies exactness of the accumulation, FSDP accum vs DDP accum verifies the
    micro-VJP's stage-A update VJP and stage-B per-micro-batch VJPs are
    shard-correct.
    """
    ddp_noclip = noclip_scores["ddp"]

    run_magic(magic_cfg(f"{tmp_path}/ddp", fsdp=False, clip=False, grad_accum=2))
    run_magic(magic_cfg(f"{tmp_path}/fsdp", fsdp=True, clip=False, grad_accum=2))

    ddp_scores = torch.load(f"{tmp_path}/ddp/scores.pt", weights_only=True)
    fsdp_scores = torch.load(f"{tmp_path}/fsdp/scores.pt", weights_only=True)

    assert fsdp_scores.shape == ddp_noclip.shape

    # Scale-free checks: scores are tiny (~1e-6), so bound the deviations by a
    # fraction of the scores' own magnitude rather than an absolute atol.
    scale = ddp_noclip.abs().max()
    accum_effect = (ddp_scores - ddp_noclip).abs().max()
    fsdp_ddp_diff = (fsdp_scores - ddp_scores).abs().max()

    assert accum_effect < 0.05 * scale, (
        f"grad accumulation changed DDP scores by {accum_effect:.2e} "
        f"(scale {scale:.2e}) — accumulation is not reproducing the "
        "full-batch gradient"
    )
    assert fsdp_ddp_diff < 0.05 * scale, (
        f"FSDP and DDP scores differ with grad accumulation: {fsdp_ddp_diff:.2e} "
        f"(scale {scale:.2e}) — the micro-VJP is likely not shard-correct"
    )


@requires_multi_gpu
def test_grad_accum_matches_across_fsdp_ddp_under_dropout(tmp_path):
    """Micro-batched attribution stays shard-correct when the model is stochastic.

    Every rank has to draw the same dropout masks for the sharded and the
    replicated run to agree, so this pins the accumulation path's RNG handling
    across parallelism modes end to end. The sharp check on the replay rewinding
    the CUDA generator is
    ``test_metagrad_step_matches_single_shot_under_dropout[cuda]``, which
    compares the micro-VJP against the single-shot one exactly; here the
    comparison is across whole runs, so it can only be a loose bound.

    Runs at the config default lr rather than this module's inflated one. At
    8e-4 with dropout the trajectory is unstable enough that fp-level
    differences in the accumulation path compound into a ~1e-2 relative FSDP/DDP
    gap — real, but noise, and it leaves no headroom to call a genuine
    desynchronization. At 1e-5 the same gap is ~5e-6.

    Comparing accum=2 against accum=1 doubles as the control that dropout is
    live: micro-batching changes the tensor shapes the masks are drawn for, so
    the two disagree by ~their own size here, while
    :func:`test_grad_accum_matches_full_batch` has them matching to 5% with
    dropout off.
    """

    def run(name: str, *, fsdp: bool, grad_accum: int) -> torch.Tensor:
        path = f"{tmp_path}/{name}"
        run_magic(
            magic_cfg(
                path,
                fsdp=fsdp,
                clip=False,
                grad_accum=grad_accum,
                dropout=0.5,
                lr=1e-5,
            )
        )
        return torch.load(f"{path}/scores.pt", weights_only=True)

    ddp1 = run("ddp1", fsdp=False, grad_accum=1)
    ddp2 = run("ddp2", fsdp=False, grad_accum=2)
    fsdp2 = run("fsdp2", fsdp=True, grad_accum=2)

    assert fsdp2.shape == ddp1.shape

    scale = ddp2.abs().max()
    dropout_effect = (ddp2 - ddp1).abs().max()
    fsdp_ddp_diff = (fsdp2 - ddp2).abs().max()

    assert dropout_effect > 0.1 * scale, (
        f"accum=2 and accum=1 agree to {dropout_effect:.2e} (scale {scale:.2e}); "
        "dropout is probably not active, making this test degenerate"
    )
    assert fsdp_ddp_diff < 1e-3 * scale, (
        f"FSDP and DDP scores differ under dropout: {fsdp_ddp_diff:.2e} "
        f"(scale {scale:.2e}) — the ranks are likely drawing different masks"
    )


@requires_multi_gpu
def test_save_optimizer_state_completes_under_fsdp(tmp_path):
    """save_optimizer_state must not hang under FSDP.

    Gathering each rank's sharded second moments (DTensor.full_tensor()) is a
    collective, so every rank must call save_second_moments_as_optimizer_pt;
    if worker() ever gates that call behind `global_rank == 0` again, this
    hangs instead of failing cleanly, since the rank-0 all-gather then waits
    forever for peers that never join it.
    """
    cfg = magic_cfg(f"{tmp_path}/fsdp_opt", fsdp=True, clip=False)
    cfg.save_optimizer_state = True
    run_magic(cfg)

    optimizer_pt = torch.load(f"{tmp_path}/fsdp_opt/optimizer.pt", weights_only=False)
    assert optimizer_pt["state"], "optimizer.pt has no second-moment entries"
    for entry in optimizer_pt["state"].values():
        assert torch.isfinite(entry["exp_avg_sq"]).all()
