"""`overwrite` must recompute every SOURCE pipeline step, not reuse stale output.

`resume=True` means "skip if the output already exists", so passing `overwrite`
straight through inverts it for every step that takes `resume`.
"""

from unittest.mock import patch

import pytest

from bergson.approx_unrolling import pipeline as pl
from bergson.config import ApproxUnrollingConfig, HessianConfig, IndexConfig


class _Stop(Exception):
    """Raised by the last stubbed step, to end the run before it does work."""


def _run(overwrite: bool, tmp_path) -> dict[str, bool]:
    """Run the pipeline with every step stubbed; return each step's resume flag."""
    seen: dict[str, bool] = {}

    def record(step: str, *, last: bool = False):
        def stub(*args, **kwargs):
            if "resume" in kwargs:
                seen[step] = kwargs["resume"]
            elif "overwrite" in kwargs:
                # step 1 takes overwrite; normalize to the same meaning
                seen[step] = not kwargs["overwrite"]
            if last:
                raise _Stop()

        return stub

    index_cfg = IndexConfig(
        run_path=str(tmp_path / "run"), model="sshleifer/tiny-gpt2", overwrite=overwrite
    )
    hessian_cfg = HessianConfig(method="ekfac", ev_correction=True)
    au_cfg = ApproxUnrollingConfig(
        checkpoints=[f"checkpoint-{i}" for i in (10, 20, 30, 40)],
        segments=2,
        # Given explicitly so the pipeline doesn't read trainer_state.json.
        lr_list=[1e-4, 1e-4],
        step_size_list=[10, 10],
    )

    with (
        patch.object(pl, "precompute_checkpoint_hessians", record("step1")),
        patch.object(pl, "aggregate_segment_covariances", record("step2")),
        patch.object(pl, "precompute_checkpoint_averaged_lambdas", record("step3")),
        patch.object(pl, "aggregate_segment_lambdas", record("step4", last=True)),
    ):
        with pytest.raises(_Stop):
            pl.approx_unrolling_pipeline(index_cfg, hessian_cfg, au_cfg)

    return seen


@pytest.mark.parametrize("overwrite", [True, False])
def test_resume_is_the_negation_of_overwrite(overwrite: bool, tmp_path):
    seen = _run(overwrite, tmp_path)

    assert set(seen) == {"step1", "step2", "step3", "step4"}, seen
    for step, resume in seen.items():
        assert resume is (not overwrite), (
            f"{step}: resume={resume} with overwrite={overwrite}; "
            "overwrite=True must recompute, not reuse"
        )
