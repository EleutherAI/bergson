"""Upload LESS warmup checkpoints to a HF repo as Pythia-style revisions.

Each warmup ``checkpoint-*`` directory becomes its own branch on a single
HuggingFace repo. The branch name is derived from the trainer's epoch at that
checkpoint, e.g. ``epoch-1``, ``epoch-2``. Optionally also force-updates
``main`` to point at the final epoch.

Run:
    python -m examples.replications.less_replication.push_warmup_to_hub \
        --warmup_path runs/less/Llama-2-7b-hf<...>/warmup \
        --repo_id EleutherAI/less-replication-7b-warmup
"""

import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from simple_parsing import parse


@dataclass
class PushWarmupConfig:
    warmup_path: Path
    """Directory containing checkpoint-* subdirectories produced by run_sft."""

    repo_id: str
    """Target HF repo, e.g. EleutherAI/less-replication-7b-warmup."""

    private: bool = False
    """Whether to create the repo as private if it doesn't exist."""

    branch_prefix: str = "epoch-"
    """Branch name prefix; the suffix is the integer epoch from trainer_state.json."""

    update_main: bool = True
    """If True, also upload the highest-epoch checkpoint to the main branch."""

    branch: str = ""
    """If set, upload every checkpoint to this branch instead of per-epoch
    branches. Useful for smoke tests that don't pollute real epoch-N revisions."""

    overwrite: bool = False
    """If a target branch already exists, force the upload anyway. By default
    we skip branches that already have files."""


def epoch_for_checkpoint(ckpt: Path) -> int:
    """Read trainer_state.json's ``epoch`` field and round to int. HF Trainer
    with ``save_strategy='epoch'`` writes integer epochs at boundaries."""
    state = json.loads((ckpt / "trainer_state.json").read_text())
    epoch = state.get("epoch")
    assert epoch is not None, f"trainer_state.json in {ckpt} has no epoch"
    rounded = round(epoch)
    assert (
        abs(epoch - rounded) < 1e-6
    ), f"checkpoint {ckpt.name} has non-integer epoch {epoch}"
    return rounded


def push_checkpoint(
    api: HfApi,
    repo_id: str,
    ckpt: Path,
    branch: str,
    overwrite: bool,
) -> None:
    try:
        api.create_branch(repo_id=repo_id, branch=branch)
    except HfHubHTTPError as e:
        # 409 = branch exists. Continue if user asked for overwrite.
        if "exists" not in str(e).lower():
            raise
        if not overwrite:
            print(
                f"  branch {branch} already exists, skipping "
                f"(pass --overwrite to force re-upload)"
            )
            return

    print(f"  uploading {ckpt} -> {repo_id}@{branch}")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(ckpt),
        revision=branch,
        commit_message=f"Warmup checkpoint {ckpt.name}",
    )


def main(push_cfg: PushWarmupConfig) -> None:
    checkpoint_dirs = sorted(push_cfg.warmup_path.glob("checkpoint-*"))
    assert checkpoint_dirs, f"No checkpoint-* dirs in {push_cfg.warmup_path}"

    api = HfApi()
    try:
        api.create_repo(
            repo_id=push_cfg.repo_id,
            repo_type="model",
            private=push_cfg.private,
            exist_ok=True,
        )
    except HfHubHTTPError as e:
        raise RuntimeError(f"Failed to create/access repo {push_cfg.repo_id}: {e}")

    print(f"Pushing {len(checkpoint_dirs)} checkpoints to {push_cfg.repo_id}")

    last_ckpt = checkpoint_dirs[-1]
    last_epoch = epoch_for_checkpoint(last_ckpt)

    for ckpt in checkpoint_dirs:
        if push_cfg.branch:
            branch = push_cfg.branch
        else:
            branch = f"{push_cfg.branch_prefix}{epoch_for_checkpoint(ckpt)}"
        push_checkpoint(api, push_cfg.repo_id, ckpt, branch, push_cfg.overwrite)

    if push_cfg.update_main and not push_cfg.branch:
        print(f"  updating main to mirror epoch-{last_epoch} ({last_ckpt.name})")
        api.upload_folder(
            repo_id=push_cfg.repo_id,
            folder_path=str(last_ckpt),
            revision="main",
            commit_message=f"Final warmup checkpoint ({last_ckpt.name})",
        )

    print("Done.")


if __name__ == "__main__":
    push_cfg = parse(PushWarmupConfig)
    main(push_cfg)
