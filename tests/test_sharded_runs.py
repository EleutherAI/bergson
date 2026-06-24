"""Sharded (embarrassingly parallel) runs that share one run_path.

Unit tests cover the sharding helpers, the concatenated reader view, and
the canonical config protocol. The GPU integration test simulates the
SLURM job-array workflow on one machine: several independent ``bergson
build`` invocations with ``--num_shards``/``--shard_id`` publish into one
run_path, one of them is killed mid-build and restarted, a finished shard
is re-run to check idempotency, and a config mismatch is rejected. The
resulting index must match a non-sharded build of the same dataset.
"""

import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from datasets import Dataset

from bergson.config.config import IndexConfig, PreprocessConfig
from bergson.config.config_io import publish_canonical_config
from bergson.data import load_gradient_dataset, load_gradients
from bergson.gradients import GradientProcessor
from bergson.sharding import (
    ShardedMemmap,
    published_shard_dirs,
    shard_dir_name,
    shard_row_range,
    shard_status,
)

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_shard_row_range_matches_hf_contiguous_shard():
    ds = Dataset.from_dict({"x": list(range(103))})
    for num_shards in (1, 4, 7, 103):
        previous_end = 0
        for shard_id in range(num_shards):
            start, end = shard_row_range(len(ds), num_shards, shard_id)
            piece = ds.shard(num_shards=num_shards, index=shard_id, contiguous=True)
            assert start == previous_end
            assert end - start == len(piece)
            assert piece["x"] == list(range(start, end))
            previous_end = end
        assert previous_end == len(ds)


@pytest.mark.parametrize("structured", [False, True])
def test_sharded_memmap_matches_concatenate(structured: bool):
    rng = np.random.default_rng(0)
    if structured:
        dtype = np.dtype({"names": ["a", "b"], "formats": ["(3,)f4", "f4"]})
        parts = []
        for n in (3, 1, 4):
            arr = np.zeros(n, dtype=dtype)
            arr["a"] = rng.normal(size=(n, 3))
            arr["b"] = rng.normal(size=n)
            parts.append(arr)
    else:
        parts = [rng.normal(size=(n, 5)).astype(np.float32) for n in (3, 1, 4)]

    ref = np.concatenate(parts)
    view = ShardedMemmap(parts)

    assert len(view) == len(ref)
    assert view.shape == ref.shape
    assert view.dtype == ref.dtype
    assert np.array_equal(view[:], ref)
    assert np.array_equal(view[2:7], ref[2:7])
    assert np.array_equal(view[::3], ref[::3])
    assert np.array_equal(view[-1], ref[-1])
    assert np.array_equal(view[4], ref[4])
    fancy = [7, 0, 3, 3, -2]
    assert np.array_equal(view[fancy], ref[fancy])
    mask = np.arange(len(ref)) % 2 == 0
    assert np.array_equal(view[mask], ref[mask])
    assert np.array_equal(view.copy(), ref)
    assert np.array_equal(np.asarray(view), ref)

    if structured:
        assert view.dtype.names == ("a", "b")
        assert np.array_equal(view["a"], ref["a"])

    with pytest.raises(IndexError):
        view[len(ref)]


def test_sharded_memmap_rejects_mismatched_shards():
    with pytest.raises(ValueError, match="disagree"):
        ShardedMemmap([np.zeros((2, 3)), np.zeros((2, 4))])


def _make_shard_dirs(run_path: Path, names: list[str]) -> None:
    for name in names:
        (run_path / "shards" / name).mkdir(parents=True)


def test_shard_status_and_published_dirs(tmp_path: Path):
    _make_shard_dirs(
        tmp_path, ["00000-of-00003", "00001-of-00003.part", "junk", "notes.txt"]
    )
    (tmp_path / "shards" / "notes.txt").rmdir()
    (tmp_path / "shards" / "notes.txt").touch()

    published, partial, num_shards = shard_status(tmp_path)
    assert num_shards == 3
    assert sorted(published) == [0]
    assert sorted(partial) == [1]

    # Incomplete runs raise by default and list the missing shards
    with pytest.raises(RuntimeError, match=r"missing shards \[1, 2\]"):
        published_shard_dirs(tmp_path)

    assert [p.name for p in published_shard_dirs(tmp_path, allow_partial=True)] == [
        "00000-of-00003"
    ]

    # Mixing different --num_shards in one run_path is rejected
    _make_shard_dirs(tmp_path, ["00000-of-00002"])
    with pytest.raises(ValueError, match="Inconsistent shard counts"):
        shard_status(tmp_path)


class FakeBuild:
    """Minimal stand-in for the Build command dataclass."""

    def __init__(self, index_cfg: IndexConfig, preprocess_cfg: PreprocessConfig):
        self.index_cfg = index_cfg
        self.preprocess_cfg = preprocess_cfg

    def to_dict(self):
        return {
            "index_cfg": self.index_cfg.to_dict(),
            "preprocess_cfg": self.preprocess_cfg.to_dict(),
        }


def test_publish_canonical_config(tmp_path: Path):
    run_path = tmp_path / "run"

    def make_command(shard_id: int, **kwargs) -> FakeBuild:
        index_cfg = IndexConfig(
            run_path=str(run_path), num_shards=4, shard_id=shard_id, **kwargs
        )
        return FakeBuild(index_cfg, PreprocessConfig())

    publish_canonical_config(make_command(0), run_path)
    config_path = run_path / "config.yaml"
    assert config_path.exists()

    with config_path.open() as f:
        doc = yaml.safe_load(f)
    index_dict = doc["steps"][0]["fakebuild"]["index_cfg"]
    assert "shard_id" not in index_dict
    assert "overwrite" not in index_dict
    assert "node_rank" not in index_dict["distributed"]
    assert index_dict["num_shards"] == 4

    # Another shard with per-invocation differences only: accepted
    other = make_command(2)
    other.index_cfg.overwrite = True
    other.index_cfg.distributed.node_rank = 5
    publish_canonical_config(other, run_path)

    # A shard with a different run configuration: rejected
    with pytest.raises(ValueError, match="different configuration"):
        publish_canonical_config(make_command(1, projection_dim=99), run_path)


def test_sharded_config_validation():
    with pytest.raises(ValueError, match="shard_id requires"):
        IndexConfig(run_path="x", shard_id=0)

    with pytest.raises(ValueError, match="shard_id must be in"):
        IndexConfig(run_path="x", num_shards=2, shard_id=2)

    cfg = IndexConfig(run_path="x", num_shards=2)
    cfg.distributed.nnode = 2
    with pytest.raises(ValueError, match="cannot be combined with nnode"):
        cfg.__post_init__()


# ---------------------------------------------------------------------------
# GPU integration: the SLURM job-array workflow on one machine
# ---------------------------------------------------------------------------

NUM_EXAMPLES = 40
NUM_SHARDS = 3


def build_command(run_path: Path, dataset_path: Path, **overrides) -> list[str]:
    args = {
        "model": "gpt2",
        "dataset": str(dataset_path),
        "prompt_column": "text",
        "projection_dim": "8",
        "token_batch_size": "512",
        "nproc_per_node": "1",
        "force_math_sdp": None,  # batch-composition-invariant gradients
        **overrides,
    }
    cmd = ["bergson", "build", str(run_path)]
    for key, value in args.items():
        cmd.append(f"--{key}")
        if value is not None:
            cmd.append(str(value))
    return cmd


def run_checked(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert (
        result.returncode == 0
    ), f"{' '.join(cmd)} failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sharded_build_lifecycle(tmp_path: Path):
    texts = [
        f"The number {i} comes before the number {i + 1}." for i in range(NUM_EXAMPLES)
    ]
    dataset_path = tmp_path / "data"
    Dataset.from_dict({"text": texts}).save_to_disk(str(dataset_path))

    single_path = tmp_path / "single"
    sharded_path = tmp_path / "sharded"

    # Reference: ordinary non-sharded build
    run_checked(build_command(single_path, dataset_path))

    shard_args = {"num_shards": str(NUM_SHARDS)}

    # ── Crash: kill shard 0 mid-build, before it can publish ────────────────
    crash_cmd = build_command(sharded_path, dataset_path, **shard_args, shard_id="0")
    proc = subprocess.Popen(
        crash_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # so we can kill the whole process group
    )
    part_dir = sharded_path / "shards" / (shard_dir_name(0, NUM_SHARDS) + ".part")
    deadline = time.monotonic() + 120
    while not part_dir.exists():
        assert proc.poll() is None, "shard finished before it could be killed"
        assert time.monotonic() < deadline, ".part dir never appeared"
        time.sleep(0.05)
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()

    published, partial, _ = shard_status(sharded_path)
    assert 0 not in published, "killed shard must not be published"
    assert 0 in partial, "killed shard should leave a .part dir behind"

    # ── Restart: re-running the same command rebuilds the crashed shard ─────
    for shard_id in range(NUM_SHARDS):
        run_checked(
            build_command(
                sharded_path, dataset_path, **shard_args, shard_id=str(shard_id)
            )
        )

    published, partial, num_shards = shard_status(sharded_path)
    assert num_shards == NUM_SHARDS and len(published) == NUM_SHARDS and not partial

    # ── Idempotency: re-running a published shard is a no-op ────────────────
    output = run_checked(
        build_command(sharded_path, dataset_path, **shard_args, shard_id="1")
    )
    assert "already published" in output

    # ── Mismatch: a different config may not add shards to this run_path ────
    bad = subprocess.run(
        build_command(
            sharded_path, dataset_path, **shard_args, shard_id="2", projection_dim="16"
        ),
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0
    assert "different configuration" in bad.stderr

    # ── The sharded index reads back identical to the non-sharded one ───────
    single = load_gradients(single_path, structured=False)
    sharded = load_gradients(sharded_path, structured=False)
    assert isinstance(sharded, ShardedMemmap)
    assert sharded.shape == single.shape
    torch.testing.assert_close(
        torch.from_numpy(sharded[:]).float(),
        torch.from_numpy(single.copy()).float(),
    )

    # Structured view, dataset view, and processor artifacts all resolve
    structured = load_gradients(sharded_path)
    assert structured.dtype.names == load_gradients(single_path).dtype.names
    ds = load_gradient_dataset(sharded_path, structured=False)
    assert len(ds) == NUM_EXAMPLES
    GradientProcessor.load(sharded_path)

    # Canonical + per-shard configs and provenance records are in place
    assert (sharded_path / "config.yaml").exists()
    shard_dirs = published_shard_dirs(sharded_path)
    ranges = []
    for shard_dir in shard_dirs:
        assert (shard_dir / "config.yaml").exists()
        with (shard_dir / "shard.json").open() as f:
            record = yaml.safe_load(f)
        ranges.append((record["row_start"], record["row_end"]))
    assert ranges[0][0] == 0 and ranges[-1][1] == NUM_EXAMPLES
    assert all(end == nxt for (_, end), (nxt, _) in zip(ranges, ranges[1:]))

    # `bergson status` agrees
    status_output = run_checked(["bergson", "status", str(sharded_path)])
    assert f"{NUM_SHARDS}/{NUM_SHARDS} shards published" in status_output
