import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from bergson.config import IndexConfig, PreprocessConfig
from bergson.data import _allocate_batches_world
from bergson.utils.batch_size import _cache_key, probe_batches

from .cli_command import bergson_cmd, bergson_env


def test_probe_batches_cover_every_rank():
    lengths = [512] * 4 + [8] * 300 + [100] * 40
    world_size = 4
    probes = probe_batches(lengths, 1024, world_size, None)
    batches = [
        b for rank in _allocate_batches_world(lengths, 1024, world_size) for b in rank
    ]

    def docs(b):
        return len(b)

    def padded(b):
        return len(b) * max(lengths[i] for i in b)

    assert padded(probes[0]) == max(padded(b) for b in batches)
    assert docs(probes[1]) == max(docs(b) for b in batches)
    # The many-short-documents batch, which a probe of full-length rows misses
    assert docs(probes[1]) == 1024 // 8


def test_cache_key_tracks_memory():
    cfg = IndexConfig(run_path="a", model="EleutherAI/pythia-14m")
    pre = PreprocessConfig()
    lengths = [10, 20, 30]
    key = _cache_key(cfg, pre, lengths)

    assert _cache_key(replace(cfg, run_path="b", overwrite=True), pre, lengths) == key
    assert _cache_key(replace(cfg, token_batch_size=4096), pre, lengths) == key
    for changed in (
        replace(cfg, attribute_tokens=True),
        replace(cfg, include_bias=True),
        replace(cfg, filter_modules="*.mlp.*"),
        replace(cfg, loss_fn="kl"),
        replace(cfg, max_batch_size=4),
    ):
        assert _cache_key(changed, pre, lengths) != key
    assert _cache_key(cfg, replace(pre, aggregation="mean"), lengths) != key
    assert _cache_key(cfg, pre, [10, 20, 31]) != key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("extra", [[], ["--attribute_tokens"]])
def test_auto_batch_size_build(tmp_path: Path, extra: list[str]):
    def build(run: str):
        return subprocess.run(
            bergson_cmd(
                "build",
                run,
                "--model",
                "EleutherAI/pythia-14m",
                "--dataset",
                "NeelNanda/pile-10k",
                "--split",
                "train[:100]",
                "--truncation",
                "--projection_dim",
                "16",
                "--token_batch_size",
                "512",
                "--auto_batch_size",
                *extra,
            ),
            cwd=tmp_path,
            env=bergson_env(),
            capture_output=True,
            text=True,
        )

    first = build("first")
    assert first.returncode == 0, first.stderr
    assert "Optimal token_batch_size" in first.stdout
    assert (tmp_path / "first" / "gradients.bin").exists()

    second = build("second")
    assert second.returncode == 0, second.stderr
    assert "Loaded token_batch_size from cache" in second.stdout
