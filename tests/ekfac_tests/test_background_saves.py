"""Shards saved on a thread read back whole, and a write that fails raises
where the fit waits for it."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from bergson.hessians.sharded_computation import BackgroundSaves, save_shards


def test_background_saves_read_back(tmp_path: Path):
    tensors = {"a": torch.arange(6.0).view(2, 3), "b": torch.ones(4, dtype=torch.int64)}
    saves = BackgroundSaves()
    saves.save(tensors, str(tmp_path / "one.safetensors"))
    # A view saves as its own rows
    saves.save({"a": tensors["a"].T}, str(tmp_path / "two.safetensors"))
    saves.wait()
    assert saves.threads == []

    one = load_file(str(tmp_path / "one.safetensors"))
    assert one.keys() == tensors.keys()
    for key in tensors:
        torch.testing.assert_close(one[key], tensors[key])
    torch.testing.assert_close(
        load_file(str(tmp_path / "two.safetensors"))["a"], tensors["a"].T
    )


def test_background_saves_raise_at_wait(tmp_path: Path):
    saves = BackgroundSaves()
    saves.save({"a": torch.zeros(2)}, str(tmp_path / "missing" / "x.safetensors"))
    with pytest.raises(FileNotFoundError):
        saves.wait()
    assert saves.threads == []


def test_save_shards_without_a_thread_saves_now(tmp_path: Path):
    path = tmp_path / "now.safetensors"
    save_shards({"a": torch.zeros(3)}, str(path), None)
    torch.testing.assert_close(load_file(str(path))["a"], torch.zeros(3))
