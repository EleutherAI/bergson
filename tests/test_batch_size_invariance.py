"""
Test that gradient scales are invariant to batch size.

This test verifies the fix for issue #112 where gradients would vary in scale
depending on whether data was processed separately or together.
"""

import subprocess

import pytest
import torch
from datasets import Dataset

from bergson.data import load_gradients

from .cli_command import bergson_cmd, bergson_env


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("batch_size_a,batch_size_b", [(100, 100), (50, 150)])
def test_gradient_scale_invariance(tmp_path, batch_size_a, batch_size_b):
    """
    Test that gradient scales don't depend on how we batch the data.

    This reproduces the bug from issue #112: when computing gradients for the same
    set of datapoints, the gradient magnitudes should be consistent regardless of
    whether we process them separately or together.

    The fix changes loss.mean().backward() to loss.sum().backward() to make
    gradient scales invariant to batch size.
    """
    # Create two simple datasets
    texts_a = [
        f"The quick brown fox jumps over the lazy dog {i}" for i in range(batch_size_a)
    ]
    texts_b = [
        f"A journey of a thousand miles begins with a single step {i}"
        for i in range(batch_size_b)
    ]

    ds_a = Dataset.from_dict({"text": texts_a})
    ds_b = Dataset.from_dict({"text": texts_b})
    ds_combined = Dataset.from_dict({"text": texts_a + texts_b})

    # Save datasets
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ds_a.save_to_disk(str(data_dir / "data_a"))
    ds_b.save_to_disk(str(data_dir / "data_b"))
    ds_combined.save_to_disk(str(data_dir / "data_combined"))

    # Build three indices with minimal settings for speed
    index_dir = tmp_path / "indices"
    index_dir.mkdir()

    def start_bergson_build(index_name: str, dataset_path: str):
        index_path = index_dir / index_name
        cmd = bergson_cmd(
            "build",
            str(index_path),
            "--model",
            "gpt2",  # Use small model for testing
            "--dataset",
            dataset_path,
            "--prompt_column",
            "text",
            "--projection_dim",
            "8",  # Small for speed
            "--token_batch_size",
            "1000",
            "--nproc_per_node",
            "1",
        )
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=bergson_env(),
        )
        return index_path, proc

    # Build the three independent indices concurrently
    index_a_path, proc_a = start_bergson_build("a", str(data_dir / "data_a"))
    index_b_path, proc_b = start_bergson_build("b", str(data_dir / "data_b"))
    index_combined_path, proc_combined = start_bergson_build(
        "combined", str(data_dir / "data_combined")
    )
    for proc in (proc_a, proc_b, proc_combined):
        _, stderr = proc.communicate()
        assert proc.returncode == 0, f"bergson build failed:\n{stderr}"

    # Load gradients
    grads_a = torch.from_numpy(load_gradients(index_a_path).copy()).float()
    grads_b = torch.from_numpy(load_gradients(index_b_path).copy()).float()
    grads_combined = torch.from_numpy(
        load_gradients(index_combined_path).copy()
    ).float()

    # Split combined to match a and b
    grads_a_in_combined = grads_combined[:batch_size_a]
    grads_b_in_combined = grads_combined[batch_size_a:]

    # Compute standard deviations
    std_a_sep = grads_a.std()
    std_a_comb = grads_a_in_combined.std()
    std_b_sep = grads_b.std()
    std_b_comb = grads_b_in_combined.std()

    # atol=0: the invariant is scale-free, so the check must not depend on the
    # gradients' magnitude.
    torch.testing.assert_close(std_a_sep, std_a_comb, rtol=1e-4, atol=0.0)
    torch.testing.assert_close(std_b_sep, std_b_comb, rtol=1e-4, atol=0.0)

    # Also check that cosine similarity is high (gradients point in the same direction)
    a_norm = grads_a / grads_a.norm(dim=1, keepdim=True)
    a_comb_norm = grads_a_in_combined / grads_a_in_combined.norm(dim=1, keepdim=True)
    cosines = (a_norm * a_comb_norm).sum(dim=1)

    torch.testing.assert_close(cosines.mean(), torch.tensor(1.0))
