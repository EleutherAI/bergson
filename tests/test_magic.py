import math
import subprocess
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.utils.math import weighted_causal_lm_ce


def _forward_loss(model, input_ids, attention_mask, labels, example_weight):
    """Get logits and compute weighted loss externally."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return weighted_causal_lm_ce(logits, labels, example_weight=example_weight)


def _ground_truth(model, inputs, batch_size):
    """Single rank, no padding, uniform weights."""
    ids = inputs["input_ids"][:batch_size]
    mask = inputs["attention_mask"][:batch_size]
    labels = ids.clone()
    w = torch.ones(batch_size, device=ids.device)
    loss = _forward_loss(model, ids, mask, labels, w)
    loss.backward()
    grads = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    model.zero_grad()
    return loss.item(), grads


def _padded_distributed(model, inputs, batch_size, world_size):
    """Simulate padding with scaled weights + masked labels across ranks."""
    padded_bs = math.ceil(batch_size / world_size) * world_size
    per_rank = padded_bs // world_size
    correction = padded_bs / batch_size

    rank_grads = []
    rank_losses = []
    for r in range(world_size):
        global_indices = [r + k * world_size for k in range(per_rank)]
        is_pad = [idx >= batch_size for idx in global_indices]
        safe_indices = [idx % batch_size for idx in global_indices]

        ids = inputs["input_ids"][safe_indices]
        mask = inputs["attention_mask"][safe_indices]
        labels = ids.clone()

        w = torch.zeros(per_rank, device=ids.device)
        for k, pad in enumerate(is_pad):
            if pad:
                labels[k] = -100
            else:
                w[k] = correction

        loss = _forward_loss(model, ids, mask, labels, w)
        loss.backward()
        grads = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
        model.zero_grad()
        rank_grads.append(grads)
        rank_losses.append(loss.item())

    avg_loss = sum(rank_losses) / world_size
    result = {}
    for k in rank_grads[0]:
        result[k] = sum(rg[k] for rg in rank_grads) / world_size
    return avg_loss, result


@pytest.mark.parametrize("batch_size", [5, 7, 10])
@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 5, 7, 8])
def test_padding_gradient_invariance(batch_size, world_size):
    """Padding + weight correction + label masking produces identical
    loss and gradients to the unpadded single-rank baseline.

    Runs on CPU to avoid CUDA non-determinism in batched matmuls, which
    causes logits to differ slightly across batch compositions.
    """
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-14m", torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.pad_token = tokenizer.eos_token

    texts = [f"The quick brown fox number {i}" for i in range(20)]
    inputs = tokenizer(
        texts,
        max_length=32,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
    )

    gt_loss, gt_grads = _ground_truth(model, inputs, batch_size)
    pad_loss, pad_grads = _padded_distributed(model, inputs, batch_size, world_size)

    loss_diff = abs(pad_loss - gt_loss)
    grad_diff = max((pad_grads[k] - gt_grads[k]).abs().max().item() for k in gt_grads)

    # f32 accumulation error in a real transformer produces ~1e-4 diffs;
    # confirmed < 1e-5 in f64 (see commit message).
    assert loss_diff < 1e-3, f"Loss diff {loss_diff:.2e} too large"
    assert grad_diff < 1e-3, f"Grad diff {grad_diff:.2e} too large"


def test_final_batch_padding():
    """DataStream pads the final batch when the dataset doesn't divide evenly,
    with correct weight correction and label masking."""
    from datasets import Dataset

    from bergson.trainer import DataStream

    ds = Dataset.from_dict({"text": [f"hello world {i}" for i in range(10)]})
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.pad_token = tokenizer.eos_token

    # 10 examples, batch_size=4 per-device, world_size=1 → global=4
    # ceil(10/4) = 3 batches, last batch has 2 real + 2 padded
    stream = DataStream(ds, tokenizer, batch_size=4, max_length=16)

    assert stream.num_batches == 3
    assert stream._num_real == 10
    assert stream._pad_last_batch == 2

    # First two batches: uniform weight=1
    assert (stream.weights.data[:4] == 1.0).all()
    assert (stream.weights.data[4:8] == 1.0).all()

    # Last batch: 2 real with correction=4/2=2, 2 padded with weight=0
    assert (stream.weights.data[8:10] == 2.0).all()
    assert (stream.weights.data[10:12] == 0.0).all()

    # Padded examples get labels=-100
    last_batch = stream[2]
    assert (last_batch["labels"][2:] == -100).all()
    # Real examples in last batch keep their labels
    assert (last_batch["labels"][:2] != -100).any()

    # reset_weights restores initial state
    stream.weights.data.fill_(999.0)
    stream.reset_weights()
    assert (stream.weights.data[:8] == 1.0).all()
    assert (stream.weights.data[8:10] == 2.0).all()
    assert (stream.weights.data[10:12] == 0.0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_magic_e2e(tmp_path: Path):
    """End-to-end test of the magic (double_backward) CLI on a single GPU."""
    run_path = tmp_path / "magic_run"

    result = subprocess.run(
        [
            "bergson",
            "magic",
            str(run_path),
            "--model",
            "EleutherAI/pythia-14m",
            "--batch_size",
            "2",
            "--num_batches",
            "3",
            "--query_batches",
            "1",
            "--num_subsets",
            "3",
            "--max_length",
            "64",
            "--warmup_steps",
            "0",
            "--data.dataset",
            "roneneldan/TinyStories",
            "--data.split",
            "train[:50]",
            "--query.dataset",
            "roneneldan/TinyStories",
            "--query.split",
            "train[50:70]",
            "--nproc_per_node",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert (
        result.returncode == 0
    ), f"magic CLI failed with code {result.returncode}:\n{result.stderr}"

    import numpy as np

    assert (run_path / "run_config.json").exists()
    assert (run_path / "scores.npy").exists()
    assert (run_path / "baseline.npy").exists()
    assert (run_path / "validation.npy").exists()

    scores = np.load(str(run_path / "scores.npy"))
    baseline = float(np.load(str(run_path / "baseline.npy")))
    validation = np.load(str(run_path / "validation.npy"))

    assert scores.shape == (6,)  # 2 per-device * 1 GPU * 3 batches
    assert baseline > 0
    assert validation.shape == (3, 2)  # num_subsets x (diff, score_sum)
