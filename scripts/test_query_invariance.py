"""Test that query gradient reduction is invariant to batch_size padding.

Simulates the DataStream padding + correction factor approach and checks
the result matches a single-rank computation with no padding.
"""

import math

import torch
import torch.nn.functional as F


def loss_fn(logits, labels, ignore_index=-100):
    """Mimic weighted_causal_lm_ce: reduction='none' then .mean()"""
    tok_loss = F.cross_entropy(
        logits, labels, reduction="none", ignore_index=ignore_index
    )
    return tok_loss.mean()


def fake_query_grads(model, data, batch_size, num_batches=2):
    """Single-rank query gradient computation (ground truth)."""
    grad_accum = None
    for j in range(num_batches):
        start = j * batch_size
        logits = model(data[start : start + batch_size])
        labels = torch.zeros(batch_size, dtype=torch.long)
        loss = loss_fn(logits, labels)
        loss.backward()
        if grad_accum is None:
            grad_accum = {n: p.grad.clone() for n, p in model.named_parameters()}
        else:
            for n, p in model.named_parameters():
                grad_accum[n] += p.grad
        model.zero_grad()
    # mean over batches
    for k in grad_accum:
        grad_accum[k] /= num_batches
    return grad_accum


def fake_distributed_query_grads(model, data, batch_size, world_size, num_batches=2):
    """Simulate multi-rank query gradient computation with label masking.

    Approach:
      1. Mask padded example labels with ignore_index (-100)
      2. compute_query_gradients unchanged (accumulate, / n_batches)
      3. AVG all-reduce
      4. Multiply by padded_bs / batch_size to correct denominator
    """
    padded_bs = math.ceil(batch_size / world_size) * world_size
    per_rank = padded_bs // world_size

    counts = []
    for r in range(world_size):
        real = sum(1 for k in range(per_rank) if r + k * world_size < batch_size)
        counts.append(real)

    print(
        f"  batch_size={batch_size}, world_size={world_size}, "
        f"padded_bs={padded_bs}, per_rank={per_rank}, "
        f"real_per_rank={counts}"
    )

    rank_grads = []
    for r in range(world_size):
        grad_accum = None
        for j in range(num_batches):
            # Interleaved indices for this rank in batch j
            indices = [j * padded_bs + r + k * world_size for k in range(per_rank)]
            is_pad = [(idx % padded_bs) >= batch_size for idx in indices]
            # Map to dataset: real examples use their batch-local position,
            # padded wrap around
            dataset_indices = []
            for idx in indices:
                batch_local = idx % padded_bs
                batch_offset = (idx // padded_bs) * batch_size
                if batch_local < batch_size:
                    dataset_indices.append(batch_offset + batch_local)
                else:
                    dataset_indices.append(batch_offset + (batch_local % batch_size))

            rank_data = data[dataset_indices]
            logits = model(rank_data)

            labels = torch.zeros(per_rank, dtype=torch.long)
            for k_idx, pad in enumerate(is_pad):
                if pad:
                    labels[k_idx] = -100

            loss = loss_fn(logits, labels)
            loss.backward()

            if grad_accum is None:
                grad_accum = {n: p.grad.clone() for n, p in model.named_parameters()}
            else:
                for n, p in model.named_parameters():
                    grad_accum[n] += p.grad
            model.zero_grad()

        # / n_batches (same as original compute_query_gradients)
        for k in grad_accum:
            grad_accum[k] /= num_batches
        rank_grads.append(grad_accum)

    # AVG all-reduce
    result = {}
    for k in rank_grads[0]:
        result[k] = sum(rg[k] for rg in rank_grads) / world_size

    # Correction factor
    correction = padded_bs / batch_size
    for k in result:
        result[k] *= correction

    return result


def main():
    torch.manual_seed(42)
    model = torch.nn.Linear(16, 8, bias=False)
    data = torch.randn(100, 16)

    for batch_size in [5, 7, 10]:
        print(f"\n=== batch_size={batch_size} ===")
        print("Ground truth (single rank, no padding):")
        gt = fake_query_grads(model, data, batch_size=batch_size)
        for k, v in gt.items():
            print(f"  {k}: norm={v.norm().item():.6f}")

        all_pass = True
        for world_size in [1, 2, 3, 4, 5, 7, 8]:
            print(f"\nSimulated world_size={world_size}:")
            result = fake_distributed_query_grads(
                model, data, batch_size=batch_size, world_size=world_size
            )
            for k in gt:
                diff = (result[k] - gt[k]).abs().max().item()
                status = "PASS" if diff < 1e-5 else "FAIL"
                if status == "FAIL":
                    all_pass = False
                print(f"  {k}: max_diff={diff:.2e} [{status}]")

        print(f"\n{'All passed!' if all_pass else 'SOME FAILED!'}")


if __name__ == "__main__":
    main()
