"""
Test EKFAC accuracy for computing the Fisher Information Matrix.

Compares the K-FAC approximation F_kfac = G ⊗ A against the exact FIM
computed from per-position gradients on a toy language model.
"""

from pathlib import Path

import pytest
import torch
from torch import Tensor

from bergson.collector.collector import CollectorComputer, fwd_bwd_hessian_factory
from bergson.config import HessianConfig, IndexConfig
from bergson.hessians.kfac import CovarianceCollector
from bergson.utils.utils import get_device
from tests.ekfac_tests.test_utils import load_sharded_covariances
from tests.ekfac_tests.toy_model import (
    ToyDataConfig,
    ToyLM,
    ToyLMConfig,
    generate_batches,
    generate_dataset,
)


def compute_exact_fim(
    model: ToyLM,
    dataset,
    batches: list[list[int]],
    device: torch.device,
    sample: bool,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """
    Compute exact FIM from per-position gradients for ToyLM.

    Documents are grouped by sequence length and processed as single stacked
    tensors; the covariance and FIM sums are order-independent, so this
    matches the per-document computation up to float associativity.

    Args:
        sample: If True, sample labels from model distribution (true FIM).
                If False, use dataset labels (empirical FIM).

    Returns:
        F_exact: Exact FIM from per-position gradients
        A: Activation covariance (normalized)
        G: Gradient covariance (normalized)
        n_positions: Total number of valid positions
    """
    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size

    A_sum = torch.zeros(hidden_size, hidden_size, device=device)
    G_sum = torch.zeros(vocab_size, vocab_size, device=device)
    F_sum = torch.zeros(
        vocab_size * hidden_size, vocab_size * hidden_size, device=device
    )
    n_positions = 0

    all_ids = dataset["input_ids"]
    all_labels = dataset["labels"]
    by_len: dict[int, list[int]] = {}
    for batch_indices in batches:
        for idx in batch_indices:
            by_len.setdefault(len(all_ids[idx]), []).append(idx)

    # The cross-entropy gradient wrt logits is softmax(logits) - onehot(target),
    # so all positions of a row can be computed in one closed-form pass.
    with torch.no_grad():
        for indices in by_len.values():
            # Slabs bound the (positions, V*H) per-position gradient tensor.
            for start in range(0, len(indices), 4096):
                chunk = indices[start : start + 4096]
                input_ids = torch.tensor([all_ids[i] for i in chunk], device=device)

                # Positions 0..S-2 predict the next token, matching the loss.
                a = model.model.embed(input_ids)[:, :-1]  # (N, S-1, H)
                logits = model.model.linear(a)  # (N, S-1, V)
                probs = torch.softmax(logits, dim=-1)

                if sample:
                    # Sample from model distribution (true FIM)
                    targets = torch.multinomial(
                        probs.flatten(0, 1), num_samples=1
                    ).squeeze(1)
                else:
                    # Use dataset labels (empirical FIM)
                    targets = torch.tensor(
                        [all_labels[i] for i in chunk], device=device
                    )[:, 1:].flatten()

                g = probs.flatten(0, 1).clone()  # (P, V)
                g[torch.arange(g.shape[0], device=device), targets] -= 1.0
                a_flat = a.flatten(0, 1)  # (P, H)

                A_sum += a_flat.T @ a_flat
                G_sum += g.T @ g
                position_grads = torch.einsum("pv,ph->pvh", g, a_flat).flatten(1)
                F_sum += position_grads.T @ position_grads
                n_positions += position_grads.shape[0]

    F_exact = F_sum / n_positions
    A = A_sum / n_positions
    A = (A + A.T) / 2
    G = G_sum / n_positions
    G = (G + G.T) / 2

    return F_exact, A, G, n_positions


@pytest.mark.parametrize(
    "seq_lengths, num_batches, sample, max_rel_error",
    [
        ((4,), 2000, False, 0.10),  # rel_error = ~0.25 without collection-mask logic
        ((4,), 2000, True, 0.15),  # rel_error = ~0.25 without collection-mask logic
        ((512, 2), 100, False, 0.05),  # rel_error = ~0.6 without collection-mask logic
        ((512, 2), 100, True, 0.20),  # rel_error = ~1.2 without collection-mask logic
    ],
)
def test_kfac_fim_accuracy(seq_lengths, num_batches, max_rel_error, sample, tmp_path):
    """
    Test that KFAC approximates the FIM within tolerance.

    Args:
        sample: If True, test true FIM (sampled labels).
                If False, test empirical FIM (dataset labels).
    """
    config = ToyDataConfig(
        vocab_size=8,
        hidden_size=4,
        seq_lengths=seq_lengths,
        num_batches=num_batches,
    )
    device = torch.device(get_device())

    dataset = generate_dataset(config)
    # Covariance sums are invariant to batch partitioning (verified to ~1e-7),
    # and the collector's cost is per-batch, so pack documents into batches of
    # 32 instead of one tiny batch per generated group.
    docs = [idx for batch in generate_batches(config) for idx in batch]
    batches = [docs[i : i + 32] for i in range(0, len(docs), 32)]

    model_config = ToyLMConfig(
        vocab_size=config.vocab_size, hidden_size=config.hidden_size
    )
    model = ToyLM(
        model_config,
        training_data=dataset,
        training_batches=batches,
        device=device,
    )

    F_exact, A_exact, G_exact, total_processed_exact = compute_exact_fim(
        model, dataset, batches, device, sample=sample
    )

    run_path = Path(tmp_path) / "run"
    index_cfg = IndexConfig(run_path=str(run_path), loss_reduction="sum")

    collector = CovarianceCollector(
        model=model.base_model,
        target_modules={"linear"},
        dtype=torch.float32,
        path=str(index_cfg.partial_run_path),
    )

    hessian_cfg = HessianConfig(method="autocorrelation", use_dataset_labels=not sample)

    computer = CollectorComputer(
        model=model,
        data=dataset,
        batches=batches,
        collector=collector,
        cfg=index_cfg,
    )
    computer.forward_backward = fwd_bwd_hessian_factory(index_cfg, hessian_cfg)
    computer.run_with_collector_hooks()

    A_dict_kfac = load_sharded_covariances(
        index_cfg.partial_run_path / "activation_sharded"
    )
    G_dict_kfac = load_sharded_covariances(
        index_cfg.partial_run_path / "gradient_sharded"
    )
    total_processed_kfac = torch.load(
        index_cfg.partial_run_path / "total_processed.pt"
    ).item()

    assert total_processed_kfac == total_processed_exact

    A_kfac = list(A_dict_kfac.values())[0].float().to(device) / total_processed_kfac
    A_kfac = (A_kfac + A_kfac.T) / 2
    G_kfac = list(G_dict_kfac.values())[0].float().to(device) / total_processed_kfac
    G_kfac = (G_kfac + G_kfac.T) / 2

    # A and G should be the same when we're not sampling
    if not sample:
        torch.testing.assert_close(A_kfac, A_exact, rtol=1e-3, atol=1e-6)
        torch.testing.assert_close(G_kfac, G_exact, rtol=1e-3, atol=1e-6)

    F_kfac = torch.kron(G_kfac, A_kfac)
    rel_error = (torch.norm(F_kfac - F_exact) / torch.norm(F_exact)).item()

    assert rel_error <= max_rel_error, (
        f"KFAC rel_error {rel_error:.4f} greater than tolerated max_rel_error "
        f"{max_rel_error} for seq_lengths={seq_lengths}, num_batches={num_batches}, "
        f"sample={sample}"
    )
