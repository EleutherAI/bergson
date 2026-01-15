"""
Test EKFAC accuracy for computing the Fisher Information Matrix.

Compares the K-FAC approximation F_kfac = G ⊗ A against the exact FIM
computed from per-position gradients on a toy language model.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from bergson.collector.collector import CollectorComputer, fwd_bwd_hessian_factory
from bergson.config import HessianConfig, IndexConfig
from bergson.hessians.kfac import CovarianceCollector
from bergson.utils.utils import get_device
from tests.ekfac_tests.test_utils import load_sharded_covariances, set_all_seeds
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

    position_grads = []
    A_sum = torch.zeros(hidden_size, hidden_size, device=device)
    G_sum = torch.zeros(vocab_size, vocab_size, device=device)

    for batch_indices in batches:
        for idx in batch_indices:
            input_ids = torch.tensor(
                dataset[idx]["input_ids"], device=device
            ).unsqueeze(0)
            labels = torch.tensor(dataset[idx]["labels"], device=device)

            hidden = model.model.embed(input_ids)
            hidden.requires_grad_(True)
            logits = model.model.linear(hidden)

            for s in range(input_ids.shape[1] - 1):
                if sample:
                    # Sample from model distribution (true FIM)
                    with torch.no_grad():
                        probs = torch.softmax(logits[0, s].detach(), dim=-1)
                        target = torch.multinomial(probs, num_samples=1).squeeze()
                else:
                    # Use dataset labels (empirical FIM)
                    target = labels[s + 1]

                loss = F.cross_entropy(logits[0, s], target)

                (g,) = torch.autograd.grad(loss, logits, retain_graph=True)
                g = g[0, s]
                a = hidden[0, s].detach()

                position_grads.append(torch.outer(g, a).flatten().detach())
                A_sum += torch.outer(a, a)
                G_sum += torch.outer(g.detach(), g.detach())

    n_positions = len(position_grads)
    grads_tensor = torch.stack(position_grads)
    F_exact = grads_tensor.T @ grads_tensor / n_positions

    A = A_sum / n_positions
    A = (A + A.T) / 2
    G = G_sum / n_positions
    G = (G + G.T) / 2

    return F_exact, A, G, n_positions


@pytest.mark.parametrize(
    "seq_lengths, num_batches, sample, max_rel_error",
    [
        ((512,), 100, False, 0.05),
        ((512,), 100, True, 0.05),
        ((4,), 10000, False, 0.05),  # rel_error = ~0.25 without valid_masks logic
        ((4,), 10000, True, 0.10),  # rel_error = ~0.25 without valid_masks logic
        ((512, 2), 100, False, 0.05),  # rel_error = ~0.6 without valid_masks logic
        ((512, 2), 100, True, 0.20),  # rel_error = ~1.2 without valid_masks logic
    ],
)
def test_kfac_fim_accuracy(seq_lengths, num_batches, max_rel_error, sample, tmp_path):
    """
    Test that KFAC approximates the FIM within tolerance.

    Args:
        sample: If True, test true FIM (sampled labels).
                If False, test empirical FIM (dataset labels).
    """
    set_all_seeds(42)

    config = ToyDataConfig(
        vocab_size=8,
        hidden_size=4,
        seq_lengths=seq_lengths,
        num_batches=num_batches,
    )
    device = torch.device(get_device())

    dataset = generate_dataset(config)
    batches = generate_batches(config)

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

    hessian_cfg = HessianConfig(use_dataset_labels=not sample)

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
