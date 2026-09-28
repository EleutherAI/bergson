"""Regression test for FIM covariance scaling under label masking.

The K-FAC covariance accumulators (``A_cov = Σ aaᵀ``, ``S_cov = Σ ggᵀ``)
ingest every gradient-carrying position — each real position except the
last — regardless of loss masking, because gradients flow back through
attention to prompt positions. ``total_processed.pt`` is later used to
normalize these sums into means (see ``eigenvectors.py``), so it must
count exactly the ingested positions. It used to count supervised
(non-``-100``-label) positions instead, which diverges under prompt
masking and mis-scales the FIM.
"""

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.collector.collector import CollectorComputer
from bergson.config import HessianConfig, IndexConfig
from bergson.gradients import GradientProcessor
from bergson.hessians.autocorrelation import AutocorrelationCollector
from bergson.hessians.hessian_approximations import collect_hessians
from bergson.hessians.kfac import CovarianceCollector
from bergson.utils.utils import assert_type, get_device

INPUT_IDS = [
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10],
    [11, 12, 13, 14, 15],
]
# Prompt-masked labels: the supervised-token count (6) differs from the
# number of gradient-carrying positions (length - 1 per doc = 12).
LABELS = [
    [-100, -100, -100, 4, 5, 6],
    [-100, -100, 9, 10],
    [-100, -100, -100, -100, 15],
]


def make_model():
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    return model.to(get_device(0))


class CountingCovarianceCollector(CovarianceCollector):
    """CovarianceCollector that records how many positions each hook ingests."""

    def setup(self) -> None:
        super().setup()
        self.ingested = dict.fromkeys(self.target_info, 0)

    def forward_hook(self, module, a):
        mask = self._current_collection_mask
        assert mask is not None
        name = assert_type(str, module._name)
        self.ingested[name] += int(mask.sum())
        super().forward_hook(module, a)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_total_processed_matches_ingested_positions(tmp_path):
    model = make_model()
    data = Dataset.from_dict({"input_ids": INPUT_IDS, "labels": LABELS})

    expected_positions = sum(len(ids) - 1 for ids in INPUT_IDS)
    supervised_tokens = sum(sum(tok != -100 for tok in doc[1:]) for doc in LABELS)
    assert supervised_tokens != expected_positions, "test data must diverge"

    run_cfg = IndexConfig(run_path=str(tmp_path / "run"))
    run_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

    collector = CountingCovarianceCollector(
        model=model.base_model,
        dtype=torch.float32,
        path=str(tmp_path / "cov"),
        processor=GradientProcessor(),
    )
    computer = CollectorComputer(
        model=model,
        data=data,
        collector=collector,
        cfg=run_cfg,
    )
    computer.run_with_collector_hooks()

    # Every module's covariance ingested all gradient-carrying positions,
    # prompt included.
    for name, count in collector.ingested.items():
        assert (
            count == expected_positions
        ), f"{name} ingested {count} positions, expected {expected_positions}"

    # The saved normalizer counts the same positions the covariances ingested.
    total_processed = torch.load(run_cfg.partial_run_path / "total_processed.pt")
    assert int(total_processed) == expected_positions


class RawGramAutocorrelation(AutocorrelationCollector):
    """AutocorrelationCollector that also keeps the unnormalized Grams."""

    def setup(self) -> None:
        super().setup()
        self.raw_grams: dict[str, torch.Tensor] = {}

    def backward_hook(self, module, g):
        name = assert_type(str, module._name)
        P = self._compute_gradient(module, g).float()
        if name in self.raw_grams:
            self.raw_grams[name] += P.mT @ P
        else:
            self.raw_grams[name] = P.mT @ P
        super().backward_hook(module, g)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("attribute_tokens", [False, True])
def test_autocorrelation_normalized_by_row_count(tmp_path, attribute_tokens):
    """The autocorrelation Gram is divided by the number of gradient rows it
    summed: documents for per-sequence gradients, gradient-carrying tokens
    for per-token gradients."""
    model = make_model()
    data = Dataset.from_dict({"input_ids": INPUT_IDS, "labels": LABELS})

    expected_rows = (
        sum(len(ids) - 1 for ids in INPUT_IDS) if attribute_tokens else len(INPUT_IDS)
    )

    run_cfg = IndexConfig(run_path=str(tmp_path / "run"))
    run_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

    processor = GradientProcessor()
    collector = RawGramAutocorrelation(
        model=model.base_model,
        data=data,
        path=str(tmp_path / "processor"),
        processor=processor,
        attribute_tokens=attribute_tokens,
        target_modules={"layers.0.mlp.down_proj", "layers.1.self_attn.o_proj"},
    )
    computer = CollectorComputer(
        model=model,
        data=data,
        collector=collector,
        cfg=run_cfg,
    )
    computer.run_with_collector_hooks()

    assert processor.hessians, "no hessians accumulated"
    for name, raw in collector.raw_grams.items():
        torch.testing.assert_close(
            processor.hessians[name].to(raw.device),
            raw / expected_rows,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mean_reduction_fits_hessian_of_mean_loss(tmp_path):
    """With ``loss_reduction="mean"`` the fitted gradient covariance is the sum
    reduction's divided by the document length T, the scale of the mean loss's
    Hessian, rather than by T^2."""
    model = make_model()
    data = Dataset.from_dict({"input_ids": [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]})

    covariances = {}
    for reduction in ("sum", "mean"):
        cfg = IndexConfig(run_path=str(tmp_path / reduction), loss_reduction=reduction)
        cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
        collect_hessians(
            model,
            data,
            cfg,
            hessian_cfg=HessianConfig(method="kfac", use_dataset_labels=True),
        )
        covariances[reduction] = {
            kind: load_file(cfg.partial_run_path / kind / "shard_0.safetensors")
            for kind in ("activation_sharded", "gradient_sharded")
        }

    for name, s_sum in covariances["sum"]["gradient_sharded"].items():
        torch.testing.assert_close(
            covariances["mean"]["gradient_sharded"][name], s_sum / 5
        )
        torch.testing.assert_close(
            covariances["mean"]["activation_sharded"][name],
            covariances["sum"]["activation_sharded"][name],
        )
