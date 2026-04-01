import tempfile
from collections import defaultdict
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from datasets import Dataset

from bergson import fit_normalizers
from bergson.config import IndexConfig
from bergson.gradients import AdafactorNormalizer, AdamNormalizer, GradientProcessor
from bergson.normalizer.fit_normalizers import NormalizerCollector


def test_fit_normalizers_runs(tmp_path, model, dataset):
    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, nn.Linear)
    }
    print("len dataset", len(dataset))
    print("target_modules", target_modules)

    dataset = dataset.repeat(10)

    normalizers = fit_normalizers(
        model,
        dataset,
        cfg=IndexConfig(
            run_path=str(tmp_path),
            skip_preconditioners=True,
            normalizer="adam",
        ),
        batches=[[idx] for idx in range(len(dataset))],
        target_modules=target_modules,
    )

    assert len(normalizers) == len(target_modules)

    normalizers = fit_normalizers(
        model,
        dataset,
        cfg=IndexConfig(
            run_path=str(tmp_path),
            skip_preconditioners=True,
            normalizer="adafactor",
        ),
        batches=[[idx] for idx in range(len(dataset))],
        target_modules=target_modules,
    )

    assert len(normalizers) == len(target_modules)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("normalizer_type", ["adam", "adafactor"])
def test_fit_normalizers_with_bias(tmp_path, model, dataset, normalizer_type):
    """fit_normalizers populates bias_avg_sq when include_bias=True."""
    # tiny-Phi3 has no bias on Linear layers; add zero bias
    for m in model.base_model.modules():
        if isinstance(m, nn.Linear) and m.bias is None:
            m.bias = nn.Parameter(torch.zeros(m.out_features, device=m.weight.device))

    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, nn.Linear)
    }
    dataset = dataset.repeat(10)

    normalizers = fit_normalizers(
        model,
        dataset,
        cfg=IndexConfig(
            run_path=str(tmp_path / normalizer_type),
            skip_preconditioners=True,
            normalizer=normalizer_type,
            include_bias=True,
        ),
        batches=[[idx] for idx in range(len(dataset))],
        target_modules=target_modules,
    )
    for name, norm in normalizers.items():
        assert norm.bias_avg_sq is not None, f"{name} missing bias_avg_sq"
        assert norm.bias_avg_sq.shape[0] > 0
        assert (norm.bias_avg_sq >= 0).all()  # second moments are non-negative


@pytest.mark.parametrize("normalizer_type", ["adam", "adafactor"])
def test_normalizer_collector_bias_ground_truth(normalizer_type):
    """Verify NormalizerCollector's bias_avg_sq matches manual per-sample computation.

    For each module, bias_avg_sq should equal E[bias_grad^2] where
    bias_grad = g.sum(dim=seq) for each sample.
    """
    torch.manual_seed(42)
    N, S, I, O = 4, 6, 5, 3

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(I, O * 2, bias=True)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(O * 2, O, bias=True)

        @property
        def device(self):
            return next(self.parameters()).device

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = SimpleModel()
    x = torch.randn(N, S, I)

    # --- Ground truth: per-sample bias gradients squared, averaged ---
    model.zero_grad()
    output = model(x)
    per_sample_losses = (output**2).sum(dim=(1, 2))

    gt_bias_sq = defaultdict(list)
    for n in range(N):
        model.zero_grad()
        per_sample_losses[n].backward(retain_graph=True)
        for layer_name in ["fc1", "fc2"]:
            layer = model.get_submodule(layer_name)
            gt_bias_sq[layer_name].append(layer.bias.grad.clone().float().square())

    gt_bias_avg_sq = {
        name: torch.stack(vals).mean(dim=0) for name, vals in gt_bias_sq.items()
    }

    # --- NormalizerCollector ---
    dummy_data = Dataset.from_dict({"input_ids": [[1] * 10] * N})
    cfg = IndexConfig(
        run_path=str(Path(tempfile.mkdtemp()) / "run"),
        skip_preconditioners=True,
        normalizer=normalizer_type,
        include_bias=True,
    )
    processor = GradientProcessor(include_bias=True)
    collector = NormalizerCollector(
        model=model,
        data=dummy_data,
        cfg=cfg,
        processor=processor,
        target_modules={"fc1", "fc2"},
    )

    with collector:
        model.zero_grad()
        output = model(x)
        loss = (output**2).sum()
        loss.backward()

    collector.teardown()

    for layer_name in ["fc1", "fc2"]:
        normalizer = collector.normalizers[layer_name]
        assert normalizer.bias_avg_sq is not None, f"{layer_name} missing bias_avg_sq"
        torch.testing.assert_close(
            normalizer.bias_avg_sq,
            gt_bias_avg_sq[layer_name],
            msg=f"bias_avg_sq mismatch for {layer_name}",
        )


@pytest.mark.parametrize("normalizer_type", ["adam", "adafactor"])
def test_normalizer_collector_weight_ground_truth(normalizer_type):
    """Verify NormalizerCollector's weight second moments match manual computation.

    For Adam: weight_avg_sq = E[(g.T @ a)^2]
    For Adafactor: row/col are mean-reduced second moments
    """
    torch.manual_seed(42)
    N, S, I, O = 4, 6, 5, 3

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(I, O, bias=False)

        @property
        def device(self):
            return next(self.parameters()).device

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    x = torch.randn(N, S, I)

    # --- Ground truth: per-sample weight outer products ---
    # First, collect activations and output gradients manually
    activations = {}
    grad_outputs = {}

    def fwd_hook(name):
        def hook(module, inp, out):
            activations[name] = inp[0].detach()

        return hook

    def bwd_hook(name):
        def hook(module, grad_in, grad_out):
            grad_outputs[name] = grad_out[0].detach()

        return hook

    fwd_h = model.fc.register_forward_hook(fwd_hook("fc"))
    bwd_h = model.fc.register_full_backward_hook(bwd_hook("fc"))

    model.zero_grad()
    output = model(x)
    loss = (output**2).sum()
    loss.backward()

    fwd_h.remove()
    bwd_h.remove()

    a = activations["fc"]  # [N, S, I]
    g = grad_outputs["fc"]  # [N, S, O]
    P = g.mT @ a  # [N, O, I] per-sample weight gradient

    if normalizer_type == "adam":
        gt_weight_avg_sq = P.float().square().sum(0) / N
    else:
        sq = P.float().square().sum(0)  # [O, I]
        gt_row = sq.mean(dim=1) / N  # [O]
        gt_col = sq.mean(dim=0) / N  # [I]

    # --- NormalizerCollector ---
    dummy_data = Dataset.from_dict({"input_ids": [[1] * 10] * N})
    cfg = IndexConfig(
        run_path=str(Path(tempfile.mkdtemp()) / "run"),
        skip_preconditioners=True,
        normalizer=normalizer_type,
    )
    processor = GradientProcessor(include_bias=False)
    collector = NormalizerCollector(
        model=model,
        data=dummy_data,
        cfg=cfg,
        processor=processor,
        target_modules={"fc"},
    )

    with collector:
        model.zero_grad()
        output = model(x)
        loss = (output**2).sum()
        loss.backward()

    collector.teardown()

    normalizer = collector.normalizers["fc"]
    if normalizer_type == "adam":
        assert isinstance(normalizer, AdamNormalizer)
        torch.testing.assert_close(
            normalizer.weight_avg_sq,
            gt_weight_avg_sq,
            msg="Adam weight_avg_sq mismatch",
        )
    else:
        assert isinstance(normalizer, AdafactorNormalizer)
        torch.testing.assert_close(normalizer.row, gt_row, msg="Adafactor row mismatch")
        torch.testing.assert_close(normalizer.col, gt_col, msg="Adafactor col mismatch")


def test_normalizer_save_load_with_bias(tmp_path):
    """Verify save/load roundtrip preserves bias_avg_sq."""
    weight_sq = torch.randn(4, 8).abs()
    bias_sq = torch.randn(4).abs()

    adam = AdamNormalizer(weight_avg_sq=weight_sq, bias_avg_sq=bias_sq)
    processor = GradientProcessor(
        normalizers={"layer": adam},
        include_bias=True,
    )
    processor.save(tmp_path)

    loaded = GradientProcessor.load(tmp_path, skip_preconditioners=True)
    loaded_norm = loaded.normalizers["layer"]
    assert isinstance(loaded_norm, AdamNormalizer)
    torch.testing.assert_close(loaded_norm.weight_avg_sq, weight_sq)
    torch.testing.assert_close(loaded_norm.bias_avg_sq, bias_sq)

    # Also test Adafactor roundtrip
    ada = AdafactorNormalizer(
        row=torch.randn(4).abs(), col=torch.randn(8).abs(), bias_avg_sq=bias_sq
    )
    processor2 = GradientProcessor(
        normalizers={"layer": ada},
        include_bias=True,
    )
    ada_path = tmp_path / "adafactor"
    processor2.save(ada_path)

    loaded2 = GradientProcessor.load(ada_path, skip_preconditioners=True)
    loaded_ada = loaded2.normalizers["layer"]
    assert isinstance(loaded_ada, AdafactorNormalizer)
    torch.testing.assert_close(loaded_ada.row, ada.row)
    torch.testing.assert_close(loaded_ada.col, ada.col)
    torch.testing.assert_close(loaded_ada.bias_avg_sq, bias_sq)
