import torch

from bergson.gradients import AdafactorNormalizer, AdamNormalizer, GradientProcessor


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
