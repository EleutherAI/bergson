import tempfile
from collections import defaultdict
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig
from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientProcessor,
    LayerAdapter,
)


# Test fixtures
@pytest.fixture
def test_params():
    """Common test parameters used across gradient tests.

    Returns:
        dict: Test dimensions with keys:
            - N: Batch size (4)
            - S: Sequence length (6)
            - I: Input dimension (5)
            - O: Output dimension (3)
    """
    return {"N": 4, "S": 6, "I": 5, "O": 3}


@pytest.fixture
def simple_model_class(test_params):
    """Factory for creating test model classes.

    Creates simple neural network models for testing gradient collection.
    Supports both single-layer and two-layer architectures.

    Returns:
        callable: Factory function that takes:
            - include_bias (bool): Whether to include bias terms
            - num_layers (int): Number of linear layers (1 or 2, default 2)

    Examples:
        >>> ModelClass = simple_model_class(include_bias=True, num_layers=1)
        >>> model = ModelClass()  # Single layer: fc
        >>> ModelClass = simple_model_class(include_bias=False, num_layers=2)
        >>> model = ModelClass()  # Two layers: fc1, relu, fc2
    """
    I, O = test_params["I"], test_params["O"]

    def _make_model(include_bias: bool, num_layers: int = 2):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                if num_layers == 1:
                    self.fc = nn.Linear(I, O, bias=include_bias)
                    self.layers = nn.Sequential(self.fc)
                else:  # num_layers == 2
                    self.fc1 = nn.Linear(I, O * 2, bias=include_bias)
                    self.relu = nn.ReLU()
                    self.fc2 = nn.Linear(O * 2, O, bias=include_bias)
                    self.layers = nn.Sequential(self.fc1, self.relu, self.fc2)

            @property
            def device(self):
                return next(self.parameters()).device

            def forward(self, x):
                return self.layers(x)

        return SimpleModel

    return _make_model


@pytest.fixture
def trained_model_with_normalizers(simple_model_class, test_params):
    """Factory for creating trained models with Adam second moments.

    Creates a two-layer model, runs several training steps with Adam optimizer,
    then extracts second moments (exp_avg_sq) to create AdamNormalizers for
    both weights and biases.

    Returns:
        callable: Factory function that takes:
            - include_bias (bool): Whether to include bias normalizers

        Returns tuple of (model, normalizers) where:
            - model: Trained SimpleModel instance
            - normalizers: Dict mapping layer names to AdamNormalizer instances
                          with weight and optional bias second moments
    """
    N, S, I = test_params["N"], test_params["S"], test_params["I"]

    def _create(include_bias: bool):
        torch.manual_seed(42)
        ModelClass = simple_model_class(include_bias)
        model = ModelClass().to("cpu")

        optimizer = torch.optim.Adam(model.parameters())

        # Run a few training steps to build up second moments
        for _ in range(5):
            optimizer.zero_grad()
            out = model(torch.randn(N, S, I))
            loss = (out**2).sum()
            loss.backward()
            optimizer.step()

        # Extract normalizers from optimizer state
        normalizers = {}
        for name, param in model.named_parameters():
            if "weight" in name:
                layer_name = name.replace(".weight", "")
                exp_avg_sq = optimizer.state[param]["exp_avg_sq"]

                # Get bias second moments if bias is included
                bias_avg_sq = None
                if include_bias:
                    bias_param_name = layer_name + ".bias"
                    for p_name, p in model.named_parameters():
                        if p_name == bias_param_name:
                            bias_avg_sq = optimizer.state[p]["exp_avg_sq"]
                            break

                normalizers[layer_name] = AdamNormalizer(exp_avg_sq, bias_avg_sq)

        return model, normalizers

    return _create


def test_gradient_collector_proj_norm():
    """Test gradient collection with projection and normalization.

    Verifies that GradientCollector correctly:
    - Collects gradients with and without random projection
    - Applies Adam and Adafactor normalization
    - Saves and loads GradientProcessor state
    - Produces consistent results across save/load cycles
    """
    temp_dir = Path(tempfile.mkdtemp())
    print(temp_dir)

    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-GPTNeoXForCausalLM")
    # Explicitly use float32 so the test isn't sensitive to the config's torch_dtype
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)

    # It's important that we use a batch size of one so that we can simply use the
    # aggregate gradients from the backward itself and compare against those
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], device=model.device)
    inputs = dict(
        input_ids=tokens,
        labels=tokens,
    )
    data = Dataset.from_dict({"input_ids": tokens.tolist()})

    # Test with 16 x 16 random projection as well as with no projection
    for p in (16, None):
        cfg = IndexConfig(
            run_path=str(temp_dir / "run"),
            skip_index=True,
            skip_preconditioners=p is None,
        )
        processor = GradientProcessor(projection_dim=p)
        collector = GradientCollector(
            model=model,
            cfg=cfg,
            data=data,
            processor=processor,
        )
        with collector:
            model.zero_grad()
            model(**inputs).loss.backward()
            collected_grads = collector.mod_grads.copy()

        adafactors: dict[str, AdafactorNormalizer] = {}
        adams: dict[str, AdamNormalizer] = {}

        # Go through the motions of what GradientCollector does, but after the fact
        for name, collected_grad in collected_grads.items():
            layer = model.get_submodule(name)

            i = getattr(layer, LayerAdapter.in_attr(layer))
            o = getattr(layer, LayerAdapter.out_attr(layer))

            g = layer.weight.grad
            assert g is not None

            moments = g.square()

            if p is not None:
                A = collector.projection(name, p, o, "left", g.device, g.dtype)
                B = collector.projection(name, p, i, "right", g.device, g.dtype)
                g = A @ g @ B.T

            assert torch.isfinite(g).all()
            assert torch.isfinite(collected_grad.squeeze(0)).all()

            # The test computes A @ weight.grad @ B.T, while GradientCollector computes
            # (G @ A.T).mT @ (I @ B.T), which are mathematically equivalent.
            torch.testing.assert_close(g, collected_grad.squeeze(0).view_as(g))

            # Store normalizers for this layer
            adams[name] = AdamNormalizer(moments)
            adafactors[name] = adams[name].to_adafactor()

        # Now do it again but this time use the normalizers
        for normalizers in (adams, adafactors):
            previous_collected_grads = {}
            for do_load in (False, True):
                if do_load:
                    processor = GradientProcessor.load(temp_dir / "processor")
                else:
                    processor = GradientProcessor(
                        normalizers=normalizers, projection_dim=p
                    )
                    processor.save(temp_dir / "processor")

                collector.processor = processor
                with collector:
                    model.zero_grad()
                    model(**inputs).loss.backward()
                    collected_grads = collector.mod_grads.copy()

                for name, collected_grad in collected_grads.items():
                    layer = model.get_submodule(name)
                    i = getattr(layer, LayerAdapter.in_attr(layer))
                    o = getattr(layer, LayerAdapter.out_attr(layer))
                    g = layer.weight.grad
                    assert g is not None

                    g = normalizers[name].normalize_weight(g)
                    if p is not None:
                        A = collector.projection(name, p, o, "left", g.device, g.dtype)
                        B = collector.projection(name, p, i, "right", g.device, g.dtype)
                        g = A @ g @ B.T

                    # Compare the normalized gradient with the collected gradient. We
                    # use a higher tolerance than the default because there seems to be
                    # some non-negligible numerical error that accumulates due to the
                    # different order of operations.
                    assert torch.isfinite(g).all()
                    assert torch.isfinite(collected_grad.squeeze(0)).all()

                    torch.testing.assert_close(
                        g, collected_grad.squeeze(0).view_as(g), atol=1e-4, rtol=1e-4
                    )
                    # Check gradients are the same after loading and restoring
                    if do_load:
                        torch.testing.assert_close(
                            collected_grad, previous_collected_grads[name]
                        )

                previous_collected_grads = collected_grads.copy()


@pytest.mark.parametrize("include_bias", [True, False])
def test_gradient_collector_batched(
    include_bias: bool, trained_model_with_normalizers, test_params
):
    """Test per-sample gradient collection with Adam normalization.

    Tests gradient collection with and without bias terms by:
    - Computing ground truth gradients via individual backward passes
    - Comparing against GradientCollector's batched computation
    - Verifying proper bias normalization using Adam second moments

    Args:
        include_bias: Whether to include bias gradients in collection
    """
    temp_dir = Path(tempfile.mkdtemp())
    N, S, I = test_params["N"], test_params["S"], test_params["I"]

    model, normalizers = trained_model_with_normalizers(include_bias)

    # Create dummy dataset for GradientCollector
    dummy_data = Dataset.from_dict({"input_ids": [[1] * 10] * N})

    # Create config for GradientCollector
    cfg = IndexConfig(
        run_path=str(temp_dir / "run"),
        skip_index=True,
    )

    processor = GradientProcessor(
        normalizers=normalizers, projection_dim=None, include_bias=include_bias
    )
    collector = GradientCollector(
        model=model,
        cfg=cfg,
        data=dummy_data,
        processor=processor,
        target_modules={"fc1", "fc2"},
    )

    x = torch.randn(N, S, I)
    with collector:
        model.zero_grad()
        out = model(x)
        loss = (out**2).sum()
        loss.backward()

    # Copy collected gradients from collector.mod_grads
    collected_grads = collector.mod_grads.copy()

    def compute_ground_truth():
        """Compute gradients using individual backward passes, with normalization."""
        model.zero_grad()
        output = model(x)  # [N, S, O]

        # Per-sample losses
        per_sample_losses = (output**2).sum(dim=(1, 2))  # [N]

        ground_truth_grads = defaultdict(list)
        for n in range(N):
            model.zero_grad()
            per_sample_losses[n].backward(retain_graph=True)

            # manually normalize
            for layer_name in ["fc1", "fc2"]:
                layer = model.get_submodule(layer_name)
                grad = layer.weight.grad.clone()

                grad = normalizers[layer_name].normalize_weight(grad)

                if include_bias:
                    bias_grad = layer.bias.grad.clone()
                    # Normalize bias with bias second moments
                    # (matching GradientCollector)
                    bias_grad = bias_grad / normalizers[
                        layer_name
                    ].bias_avg_sq.sqrt().add(1e-8)
                    bias_grad = bias_grad.unsqueeze(1)
                    grad = torch.cat([grad, bias_grad], dim=1)

                # Flatten to match GradientCollector's output format
                ground_truth_grads[layer_name].append(grad.flatten())

        for layer_name in ["fc1", "fc2"]:
            ground_truth_grads[layer_name] = torch.stack(ground_truth_grads[layer_name])

        return ground_truth_grads

    ground_truth = compute_ground_truth()
    for layer_name in ["fc1", "fc2"]:
        torch.testing.assert_close(
            collected_grads[layer_name], ground_truth[layer_name]
        )


def test_bias_gradients(test_params, simple_model_class):
    """Test per-sample bias gradient computation without normalizers.

    Validates that GradientCollector correctly computes bias gradients when
    no normalizers are provided by:
    - Computing ground truth via individual backward passes
    - Collecting bias gradients using GradientCollector
    - Verifying bias gradients match (summed over sequence dimension)

    This tests the no-normalizer bias collection path added to support
    bias gradients without Adam/Adafactor second moments.
    """
    temp_dir = Path(tempfile.mkdtemp())
    torch.manual_seed(42)
    N, S, I, O = test_params["N"], test_params["S"], test_params["I"], test_params["O"]

    ModelClass = simple_model_class(include_bias=True, num_layers=1)
    model = ModelClass().to("cpu")
    x = torch.randn(N, S, I)

    # bias gradient is a sum over sequence dimension for each n
    def compute_ground_truth(model) -> torch.Tensor:
        """Compute gradients using individual backward passes."""
        model.zero_grad()
        output = model(x)  # [N, S, O]

        per_sample_losses = (output**2).sum(dim=(1, 2))  # [N]

        bias_grads = []
        for n in range(N):
            model.zero_grad()
            per_sample_losses[n].backward(retain_graph=True)
            bias_grads.append(model.fc.bias.grad.clone())

        return torch.stack(bias_grads, dim=0)  # [N, O]

    ground_truth = compute_ground_truth(model)

    # GradientCollector with include_bias=True
    # Create dummy dataset for GradientCollector
    dummy_data = Dataset.from_dict({"input_ids": [[1] * 10] * N})

    # Create config for GradientCollector
    cfg = IndexConfig(
        run_path=str(temp_dir / "run"),
        skip_index=True,
    )

    processor = GradientProcessor(include_bias=True, projection_dim=None)
    collector = GradientCollector(
        model=model,
        cfg=cfg,
        data=dummy_data,
        processor=processor,
        target_modules={"fc"},
    )

    with collector:
        model.zero_grad()
        output = model(x)
        loss = (output**2).sum()
        loss.backward()

    # Reshape from [N, O*(I+1)] to [N, O, I+1] to extract bias from last column
    collected = collector.mod_grads["fc"].reshape(N, O, I + 1)
    bias_grads = collected[..., -1]

    assert bias_grads.shape == (
        N,
        O,
    ), f"Expected shape ({N}, {O}), got {bias_grads.shape}"
    assert ground_truth.shape == (
        N,
        3,
    ), f"Expected shape ({N}, {O}), got {ground_truth.shape}"

    # Compare to ground truth
    torch.testing.assert_close(bias_grads, ground_truth)


@pytest.mark.parametrize("include_bias", [True, False])
def test_gradient_collector_with_projection(
    include_bias: bool, trained_model_with_normalizers, test_params
):
    """Test gradient collection with random projection and bias terms.

    Validates that combining random projection with bias collection works correctly:
    - Verifies output shape is [N, projection_dim²] regardless of bias inclusion
    - Checks gradients are non-zero (projection doesn't zero them out)
    - Confirms deterministic behavior (same input = same output)

    This tests the critical path where bias gradients are concatenated to weight
    gradients BEFORE applying the random projection, ensuring the projection
    accounts for the increased dimensionality.

    Args:
        include_bias: Whether to include bias gradients in collection
    """
    temp_dir = Path(tempfile.mkdtemp())
    N, S, I = test_params["N"], test_params["S"], test_params["I"]
    P = 4  # projection dimension

    model, normalizers = trained_model_with_normalizers(include_bias)

    # Create dummy dataset for GradientCollector
    dummy_data = Dataset.from_dict({"input_ids": [[1] * 10] * N})

    # Create config for GradientCollector
    cfg = IndexConfig(
        run_path=str(temp_dir / "run"),
        skip_index=True,
    )

    processor = GradientProcessor(
        normalizers=normalizers, projection_dim=P, include_bias=include_bias
    )
    collector = GradientCollector(
        model=model,
        cfg=cfg,
        data=dummy_data,
        processor=processor,
        target_modules={"fc1", "fc2"},
    )

    x = torch.randn(N, S, I)
    with collector:
        model.zero_grad()
        out = model(x)
        loss = (out**2).sum()
        loss.backward()

    # Check shapes - with projection, output should be [N, P*P]
    for layer_name in ["fc1", "fc2"]:
        collected = collector.mod_grads[layer_name]
        assert collected.shape == (
            N,
            P * P,
        ), f"Expected shape ({N}, {P*P}), got {collected.shape} for {layer_name}"

        # Check that gradients are not all zeros
        assert collected.abs().sum() > 0, f"Gradients are all zeros for {layer_name}"

        # Check determinism - running twice should give same results
        with collector:
            model.zero_grad()
            out = model(x)
            loss = (out**2).sum()
            loss.backward()

        collected2 = collector.mod_grads[layer_name]
        torch.testing.assert_close(
            collected, collected2, msg=f"Gradients not deterministic for {layer_name}"
        )


@pytest.mark.parametrize("include_bias", [False, True])
def test_adafactor_normalization_ground_truth(
    include_bias: bool, trained_model_with_normalizers, test_params
):
    """Test A: Adafactor normalization matches manually-applied factored second moments.

    Converts Adam second moments to Adafactor (rank-1), then compares:
    - Ground truth: per-sample backward + manual Adafactor normalization
    - Collected: GradientCollector with Adafactor normalizers

    Test B (include_bias=True): same but also verifies bias column is normalized
    by bias_avg_sq.
    """
    temp_dir = Path(tempfile.mkdtemp())
    N, S, I = test_params["N"], test_params["S"], test_params["I"]

    model, adam_normalizers = trained_model_with_normalizers(include_bias)

    # Convert Adam → Adafactor, preserving bias_avg_sq
    adafactor_normalizers = {
        name: norm.to_adafactor() for name, norm in adam_normalizers.items()
    }

    dummy_data = Dataset.from_dict({"input_ids": [[1] * 10] * N})
    cfg = IndexConfig(
        run_path=str(temp_dir / "run"),
        skip_index=True,
    )

    processor = GradientProcessor(
        normalizers=adafactor_normalizers,
        projection_dim=None,
        include_bias=include_bias,
    )
    collector = GradientCollector(
        model=model,
        cfg=cfg,
        data=dummy_data,
        processor=processor,
        target_modules={"fc1", "fc2"},
    )

    x = torch.randn(N, S, I)
    with collector:
        model.zero_grad()
        out = model(x)
        loss = (out**2).sum()
        loss.backward()

    collected_grads = collector.mod_grads.copy()

    # Compute ground truth via individual backward passes
    model.zero_grad()
    output = model(x)
    per_sample_losses = (output**2).sum(dim=(1, 2))

    ground_truth_grads = defaultdict(list)
    for n in range(N):
        model.zero_grad()
        per_sample_losses[n].backward(retain_graph=True)

        for layer_name in ["fc1", "fc2"]:
            layer = model.get_submodule(layer_name)
            norm = adafactor_normalizers[layer_name]
            grad = layer.weight.grad.clone()

            # Apply Adafactor normalization manually:
            # row factor: sqrt(mean(row)) / sqrt(row)
            # col factor: 1/sqrt(col)
            r = norm.row.add(1e-30)
            c = norm.col.add(1e-30)
            row_factor = r.mean().sqrt() * r.rsqrt()  # [O]
            col_factor = c.rsqrt()  # [I]
            grad = grad * row_factor[:, None] * col_factor[None, :]

            if include_bias:
                bias_grad = layer.bias.grad.clone()
                bias_grad = bias_grad * norm.bias_avg_sq.add(1e-30).rsqrt()
                grad = torch.cat([grad, bias_grad.unsqueeze(1)], dim=1)

            ground_truth_grads[layer_name].append(grad.flatten())

    for layer_name in ["fc1", "fc2"]:
        gt = torch.stack(ground_truth_grads[layer_name])
        torch.testing.assert_close(
            collected_grads[layer_name],
            gt,
            atol=1e-4,
            rtol=1e-4,
            msg=f"Adafactor normalization mismatch for {layer_name}",
        )


def test_in_features_restored_after_collector(test_params, simple_model_class):
    """Test E: module.in_features is restored after collector context exits.

    Verifies that the collector doesn't permanently mutate module metadata
    (like in_features) when appending bias columns during forward hooks.
    """
    temp_dir = Path(tempfile.mkdtemp())
    N, S, I = test_params["N"], test_params["S"], test_params["I"]

    ModelClass = simple_model_class(include_bias=True, num_layers=2)
    model = ModelClass().to("cpu")

    dummy_data = Dataset.from_dict({"input_ids": [[1] * 10] * N})
    cfg = IndexConfig(
        run_path=str(temp_dir / "run"),
        skip_index=True,
    )

    # Record original in_features for all layers
    original_in_features = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            original_in_features[name] = layer.in_features

    # Run collector with include_bias=True and NO normalizer (triggers ones-appending)
    processor = GradientProcessor(include_bias=True, projection_dim=None)
    collector = GradientCollector(
        model=model,
        cfg=cfg,
        data=dummy_data,
        processor=processor,
        target_modules=set(original_in_features.keys()),
    )

    x = torch.randn(N, S, I)

    # Run multiple batches to ensure in_features doesn't accumulate
    for _ in range(3):
        with collector:
            model.zero_grad()
            out = model(x)
            loss = (out**2).sum()
            loss.backward()

    # Verify in_features is restored
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            assert layer.in_features == original_in_features[name], (
                f"{name}.in_features changed from {original_in_features[name]} "
                f"to {layer.in_features} after collector context"
            )
