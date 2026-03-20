import os
from pathlib import Path

from torch import nn

from bergson import GradientProcessor
from bergson.gradients import AdafactorNormalizer, AdamNormalizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_MODE"] = "disabled"

import pytest
import torch
from datasets import Dataset
from transformers import (
    Adafactor,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from bergson.config import AttentionConfig
from bergson.data import load_gradients
from bergson.huggingface import (
    GradientCollectorCallback,
    prepare_for_gradient_collection,
)
from bergson.utils.utils import assert_type


class TestGradientCollectorCallback:
    @pytest.fixture
    def model(self):
        """Create a small test model."""
        config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
        return AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)

    @pytest.fixture
    def dataset(self):
        """Create a small test dataset."""
        data = {
            "input_ids": [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
            "labels": [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
        }
        return Dataset.from_dict(data)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_gpu_order_tracking(self, tmp_path, model, dataset):
        """Test that every step has an associated order record in single-GPU mode."""
        # Train the model with the callback
        training_args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
        )

        callback = GradientCollectorCallback(
            path=tmp_path / "gradients",
            track_order=True,
            use_optimizer_state=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            callbacks=[callback],
        )
        trainer = prepare_for_gradient_collection(trainer)
        trainer.train()

        # Verify order records were created
        assert callback.order is not None
        assert len(callback.order) > 0

        # Check that every step has associated order records
        steps_with_records = set()
        for record in callback.order:
            steps_with_records.add(record["global_step"])

        # Get the actual number of training steps
        expected_steps = len(dataset) // training_args.per_device_train_batch_size
        if len(dataset) % training_args.per_device_train_batch_size != 0:
            expected_steps += 1

        # Verify we have records for all expected steps
        assert len(steps_with_records) == expected_steps
        # Expected steps are 1-indexed
        assert steps_with_records == set(range(1, expected_steps + 1))

        # Verify each record has required fields
        for record in callback.order:
            assert "_idx" in record
            assert "global_step" in record
            assert "epoch" in record
            assert isinstance(record["_idx"], int)
            assert isinstance(record["global_step"], int)
            assert isinstance(record["epoch"], int)

        # Verify indices are within valid range
        for record in callback.order:
            assert 0 <= record["_idx"] < len(dataset)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_order_tracking_disabled(self, tmp_path, model, dataset):
        """Test that no order records are created when tracking is disabled."""
        # Train the model with the callback
        training_args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
        )

        callback = GradientCollectorCallback(
            path=tmp_path / "gradients", use_optimizer_state=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            callbacks=[callback],
        )
        trainer = prepare_for_gradient_collection(trainer)
        trainer.train()

        # Verify no order records were created
        assert callback.order is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_order_save_and_load(self, tmp_path, model, dataset):
        """Test that order records are properly saved and can be loaded."""
        # Train the model with the callback
        training_args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
        )

        callback = GradientCollectorCallback(
            path=tmp_path / "gradients",
            track_order=True,
            use_optimizer_state=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            callbacks=[callback],
        )
        trainer = prepare_for_gradient_collection(trainer)
        trainer.train()

        # Verify order records were created
        assert callback.order is not None
        assert len(callback.order) > 0

        # Check that order file was saved
        order_file = tmp_path / "gradients" / "order.hf"
        assert order_file.exists()

        # Load and verify the saved order
        saved_order = Dataset.load_from_disk(str(order_file))
        assert len(saved_order) == len(callback.order)

        # Verify the saved order matches the in-memory order
        for i, record in enumerate(saved_order):
            record = assert_type(dict, record)
            assert record["_idx"] == callback.order[i]["_idx"]
            assert record["global_step"] == callback.order[i]["global_step"]
            assert record["epoch"] == callback.order[i]["epoch"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sft_trainer(self, tmp_path, model, dataset):
        """Test that gradient and order files are created and
        can be loaded after training with SFTTrainer."""
        # Set up and train the model with SFT
        sft_config = SFTConfig(
            output_dir=str(tmp_path / "sft_output"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
        )

        callback = GradientCollectorCallback(
            path=tmp_path / "gradients",
            track_order=True,
            use_optimizer_state=False,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            eval_dataset=dataset,
            callbacks=[callback],
        )
        trainer = prepare_for_gradient_collection(trainer)
        trainer.train()

        # Verify training order was tracked
        assert callback.order is not None
        assert len(callback.order) > 0

        # Verify gradient files were created
        gradient_dir = tmp_path / "gradients"
        train_gradient_dir = gradient_dir / "train" / "epoch_0"

        assert gradient_dir.exists()
        assert train_gradient_dir.exists()
        assert (train_gradient_dir / "gradients.bin").exists()
        assert (train_gradient_dir / "info.json").exists()
        assert (gradient_dir / "order.hf").exists()

        # Test loading the gradient data directly
        gradients = load_gradients(train_gradient_dir)
        assert len(gradients) > 0

        # Verify order data was saved and can be loaded
        order_file = tmp_path / "gradients" / "order.hf"
        assert order_file.exists()

        saved_order = Dataset.load_from_disk(str(order_file))
        assert len(saved_order) > 0
        assert all(key in saved_order[0] for key in ["_idx", "global_step", "epoch"])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_attention_head_splitting(self, tmp_path, model, dataset):
        """Test that attention head splitting produces per-head gradients
        for all configured heads during HF Trainer callback collection."""
        # tiny-Phi3 o_proj has shape [8, 8] -> split into 2 heads of size 4
        num_heads = 2
        head_size = 4
        attention_cfgs = {
            "layers.0.self_attn.o_proj": AttentionConfig(
                num_heads=num_heads, head_size=head_size, head_dim=2
            ),
        }

        training_args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
        )

        callback = GradientCollectorCallback(
            path=tmp_path / "gradients",
            attention_cfgs=attention_cfgs,
            use_optimizer_state=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            callbacks=[callback],
        )
        trainer = prepare_for_gradient_collection(trainer)
        trainer.train()

        # Verify per-head gradient files were created
        gradient_dir = tmp_path / "gradients" / "train" / "epoch_0"
        gradients = load_gradients(gradient_dir)
        module_names = gradients.dtype.names

        head_modules = [n for n in module_names if "head_" in n]
        assert len(head_modules) == num_heads, (
            f"Expected {num_heads} per-head gradient columns, "
            f"got {len(head_modules)}: {head_modules}"
        )

        # Verify all head gradients are non-zero
        for head_name in head_modules:
            assert (
                gradients[head_name][0].sum().item() != 0.0
            ), f"Gradient for {head_name} is all zeros"

    @pytest.mark.parametrize("optimizer_name", ["adam", "adafactor"])
    @pytest.mark.parametrize("include_bias", [True, False])
    def test_optimizer_state_extraction(self, optimizer_name: str, include_bias: bool):
        """Test that normalizers are correctly extracted from optimizer state.

        This tests the huggingface.py callback by:
        1. Training a model with an optimizer
        2. Calling the callback's on_step_end method
        3. Verifying against raw optimizer state
        """
        torch.manual_seed(42)
        N = 4
        S = 6
        I = 5
        O = 3

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(I, O * 2, bias=include_bias)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(O * 2, O, bias=include_bias)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        torch.manual_seed(42)
        model = SimpleModel()

        # Create optimizer
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            optimizer = Adafactor(
                model.parameters(), scale_parameter=False, relative_step=False, lr=0.001
            )

        # Train a few steps to build up second moments
        for _ in range(5):
            optimizer.zero_grad()
            out = model(torch.randn(N, S, I))
            loss = (out**2).sum()
            loss.backward()
            optimizer.step()

        # Extract normalizers using the ACTUAL callback
        from unittest.mock import Mock, patch

        from bergson.huggingface import GradientCollectorCallback

        # Create callback with minimal setup
        callback = GradientCollectorCallback(
            path=Path("/tmp/test"),
            use_optimizer_state=True,
            include_bias=include_bias,
        )

        # Mock the collector and processor
        mock_collector = Mock()
        mock_collector.processor = GradientProcessor(
            normalizers={}, include_bias=include_bias
        )
        mock_collector.target_info = {"fc1": None, "fc2": None}  # Track these layers
        callback.collector = mock_collector

        # Mock on_substep_end to avoid needing train_grad_buffer
        with patch.object(callback, "on_substep_end"):
            # Call the ACTUAL callback method
            callback.on_step_end(
                args=Mock(),
                state=Mock(epoch=0, global_step=1),
                control=Mock(),
                model=model,
                optimizer=optimizer,
            )

        # Get the normalizers the callback extracted
        normalizers = callback.collector.processor.normalizers

        # Verify against raw optimizer state (independent ground truth)
        for layer_name in ["fc1", "fc2"]:
            layer = model.get_submodule(layer_name)
            norm = normalizers[layer_name]

            # Get raw state from optimizer
            weight_state = optimizer.state[layer.weight]
            lr = optimizer.param_groups[0]["lr"]

            lr_sq = lr**2

            if optimizer_name == "adam":
                # Check normalizer type
                assert isinstance(norm, AdamNormalizer)

                # Ground truth: Adam stores full exp_avg_sq, scaled by 1/lr²
                raw_exp_avg_sq = weight_state["exp_avg_sq"]
                expected_avg_sq = raw_exp_avg_sq / lr_sq

                torch.testing.assert_close(norm.weight_avg_sq, expected_avg_sq)

            elif optimizer_name == "adafactor":
                # Check normalizer type
                assert isinstance(norm, AdafactorNormalizer)

                # Ground truth: Adafactor row/col, scaled by 1/lr²
                raw_row = weight_state["exp_avg_sq_row"]
                raw_col = weight_state["exp_avg_sq_col"]

                expected_row = raw_row / lr_sq
                expected_col = raw_col / lr_sq

                torch.testing.assert_close(norm.row, expected_row)
                torch.testing.assert_close(norm.col, expected_col)

            # Verify bias handling
            if include_bias and layer.bias is not None:
                bias_state = optimizer.state[layer.bias]  # type: ignore
                raw_bias_exp_avg_sq = bias_state["exp_avg_sq"]
                expected_bias = raw_bias_exp_avg_sq / lr_sq

                assert (
                    norm.bias_avg_sq is not None
                ), f"Expected bias_avg_sq for {layer_name}"
                torch.testing.assert_close(norm.bias_avg_sq, expected_bias)
            else:
                assert (
                    norm.bias_avg_sq is None
                ), f"Unexpected bias_avg_sq for {layer_name}"
