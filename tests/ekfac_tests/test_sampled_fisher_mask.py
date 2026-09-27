import torch
from datasets import Dataset

from bergson.config import HessianConfig, IndexConfig
from bergson.hessians.hessian_approximations import collect_hessians
from bergson.utils.utils import get_device
from tests.ekfac_tests.test_utils import load_sharded_covariances


def test_sampled_labels_only_add_loss_where_dataset_labels_do(
    tmp_path, model, monkeypatch
):
    """With every sample equal to the true next token, sampling must give the
    same covariances as the dataset labels. The batch mixes prompt positions
    and padding, which get no loss with dataset labels."""
    model = model.to(get_device(0))
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12]],
            "labels": [[-100, -100, -100, 4, 5, 6, 7, 8], [-100, 10, 11, 12]],
            "length": [8, 4],
        }
    )
    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }

    # Return each position's next input token as its sample.
    inputs = {}
    model.register_forward_pre_hook(lambda _mod, args: inputs.update(x=args[0]))
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda probs, num_samples, replacement: inputs["x"][:, 1:].reshape(-1, 1),
    )

    covariances = {}
    for use_dataset_labels in (True, False):
        index_cfg = IndexConfig(
            run_path=str(tmp_path / str(use_dataset_labels)),
            loss_reduction="sum",
            include_bias=False,
        )
        index_cfg.partial_run_path.mkdir(parents=True)
        collect_hessians(
            model=model,
            data=ds,
            index_cfg=index_cfg,
            batches=[[0, 1]],
            target_modules=target_modules,
            hessian_cfg=HessianConfig(
                method="kfac", use_dataset_labels=use_dataset_labels
            ),
        )
        covariances[use_dataset_labels] = load_sharded_covariances(
            index_cfg.partial_run_path / "gradient_sharded"
        )

    for name, expected in covariances[True].items():
        torch.testing.assert_close(covariances[False][name], expected)
