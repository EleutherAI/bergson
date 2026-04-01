import json

from datasets import Dataset

from bergson.config import DataConfig, IndexConfig
from bergson.utils.worker_utils import estimate_advantage, setup_data_pipeline


def make_reward_dataset(prompts: list[str], rewards: list[float]) -> Dataset:
    """Create a simple dataset with prompt and reward columns."""
    return Dataset.from_dict({"prompt": prompts, "reward": rewards})


def make_data_config(
    prompt_column: str = "prompt", reward_column: str = "reward"
) -> DataConfig:
    return DataConfig(
        dataset="",
        prompt_column=prompt_column,
        reward_column=reward_column,
    )


def test_estimate_advantage_single_group():
    """Advantages within a single prompt group sum to zero."""
    ds = make_reward_dataset(
        prompts=["p1", "p1", "p1"],
        rewards=[1.0, 3.0, 5.0],
    )
    cfg = make_data_config()
    advantages = estimate_advantage(ds, cfg)

    # mean = 3.0; advantages = [-2.0, 0.0, 2.0]
    assert len(advantages) == 3
    assert abs(advantages[0] - (-2.0)) < 1e-6
    assert abs(advantages[1] - 0.0) < 1e-6
    assert abs(advantages[2] - 2.0) < 1e-6
    assert abs(sum(advantages)) < 1e-6  # advantages sum to zero within group


def test_estimate_advantage_multiple_groups():
    """Advantages are computed per prompt group, not globally."""
    ds = make_reward_dataset(
        prompts=["p1", "p1", "p2", "p2"],
        rewards=[2.0, 4.0, 10.0, 20.0],
    )
    cfg = make_data_config()
    advantages = estimate_advantage(ds, cfg)

    # Group p1: mean=3.0 → advantages: [-1.0, 1.0]
    # Group p2: mean=15.0 → advantages: [-5.0, 5.0]
    assert abs(advantages[0] - (-1.0)) < 1e-6
    assert abs(advantages[1] - 1.0) < 1e-6
    assert abs(advantages[2] - (-5.0)) < 1e-6
    assert abs(advantages[3] - 5.0) < 1e-6


def create_rewards_file(tmp_path, prompts_and_rewards):
    """Create a JSON file with text and reward columns."""
    data = [{"text": text, "reward": reward} for text, reward in prompts_and_rewards]
    path = tmp_path / "rewards.json"
    path.write_text("\n".join(json.dumps(d) for d in data))
    return str(path)


def test_advantages_computed_with_drop_columns(tmp_path):
    """Regression test for issue #96: advantage column is present in the output
    even when drop_columns=True removes the original reward column."""
    prompts_and_rewards = [
        ("hello world", 1.0),
        ("hello world", 3.0),
        ("foo bar", 10.0),
        ("foo bar", 20.0),
    ]
    dataset_path = create_rewards_file(tmp_path, prompts_and_rewards)

    cfg = IndexConfig(
        run_path=str(tmp_path / "run"),
        model="EleutherAI/pythia-14m",
        token_batch_size=2048,
        drop_columns=True,
        data=DataConfig(
            dataset=dataset_path,
            reward_column="reward",
        ),
    )

    ds, _ = setup_data_pipeline(cfg)

    # The reward column should have been dropped.
    assert "reward" not in ds.column_names

    # The advantage column must be present and have correct values.
    assert "advantage" in ds.column_names
    advantages = ds["advantage"]
    assert len(advantages) == 4
    # "hello world" group mean = 2.0: advantages = [-1.0, 1.0]
    assert abs(advantages[0] - (-1.0)) < 1e-9
    assert abs(advantages[1] - 1.0) < 1e-9
    # "foo bar" group mean = 15.0: advantages = [-5.0, 5.0]
    assert abs(advantages[2] - (-5.0)) < 1e-9
    assert abs(advantages[3] - 5.0) < 1e-9
