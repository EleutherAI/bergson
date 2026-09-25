import numpy as np
import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from bergson.cli.tracin import tracin
from bergson.config import (
    DataConfig,
    DistributedConfig,
    IndexConfig,
    QuerySetConfig,
    TracInConfig,
)
from bergson.data import load_scores

MODEL = "EleutherAI/pythia-14m"
# SDPA gives pythia wrong gradients on CPU for some sequence lengths.
ATTN = "eager"
LRS = [1e-3, 5e-4, 2.5e-4]


class _Logits(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids).logits


def _summed_ce(logits, labels):
    """Per-example next-token cross entropy summed over tokens."""
    losses = F.cross_entropy(
        logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(), reduction="none"
    )
    return losses.view(len(labels), -1).sum(1)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Rows of mixed lengths and three checkpoints from a short SGD run."""
    root = tmp_path_factory.mktemp("tracin")
    torch.manual_seed(0)
    rows = [
        torch.randint(10, 2000, (int(n),)).tolist() for n in torch.randint(8, 24, (11,))
    ]
    Dataset.from_dict({"input_ids": rows[:8], "labels": rows[:8]}).save_to_disk(
        str(root / "train")
    )
    Dataset.from_dict({"input_ids": rows[8:], "labels": rows[8:]}).save_to_disk(
        str(root / "query")
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation=ATTN
    )
    checkpoints = []
    for i, lr in enumerate(LRS):
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        for row in rows[:8]:
            x = torch.tensor([row])
            model(input_ids=x, labels=x).loss.backward()
            opt.step()
            opt.zero_grad()
        path = root / f"checkpoint-{i}"
        model.save_pretrained(path)
        checkpoints.append(str(path))
    return root, rows, checkpoints


def test_tracin_matches_captum(run):
    """TracIn scores equal Captum's TracInCP on the same checkpoints."""
    captum = pytest.importorskip("captum.influence")
    root, rows, checkpoints = run

    cfg = IndexConfig(
        run_path=str(root / "run"),
        model=MODEL,
        data=DataConfig(dataset=str(root / "train"), split="train"),
        distributed=DistributedConfig(nproc_per_node=1),
        token_batch_size=256,
        precision="fp32",
        include_bias=True,
        model_kwargs=f"attn_implementation={ATTN}",
    )
    tracin_cfg = TracInConfig(
        query=QuerySetConfig(data=DataConfig(dataset=str(root / "query"))),
        checkpoints=checkpoints,
        lr_list=LRS,
    )
    tracin(cfg, tracin_cfg)
    ours = np.asarray(load_scores(root / "run" / "scores")[:], dtype=np.float64)

    def load(model, path):
        state = AutoModelForCausalLM.from_pretrained(path).state_dict()
        model.model.load_state_dict(state)
        return LRS[checkpoints.index(path)]

    def loader(rows):
        pairs = [(torch.tensor(r), torch.tensor(r)) for r in rows]
        return DataLoader(pairs)  # type: ignore[arg-type]

    model = _Logits(
        AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation=ATTN
        )
    )
    # The modules bergson tracks: every linear layer in the transformer blocks.
    layers = [
        f"model.gpt_neox.layers.{i}.{name}"
        for i in range(model.model.config.num_hidden_layers)
        for name in (
            "attention.query_key_value",
            "attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        )
    ]
    reference = captum.TracInCP(
        model,
        loader(rows[:8]),
        checkpoints,
        load,
        layers=layers,
        loss_fn=_summed_ce,
        batch_size=1,
    )
    theirs = reference.influence(loader(rows[8:])).double().numpy().T

    assert ours.shape == (8, 3)
    # Scores cancel across ~800k gradient entries, so fp32 error scales with
    # the largest score.
    np.testing.assert_allclose(ours, theirs, atol=1e-4 * np.abs(theirs).max())


def test_tracin_needs_one_lr_per_checkpoint(tmp_path):
    cfg = IndexConfig(run_path=str(tmp_path / "run"), model=MODEL)
    with pytest.raises(ValueError, match="one learning rate each"):
        tracin(cfg, TracInConfig(checkpoints=["a", "b"], lr_list=[1e-3]))
