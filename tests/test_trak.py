import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset

from bergson.build import build
from bergson.cli.trak import (
    _average_scores,
    _open_scores,
    _train_label_probs,
    trak,
)
from bergson.collector.collector import token_losses
from bergson.config import (
    DataConfig,
    DistributedConfig,
    PreprocessConfig,
    TrakConfig,
)
from bergson.config.config import TrackstarIndexConfig
from bergson.data import column_offsets, load_gradients
from bergson.hessians.inversion import invert_psd_matrix

MODEL = "EleutherAI/pythia-14m"


def _load(scores_dir: Path) -> np.ndarray:
    info = json.loads((scores_dir / "info.json").read_text())
    mmap = np.memmap(
        scores_dir / "scores.bin",
        dtype=info["dtype"],
        mode="r",
        shape=(info["num_rows"],),
    )
    for q in range(info["num_scores"]):
        assert mmap[f"written_{q}"].all()
    return np.stack([mmap[f"score_{q}"] for q in range(info["num_scores"])], 1)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """A tiny pretokenized dataset: 12 training rows, 3 queries."""
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("trak_data")
    rows = [torch.randint(10, 2000, (24,)).tolist() for _ in range(15)]
    train = Dataset.from_dict({"input_ids": rows[:12], "labels": rows[:12]})
    query = Dataset.from_dict({"input_ids": rows[12:], "labels": rows[12:]})
    both = Dataset.from_dict({"input_ids": rows, "labels": rows})
    train.save_to_disk(str(root / "train"))
    query.save_to_disk(str(root / "query"))
    both.save_to_disk(str(root / "both"))
    return root


@pytest.fixture(scope="module")
def trak_run(tmp_path_factory, data_dir) -> TrackstarIndexConfig:
    """One full (1 - p)-weighted TRAK run shared by the GPU tests. Every
    pipeline step reloads the model (~1 s each), so the tests below check
    the run's pieces instead of re-running TRAK per property."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = _index_cfg(tmp_path_factory.mktemp("trak_run") / "weighted", data_dir)
    trak(cfg, TrakConfig(query=_query_cfg(data_dir)))
    return cfg


def _index_cfg(run_path: Path, data_dir: Path) -> TrackstarIndexConfig:
    return TrackstarIndexConfig(
        run_path=str(run_path),
        model=MODEL,
        data=DataConfig(dataset=str(data_dir / "train"), split="train"),
        distributed=DistributedConfig(nproc_per_node=1),
        projection_dim=64,
        projection_target="global",
        token_batch_size=256,
        precision="fp32",
        loss_fn="margin",
    )


def _query_cfg(data_dir: Path) -> DataConfig:
    return DataConfig(dataset=str(data_dir / "query"), split="train")


def test_trak_weights_rows_by_one_minus_p(trak_run):
    """The Q term scales each training row by 1 - p_i, p_i the row's mean
    label-token probability; the pipeline saves those weights."""
    run_path = Path(trak_run.run_path)
    assert _load(run_path / "scores").shape == (12, 3)
    probs = _train_label_probs(trak_run, batch_size=4)
    assert probs.shape == (12,) and (0 < probs).all() and (probs < 1).all()
    saved = np.load(run_path / "scores" / "trak_weights.npy")
    np.testing.assert_allclose(saved, 1 - probs)


def test_trak_rejects_cross_entropy_features(tmp_path, data_dir):
    cfg = _index_cfg(tmp_path / "ce", data_dir)
    cfg.loss_fn = "ce"
    with pytest.raises(ValueError, match="loss_fn='margin'"):
        trak(cfg, TrakConfig(query=_query_cfg(data_dir)))


def test_margin_token_loss_is_negative_log_odds():
    """The margin loss is -(log p - log(1 - p)) per label token, zero on padding."""
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    labels = torch.tensor([[1, 4, -100], [0, 2, 3]])
    got = token_losses("margin", logits, labels)
    p = torch.softmax(logits, -1).gather(-1, labels.clamp(min=0).unsqueeze(-1))[..., 0]
    expected = -(torch.log(p) - torch.log(1 - p)) * (labels != -100)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_trak_rejects_per_module_projection(tmp_path, data_dir):
    cfg = _index_cfg(tmp_path / "per_module", data_dir)
    cfg.projection_target = "per_module"
    with pytest.raises(ValueError, match="projection_target='global'"):
        trak(cfg, TrakConfig(query=_query_cfg(data_dir)))


def test_trak_ensemble_averages_members(tmp_path, trak_run):
    """The ensemble score store is the mean of the members' stores."""
    src = Path(trak_run.run_path) / "scores"
    members = [tmp_path / "member_0", tmp_path / "member_1"]
    for member, scale in zip(members, (1.0, 3.0)):
        shutil.copytree(src, member)
        mmap, info = _open_scores(member, "r+")
        for q in range(info["num_scores"]):
            mmap[f"score_{q}"] = mmap[f"score_{q}"] * scale
        mmap.flush()
    _average_scores(members, tmp_path / "scores")
    np.testing.assert_allclose(_load(tmp_path / "scores"), 2 * _load(src), rtol=1e-6)


def _index_matrix(run_path: Path) -> np.ndarray:
    """All projected gradients of an index concatenated in stored module order."""
    info = json.loads((run_path / "info.json").read_text())
    mmap = load_gradients(run_path)
    cols = column_offsets(info["grad_sizes"])
    return np.concatenate(
        [np.asarray(mmap[:, lo:hi], dtype=np.float64) for _, (lo, hi) in cols.items()],
        axis=1,
    )


def test_trak_matches_explicit_formula(tmp_path, trak_run, data_dir):
    """Scores are (1 - p_i) phi_q^T (Phi^T Phi / N)^-1 phi_i over the global
    gradient sketch, with no damping by default. One index over the training
    and query rows together gives both Phi and phi_q."""
    run_path = Path(trak_run.run_path)
    scores = _load(run_path / "scores")
    weights = np.load(run_path / "scores" / "trak_weights.npy")

    both_cfg = _index_cfg(tmp_path / "both_index", data_dir)
    both_cfg.data = DataConfig(dataset=str(data_dir / "both"), split="train")
    build(both_cfg, PreprocessConfig())
    grads = _index_matrix(tmp_path / "both_index")
    phi, phi_q = grads[:12], grads[12:]
    assert phi.shape[0] == 12 and phi_q.shape[0] == 3

    gram = torch.from_numpy(phi.T @ phi / phi.shape[0]).float()
    h_inv = invert_psd_matrix(
        gram, inversion="damped_inverse", damping_factor=0.0, power=-1.0
    )
    expected = (phi @ h_inv.double().numpy() @ phi_q.T) * weights[:, None]
    # The index stores gradients in reduced precision; the Gram is fp32.
    np.testing.assert_allclose(scores, expected, rtol=2e-2, atol=1e-3)
