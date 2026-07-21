import numpy as np
import pytest
from datasets import Dataset, load_from_disk

from bergson.cli.commands import Score
from bergson.config.config import (
    IndexConfig,
    PreprocessConfig,
    RecallConfig,
    RecallDataConfig,
    ScoreConfig,
)
from bergson.config.config_io import save_run_config
from bergson.recall import recall as recall_mod
from bergson.recall.generate import (
    NUM_FIELDS,
    ensure_recall_datasets,
    recall_dataset_paths,
)
from bergson.recall.recall import gold_ranks, run_recall

# ---------------------------------------------------------------------------
# gold_ranks: 1-indexed ranks under a stable descending sort
# ---------------------------------------------------------------------------


def test_gold_ranks_distinct_scores():
    # scores 0.9, 0.1, 0.5, 0.7 -> descending order of rows: 0, 3, 2, 1
    scores = np.array([0.9, 0.1, 0.5, 0.7])
    # row 0 is rank 1, row 3 is rank 2, row 1 is rank 4
    assert gold_ranks(scores, np.array([0])).tolist() == [1]
    assert gold_ranks(scores, np.array([3])).tolist() == [2]
    assert gold_ranks(scores, np.array([1])).tolist() == [4]
    # multiple gold indices at once, in the order given
    assert gold_ranks(scores, np.array([1, 0])).tolist() == [4, 1]


def test_gold_ranks_ties_break_by_row_order():
    # All tied: rank is 1-indexed original position (stable sort).
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert gold_ranks(scores, np.array([0, 1, 2, 3])).tolist() == [1, 2, 3, 4]

    # A clear winner, then a tie for the next positions.
    scores = np.array([0.5, 0.9, 0.5])
    # row 1 (0.9) is rank 1; rows 0 and 2 tie -> ranks 2 and 3 by row order
    assert gold_ranks(scores, np.array([0])).tolist() == [2]
    assert gold_ranks(scores, np.array([2])).tolist() == [3]


# ---------------------------------------------------------------------------
# run_recall end-to-end with tiny on-disk datasets and a fake Scores buffer
# ---------------------------------------------------------------------------


class _FakeScores:
    """Minimal stand-in for bergson.data.Scores backing run_recall."""

    def __init__(self, matrix: np.ndarray):
        # matrix shape: [num_statements, num_questions]
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.num_scores = self.matrix.shape[1]

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def get(self, key, score_idx: int = 0):
        return self.matrix[key, score_idx]

    def is_written(self) -> bool:
        return True


def _write_datasets(tmp_path):
    """Two people, one field, two paraphrases each -> 4 statements, 2 questions."""
    statements = Dataset.from_list(
        [
            {"identifier": 0, "field": "birthdate", "template": 0},
            {"identifier": 0, "field": "birthdate", "template": 1},
            {"identifier": 1, "field": "birthdate", "template": 0},
            {"identifier": 1, "field": "birthdate", "template": 1},
        ]
    )
    questions = Dataset.from_list(
        [
            {"identifier": 0, "field": "birthdate", "answer": "a0"},
            {"identifier": 1, "field": "birthdate", "answer": "a1"},
        ]
    )
    s_path = tmp_path / "statements.hf"
    q_path = tmp_path / "questions.hf"
    statements.save_to_disk(str(s_path))
    questions.save_to_disk(str(q_path))
    return s_path, q_path


def _patch_io(monkeypatch, s_path, q_path, matrix):
    monkeypatch.setattr(
        recall_mod, "ensure_recall_datasets", lambda _cfg: (s_path, q_path)
    )
    monkeypatch.setattr(recall_mod, "load_scores", lambda _p: _FakeScores(matrix))


def _cfg(tmp_path) -> RecallConfig:
    return RecallConfig(
        run_path=str(tmp_path / "out"),
        scores="unused",
        data=RecallDataConfig(num_people=2),
        k=2,
    )


def test_run_recall_perfect_ranking(tmp_path, monkeypatch):
    s_path, q_path = _write_datasets(tmp_path)
    # Q0 gold = rows {0,1}; Q1 gold = rows {2,3}. Give each question's gold rows
    # the top scores so every gold statement ranks 1 or 2 (k=2).
    matrix = np.array(
        [
            [1.0, 0.0],  # stmt row 0 (gold for Q0)
            [0.9, 0.0],  # stmt row 1 (gold for Q0)
            [0.0, 1.0],  # stmt row 2 (gold for Q1)
            [0.0, 0.9],  # stmt row 3 (gold for Q1)
        ]
    )
    _patch_io(monkeypatch, s_path, q_path, matrix)

    metrics = run_recall(_cfg(tmp_path))

    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["recall_at_2"] == pytest.approx(1.0)
    # Both gold rows of each question are within top-2 -> strict recall 1.0
    assert metrics["strict_recall_at_2"] == pytest.approx(1.0)

    # CSV artifacts written
    assert (tmp_path / "out" / "recall.csv").exists()
    assert (tmp_path / "out" / "summary.csv").exists()


def test_run_recall_miss_pushes_rank_down(tmp_path, monkeypatch):
    s_path, q_path = _write_datasets(tmp_path)
    # For Q0, put a non-gold row (row 2) above both gold rows so the best gold
    # rank is 2 (hit at k=2) but strict recall is 0.5 (only one gold in top-2).
    matrix = np.array(
        [
            [0.5, 0.0],  # gold Q0, rank 2
            [0.1, 0.0],  # gold Q0, rank 3 (outside k=2)
            [0.9, 1.0],  # non-gold for Q0 ranks 1; gold Q1 rank 1
            [0.0, 0.9],  # gold Q1 rank 2
        ]
    )
    _patch_io(monkeypatch, s_path, q_path, matrix)

    metrics = run_recall(_cfg(tmp_path))

    # Q0: first gold rank 2 -> rr 0.5, hit; Q1: first gold rank 1 -> rr 1.0, hit
    assert metrics["mrr"] == pytest.approx((0.5 + 1.0) / 2)
    assert metrics["recall_at_2"] == pytest.approx(1.0)
    # Q0 strict 0.5 (one of two gold in top-2), Q1 strict 1.0 -> mean 0.75
    assert metrics["strict_recall_at_2"] == pytest.approx(0.75)


def test_run_recall_higher_is_better_false_flips_sign(tmp_path, monkeypatch):
    s_path, q_path = _write_datasets(tmp_path)
    # With higher_is_better=False the most-negative score should rank first.
    matrix = np.array(
        [
            [-1.0, 0.0],  # gold Q0 -> most negative -> rank 1 when flipped
            [-0.9, 0.0],  # gold Q0
            [0.0, -1.0],  # gold Q1
            [0.0, -0.9],  # gold Q1
        ]
    )
    _patch_io(monkeypatch, s_path, q_path, matrix)

    cfg = _cfg(tmp_path)
    cfg.higher_is_better = False
    metrics = run_recall(cfg)

    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["recall_at_2"] == pytest.approx(1.0)


def _write_score_dir(tmp_path, higher_is_better: bool) -> str:
    """A score directory whose ``config.yaml`` records ``higher_is_better``."""
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    save_run_config(
        Score(
            ScoreConfig(query_path="q", higher_is_better=higher_is_better),
            IndexConfig(run_path=str(score_dir)),
            PreprocessConfig(),
        ),
        score_dir,
    )
    return str(score_dir)


def test_run_recall_reads_higher_is_better_from_score_dir(tmp_path, monkeypatch):
    """Regression: recall must honour the orientation the scoring run recorded.

    SOURCE (approx unrolling) writes ``score_cfg.higher_is_better=False``. With
    ``RecallConfig.higher_is_better`` left unset, recall used to default to
    True and rank the true proponent last.
    """
    s_path, q_path = _write_datasets(tmp_path)
    # Loss-diff convention: the most negative score is the best proponent.
    matrix = np.array(
        [
            [-1.0, 0.0],  # gold Q0
            [-0.9, 0.0],  # gold Q0
            [0.0, -1.0],  # gold Q1
            [0.0, -0.9],  # gold Q1
        ]
    )
    _patch_io(monkeypatch, s_path, q_path, matrix)

    cfg = _cfg(tmp_path)
    cfg.scores = _write_score_dir(tmp_path, higher_is_better=False)
    # cfg.higher_is_better is left at its default: the score dir decides.

    metrics = run_recall(cfg)

    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["recall_at_2"] == pytest.approx(1.0)


def test_run_recall_explicit_higher_is_better_overrides_score_dir(
    tmp_path, monkeypatch
):
    """An explicitly set config value still wins over the saved score_cfg."""
    s_path, q_path = _write_datasets(tmp_path)
    matrix = np.array(
        [
            [-1.0, 0.0],
            [-0.9, 0.0],
            [0.0, -1.0],
            [0.0, -0.9],
        ]
    )
    _patch_io(monkeypatch, s_path, q_path, matrix)

    cfg = _cfg(tmp_path)
    cfg.scores = _write_score_dir(tmp_path, higher_is_better=False)
    cfg.higher_is_better = True  # explicit: do not negate

    metrics = run_recall(cfg)

    # Gold rows are the most negative, so ranking descending puts them last:
    # Q0 gold ranks 3 and 4, Q1 gold ranks 3 and 4 -> no hits at k=2.
    assert metrics["recall_at_2"] == pytest.approx(0.0)


def test_resolve_higher_is_better_defaults_true_without_score_cfg(tmp_path):
    """No score directory / no saved score_cfg -> default True."""
    resolve = recall_mod.resolve_higher_is_better
    assert resolve(str(tmp_path / "missing"), None) is True
    assert resolve("", None) is True


def test_run_recall_statement_count_mismatch_raises(tmp_path, monkeypatch):
    s_path, q_path = _write_datasets(tmp_path)
    # 3 statement rows in the score buffer, but the dataset has 4 -> error.
    matrix = np.zeros((3, 2))
    _patch_io(monkeypatch, s_path, q_path, matrix)

    with pytest.raises(ValueError, match="statements"):
        run_recall(_cfg(tmp_path))


def test_run_recall_question_count_mismatch_raises(tmp_path, monkeypatch):
    s_path, q_path = _write_datasets(tmp_path)
    # 4 statements (correct) but 3 score columns vs 2 questions -> error.
    matrix = np.zeros((4, 3))
    _patch_io(monkeypatch, s_path, q_path, matrix)

    with pytest.raises(ValueError, match="quer"):
        run_recall(_cfg(tmp_path))


# ---------------------------------------------------------------------------
# dataset generation (hermetic: builds a tiny data dir, never touches repo data/)
# ---------------------------------------------------------------------------

# birthdate has two paraphrase templates, the other fields one each.
_TEMPLATES = {
    "birthdate": [
        "{first_name} {last_name} was born on {birthdate}.",
        "{first_name} {last_name}'s birthday is {birthdate}.",
    ],
    "birthplace": ["{first_name} {last_name} was born in {birthplace}."],
    "employer": ["{first_name} {last_name} works at {employer}."],
    "university": ["{first_name} {last_name} studied at {university}."],
}


def _make_data_dir(tmp_path, n_names=5):
    data_dir = tmp_path / "data"
    (data_dir / "names").mkdir(parents=True)
    (data_dir / "templates").mkdir(parents=True)

    names = {
        "first_name.txt": [f"First{i}" for i in range(n_names)],
        "last_name.txt": [f"Last{i}" for i in range(n_names)],
        "employer.txt": ["Acme", "Globex"],
        "town.txt": ["Springfield", "Shelbyville"],
        "university.txt": ["State U", "Tech U"],
    }
    for fname, lines in names.items():
        (data_dir / "names" / fname).write_text("\n".join(lines) + "\n")
    for field, templates in _TEMPLATES.items():
        (data_dir / "templates" / f"{field}.txt").write_text(
            "\n".join(templates) + "\n"
        )

    return data_dir


def test_recall_dataset_paths_naming(tmp_path):
    cfg = RecallDataConfig(num_people=1000, seed=0, data_dir="data")
    s, q = recall_dataset_paths(cfg)
    assert s.name == "statements_1000p_seed0.hf"
    assert q.name == "questions_1000p_seed0.hf"

    single = RecallDataConfig(num_people=1000, seed=0, single_paraphrase=True)
    s2, _ = recall_dataset_paths(single)
    assert s2.name == "statements_1000p_seed0_single.hf"


def test_ensure_recall_datasets_generates_and_caches(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    cfg = RecallDataConfig(num_people=3, seed=0, data_dir=str(data_dir))

    s_path, q_path = ensure_recall_datasets(cfg)
    assert s_path.exists() and q_path.exists()

    statements = load_from_disk(str(s_path))
    questions = load_from_disk(str(q_path))

    # 3 people x 4 fields = 12 questions, one per (person, field).
    assert len(questions) == NUM_FIELDS * 3 == 12
    # Per person: 2 birthdate paraphrases + 1 each for the other 3 fields = 5.
    assert len(statements) == 5 * 3 == 15
    assert len({s["identifier"] for s in statements}) == 3

    # Re-running reuses the cache (identical paths, no regeneration error).
    s_again, q_again = ensure_recall_datasets(cfg)
    assert (s_again, q_again) == (s_path, q_path)


def test_ensure_recall_datasets_single_paraphrase(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    cfg = RecallDataConfig(
        num_people=3, seed=0, single_paraphrase=True, data_dir=str(data_dir)
    )
    s_path, q_path = ensure_recall_datasets(cfg)

    statements = load_from_disk(str(s_path))
    questions = load_from_disk(str(q_path))
    # One statement per (person, field) -> exactly one gold per question.
    assert len(statements) == NUM_FIELDS * 3 == 12
    assert len(questions) == NUM_FIELDS * 3 == 12


def test_ensure_recall_datasets_too_few_names_raises(tmp_path):
    # Only 2 usable name pairs but 3 people requested -> explicit error.
    data_dir = _make_data_dir(tmp_path, n_names=2)
    cfg = RecallDataConfig(num_people=3, seed=0, data_dir=str(data_dir))
    with pytest.raises(ValueError, match="name lists"):
        ensure_recall_datasets(cfg)
