"""Tests for YAML-based dataset formatting."""

import textwrap

import pytest
from datasets import Dataset

from bergson.format import apply_format


@pytest.fixture
def mcqa_dataset():
    return Dataset.from_dict(
        {
            "question": [
                "What is 2+2?",
                "Capital of France?",
            ],
            "choices": [
                ["3", "4", "5", "6"],
                ["London", "Paris", "Berlin", "Madrid"],
            ],
            "answer": [1, 1],
        }
    )


@pytest.fixture
def plain_dataset():
    return Dataset.from_dict(
        {
            "title": ["Hello", "World"],
            "body": ["body one", "body two"],
        }
    )


# ── MCQA with doc_to_choice ──────────────────────────────────────────────


def test_mcqa_format(mcqa_dataset, tmp_path):
    yaml_path = tmp_path / "mcqa.yaml"
    yaml_path.write_text(textwrap.dedent("""\
        doc_to_text: >-
          {{question}}
          A. {{choices[0]}}
          B. {{choices[1]}}
          C. {{choices[2]}}
          D. {{choices[3]}}
          Answer:
        doc_to_target: answer
        doc_to_choice: ["A", "B", "C", "D"]
    """))

    result = apply_format(mcqa_dataset, str(yaml_path))

    assert "prompt" in result.column_names
    assert "completion" in result.column_names
    assert result[0]["completion"] == "B"
    assert result[1]["completion"] == "B"
    assert result[0]["prompt"].startswith("What is 2+2?")
    assert "A. 3" in result[0]["prompt"]
    assert result[0]["prompt"].endswith("Answer:")


# ── Text-only (no target) ────────────────────────────────────────────────


def test_text_only_format(plain_dataset, tmp_path):
    yaml_path = tmp_path / "text.yaml"
    yaml_path.write_text(textwrap.dedent("""\
        doc_to_text: "{{title}}: {{body}}"
    """))

    result = apply_format(plain_dataset, str(yaml_path))

    assert "text" in result.column_names
    assert "prompt" not in result.column_names
    assert result[0]["text"] == "Hello: body one"
    assert result[1]["text"] == "World: body two"


# ── Jinja2 template as target ────────────────────────────────────────────


def test_jinja_target(plain_dataset, tmp_path):
    yaml_path = tmp_path / "jinja_target.yaml"
    yaml_path.write_text(textwrap.dedent("""\
        doc_to_text: "{{title}}"
        doc_to_target: "{{body.upper()}}"
    """))

    result = apply_format(plain_dataset, str(yaml_path))

    assert result[0]["prompt"] == "Hello"
    assert result[0]["completion"] == "BODY ONE"


# ── Column-name target (no Jinja) ────────────────────────────────────────


def test_column_name_target(plain_dataset, tmp_path):
    yaml_path = tmp_path / "col_target.yaml"
    yaml_path.write_text(textwrap.dedent("""\
        doc_to_text: "{{title}}"
        doc_to_target: body
    """))

    result = apply_format(plain_dataset, str(yaml_path))

    assert result[0]["prompt"] == "Hello"
    assert result[0]["completion"] == "body one"


# ── Shipped mcqa.yaml template ────────────────────────────────────────────


def test_shipped_mcqa_template(mcqa_dataset):
    result = apply_format(mcqa_dataset, "bergson/templates/mcqa.yaml")

    assert result[0]["completion"] == "B"
    assert "A. 3" in result[0]["prompt"]
    assert "D. 6" in result[0]["prompt"]
