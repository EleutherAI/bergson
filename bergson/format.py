"""Dataset formatting via Jinja2 templates. See lm-evaluation-harness for
examples of Jinja2 templates.

A format YAML contains:
    doc_to_text: Jinja2 template producing the prompt string.
    doc_to_target: (optional) Field name or Jinja2 template for the target.
    doc_to_choice: (optional) List of choice labels for MCQ tasks.

When both ``doc_to_target`` and ``doc_to_choice`` are present, the target
is resolved as ``doc_to_choice[doc_to_target]``.
"""

from pathlib import Path

import yaml
from datasets import Dataset
from jinja2 import BaseLoader, Environment


def _render(template_str: str, env: Environment, row: dict) -> str:
    return env.from_string(template_str).render(**row)


def apply_format(ds: Dataset, format_path: str) -> Dataset:
    """Apply a YAML format template to *ds*.

    Returns a dataset with either a ``text`` column (prompt-only) or
    ``prompt`` + ``completion`` columns (when ``doc_to_target`` is set).
    """
    spec = yaml.safe_load(Path(format_path).read_text())
    env = Environment(loader=BaseLoader())

    doc_to_text: str = spec["doc_to_text"]
    doc_to_target: str | None = spec.get("doc_to_target")
    doc_to_choice: list[str] | None = spec.get("doc_to_choice")

    has_target = doc_to_target is not None

    def fmt(row: dict) -> dict:
        prompt = _render(doc_to_text, env, row)

        if not has_target:
            return {"text": prompt}

        # Resolve target — may be a column name or a Jinja2 template
        assert doc_to_target is not None
        if "{{" in doc_to_target or "{%" in doc_to_target:
            target_val = _render(doc_to_target, env, row)
        else:
            target_val = row[doc_to_target]

        # For MCQ: target is an index into doc_to_choice
        if doc_to_choice is not None:
            target_val = doc_to_choice[int(target_val)]

        return {"prompt": prompt, "completion": str(target_val)}

    return ds.map(fmt)
