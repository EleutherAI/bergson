# Contrastive queries

A contrastive query scores training data by how it moves one loss relative to another. Here the query is the Anthropic power-seeking evaluation with the power-seeking answer as the completion, and the control is the same questions with the other answer, so proponents are the training items that make the model prefer the power-seeking answer.

- `build_power_seeking_queries.py` writes `queries/power_seeking.jsonl`.
- `formats/power_seeking.yaml` and `formats/power_seeking_control.yaml` render a row into the prompt and either completion; pass one as `format_template` on the query or contrast `DataConfig`.
- `magic_contrast.yaml` scores the training set with MAGIC using `query.contrast`. The field is part of every pipeline's query set (`magic`, `validate`, `trackstar`, `ekfac`, `trak`, `approxunrolling`).

```bash
python -m examples.contrastive_queries.build_power_seeking_queries --out queries/
bergson examples/contrastive_queries/magic_contrast.yaml
```

The prompt and completion are rendered through the tokenizer's chat template as a user and an assistant turn, and the loss covers the assistant turn only.
