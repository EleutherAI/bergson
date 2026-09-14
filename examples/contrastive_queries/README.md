# Contrastive queries

A contrastive query attributes a query loss minus a control loss. Here the query is the Anthropic power-seeking evaluation with the power-seeking answer as the completion, and the control is the same questions with the other answer, so proponents are the training items that make the model prefer the power-seeking answer.

- `build_power_seeking_queries.py` writes `queries/power_seeking.jsonl` and `queries/power_seeking_control.jsonl`, the same questions with either answer.
- `formats/question_answer.yaml` renders a row's `question` as the prompt and `answer` as the completion, the rendering lm-evaluation-harness uses for `advanced_ai_risk` with the dialogue turns supplied by the chat template.
- `magic_contrast.yaml` scores the training set with MAGIC using `query.contrast`. The field is part of every pipeline's query set (`magic`, `validate`, `trackstar`, `ekfac`, `trak`, `approxunrolling`).

```bash
python -m examples.contrastive_queries.build_power_seeking_queries --out queries/
bergson examples/contrastive_queries/magic_contrast.yaml
```
