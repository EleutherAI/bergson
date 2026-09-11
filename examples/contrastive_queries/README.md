# Contrastive queries

A contrastive query scores training data by how it moves one loss relative to another: the loss on a behaviour evaluation minus the loss on a general-capability control. Proponents of that query are the training items that make the behaviour likelier without helping general capability, which is what a data filter for that behaviour should remove.

- `build_eval_queries.py` writes five positive sets (sycophancy, toxicity, consciousness claims, self-awareness, power-seeking) and an MMLU control as JSONL with `prompt`, `completion` and `text` columns.
- `magic_contrast.yaml` scores one of them with MAGIC using `query_contrast`. The same fields exist on the `build`-based pipelines (`contrast` on `build`, `query_contrast` on `trackstar`, `ekfac` and `approxunrolling`).

```bash
python -m examples.contrastive_queries.build_eval_queries --out queries/
bergson examples/contrastive_queries/magic_contrast.yaml
```

The loss covers the completion only when `prompt_column` and `completion_column` are set, which needs a tokenizer with a chat template. For a base model use `prompt_column: text` instead.
