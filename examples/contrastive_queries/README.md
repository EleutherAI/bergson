# Contrastive queries

A contrastive query attributes a query loss minus a control loss. Here the query is the Anthropic power-seeking evaluation with the power-seeking answer as the completion, and the control is the same questions with the other answer, so proponents are the training items that make the model prefer the power-seeking answer.

- `build_power_seeking_queries.py` writes the datasets to disk.
- `magic_contrast.yaml` scores the training set with MAGIC using `query.contrast`.

```bash
python -m examples.contrastive_queries.build_power_seeking_queries --out queries/
bergson examples/contrastive_queries/magic_contrast.yaml
```
