# Contrastive queries

A contrastive query attributes a query loss minus a control loss. The query is the Anthropic power-seeking evaluation with the power-seeking answer as the completion, and the control is the same questions with the other answer, so proponents are the training items that make the model prefer the power-seeking answer.

- `build_power_seeking_queries.py` writes the datasets to disk.
- `magic_contrast.yaml` scores the training set with MAGIC using `query.contrast`.

```bash
python -m examples.powerseeking_contrastive_query.build_power_seeking_queries --out queries/
bergson examples/powerseeking_contrastive_query/magic_contrast.yaml
```
