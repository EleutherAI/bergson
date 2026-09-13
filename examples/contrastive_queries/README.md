# Contrastive queries

A contrastive query scores training data by how it moves one loss relative to another: the loss on a behaviour evaluation minus the loss on a general-capability control. Proponents of that query are the training items that make the behaviour likelier without helping general capability, which is what a data filter for that behaviour should remove.

- `build_eval_queries.py` writes five positive sets (sycophancy, toxicity, consciousness claims, self-awareness, power-seeking) and an MMLU control as JSONL of source rows.
- `formats/*.yaml` are the Jinja templates (`doc_to_text`, `doc_to_target`) that render a row into a prompt and a completion; pass one as `format_template` on the query `DataConfig`.
- `magic_contrast.yaml` scores one set with MAGIC using `query_contrast`. The same field exists on `trackstar`, `ekfac` and `approxunrolling`.

```bash
python -m examples.contrastive_queries.build_eval_queries --out queries/
bergson examples/contrastive_queries/magic_contrast.yaml
```

The prompt and completion are rendered through the tokenizer's chat template as a user and an assistant turn, and the loss covers the assistant turn only.

Base models whose chat tokens are untrained (Qwen2.5 base) need a plain template instead: copy the tokenizer and replace its `chat_template` with `formats/plain_chat_template.jinja`, which joins the prompt and completion with a space and no special tokens, then point `tokenizer` at that copy.
