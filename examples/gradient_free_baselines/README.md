# Gradient-free attribution baselines

Text similarity baselines evaluated by mean LDS over 50 test queries.

- `bm25_baseline.py` — BM25 lexical overlap: term overlap, no model or embedding.
- `dsir_baseline.py` — DSIR importance weights: hashed-n-gram likelihood ratio of the query set against the training corpus (Xie et al., 2023).
- `activation_baseline.py` — activation similarity: each doc is the mean-pooled input activation to every linear matrix of the model, L2-normalized per matrix and concatenated, then cosine similarity.
- `semantic_baseline.py` — semantic search with `jinaai/jina-embeddings-v5-text-small` (asymmetric retrieval query/document prompts; `--model` to swap, e.g. `jinaai/jina-embeddings-v3`).
- `qwen3_baseline.py` — semantic search with `Qwen/Qwen3-Embedding-8B`, a SOTA decoder embedder (`--model` to swap).

Each produces a `[num_train_docs, num_queries]` score matrix.

## Running

Point a baseline at a re-train bank (written with `save_models=true`); it reads the model and dataset from the bank's `config.yaml`:

```bash
python -m examples.gradient_free_baselines.bm25_baseline        --bank runs/retrain_bank_path
python -m examples.gradient_free_baselines.dsir_baseline        --bank runs/retrain_bank_path
python -m examples.gradient_free_baselines.activation_baseline  --bank runs/retrain_bank_path
python -m examples.gradient_free_baselines.semantic_baseline    --bank runs/retrain_bank_path
python -m examples.gradient_free_baselines.qwen3_baseline       --bank runs/retrain_bank_path
```

Omit `--bank` to build the default GPT-2/WikiText bank (`examples/magic/gpt2_wikitext_bank.yaml`) first. `--query_split` sets the query set (default `test[1:51]`), `--out` the output dir (default `runs/gradient_free_baselines/`).

## Results

Results are on the [leaderboard](../../LEADERBOARD.md).
