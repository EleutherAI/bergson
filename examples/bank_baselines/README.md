# Non-gradient attribution baselines

Text similarity baselines evaluated by mean LDS over 50 test queries.

- `bm25_baseline.py` — BM25 lexical overlap: pure surface-form term overlap, no model or embedding.
- `gradient_baseline.py` — gradient cosine similarity: cosine of the full per-example loss gradients on the bank's model (TracIn-style, unpreconditioned).
- `activation_baseline.py` — activation similarity: each doc is the mean-pooled input activation to every linear matrix of the model, L2-normalized per matrix and concatenated, then cosine similarity.
- `semantic_baseline.py` — semantic search with `jinaai/jina-embeddings-v3` (asymmetric `retrieval.query`/`retrieval.passage`).
- `qwen3_baseline.py` — semantic search with `Qwen/Qwen3-Embedding-8B`, a SOTA decoder embedder (`--model` to swap).

Each produces a `[num_train_docs, num_queries]` score matrix.

## Running

Point a baseline at a re-train bank (written with `save_retrained_models=true`); it reads the model and dataset from the bank's `config.yaml`:

```bash
python -m examples.bank_baselines.bm25_baseline        --bank runs/retrain_bank_path
python -m examples.bank_baselines.gradient_baseline    --bank runs/retrain_bank_path
python -m examples.bank_baselines.activation_baseline  --bank runs/retrain_bank_path
python -m examples.bank_baselines.semantic_baseline    --bank runs/retrain_bank_path
python -m examples.bank_baselines.qwen3_baseline       --bank runs/retrain_bank_path
```

Omit `--bank` to build the default GPT-2/WikiText bank (`examples/magic/gpt2_wikitext_bank.yaml`) first. `--query_split` sets the query set (default `test[1:51]`), `--out` the output dir (default `runs/bank_baselines/`).

## Results

GPT-2 / WikiText bank (100 subsets, 1% leave-out, `eps_root 1e-8`):

| Method | mean ρ |
| --- | --- |
| BM25 lexical overlap | 0.16 |
| Qwen3-Embedding-8B | 0.11 |
| activation similarity | 0.09 |
| Jina v3 semantic search | 0.06 |
| gradient cosine similarity | 0.05 |

## Notes

`jina-embeddings-v3`'s custom code predates transformers 5.x; `semantic_baseline.load_model` sets an `all_tied_weights_keys` default and resets the NaN LoRA `lora_dropout_mask` buffers to ones so it loads and runs. NVIDIA's NV-Embed-v2 is a comparable SOTA embedder but its custom code is incompatible with transformers 5.x, so `qwen3_baseline.py` uses Qwen3-Embedding instead.
