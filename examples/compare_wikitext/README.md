GPT-2 fine-tuned on WikiText (`EleutherAI/bergson-wikitext-512-chunks`, 4,608 training chunks) with plain AdamW: eps_root 1e-17, betas 0.9/0.999, lr 4e-4 polynomial (warmup 25%), batch 256, 4 epochs, seed 42. Every method scores the same 50 test chunks (`test[0:50]`); LDS is the per-query Spearman correlation between a method's summed scores and the measured query-loss change over 100 random 1%-drop retrains (mean over queries, 95% CI from a 10k bootstrap over subsets); the proponent QLD is the mean query-loss increase after retraining without the query's top 1% (46) training chunks by that method's scores (95% CI from a 10k bootstrap over queries). Removing a random 1% changes query loss by 0.0008 on average. Held-out loss (test chunks 50 onward) dropped from 3.545 to 3.111 over training; metasmoothness 0.989.

| method | proponent QLD | 95% CI | LDS | 95% CI | median | min | max | queries p<.05 |
|---|---|---|---|---|---|---|---|---|
| MAGIC (per-query) | 0.100 | [0.090, 0.112] | 0.931 | [0.925, 0.936] | 0.933 | 0.804 | 0.970 | 50/50 |
| Shampoo | 0.071 | [0.060, 0.082] | 0.517 | [0.491, 0.539] | 0.532 | 0.294 | 0.703 | 50/50 |
| EK-FAC | 0.070 | [0.058, 0.082] | 0.454 | [0.426, 0.479] | 0.453 | 0.095 | 0.664 | 49/50 |
| BM25 | 0.062 | [0.048, 0.076] | 0.220 | [0.185, 0.252] | 0.253 | -0.168 | 0.486 | 28/50 |
| Jina v5 semantic search | 0.046 | [0.035, 0.059] | 0.124 | [0.087, 0.160] | 0.115 | -0.108 | 0.483 | 11/50 |
| TrackStar (projection 64) | 0.045 | [0.036, 0.055] | 0.270 | [0.240, 0.295] | 0.294 | -0.011 | 0.513 | 37/50 |
| TrackStar (Adam, projection 64) | 0.043 | [0.034, 0.052] | 0.225 | [0.195, 0.252] | 0.234 | -0.011 | 0.434 | 29/50 |
| TRAK (per-module kernel, projection 32) | 0.036 | [0.028, 0.045] | 0.215 | [0.185, 0.244] | 0.210 | 0.011 | 0.417 | 28/50 |
| TrackStar (projection 32) | 0.035 | [0.027, 0.044] | 0.211 | [0.183, 0.238] | 0.222 | 0.005 | 0.386 | 28/50 |
| TrackStar (Adam, projection 32) | 0.032 | [0.025, 0.041] | 0.173 | [0.144, 0.201] | 0.179 | -0.094 | 0.467 | 20/50 |
| SOURCE-Adam | 0.024 | [0.018, 0.030] | 0.154 | [0.126, 0.181] | 0.147 | -0.144 | 0.412 | 15/50 |
| SOURCE | 0.022 | [0.017, 0.027] | 0.165 | [0.138, 0.191] | 0.171 | -0.120 | 0.419 | 16/50 |
| Activation similarity | 0.000 | [-0.000, 0.001] | 0.110 | [0.070, 0.149] | 0.106 | -0.137 | 0.361 | 11/50 |
| Gradient cosine similarity | -0.001 | [-0.001, -0.000] | 0.019 | [-0.008, 0.045] | 0.046 | -0.190 | 0.246 | 2/50 |
| TRAK (joint kernel, projection 16) | — | — | 0.110 | [0.079, 0.141] | 0.123 | -0.132 | 0.325 | 12/50 |

The README table reports the per-module TRAK kernel; TRAK's joint Gram over the concatenated sketch scores lower here and on an 8k SmolLM2 bank (0.08 vs 0.21). TrackStar is swept over projection dims 16/32/64 (per-module random projection of the gradients), plain and Adam-normalized. The four non-gradient-method baselines come from `examples/bank_baselines` run with `--bank runs/compare_wikitext/random --query_split "test[0:50]"`: BM25 lexical overlap (`bm25_baseline.py`), semantic search with `jinaai/jina-embeddings-v5-text-small` (`semantic_baseline.py`), cosine similarity between each training chunk's full-parameter loss gradient and the query's on the trained model (`gradient_baseline.py`, TracIn-style, no preconditioning), and cosine similarity between the mean-pooled input activations of the attributed linear modules, per-module L2-normalized and concatenated (`activation_baseline.py`).

Reproduce:

```bash
bergson examples/compare_wikitext/magic.yaml        # train, MAGIC scores, the retrain bank
bergson examples/compare_wikitext/interval.yaml     # evenly spaced checkpoints for SOURCE
python -c "from bergson.utils.trainer_export import export_checkpoints; export_checkpoints('runs/compare_wikitext/interval', steps=[72])"
bergson examples/compare_wikitext/ekfac.yaml
bergson examples/compare_wikitext/shampoo.yaml     # EK-FAC pipeline with Shampoo factors
bergson examples/compare_wikitext/trackstar.yaml    # projection 16/32/64, plain and Adam-normalized
bergson examples/compare_wikitext/source.yaml       # plain and Adam-preconditioned
bergson examples/compare_wikitext/trak.yaml
bergson examples/compare_wikitext/metasmoothness.yaml
for b in bm25 semantic gradient activation; do   # each also writes baselines/${b}_scores/scores for the filters
  python -m examples.bank_baselines.${b}_baseline --bank runs/compare_wikitext/random --query_split "test[0:50]" --out runs/compare_wikitext/baselines
done
for m in magic ekfac shampoo trak trackstar_p16 trackstar_adam_p16 trackstar_p32 trackstar_adam_p32 trackstar_p64 trackstar_adam_p64 source source_adam bm25 semantic gradient activation; do
  bergson examples/compare_wikitext/filters/filter_$m.yaml
done
python examples/compare_wikitext/lds_from_bank.py --validation runs/compare_wikitext/random/validation.csv --out runs/compare_wikitext/lds_magic.json
for m in ekfac shampoo trak trackstar_p16 trackstar_adam_p16 trackstar_p32 trackstar_adam_p32 trackstar_p64 trackstar_adam_p64; do python examples/compare_wikitext/lds_from_bank.py --sign grad --scores runs/compare_wikitext/$m/scores --bank runs/compare_wikitext/random --out runs/compare_wikitext/lds_$m.json; done
for b in bm25 semantic gradient activation; do python examples/compare_wikitext/lds_from_bank.py --sign loss --npy runs/compare_wikitext/baselines/${b}_scores.npy --bank runs/compare_wikitext/random --out runs/compare_wikitext/lds_$b.json; done
for m in source source_adam; do python examples/compare_wikitext/lds_from_bank.py --sign loss --scores runs/compare_wikitext/$m/scores --bank runs/compare_wikitext/random --out runs/compare_wikitext/lds_$m.json; done
python examples/compare_wikitext/qld_from_filters.py runs/compare_wikitext runs/compare_wikitext/random
python examples/compare_wikitext/lds_tables.py runs/compare_wikitext
```

EK-FAC, Shampoo, TRAK and TrackStar scores are influence-signed (higher = proponent) and MAGIC and SOURCE loss-signed, hence `--sign`; the baseline scripts already write loss-signed matrices (negated similarity), so their score directories need no sign flip. `filters/` reuses the bank's random retrains as the matched control.
