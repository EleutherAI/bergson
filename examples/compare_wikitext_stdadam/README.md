GPT-2 fine-tuned on WikiText (`EleutherAI/bergson-wikitext-512-chunks`, 4,608 training chunks) with plain AdamW: eps_root 1e-17, betas 0.9/0.999, lr 4e-4 polynomial (warmup 25%), batch 256, 4 epochs, seed 42. Every method scores the same 50 test chunks (`test[0:50]`); LDS is the per-query Spearman correlation between a method's summed scores and the measured query-loss change over 100 random 1%-drop retrains (mean over queries, 95% CI from a 10k bootstrap over subsets); the proponent QLD is the mean query-loss increase after retraining without the query's top 1% (46) training chunks by that method's scores (95% CI from a 10k bootstrap over queries). Removing a random 1% changes query loss by 0.0008 on average. Held-out loss (test chunks 50 onward) dropped from 3.545 to 3.111 over training; metasmoothness 0.989.

| method | LDS | 95% CI | median | min | max | queries p<.05 | proponent QLD | 95% CI |
|---|---|---|---|---|---|---|---|---|
| MAGIC (per-query) | 0.931 | [0.925, 0.936] | 0.933 | 0.804 | 0.970 | 50/50 | 0.100 | [0.090, 0.112] |
| EK-FAC | 0.454 | [0.426, 0.479] | 0.453 | 0.095 | 0.664 | 49/50 | 0.070 | [0.058, 0.082] |
| TRAK (per-module kernel, projection 32) | 0.215 | [0.185, 0.244] | 0.210 | 0.011 | 0.417 | 28/50 | 0.036 | [0.028, 0.045] |
| TrackStar | 0.211 | [0.183, 0.238] | 0.222 | 0.005 | 0.386 | 28/50 | 0.035 | [0.027, 0.044] |
| TrackStar+Adam | 0.173 | [0.144, 0.201] | 0.179 | -0.094 | 0.467 | 20/50 | 0.032 | [0.025, 0.041] |
| SOURCE | 0.165 | [0.138, 0.191] | 0.171 | -0.120 | 0.419 | 16/50 | 0.022 | [0.017, 0.027] |
| SOURCE-Adam | 0.154 | [0.126, 0.181] | 0.147 | -0.144 | 0.412 | 15/50 | 0.024 | [0.018, 0.030] |
| TRAK (joint kernel, projection 16) | 0.110 | [0.079, 0.141] | 0.123 | -0.132 | 0.325 | 12/50 | — | — |

The README table reports the per-module TRAK kernel; TRAK's joint Gram over the concatenated sketch scores lower here and on an 8k SmolLM2 bank (0.08 vs 0.21).

Reproduce:

```bash
bergson examples/compare_wikitext_stdadam/magic.yaml        # train, MAGIC scores, the retrain bank
bergson examples/compare_wikitext_stdadam/interval.yaml     # evenly spaced checkpoints for SOURCE
python -c "from bergson.utils.trainer_export import export_checkpoints; export_checkpoints('runs/compare_wikitext_stdadam/interval', steps=[72])"
bergson examples/compare_wikitext_stdadam/ekfac.yaml
bergson examples/compare_wikitext_stdadam/trackstar.yaml    # plain and Adam-normalized
bergson examples/compare_wikitext_stdadam/source.yaml       # plain and Adam-preconditioned
bergson examples/compare_wikitext_stdadam/trak.yaml
bergson examples/compare_wikitext_stdadam/metasmoothness.yaml
for m in magic ekfac trak trackstar trackstar_adam source source_adam; do
  bergson examples/compare_wikitext_stdadam/filters/filter_$m.yaml
done
python scripts/lds_from_bank.py --validation runs/compare_wikitext_stdadam/random/validation.csv --out runs/compare_wikitext_stdadam/lds_magic.json
for m in ekfac trak trackstar trackstar_adam; do python scripts/lds_from_bank.py --sign grad --scores runs/compare_wikitext_stdadam/$m/scores --bank runs/compare_wikitext_stdadam/random --out runs/compare_wikitext_stdadam/lds_$m.json; done
for m in source source_adam; do python scripts/lds_from_bank.py --sign loss --scores runs/compare_wikitext_stdadam/$m/scores --bank runs/compare_wikitext_stdadam/random --out runs/compare_wikitext_stdadam/lds_$m.json; done
python scripts/qld_from_filters.py runs/compare_wikitext_stdadam runs/compare_wikitext_stdadam/random
python scripts/lds_tables.py runs/compare_wikitext_stdadam
```

EK-FAC and TrackStar scores are influence-signed (higher = proponent) and MAGIC and SOURCE loss-signed, hence `--sign`. `filters/` reuses the bank's random retrains as the matched control.
