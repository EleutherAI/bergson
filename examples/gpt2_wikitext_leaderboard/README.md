This directory contains YAMLs sufficient to replicate the results in the Bergson finetuning leaderboard. The YAMLs for each method are a good jumping-off point for applying the method to your use case, with sensible configuration values. Reproduce any results by re-running the YAMLS with `bergson <path to YAML>`.

## Methods

The leaderboard setting is GPT-2 fine-tuned on WikiText. See `examples/gpt2_wikitext_leaderboard/1_magic.yaml` to see the training hyperparameters, and reproduce the training run in the CLI with `bergson examples/gpt2_wikitext_leaderboard/1_magic.yaml`. Held-out loss (test chunks 50 onward) dropped from 3.545 to 3.111 over training and the empirical metasmoothness of the training run was 0.989.

Every method scores the same 50 test chunks (`test[0:50]`). We report the QLD and mean LDS. LDS is the per-query Spearman correlation between a method's summed scores and the measured query-loss change over 100 random leave-1%-out retrains (mean over queries, 95% CI from a 10k bootstrap over subsets); the proponent QLD is the mean increase in query loss after retraining without the query's top 1% (46) training chunks by that method's scores (95% CI from a 10k bootstrap over queries).

## Results

| method | proponent QLD | 95% CI | LDS | 95% CI | median | min | max | queries p<.05 |
|---|---|---|---|---|---|---|---|---|
| MAGIC (per-query) | 0.100 | [0.090, 0.112] | 0.931 | [0.925, 0.936] | 0.933 | 0.804 | 0.970 | 50/50 |
| MAGIC (cross-seed) | 0.098 | [0.087, 0.110] | 0.829 | [0.815, 0.840] | 0.836 | 0.658 | 0.932 | 50/50 |
| EKFAC + CG | 0.077 | [0.066, 0.089] | 0.685 | [0.664, 0.702] | 0.699 | 0.457 | 0.805 | 50/50 |
| EKFAC + ASTRA | 0.073 | [0.061, 0.087] | 0.579 | [0.554, 0.601] | 0.601 | -0.042 | 0.811 | 48/50 |
| Eigenvalue-corrected Shampoo + ASTRA | 0.072 | [0.061, 0.085] | 0.625 | [0.604, 0.643] | 0.644 | 0.095 | 0.805 | 49/50 |
| EKFAC + ASTRA (exact paper settings) | 0.071 | [0.060, 0.084] | 0.629 | [0.607, 0.647] | 0.646 | 0.435 | 0.786 | 50/50 |
| Eigenvalue-corrected Shampoo | 0.071 | [0.060, 0.082] | 0.517 | [0.491, 0.539] | 0.532 | 0.294 | 0.703 | 50/50 |
| SOURCE (Adam) | 0.071 | [0.060, 0.084] | 0.473 | [0.446, 0.498] | 0.466 | 0.120 | 0.676 | 49/50 |
| EKFAC | 0.070 | [0.058, 0.082] | 0.454 | [0.426, 0.479] | 0.453 | 0.095 | 0.664 | 49/50 |
| KFAC | 0.067 | [0.056, 0.080] | 0.420 | [0.391, 0.446] | 0.412 | 0.041 | 0.646 | 48/50 |
| LiSSA (Gauss-Newton Hessian) | 0.065 | [0.054, 0.076] | 0.572 | [0.549, 0.592] | 0.572 | 0.391 | 0.783 | 50/50 |
| BM25 | 0.062 | [0.048, 0.076] | 0.220 | [0.185, 0.252] | 0.253 | -0.168 | 0.486 | 28/50 |
| [Qwen3-Embedding-8B](https://huggingface.co/spaces/mteb/leaderboard) semantic search | 0.049 | [0.038, 0.061] | 0.132 | [0.093, 0.169] | 0.132 | -0.119 | 0.483 | 17/50 |
| Jina v5 semantic search | 0.046 | [0.035, 0.059] | 0.124 | [0.087, 0.160] | 0.115 | -0.108 | 0.483 | 11/50 |
| TrackStar (no optimizer correction, projection 64) | 0.045 | [0.036, 0.055] | 0.276 | [0.252, 0.299] | 0.280 | -0.017 | 0.487 | 42/50 |
| TrackStar (Adam, projection 64) | 0.043 | [0.034, 0.052] | 0.242 | [0.216, 0.267] | 0.249 | -0.072 | 0.392 | 36/50 |
| TrackStar (no optimizer correction, projection 32) | 0.035 | [0.027, 0.044] | 0.194 | [0.171, 0.216] | 0.195 | -0.098 | 0.455 | 25/50 |
| TrackStar (Adam, projection 32) | 0.032 | [0.025, 0.041] | 0.182 | [0.156, 0.209] | 0.191 | -0.074 | 0.458 | 24/50 |
| TRAK (8-model ensemble) | 0.032 | [0.024, 0.040] | 0.138 | [0.111, 0.165] | 0.141 | -0.131 | 0.430 | 13/50 |
| KFAC (projection 64) | 0.023 | [0.016, 0.032] | 0.103 | [0.076, 0.128] | 0.107 | -0.137 | 0.309 | 9/50 |
| SOURCE | 0.022 | [0.017, 0.027] | 0.165 | [0.138, 0.191] | 0.171 | -0.120 | 0.419 | 16/50 |
| TrackStar (no optimizer correction, projection 16) | 0.022 | [0.016, 0.028] | 0.120 | [0.090, 0.149] | 0.116 | -0.091 | 0.314 | 12/50 |
| Gradient cosine similarity | 0.021 | [0.016, 0.027] | 0.156 | [0.131, 0.181] | 0.154 | -0.142 | 0.396 | 15/50 |
| TrackStar (Adam, projection 16) | 0.020 | [0.014, 0.026] | 0.114 | [0.084, 0.143] | 0.087 | -0.104 | 0.330 | 13/50 |
| Gradient dot product | 0.019 | [0.015, 0.024] | 0.156 | [0.130, 0.180] | 0.151 | -0.138 | 0.390 | 16/50 |
| Projected gradient cosine similarity | 0.016 | [0.012, 0.020] | 0.132 | [0.103, 0.159] | 0.133 | -0.214 | 0.360 | 13/50 |
| Activation similarity (activation covariance preconditioner) | 0.055 | [0.043, 0.069] | 0.157 | [0.120, 0.192] | 0.152 | -0.110 | 0.476 | 19/50 |
| Activation similarity (activation covariance preconditioner, projection 64) | 0.055 | [0.043, 0.069] | 0.152 | [0.116, 0.187] | 0.158 | -0.123 | 0.465 | 20/50 |
| Activation similarity | 0.049 | [0.037, 0.063] | 0.110 | [0.070, 0.149] | 0.106 | -0.137 | 0.361 | 11/50 |

## Notes on the Methods

MAGIC (cross-seed) applies the seed-42 MAGIC scores to retrains with seed 43 (different data order and dropout): its LDS uses the same 100 subsets retrained at seed 43 and its QLD retrains the proponent filters at seed 43.

KFAC is `kfac.yaml`: the Kronecker factors and damping of `ekfac.yaml` without the eigenvalue correction (`ev_correction: false`), so the two rows isolate what the correction adds.

LiSSA is `lissa.yaml`, the influence function of Bae et al. (2022): each query's damped Gauss-Newton Hessian inverse `(G + 0.01 I)^-1 q`, estimated by LiSSA with the paper's Transformer settings (5,000 terms, 5 repeats) on fresh 8-chunk batches, with each chunk's loss the mean over its tokens as in the paper; the scale 50 is the smallest of the paper's grid (10 to 500) that stays bounded over 5,000 terms (10 and 25 diverge on some queries). At depth 5,000 the estimates have not fully converged: their norm still changes by 5.8% (median) over the last 500 terms. It ran 25.4 h on 8 A40s.

KFAC (projection 64) is `kfac_p64.yaml`: the same factors and damping with the gradients random-projected per module to 64 dimensions as in the TrackStar rows.

TrackStar is swept over projection dims 16/32/64 (per-module random projection of the gradients), plain and Adam-normalized.

TRAK follows Park et al. (2023): eight GPT-2 models trained on independent random 50% subsets of the training set (`trak_ensemble/train_s*.yaml`, seeds 101-104), scored by `trak.yaml` with one global random projection (dim 4096, a different seed per member) of the negative log-odds gradients whitened by the undamped Gram over the training set, each member's scores scale-normalized by the mean absolute entry of its inverse Gram, averaged, and weighted by the members' mean 1 - p.

Gradient cosine similarity is `gradient_cosine.yaml`: bergson `build` of the query gradients and `score` of the training chunks, unprojected and unit-normalized with no Hessian, so each score is the cosine between a training chunk's gradient and a query's over the attributed modules. Projected gradient cosine similarity (the projected steps of the same yaml) is the same with the gradients random-projected per module first, i.e. TrackStar without preconditioning. Gradient dot product is `gradient_dot.yaml`: the gradient cosine pipeline without unit normalization, so each score is the raw inner product of a training chunk's gradient and a query's.

The four gradient-free baselines come from `examples/gpt2_wikitext_leaderboard/gradient_free_baselines` run with `--bank runs/compare_wikitext/random --query_split "test[0:50]"`: BM25 lexical overlap (`bm25_baseline.py`), semantic search with `jinaai/jina-embeddings-v5-text-small` (`semantic_baseline.py`) and with `Qwen/Qwen3-Embedding-8B` (`qwen3_baseline.py`), and cosine similarity between the mean-pooled input activations of the attributed linear modules, per-module L2-normalized and concatenated (`activation_baseline.py`). The activation covariance preconditioner rows multiply each module's pooled activations by (A + 0.1 mean(eig A) I)^-1/2 before normalizing, with A the module's activation factor from `ekfac.yaml` (`--hessian runs/compare_wikitext/ekfac/hessian/kfac`), and the projected row then applies bergson's per-module random projection to 64 dimensions (`--projection_dim 64`).

## Reproducibility

To reproduce (run the numbered yamls first):

```bash
bergson examples/gpt2_wikitext_leaderboard/1_magic.yaml       # train, MAGIC scores, the retrain bank
bergson examples/gpt2_wikitext_leaderboard/2_interval.yaml    # evenly spaced checkpoints for SOURCE and the scored model
python -c "from bergson.utils.trainer_export import export_checkpoints; export_checkpoints('runs/compare_wikitext/interval', steps=[72])"
bergson examples/gpt2_wikitext_leaderboard/ekfac.yaml
bergson examples/gpt2_wikitext_leaderboard/ekfac_astra_cg.yaml
bergson examples/gpt2_wikitext_leaderboard/ekfac_astra_paper.yaml
bergson examples/gpt2_wikitext_leaderboard/ekfac_astra.yaml
bergson examples/gpt2_wikitext_leaderboard/lissa.yaml
bergson examples/gpt2_wikitext_leaderboard/shampoo_astra.yaml
bergson examples/gpt2_wikitext_leaderboard/kfac.yaml
bergson examples/gpt2_wikitext_leaderboard/kfac_p64.yaml
bergson examples/gpt2_wikitext_leaderboard/shampoo.yaml
bergson examples/gpt2_wikitext_leaderboard/trackstar.yaml       # projection 16/32/64
bergson examples/gpt2_wikitext_leaderboard/trackstar_adam.yaml  # projection 16/32/64
bergson examples/gpt2_wikitext_leaderboard/source.yaml          # plain and Adam-preconditioned
for i in 0 1 2 3; do bergson examples/gpt2_wikitext_leaderboard/trak_ensemble/train_s$i.yaml; done           # eight 50%-subset models
bergson examples/gpt2_wikitext_leaderboard/trak.yaml                # scores the eight subset models and averages them
bergson examples/gpt2_wikitext_leaderboard/gradient_cosine.yaml # full and projected gradients
bergson examples/gpt2_wikitext_leaderboard/gradient_dot.yaml    # unnormalized gradient dot products
bergson examples/gpt2_wikitext_leaderboard/magic_seed43.yaml
bergson examples/gpt2_wikitext_leaderboard/metasmoothness.yaml
for b in bm25 semantic qwen3 activation; do   # writes baselines/${b}_scores/scores
  python -m examples.gpt2_wikitext_leaderboard.gradient_free_baselines.${b}_baseline --bank runs/compare_wikitext/random --query_split "test[0:50]" --out runs/compare_wikitext/baselines
done
for p in 0 64; do   # writes baselines/activation_ekfac_scores and baselines/activation_ekfac_p64_scores
  python -m examples.gpt2_wikitext_leaderboard.gradient_free_baselines.activation_baseline --bank runs/compare_wikitext/random --query_split "test[0:50]" --out runs/compare_wikitext/baselines --hessian runs/compare_wikitext/ekfac/hessian/kfac --projection_dim $p
done
for f in magic magic_seed43 ekfac lissa ekfac_astra_cg ekfac_astra_paper ekfac_astra kfac kfac_p64 shampoo trak_ens gradient_dot trackstar trackstar_adam source gradient_cosine bm25 semantic qwen3 activation activation_ekfac activation_ekfac_p64; do
  bergson examples/gpt2_wikitext_leaderboard/filters/filter_$f.yaml
done
python examples/gpt2_wikitext_leaderboard/lds_from_bank.py --validation runs/compare_wikitext/random/validation.csv --out runs/compare_wikitext/lds_magic.json
python examples/gpt2_wikitext_leaderboard/lds_from_bank.py --validation runs/compare_wikitext/magic_seed43/validation.csv --out runs/compare_wikitext/lds_magic_seed43.json
for m in ekfac lissa ekfac_astra_cg ekfac_astra_paper ekfac_astra kfac kfac_p64 shampoo trackstar_p16 trackstar_p32 trackstar_p64 trackstar_adam_p16 trackstar_adam_p32 trackstar_adam_p64 gradient_cosine gradient_cosine_projected gradient_dot; do python examples/gpt2_wikitext_leaderboard/lds_from_bank.py --sign grad --scores runs/compare_wikitext/$m/scores --bank runs/compare_wikitext/random --out runs/compare_wikitext/lds_$m.json; done
python examples/gpt2_wikitext_leaderboard/lds_from_bank.py --sign grad --scores runs/compare_wikitext/trak/scores --bank runs/compare_wikitext/random --out runs/compare_wikitext/lds_trak_ens.json
for b in bm25 semantic qwen3 activation activation_ekfac activation_ekfac_p64; do python examples/gpt2_wikitext_leaderboard/lds_from_bank.py --sign loss --scores runs/compare_wikitext/baselines/${b}_scores/scores --bank runs/compare_wikitext/random --out runs/compare_wikitext/lds_$b.json; done
for m in source source_adam; do python examples/gpt2_wikitext_leaderboard/lds_from_bank.py --sign loss --scores runs/compare_wikitext/$m/scores --bank runs/compare_wikitext/random --out runs/compare_wikitext/lds_$m.json; done
python examples/gpt2_wikitext_leaderboard/qld_from_filters.py runs/compare_wikitext runs/compare_wikitext/random
python examples/gpt2_wikitext_leaderboard/lds_tables.py runs/compare_wikitext
```

## Notes

EKFAC, KFAC, Eigenvalue-corrected Shampoo, TRAK, TrackStar and gradient dot-product scores are influence-signed (higher = proponent) and MAGIC and SOURCE loss-signed, hence `--sign`; the baseline scripts already write loss-signed matrices (negated similarity), so their score directories need no sign flip. `filters/` reuses the bank's random retrains as the matched control.
