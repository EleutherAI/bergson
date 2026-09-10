# Leaderboard

[Linear datamodeling score](https://arxiv.org/abs/2303.14186) (LDS) and 1% proponent-filter query loss difference (QLD) values for GPT-2 fine-tuned on WikiText. QLD is the mean increase in a query's loss after retraining without the most influential training items as determined by a data filtering method.

| Method | Proponent QLD [95% CI] | LDS [95% CI] |
|:---|:---:|:---:|
| MAGIC | 0.100 [0.090, 0.112] | 0.931 [0.925, 0.936] |
| Shampoo | 0.071 [0.060, 0.082] | 0.517 [0.491, 0.539] |
| EK-FAC | 0.070 [0.058, 0.082] | 0.454 [0.426, 0.479] |
| BM25 | 0.062 [0.048, 0.076] | 0.220 [0.185, 0.252] |
| Jina v5 semantic search | 0.046 [0.035, 0.059] | 0.124 [0.087, 0.160] |
| TrackStar (projection 64) | 0.045 [0.036, 0.055] | 0.270 [0.240, 0.295] |
| TrackStar (Adam, projection 64) | 0.043 [0.034, 0.052] | 0.225 [0.195, 0.252] |
| TRAK | 0.036 [0.028, 0.045] | 0.215 [0.185, 0.244] |
| TrackStar (projection 32) | 0.035 [0.027, 0.044] | 0.211 [0.183, 0.238] |
| TrackStar (Adam, projection 32) | 0.032 [0.025, 0.041] | 0.173 [0.144, 0.201] |
| SOURCE-Adam | 0.024 [0.018, 0.030] | 0.154 [0.126, 0.181] |
| SOURCE | 0.022 [0.017, 0.027] | 0.165 [0.138, 0.191] |
| Activation similarity | 0.000 [-0.000, 0.001] | 0.110 [0.070, 0.149] |
| Gradient cosine similarity | -0.001 [-0.001, -0.000] | 0.019 [-0.008, 0.045] |

Held-out loss on the WikiText corpus dropped from 3.545 to 3.111 over training. Every row scores the same model and query set; per-query statistics, run configs and reproduction steps are in [examples/compare_wikitext_stdadam](examples/compare_wikitext_stdadam/README.md).
