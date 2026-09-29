Leaderboard
===========

Indicative performance of data attribution methods in the finetuning regime.

`Linear datamodeling score <https://arxiv.org/abs/2303.14186>`_ (LDS) is the accuracy of a method for producing global data rankings by influence. The query loss difference (QLD) shows how much model loss can be increased by retraining without the most highly ranked data by influence (here the top 1%), compared to a random removal baseline.

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Method
     - Proponent QLD [95% CI]
     - LDS [95% CI]
   * - MAGIC
     - 0.100 [0.090, 0.112]
     - 0.931 [0.925, 0.936]
   * - MAGIC (cross-seed)
     - 0.098 [0.087, 0.110]
     - 0.829 [0.815, 0.840]
   * - EK-FAC + ASTRA
     - 0.074 [0.063, 0.087]
     - 0.643 [0.624, 0.660]
   * - Eigenvalue-corrected Shampoo + ASTRA
     - 0.072 [0.061, 0.085]
     - 0.625 [0.604, 0.643]
   * - Eigenvalue-corrected Shampoo
     - 0.071 [0.060, 0.082]
     - 0.517 [0.491, 0.539]
   * - EK-FAC
     - 0.070 [0.058, 0.082]
     - 0.454 [0.426, 0.479]
   * - KFAC
     - 0.067 [0.056, 0.080]
     - 0.420 [0.391, 0.446]
   * - BM25
     - 0.062 [0.048, 0.076]
     - 0.220 [0.185, 0.252]
   * - `Qwen3-Embedding-8B <https://huggingface.co/spaces/mteb/leaderboard>`_ semantic search
     - 0.049 [0.038, 0.061]
     - 0.132 [0.093, 0.169]
   * - Jina v5 semantic search
     - 0.046 [0.035, 0.059]
     - 0.124 [0.087, 0.160]
   * - TrackStar (no optimizer correction, projection 64)
     - 0.045 [0.036, 0.055]
     - 0.270 [0.240, 0.295]
   * - TrackStar (Adam, projection 64)
     - 0.043 [0.034, 0.052]
     - 0.225 [0.195, 0.252]
   * - TrackStar (no optimizer correction, projection 32)
     - 0.035 [0.027, 0.044]
     - 0.211 [0.183, 0.238]
   * - TrackStar (Adam, projection 32)
     - 0.032 [0.025, 0.041]
     - 0.173 [0.144, 0.201]
   * - TRAK (8-model ensemble)
     - 0.032 [0.024, 0.040]
     - 0.138 [0.111, 0.165]
   * - SOURCE (Adam)
     - 0.024 [0.018, 0.030]
     - 0.154 [0.126, 0.181]
   * - KFAC (projection 64)
     - 0.023 [0.016, 0.032]
     - 0.103 [0.076, 0.128]
   * - SOURCE
     - 0.022 [0.017, 0.027]
     - 0.165 [0.138, 0.191]
   * - TrackStar (no optimizer correction, projection 16)
     - 0.022 [0.016, 0.028]
     - 0.143 [0.113, 0.173]
   * - Gradient cosine similarity
     - 0.021 [0.016, 0.027]
     - 0.156 [0.131, 0.181]
   * - TrackStar (Adam, projection 16)
     - 0.020 [0.014, 0.026]
     - 0.103 [0.072, 0.133]
   * - Gradient dot product
     - 0.019 [0.015, 0.024]
     - 0.156 [0.130, 0.180]
   * - Projected gradient cosine similarity
     - 0.016 [0.012, 0.020]
     - 0.132 [0.103, 0.159]
   * - Activation similarity
     - 0.000 [-0.000, 0.001]
     - 0.110 [0.070, 0.149]

Results for GPT-2 finetuned on 4 epochs of the WikiText corpus. Held-out loss dropped from 3.545 to 3.111 over training. Every row scores the same model and query set; per-query statistics, run configs and reproduction steps are in `examples/compare_wikitext <https://github.com/EleutherAI/bergson/tree/main/examples/compare_wikitext>`_.

Per-token
---------

Per-token scores attribute each training token's loss term. The proponent QLD masks the query's top 1% of training tokens by each method's per-token scores (23,484 loss terms per query, spread over about 4,000 of the 4,608 chunks) and retrains; the control retrains without a random 1% of tokens, three retrains per query, which changes query loss by 0.0006 on average. Removing the top 1% of whole chunks by document-level EK-FAC changes it by 0.070. A per-token LDS needs a token-level retrain bank and is left for later.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Method
     - Proponent QLD [95% CI]
   * - MAGIC
     - 1.375 [1.348, 1.403]
   * - EK-FAC + ASTRA
     - 1.082 [1.047, 1.116]
   * - EK-FAC
     - 0.863 [0.833, 0.894]

Configs and per-query statistics are in `examples/compare_wikitext/tokens <https://github.com/EleutherAI/bergson/tree/main/examples/compare_wikitext/tokens>`_.
