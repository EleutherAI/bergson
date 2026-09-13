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
   * - Shampoo
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
   * - TRAK (8-model ensemble)
     - 0.007 [0.005, 0.010]
     - 0.045 [0.015, 0.073]
   * - TRAK (4-model ensemble, DsDm averaging)
     - 0.005 [0.003, 0.007]
     - 0.037 [0.009, 0.064]
   * - `DDA <https://arxiv.org/abs/2410.01285>`_ (debiased gradient dot product)
     - 0.000 [0.000, 0.001]
     - -0.054 [-0.088, -0.020]
   * - Activation similarity
     - 0.000 [-0.000, 0.001]
     - 0.110 [0.070, 0.149]

Results for GPT-2 finetuned on 4 epochs of the WikiText corpus. Held-out loss dropped from 3.545 to 3.111 over training. Every row scores the same model and query set; per-query statistics, run configs and reproduction steps are in `examples/compare_wikitext <https://github.com/EleutherAI/bergson/tree/main/examples/compare_wikitext>`_.
