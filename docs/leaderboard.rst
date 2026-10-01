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
   * - SOURCE (Adam)
     - 0.071 [0.060, 0.084]
     - 0.473 [0.446, 0.498]
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
   * - Activation similarity + MAC
     - TODO
     - 0.157 [0.120, 0.192]
   * - Activation similarity
     - 0.000 [-0.000, 0.001]
     - 0.110 [0.070, 0.149]

Results for GPT-2 finetuned on 4 epochs of the WikiText corpus. Held-out loss dropped from 3.545 to 3.111 over training. Every row scores the same model and query set; per-query statistics, run configs and reproduction steps are in `examples/compare_wikitext <https://github.com/EleutherAI/bergson/tree/main/examples/compare_wikitext>`_.

Per-token
---------

Per-token attribution scores, which attribute each training token's loss term, enable substantially more efficacious data filtering. The proponent QLD masks the query's top 1% of training tokens by each method's per-token scores while the control masks a random 1% of tokens. Each method's per-token scores are the forward-mode equivalent of its document scores: a chunk's rows split its score over the loss terms of its tokens.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Method
     - Proponent QLD [95% CI]
   * - MAGIC
     - 1.375 [1.348, 1.403]
   * - EK-FAC + ASTRA
     - 1.082 [1.047, 1.116]
   * - Eigenvalue-corrected Shampoo
     - 0.922 [0.887, 0.957]
   * - SOURCE (Adam, EK-FAC)
     - 0.908 [0.877, 0.942]
   * - EK-FAC
     - 0.863 [0.833, 0.894]
   * - KFAC
     - 0.786 [0.758, 0.815]
   * - BM25
     - 0.677 [0.650, 0.704]
   * - TrackStar (no optimizer correction, projection 64)
     - 0.489 [0.465, 0.515]
   * - Gradient cosine similarity
     - 0.217 [0.206, 0.229]
   * - TRAK (8-model ensemble)
     - 0.119 [0.112, 0.125]
   * - Activation similarity
     - 0.028 [0.018, 0.039]
   * - Qwen3-Embedding-8B semantic search
     - 0.008 [0.006, 0.010]

Configs and per-query statistics are in `examples/compare_wikitext/tokens <https://github.com/EleutherAI/bergson/tree/main/examples/compare_wikitext/tokens>`_.

Notes
-----

There are many method selection considerations not captured by LDS or QLD. Here is some additional information.

MAGIC
   This method backpropagates through the training process once per query and doesn't support query batching. It works well in FP32 or TF32 but may significantly degrade when used to attribute BF16 training runs, or training runs with low batch sizes.

ASTRA
   ASTRA improves any existing Kronecker-factored Hessian approximation during each query's application stage, so it adds a constant amount of compute to each query on top of a basic influence function like EK-FAC.

EK-FAC
   This is a classic influence function in the form $g_t H^{-1} g_q$ using the EK-FAC Hessian approximation for H. The H can be swapped out for other matrices including KFAC, Shampoo, and MAC.

KFAC
   KFAC supports gradient compression with ``projection_dim``. This produces a highly efficient variant based on a reusable gradient store, suitable for the retrieval stage of a data attribution pipeline, at the cost of some LDS/QLD.

Eigenvalue-corrected Shampoo
   This is a custom influence function in the form $g_t H^{-1} g_q$ using the Shampoo preconditioner as the H. Its mathematical relationship with the true Hessian is somewhere between "null" and "extremely tenuous", but it performs well anyway. It uses an eigenvalue-correction similar to EK-FAC's, and to the variant used in SOAP optimization.

SOURCE
   SOURCE uses a weighted combination of Hessian approximations fit on different segments of the training trajectory, so it may perform relatively better in longer or multi-phase training runs.

TrackStar
   TrackStar is a highly efficient method that can more readily scale to pretraining via a reusable compressed gradient store. Presumably it's designed to be used as the first stage of a data search pipeline that eventually uses a more powerful influence function to re-rank the top-k items.

TRAK
   TRAK's performance matches `existing results <https://arxiv.org/pdf/2405.12186>`_ in the literature.

Gradient similarity
   A weak baseline that considers only the direct effect of the parameter update on the query.

Semantic search
   A strong baseline that uses a model optimized to retrieve documents with a similar meaning to the query document.

BM25
   A strong baseline that uses a model optimized to retrieve documents with words matching those in the query document.

Activation similarity
   A novel baseline that uses the model of interest's activations rather than gradients. To some extent any representation space can be used to compute data similarities, so to the extent model gradients are especially good data for making counterfactual predictions we expect that they will outperform model activations.
