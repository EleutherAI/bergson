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
   * - EKFAC + CG
     - 0.077 [0.066, 0.089]
     - 0.685 [0.664, 0.702]
   * - EKFAC + ASTRA
     - 0.076 [0.064, 0.089]
     - 0.605 [0.581, 0.625]
   * - Eigenvalue-corrected Shampoo + ASTRA
     - 0.072 [0.061, 0.085]
     - 0.625 [0.604, 0.643]
   * - EKFAC + ASTRA (exact replication)
     - 0.071 [0.060, 0.084]
     - 0.629 [0.607, 0.647]
   * - Eigenvalue-corrected Shampoo
     - 0.071 [0.060, 0.082]
     - 0.517 [0.491, 0.539]
   * - SOURCE (Adam)
     - 0.071 [0.060, 0.084]
     - 0.473 [0.446, 0.498]
   * - EKFAC
     - 0.070 [0.058, 0.082]
     - 0.454 [0.426, 0.479]
   * - KFAC
     - 0.067 [0.056, 0.080]
     - 0.420 [0.391, 0.446]
   * - LiSSA (Gauss-Newton Hessian)
     - 0.065 [0.054, 0.076]
     - 0.572 [0.549, 0.592]
   * - BM25
     - 0.062 [0.048, 0.076]
     - 0.220 [0.185, 0.252]
   * - Activation similarity (activation covariance preconditioner)
     - 0.055 [0.043, 0.069]
     - 0.157 [0.120, 0.192]
   * - Activation similarity (activation covariance preconditioner, projection 64)
     - 0.055 [0.043, 0.069]
     - 0.152 [0.116, 0.187]
   * - `Gemma Scope 2 <https://huggingface.co/google/gemma-scope-2-27b-pt>`_ SAE feature similarity
     - 0.053 [0.040, 0.067]
     - 0.165 [0.129, 0.200]
   * - `Qwen3-Embedding-8B <https://huggingface.co/spaces/mteb/leaderboard>`_ semantic search
     - 0.049 [0.038, 0.061]
     - 0.132 [0.093, 0.169]
   * - Activation similarity
     - 0.049 [0.037, 0.063]
     - 0.110 [0.070, 0.149]
   * - Jina v5 semantic search
     - 0.046 [0.035, 0.059]
     - 0.124 [0.087, 0.160]
   * - TrackStar (no optimizer correction, projection 64)
     - 0.045 [0.036, 0.055]
     - 0.276 [0.252, 0.299]
   * - KFAC (projection 64)
     - 0.045 [0.036, 0.055]
     - 0.244 [0.216, 0.270]
   * - TrackStar (Adam, projection 64)
     - 0.043 [0.034, 0.052]
     - 0.242 [0.216, 0.267]
   * - TrackStar (no optimizer correction, projection 32)
     - 0.035 [0.027, 0.044]
     - 0.194 [0.171, 0.216]
   * - TrackStar (Adam, projection 32)
     - 0.032 [0.025, 0.041]
     - 0.182 [0.156, 0.209]
   * - TRAK (8-model ensemble)
     - 0.032 [0.024, 0.040]
     - 0.138 [0.111, 0.165]
   * - SOURCE
     - 0.022 [0.017, 0.027]
     - 0.165 [0.138, 0.191]
   * - TrackStar (no optimizer correction, projection 16)
     - 0.022 [0.016, 0.028]
     - 0.120 [0.090, 0.149]
   * - Gradient cosine similarity
     - 0.021 [0.016, 0.027]
     - 0.156 [0.131, 0.181]
   * - TrackStar (Adam, projection 16)
     - 0.020 [0.014, 0.026]
     - 0.114 [0.084, 0.143]
   * - Gradient dot product
     - 0.019 [0.015, 0.024]
     - 0.156 [0.130, 0.180]
   * - Projected gradient cosine similarity
     - 0.016 [0.012, 0.020]
     - 0.132 [0.103, 0.159]

Results for GPT-2 finetuned on 4 epochs of the WikiText corpus. Held-out loss dropped from 3.545 to 3.111 over training. Every row scores the same model and query set; per-query statistics, run configs and reproduction steps are in `examples/gpt2_wikitext_leaderboard <https://github.com/EleutherAI/bergson/tree/main/examples/gpt2_wikitext_leaderboard>`_.

Per-token
---------

Per-token attribution scores, which attribute each training token's loss term, enable substantially more efficacious data filtering. The proponent QLD masks the query's top 1% of training tokens by each method's per-token scores while the control masks a random 1% of tokens.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Method
     - Proponent QLD [95% CI]
   * - MAGIC
     - 1.375 [1.348, 1.403]
   * - EKFAC + CG
     - 1.014 [0.986, 1.043]
   * - Eigenvalue-corrected Shampoo
     - 0.922 [0.887, 0.957]
   * - SOURCE (Adam, EKFAC)
     - 0.908 [0.877, 0.942]
   * - EKFAC + ASTRA
     - 0.907 [0.863, 0.943]
   * - LiSSA (Gauss-Newton Hessian)
     - 0.880 [0.850, 0.912]
   * - EKFAC
     - 0.863 [0.833, 0.894]
   * - KFAC
     - 0.786 [0.758, 0.815]
   * - BM25
     - 0.677 [0.650, 0.704]
   * - TrackStar (no optimizer correction, projection 64)
     - 0.492 [0.467, 0.517]
   * - Gradient cosine similarity
     - 0.217 [0.206, 0.229]
   * - TRAK (8-model ensemble)
     - 0.119 [0.112, 0.125]
   * - Activation similarity (activation covariance preconditioner)
     - 0.073 [0.056, 0.092]
   * - Gemma Scope 2 SAE feature similarity
     - 0.047 [0.037, 0.058]
   * - Activation similarity
     - 0.028 [0.018, 0.039]
   * - Qwen3-Embedding-8B semantic search
     - 0.008 [0.006, 0.010]

Configs and per-query statistics are in `examples/gpt2_wikitext_leaderboard/tokens <https://github.com/EleutherAI/bergson/tree/main/examples/gpt2_wikitext_leaderboard/tokens>`_.

Per-span
--------

Per-span scores sum each method's per-token scores over spans of each training chunk: 64-token spans, or natural spans cut after sentence ends and line breaks and merged to at least 32 tokens (median 46). The proponent QLD masks the query's top 1% of training tokens by span score, about 369 whole 64-token spans or 413 whole natural spans; the control retrains without a random 1% of tokens.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Method
     - Proponent QLD, natural spans [95% CI]
     - Proponent QLD, 64-token spans [95% CI]
   * - MAGIC
     - 0.239 [0.226, 0.253]
     - 0.226 [0.214, 0.240]
   * - EKFAC + ASTRA
     - 0.153 [0.140, 0.168]
     - 0.147 [0.133, 0.161]
   * - EKFAC
     - 0.135 [0.121, 0.149]
     - 0.129 [0.116, 0.143]

Configs and per-query statistics are in `examples/gpt2_wikitext_leaderboard/spans <https://github.com/EleutherAI/bergson/tree/main/examples/gpt2_wikitext_leaderboard/spans>`_.

Notes
-----

There are many method selection considerations not captured by LDS or QLD. Here is some additional information.

MAGIC
   MAGIC backpropagates through the training process once per query - it doesn't support query batching. It works well in FP32 or TF32 but may significantly degrade when used to attribute BF16 training runs, or training runs with low batch sizes.

EKFAC
   A classic influence function in the form ``g_t H^{-1} g_q`` using the EKFAC Hessian approximation for H. The H can be swapped out for other matrices including KFAC, Shampoo, and MAC.

ASTRA
   ASTRA refines a Kronecker-factored inverse Hessian vector product such as EKFAC's, so it adds a constant amount of compute to each query on top of EKFAC's. Our mainline ASTRA result uses a learning rate tuned via a small grid search against the ASTRA objective, and identically scaled EKFAC and Gauss-Newton Hessians. We also include a reference result matching the implementation in the original paper, which uses a Gauss-Newton Hessian 1/511th the scale of the EKFAC Hessian. Results on this benchmark could likely be improved with careful per-query learning rate tuning - we recommend always tuning ASTRA's learning rate.

LiSSA
   The influence function of `Bae et al. <https://arxiv.org/abs/2209.05364>`_: the damped Gauss-Newton Hessian inverse applied to each query by the LiSSA solver. Unlike ASTRA, it doesn't precondition with an existing Hessian approximation, and costs significantly more per query.

KFAC
   A classic influence function in the form ``g_t H^{-1} g_q``. KFAC supports gradient compression with ``projection_dim``, enabling a highly efficient variant based on a reusable gradient store, suitable for the retrieval stage of a data attribution pipeline, at the cost of some LDS/QLD.

Eigenvalue-corrected Shampoo
   A novel influence function in the form ``g_t H^{-1} g_q`` using the Shampoo preconditioner as the H. Its mathematical relationship with the true Hessian is somewhere between "null" and "extremely tenuous" but it performs well anyway. It uses an eigenvalue correction similar to EKFAC's, and to the procedure in SOAP optimization.

SOURCE
   SOURCE uses a weighted combination of Hessian approximations fit on different segments of the training trajectory, so it may perform relatively better in longer or multi-phase training runs.

TrackStar
   TrackStar is a highly efficient method that can more readily scale to pretraining via a reusable compressed gradient store. Presumably it's designed to be used as the first stage of a data search pipeline that eventually uses a more powerful influence function to re-rank the top-k items.

TRAK
   A gradient-based method. TRAK's performance matches `existing results <https://arxiv.org/pdf/2405.12186>`_ in the literature.

Gradient similarity
   A baseline that considers only the direct effect of the parameter update on the query.

Semantic search
   A baseline that uses a model optimized to retrieve documents with a similar meaning to the query document.

SAE feature similarity
   A baseline that compares documents by the mean of a sparse autoencoder's features on a large language model's residual stream, here the Gemma Scope 2 SAE at layer 31 of Gemma 3 27B.

BM25
   A baseline that retrieve documents with words matching those in the query document.

Activation similarity
   A novel baseline that uses the model's activations rather than its gradients. Any embedding of data into a representation space can be used to compute data similarities, so comparing activation-based similarities to gradient-based similarities sheds light on the extent to which gradient-based notions of similarity are especially suited to making counterfactual predictions. Here we see that the best gradient-based methods outperform model activations, but that suitably preconditioned model activations seem to outperform gradient-based methods optimized for efficiency, like gradient cosine similarity and TrackStar, along with semantic search.
