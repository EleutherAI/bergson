Influence Functions & Preconditioners
=====================================

Influence functions are tools borrowed from logistic regression, where they may be applied to compute the exact leave-one-out effects of training data points on test-time model behavior. In this context they have the formula :math:`g_q H^{-1} g_t^\top`, where :math:`g_q` is the query gradient for the test-time behavior, :math:`g_t` is a training item gradient, and :math:`H` is the Hessian. These quantities are all evaluated on the trained model.

Neural networks do not have invertible Hessians, but it `has been shown <https://arxiv.org/abs/1703.04730>`_ that swapping out the inverse Hessian with a damped inverse Hessian approximation can still noisily predict leave-one-out effects in some cases. Later results showed that leave-one-out effects can also be predicted when the matrices in the `H` position do not approximate the Hessian at all: see the empirical Fisher-based preconditioners used in `TRAK <https://arxiv.org/abs/2303.14186>`_, `LoGra <https://arxiv.org/abs/2405.13954>`_, `TrackStar <https://arxiv.org/html/2410.17413v1>`_, and even simple `gradient cosine similarity <https://openreview.net/pdf?id=fQvVV6UN4p>`_.

Work on the theory of influence functions continues, for example `Mlodozeniec et al. <https://proceedings.neurips.cc/paper_files/paper/2025/file/0e8909cae8248c98279f6cd82074aa6d-Paper-Conference.pdf>`_. But for now, I think applied researchers could do worse than viewing influence functions as simply computing a similarity metric with an empirically useful definition of similarity. Theory that aligns more closely with this view tends to use terms like `kernels, metric matrices <https://proceedings.neurips.cc/paper_files/paper/1998/file/db1915052d15f7815c8b88e879465a1e-Paper.pdf>`_, or `preconditioners <https://docs.modula.systems/algorithms/manifold/>`_.

We offer several such matrices, which we refer to as either Hessians or preconditioners, and expose other hyperparameters such as the inversion damping factor that affect results in tiny models. We also provide gradient unit normalization, which can be interpreted as inducing the normalized linear kernel.

Preconditioners
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Preconditioner
     - Fit Preconditioner in CLI
   * - :ref:`Gradient autocorrelation <methods/influence-functions:Gradient autocorrelation>` (second moment)
     - ``bergson hessian <run_path> --method autocorrelation``
   * - :ref:`KFAC <methods/influence-functions:KFAC>`
     - ``bergson hessian <run_path> --method kfac``
   * - :ref:`TKFAC <methods/influence-functions:TKFAC>`
     - ``bergson hessian <run_path> --method tkfac``
   * - :ref:`EK-FAC <methods/influence-functions:EKFAC>`
     - ``bergson hessian <run_path> --method kfac --hessian_cfg.ev_correction true``
   * - :ref:`Shampoo <methods/influence-functions:Shampoo>`
     - ``bergson hessian <run_path> --method shampoo``
   * - :ref:`Optimizer state <methods/influence-functions:Optimizer state>` (Adam / Adafactor)
     - Load a training buffer with ``--optimizer_state <path>`` on ``build`` / ``score``
   * - Gauss-Newton hessian
     - Not fitted. Apply per-query with ``bergson lissa`` or ASTRA

Refine a Preconditioner with ASTRA
----------------------------------

`ASTRA <https://arxiv.org/abs/2507.14740>`_ refines a query's inverse-Hessian-vector product :math:`(H + D)^{-1} g_q`, where :math:`H` is the Gauss-Newton Hessian estimated on a random batch of training documents per step and :math:`D` is the damping, by momentum SGD on :math:`\tfrac12 x^\top (H + D) x - x^\top g_q`. It starts from an existing Kronecker-factored Hessian approximation like EKFAC and preconditions every step with the factored inverse. It can yield a more accurate inverse-Hessian-vector product, but adds some computation per query.

Enable it in the CLI or in a YAML in the ``influence`` command, by setting ``hessian_pipeline_cfg.astra.num_steps``.

Setting ``astra.solver: cg`` enables an alternative to the canonical ASTRA implementation that uses the CG solver (preconditioned conjugate gradients) on one batch of ``batch_size`` documents.

The ASTRA solver learning rate should be tuned to minimize solver error. If you use multiple solver seeds, use the seed with the lowest final objective.

See ``examples/replications/bae_approx_unrolling_source_replication/wikitext_gpt2_astra.yaml`` and ``examples/gpt2_wikitext_leaderboard/ekfac_astra_cg.yaml``.

Rerank Candidates
--------------------

For applications where only the :math:`k` most influential items are of interest, such as interpretability and data filtering, it's possible to achieve improved accuracy at a low cost by ranking all the documents using a highly efficient method like TrackStar, and then reranking the top :math:`k * n` documents, where n is a free hyperparameter, using a less efficient but more accurate method like EKFAC or EKFAC + ASTRA.

To perform reranking for a query, add a second step to the bergson YAML with the field ``score_cfg.candidates`` set on ``score``, or ``influence``. In this example, ``trackstar`` scores all the rows, and then the influence function scores (reranks) only the 1000 rows TrackStar ranked highest:

.. code-block:: yaml

   steps:
     - trackstar:
         index_cfg: {run_path: runs/two_stage/trackstar, ...}
         ...
     - influence:
         index_cfg: {run_path: runs/two_stage/shampoo, ...}
         hessian_cfg: {method: shampoo, ev_correction: true}
         score_cfg:
           candidates:
             scores: runs/two_stage/trackstar/scores
             top_k: 1000
         ...

The candidates are the union over the earlier run's query columns of each column's ``top_k`` rows (or ``fraction``) at the ``direction`` end. ``scores`` may also be the CSV written by ``bergson query --record``, whose ``Top`` (or ``Bottom``) rows per query are the candidates, the first ``top_k`` of each when set. The Hessian is still fitted on the whole training set. The store has one row per candidate in training order; ``candidates.npy`` beside it holds their training rows, which ``load_scores`` returns as ``candidates``.

See ``examples/pretraining_scale_attribution/query_and_rerank.yaml``.

Attribute Tokens
----------------

With ``attribute_tokens``, ``score_cfg.token_influence`` on ``score`` or ``influence`` chooses what each per-token row scores. Both kinds of row sum to the per-example score.

* ``gradient`` (the default) scores row ``t`` by the per-token gradient at position ``t``. This is the simpler tokenwise attribution of `Studying Large Language Model Generalization with Influence Functions <https://arxiv.org/abs/2308.03296>`_ (Grosse et al., 2023, Eq. 31). Row ``t`` is position ``t``'s effect on the loss of every later token, since later tokens attend to it, so it isn't the influence of any one token.
* ``output`` scores row ``t`` by the loss on token ``t + 1`` alone, which is the term that loss masking removes. This is the paper's output token influence (Appendix B.1, Eq. 36). Rows without a label are zero.

The paper estimates output token influence by moving the weights a small step along the query and comparing each token's loss before and after. Bergson computes the rate of change exactly instead, with one forward-mode pass (a Jacobian-vector product) per query column that covers every token in the batch, so aggregate the query when you can. Without ``attribute_tokens``, ``output`` gives the same per-example scores as ``gradient`` by a different computation.

``output`` needs an unprojected query (``projection_dim: 0``) and dot-product scoring, and doesn't support ``loss_fn: kl``, ``optimizer_state``, ``split_attention_modules``, quantized models, FSDP, or fused MoE experts. The fused attention kernels don't implement forward-mode derivatives, so the model is loaded with eager attention. The run saves ``data.hf`` with each example's loss, as the gradient path does, but not ``total_processed.pt``, which only Hessian fitting reads.

Span attribution
----------------

``data.span_column`` names a column holding each document's span start positions, which makes a gradient row a span -- a contiguous run of token positions, such as a sentence, a paragraph or a fixed-width window -- instead of a whole document. The starts must be increasing and begin at 0, so the spans partition the document:

.. code-block:: yaml

   steps:
     - score:
         index_cfg:
           data:
             dataset: my_pretokenized_dataset
             span_column: span_starts     # e.g. [0, 46, 91, ...] per document
         score_cfg:
           query_path: runs/ekfac/kfac_query

Nothing generates the column, so any rule can cut the documents. Fixed windows are one ``map``:

.. code-block:: python

   ds = ds.map(lambda row: {"span_starts": list(range(0, len(row["input_ids"]), 64))})

A span's gradient is the sum of its positions', so its score is the sum of the per-token scores over the same rows, and a document's spans still sum to the document. Span attribution therefore changes only the cost and the size of the store, never the numbers.

That cost sits between the two granularities it interpolates. A per-document gradient is one ``[out, in]`` matrix formed by a matmul over the document's positions, and scoring it is one dot product per query; per-token scoring cannot form ``[tokens, out, in]``, so it contracts the query with each position separately and the token count multiplies the query count. Spans form one matrix per span, which is affordable again, so the query cost scales with the number of spans. For 512-token documents and 50 queries, on one ``[768, 3072]`` module:

.. list-table::
   :header-rows: 1

   * - Rows per document
     - Scoring FLOPs
     - Relative
   * - 1 (document)
     - 2.65 G
     - 1.00x
   * - 4 (128-token spans)
     - 3.36 G
     - 1.27x
   * - 8 (64-token spans)
     - 4.30 G
     - 1.62x
   * - 16 (32-token spans)
     - 6.19 G
     - 2.33x
   * - 512 (per token)
     - 120.84 G
     - 45.57x

Uneven spans cost a little more than even ones: a batch's spans are padded to its longest before they are summed, so the padding is the ratio of the longest span to the mean one.

``score_cfg.token_influence`` decides which rows a span owns. Under ``gradient`` row ``t`` is position ``t``, so a span over positions ``[a, b)`` owns rows ``[a, b)``; under ``output`` row ``t`` is the loss on token ``t + 1``, so it owns rows ``[a - 1, b - 1)`` -- the loss terms of its own tokens. The store records the resolved row ranges in ``spans.npy``, so readers need not repeat that reasoning, and a span whose tokens carry no loss term keeps a row with a zero score rather than dropping out, which would make the store's shape depend on the token influence. ``output`` costs one forward-mode pass per query column whatever the granularity, so spans save storage there rather than compute.

The unit carries into a dense (``autocorrelation``) Hessian fitted with the same ``span_column``, whose Gram is then a second moment over span gradients, normalized by their count. The Kronecker-factored methods fit over token activations and gradients either way.

``load_scores`` marks a span store with ``info["attribute_spans"]``, and ``Scores.to_grid`` lays each span's score back over the token positions it covers. Leave-out validation and the filters therefore rank token rows as they always have, and removing a tail of them removes whole spans.

Span attribution needs a pre-tokenized dataset, since the positions index ``input_ids``. It is incompatible with ``attribute_tokens`` and ``chunk_length``, which set or re-cut the same unit, and with ``score_cfg.candidates``, fused MoE experts, and MAGIC -- for MAGIC, attribute tokens and sum each span's rows.

Compress Gradients
------------------

``index_cfg.projection_dim`` randomly down-projects the gradients before they are scored. ``projection_target: per_module`` compresses each module's gradient to its own ``[projection_dim, projection_dim]`` block; ``global`` compresses the whole gradient to one ``[projection_dim]`` vector:

.. code-block:: yaml

   steps:
     - influence:
         index_cfg:
           projection_dim: 4096
           projection_target: global
         hessian_cfg: {method: kfac}

The factored preconditioners apply :math:`H^{-1}` to the unprojected query and down-project the result to match the index. EK-FAC (``ev_correction``) doesn't support projection.

Method details
--------------

Gradient autocorrelation
^^^^^^^^^^^^^^^^^^^^^^^^

The simplest preconditioner: for each module, the per-module second moments :math:`H = \sum_n g_n^\top g_n` of the per-example training gradients. It is the primary preconditioner for compressed gradients, including TrackStar's per-module Grams and TRAK's joint Gram over the whole projected gradient. See :doc:`trackstar` and :doc:`trak`.

KFAC
^^^^

KFAC is a simple influence function with moderate accuracy and a fairly low computational cost.

KFAC's :math:`H` is a damped Kronecker-factored approximation to the Gauss-Newton Hessian. The :math:`H` is fitted on the training set with the fully trained model, so KFAC does not require access to training checkpoints or any other training artifacts.

EKFAC
^^^^^

EKFAC is the workhorse of influence functions, providing fairly good accuracy and a fairly low computational cost.

EKFAC's :math:`H` is a damped Kronecker-factored approximation to the Gauss-Newton Hessian. The :math:`H` is fitted on the training set with the fully trained model, so EKFAC does not require access to training checkpoints or any other training artifacts. Compared to KFAC, EKFAC's :math:`H` applies an additional correction step, where the Kronecker product's eigenvalues are set to the second moments of the gradients in the Kronecker eigenbasis.

TKFAC
^^^^^

KFAC with each batch's activation covariance weighted by the trace of that batch's gradient covariance, and vice versa, so the Kronecker product keeps the trace of the block it approximates.

Shampoo
^^^^^^^

A Kronecker-factored :math:`H` built from the batch parameter gradients :math:`\nabla W = G^\top X` rather than from activations and output gradients separately: the left factor sums :math:`\nabla W \nabla W^\top` over batches and the right factor sums :math:`\nabla W^\top \nabla W`, divided by its trace. These are the preconditioner factors of the Shampoo optimizer.

Optimizer state
^^^^^^^^^^^^^^^

These states can be loaded from Adam training buffers, i.e. the ``optimizer.pt`` generated by the Hugging Face Trainer or its ``hf://`` URI. Each gradient element is divided by the square root of the training optimizer's second-moment estimate, which is the per-parameter scaling the optimizer applied to the same gradients during training. The same ``--optimizer_state`` must be given for the index and the query gradients. See :doc:`../preprocessing`.

LiSSA Solver
^^^^^^^^^^^^

``bergson lissa`` estimates each query's :math:`(G + \lambda I)^{-1} g_q`, where :math:`G` is the Gauss-Newton Hessian approximation of the training loss, with the `LiSSA <https://arxiv.org/abs/1602.03943>`_ iterative solver's truncated series :math:`\sigma^{-1} \sum_t (I - (G + \lambda I) / \sigma)^t g_q`, each term estimating :math:`G` on a random batch of training documents, averaged over repeats.

This is the influence estimator of `Bae et al. <https://arxiv.org/abs/2209.05364>`_.

The series converges when the scale :math:`\sigma` exceeds half the largest eigenvalue of :math:`G + \lambda I`. Each term costs about five forward passes over the batch, per query. See ``examples/gpt2_wikitext_leaderboard/lissa.yaml``.
