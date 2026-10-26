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
   * - Gradient autocorrelation (second moment)
     - ``bergson hessian <run_path> --method autocorrelation``
   * - :ref:`KFAC <methods/influence-functions:KFAC>`
     - ``bergson hessian <run_path> --method kfac``
   * - TKFAC
     - ``bergson hessian <run_path> --method tkfac``
   * - :ref:`EK-FAC <methods/influence-functions:EKFAC>`
     - ``bergson hessian <run_path> --method kfac --hessian_cfg.ev_correction true``
   * - Shampoo
     - ``bergson hessian <run_path> --method shampoo``
   * - Optimizer state (Adam / Adafactor)
     - Load a training buffer with ``--optimizer_state <path>`` on ``build`` / ``score``

KFAC
----

KFAC is a simple influence function with moderate accuracy and a fairly low computational cost.

KFAC's :math:`H` is a damped Kronecker-factored approximation to the Gauss-Newton Hessian. The :math:`H` is fitted on the training set with the fully trained model, so KFAC does not require access to training checkpoints or any other training artifacts.

EKFAC
-----

EKFAC is the workhorse of influence functions, providing fairly good accuracy and a fairly low computational cost.

EKFAC's :math:`H` is a damped Kronecker-factored approximation to the Gauss-Newton Hessian. The :math:`H` is fitted on the training set with the fully trained model, so EKFAC does not require access to training checkpoints or any other training artifacts. Compared to KFAC, EKFAC's :math:`H` applies an additional correction step, where the Kronecker product's eigenvalues are set to the second moments of the gradients in the Kronecker eigenbasis.

ASTRA
-----

`ASTRA <https://arxiv.org/abs/2507.14740>`_ refines a Kronecker-factored approximate Hessian :math:`(H + D)^{-1} g_q` with momentum SGD on :math:`\tfrac12 x^\top (H + D) x - x^\top g_q`, preconditioned by the damped Kronecker-factored inverse, where :math:`H` is the Gauss-Newton Hessian estimated on a random batch of training documents per step.

Enable it in the CLI or in a YAML in the ``influence`` command, by setting ``hessian_pipeline_cfg.astra.num_steps``. Each step costs about five forward passes over the batch, per query.

``astra.solver: cg`` is an alternative to the canonical ASTRA implementation that uses the CG solver (preconditioned conjugate gradients) on one batch of ``batch_size`` documents.

See ``examples/replications/bae_approx_unrolling_source_replication/wikitext_gpt2_astra.yaml`` and ``examples/gpt2_wikitext_leaderboard/ekfac_astra_cg.yaml``.

LiSSA
-----

``bergson lissa`` estimates each query's :math:`(G + \lambda I)^{-1} g_q`, where :math:`G` is the Gauss-Newton Hessian approximation of the training loss, with the `LiSSA <https://arxiv.org/abs/1602.03943>`_ iterative solver's truncated series :math:`\sigma^{-1} \sum_t (I - (G + \lambda I) / \sigma)^t g_q`, each term estimating :math:`G` on a random batch of training documents, averaged over repeats.

This is the influence estimator of `Bae et al. <https://arxiv.org/abs/2209.05364>`_.

The series converges when the scale :math:`\sigma` exceeds half the largest eigenvalue of :math:`G + \lambda I`. Each term costs about five forward passes over the batch, per query. See ``examples/gpt2_wikitext_leaderboard/lissa.yaml``.

Reranking candidates
--------------------

``score_cfg.candidates`` on ``score``, ``influence`` or ``trackstar`` scores only the rows an earlier run ranked highest, e.g. TrackStar over every row, then Eigenvalue-corrected Shampoo over its top rows:

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

Output token influence
----------------------

With ``attribute_tokens``, ``score_cfg.token_influence`` on ``score`` or ``influence`` chooses what each per-token row scores. Both kinds of row sum to the per-example score.

* ``gradient`` (the default) scores row ``t`` by the per-token gradient at position ``t``. This is the simpler tokenwise attribution of `Studying Large Language Model Generalization with Influence Functions <https://arxiv.org/abs/2308.03296>`_ (Grosse et al., 2023, Eq. 31). Row ``t`` is position ``t``'s effect on the loss of every later token, since later tokens attend to it, so it isn't the influence of any one token.
* ``output`` scores row ``t`` by the loss on token ``t + 1`` alone, which is the term that loss masking removes. This is the paper's output token influence (Appendix B.1, Eq. 36). Rows without a label are zero.

The paper estimates output token influence by moving the weights a small step along the query and comparing each token's loss before and after. Bergson computes the rate of change exactly instead, with one forward-mode pass (a Jacobian-vector product) per query column that covers every token in the batch, so aggregate the query when you can. Without ``attribute_tokens``, ``output`` gives the same per-example scores as ``gradient`` by a different computation.

``output`` needs an unprojected query (``projection_dim: 0``) and dot-product scoring, and doesn't support ``loss_fn: kl``, ``optimizer_state``, ``split_attention_modules``, quantized models, FSDP, or fused MoE experts. The fused attention kernels don't implement forward-mode derivatives, so the model is loaded with eager attention. The run saves ``data.hf`` with each example's loss, as the gradient path does, but not ``total_processed.pt``, which only Hessian fitting reads.

Compressing the gradients
-------------------------

``index_cfg.projection_dim`` randomly down-projects the gradients before they are scored. ``projection_target: per_module`` compresses each module's gradient to its own ``[projection_dim, projection_dim]`` block; ``global`` compresses the whole gradient to one ``[projection_dim]`` vector:

.. code-block:: yaml

   steps:
     - influence:
         index_cfg:
           projection_dim: 4096
           projection_target: global
         hessian_cfg: {method: kfac}

The factored preconditioners apply :math:`H^{-1}` to the unprojected query and down-project the result to match the index. EK-FAC (``ev_correction``) doesn't support projection.
