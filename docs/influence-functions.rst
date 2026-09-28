Influence Functions & Preconditioners
=====================================

Influence functions are tools borrowed from logistic regression, where they may be applied to compute the exact leave-one-out effects of training data points on test-time model behavior. In this context they have the formula :math:`g_q H^{-1} g_t^\top`, where :math:`g_q` is the query gradient for the test-time behavior, :math:`g_t` is a training item gradient, and :math:`H` is the Hessian. These quantities are all evaluated on the trained model.

Neural networks do not have invertible Hessians, but it `has been shown <https://arxiv.org/abs/1703.04730>`_ that swapping out the inverse Hessian with a damped inverse Hessian approximation can still noisily predict leave-one-out effects in some cases. Later results showed that leave-one-out effects can also be predicted when the matrices in the `H` position do not approximate the Hessian at all: see the empirical Fisher-based preconditioners used in `TRAK <https://arxiv.org/abs/2303.14186>`_, `LoGra <https://arxiv.org/abs/2405.13954>`_, `TrackStar <https://arxiv.org/html/2410.17413v1>`_, and even simple `gradient cosine similarity <https://openreview.net/pdf?id=fQvVV6UN4p>`_.

Work on the theory of influence functions continues, for example `Mlodozeniec et al. <https://proceedings.neurips.cc/paper_files/paper/2025/file/0e8909cae8248c98279f6cd82074aa6d-Paper-Conference.pdf>`_. But for now, I think applied researchers could do worse than viewing influence functions as simply computing a similarity metric with an empirically useful definition of similarity. Theory that aligns more closely with this view tends to use terms like `kernels, metric matrices <https://proceedings.neurips.cc/paper_files/paper/1998/file/db1915052d15f7815c8b88e879465a1e-Paper.pdf>`_, or `preconditioners <https://docs.modula.systems/algorithms/manifold/>`_.

We offer several such matrices, which we refer to as either Hessians or preconditioners, and expose other hyperparameters such as the inversion damping factor that affect some results. We also provide gradient unit normalization, which can be interpreted as inducing the normalized linear kernel.

Preconditioners
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Preconditioner
     - CLI
   * - Gradient autocorrelation (second moment)
     - ``bergson hessian <run_path> --method autocorrelation``
   * - KFAC
     - ``bergson hessian <run_path> --method kfac``
   * - TKFAC
     - ``bergson hessian <run_path> --method tkfac``
   * - EK-FAC
     - ``bergson hessian <run_path> --method kfac --ev_correction`` (end-to-end pipeline: ``bergson ekfac``)
   * - Shampoo
     - ``bergson hessian <run_path> --method shampoo``
   * - Optimizer state (Adam / Adafactor)
     - ``--optimizer_state <path>`` on ``build`` / ``score``

ASTRA
-----

`ASTRA <https://arxiv.org/abs/2507.14740>`_ refines the EK-FAC solution :math:`(H + D)^{-1} g_q` with momentum SGD on :math:`\tfrac12 x^\top (H + D) x - x^\top g_q`, preconditioned by the damped EK-FAC inverse, where :math:`H` is the Gauss-Newton Hessian estimated on a random batch of training documents per step. Enable it on ``ekfac`` with ``hessian_pipeline_cfg.astra.num_steps``; each step costs about five forward passes over the batch, per query. See ``examples/replicate_bae_approx_unrolling_source/wikitext_gpt2_astra.yaml``.

Reranking candidates
--------------------

``score_cfg.candidates`` on ``score``, ``ekfac`` or ``trackstar`` scores only the rows an earlier run ranked highest, e.g. TrackStar over every row, then Eigenvalue-corrected Shampoo over its top rows:

.. code-block:: yaml

   steps:
     - trackstar:
         index_cfg: {run_path: runs/two_stage/trackstar, ...}
         ...
     - ekfac:
         index_cfg: {run_path: runs/two_stage/shampoo, ...}
         hessian_cfg: {method: shampoo, ev_correction: true}
         score_cfg:
           candidates:
             scores: runs/two_stage/trackstar/scores
             top_k: 1000
         ...

The candidates are the union over the earlier run's query columns of each column's ``top_k`` rows (or ``fraction``) at the ``direction`` end. ``scores`` may also be the CSV written by ``bergson query --record``, whose ``Top`` (or ``Bottom``) rows per query are the candidates, the first ``top_k`` of each when set. The Hessian is still fitted on the whole training set. The store has one row per candidate in training order; ``candidates.npy`` beside it holds their training rows, which ``load_scores`` returns as ``candidates``.

See ``examples/pipelines/trackstar_then_shampoo.yaml``.

Compressing the gradients
-------------------------

``index_cfg.projection_dim`` randomly down-projects the gradients before they are scored. ``projection_target: per_module`` compresses each module's gradient to its own ``[projection_dim, projection_dim]`` block; ``global`` compresses the whole gradient to one ``[projection_dim]`` vector:

.. code-block:: yaml

   steps:
     - ekfac:
         index_cfg:
           projection_dim: 4096
           projection_target: global
         hessian_cfg: {method: kfac}

The factored preconditioners apply :math:`H^{-1}` to the unprojected query and down-project the result to match the index. EK-FAC (``ev_correction``) doesn't support projection.
