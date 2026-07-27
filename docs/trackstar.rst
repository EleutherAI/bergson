Trackstar
=========

``trackstar`` is a high-level pipeline that computes and mixes hessian preconditioners, then scores queries on the fly, following the methodology described in
`Scalable Influence and Fact Tracing for Large Language Model Pretraining <https://arxiv.org/abs/2410.17413>`_
(Bae et al., 2024). TrackStar is designed for use in large-scale language model attribution, so it's fast and efficient. It can be used as the initial recall-optimized step of a two-stage pipeline as proposed in `Efficient Retrieval of Influential LLM Training Examples <https://simons.berkeley.edu/talks/roger-grosse-university-toronto-2026-04-13>`_.

What It Produces
----------------

A directory at ``run_path`` with the following subdirectories:

- ``value_hessian/`` — hessian fitted on the value (training) dataset.
- ``query_hessian/`` — hessian fitted on the query dataset.
- ``mixed_hessian/`` — mixed hessian combining value and query statistics.
  Contains ``config.yaml``, ``normalizers.pth``, ``hessians.pth``,
  ``hessians_eigen.pth``, and ``processor_config.yaml``.
- ``query/`` — gradient index built on the query dataset (same artifacts as ``build``).
- ``scores/`` — scores for the value dataset (same artifacts as ``score``).

Key Options
-----------

- ``--data.dataset``: the value (training) dataset.
- ``--query.dataset``: the query dataset.
- ``--target_downweight_components``: number of gradient components to downweight when
  mixing hessians (default 1000).

Example
-------

.. code-block:: bash

   bergson trackstar runs/my-trackstar \
       --model EleutherAI/pythia-14m \
       --data.dataset NeelNanda/pile-10k \
       --data.truncation \
       --query.dataset NeelNanda/pile-10k \
       --query.truncation \
       --projection_dim 16

See :doc:`cli` for the full ``TrackstarConfig`` API reference.
