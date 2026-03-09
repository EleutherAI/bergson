Trackstar
=========

``trackstar`` is a high-level composed pipeline that runs preconditioners, build, and
score as a single command. It computes separate preconditioners on the value (training)
and query datasets, mixes them, builds a query gradient index, and scores the value
dataset.

This implements the methodology described in
`Scalable Influence and Fact Tracing for Large Language Model Pretraining <https://arxiv.org/abs/2410.17413>`_
(Bae et al., 2024), which extends `TRAK <https://arxiv.org/abs/2303.14186>`_ with mixed
preconditioners for large-scale language model attribution.

What It Produces
----------------

A directory at ``run_path`` with the following subdirectories:

- ``value_preconditioner/`` — preconditioner fitted on the value (training) dataset.
- ``query_preconditioner/`` — preconditioner fitted on the query dataset.
- ``mixed_preconditioner/`` — mixed preconditioner combining value and query statistics.
  Contains ``mix_config.json``, ``normalizers.pth``, ``preconditioners.pth``,
  ``preconditioners_eigen.pth``, and ``processor_config.json``.
- ``query/`` — gradient index built on the query dataset (same artifacts as ``build``).
- ``scores/`` — scores for the value dataset (same artifacts as ``score``).

Key Options
-----------

- ``--data.dataset``: the value (training) dataset.
- ``--query.dataset``: the query dataset.
- ``--target_downweight_components``: number of gradient components to downweight when
  mixing preconditioners (default 1000).

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
