TRAK
====

``trak`` implements `TRAK: Attributing Model Behavior at Scale <https://arxiv.org/abs/2303.14186>`_
(2023), a method which randomly projects whole-model gradients from ``model_dim`` to ``projection_dim``.

A training example ``z_i`` is scored for a query ``z_q`` as

.. math::

   \phi(z_q)^\top (\Phi^\top \Phi)^{-1} \phi(z_i) \,(1 - p_i)

where :math:`\phi` is the projected per-example gradient of the output function :math:`\log p - \log(1 - p)`
summed over the example's label tokens, :math:`\Phi` is the matrix of projected training gradients, one row
per example, and :math:`1 - p_i` is the mean derivative of the loss w.r.t. the output function.

The Gram :math:`\Phi^\top \Phi` inverse is applied to the query gradients, for computational efficiency,
and is undamped by default. Pass several independently trained checkpoints to enable ensembling (score
averaging).

TRAK does not support per-token attribution.

What It Produces
----------------

A directory at ``run_path`` with the following subdirectories:

- ``train_hessian/`` — the projected-gradient Gram fit on the training set
  (``hessians.pth``, ``hessians_eigen.pth``, ``normalizers.pth``,
  ``joint_layout.json``).
- ``query/`` — the Gram-whitened query gradient index (same artifacts as ``build``).
- ``scores/`` — scores for the training set (same artifacts as ``score``), with
  ``trak_weights.npy`` holding the ``1 - p_i`` weights that were applied.
- ``checkpoint_<i>/`` — the same layout per ensemble member when
  ``checkpoints`` is set; ``scores/`` is then their mean.

Key Options
-----------

- ``--data.dataset``: the training dataset.
- ``--query.dataset``: the query dataset.
- ``--projection_dim``: size of the global gradient sketch.
- ``--projection_target``: must be ``global``.
- ``--loss_fn``: must be ``log_odds``.
- ``--trak_cfg.preprocess_cfg.inversion_cfg.damping_factor``: Gram damping
  relative to the mean eigenvalue (default 0, the paper's plain inverse).
- ``--trak_cfg.checkpoints``: model checkpoints to ensemble.

Example
-------

.. code-block:: bash

   bergson trak runs/my-trak \
       --model EleutherAI/pythia-14m \
       --data.dataset NeelNanda/pile-10k \
       --data.truncation \
       --query.dataset NeelNanda/pile-10k \
       --query.truncation \
       --query.split "train[:20]" \
       --projection_target global \
       --projection_dim 512 \
       --loss_fn margin

Scores are ``higher_is_better``, so a positive score indicates a query proponent.
See :doc:`cli` for the full ``TrakConfig`` API reference.
