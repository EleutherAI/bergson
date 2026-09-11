TRAK
====

``trak`` is a high-level pipeline implementing
`TRAK: Attributing Model Behavior at Scale <https://arxiv.org/abs/2303.14186>`_
(Park et al., 2023) on Bergson's random-projected gradient index. A training
example ``z_i`` is scored for a query ``z_q`` as

.. math::

   \phi(z_q)^\top (\Phi^\top \Phi)^{-1} \phi(z_i) \,(1 - p_i)

where :math:`\phi` is the projected per-example gradient of the margin output
function :math:`\log p - \log(1 - p)` summed over the example's label tokens
(``loss_fn: margin``, which ``bergson trak`` requires), :math:`\Phi` stacks the
projected training gradients and :math:`1 - p_i` is the derivative of the loss
w.r.t. the margin, averaged over the training example's label tokens.
:math:`\phi` is one random projection of the whole gradient: the index must be
built with ``projection_target: global``, which projects each module's
flattened gradient with its own block of a single ``k x d`` Rademacher matrix
and sums the blocks, so ``projection_dim`` is the size of the sketch; the
blocks are generated on the fly, so ``k`` is bounded by the ``k x k`` Gram
rather than by GPU memory. ``bergson trak`` refuses other projection targets.
The Gram :math:`\Phi^\top \Phi` is fit over that sketch on the training
examples' own labels (the ``autocorrelation`` Hessian with ``scope: joint``)
and its inverse is applied to the query gradients, undamped by default;
``preprocess_cfg.inversion_cfg.damping_factor`` adds a multiple of the mean
eigenvalue. Passing several independently trained checkpoints averages their
score stores, as in the paper's ensembles.

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
- ``--loss_fn``: must be ``margin``.
- ``--trak_cfg.preprocess_cfg.inversion_cfg.damping_factor``: Gram damping
  relative to the mean eigenvalue (default 0, the paper's plain inverse).
- ``--trak_cfg.q_weighting``: ``one_minus_p`` (default) or ``none``.
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

Scores are influence-signed (``higher_is_better``): a positive score marks a
proponent of the query. See :doc:`cli` for the full ``TrakConfig`` API reference.
