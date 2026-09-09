TRAK
====

``trak`` is a high-level pipeline implementing
`TRAK: Attributing Model Behavior at Scale <https://arxiv.org/abs/2303.14186>`_
(Park et al., 2023) on Bergson's random-projected gradient index. A training
example ``z_i`` is scored for a query ``z_q`` as

.. math::

   \phi(z_q)^\top (\Phi^\top \Phi + \lambda I)^{-1} \phi(z_i) \,(1 - p_i)

where :math:`\phi` is the projected per-example gradient, :math:`\Phi` stacks the
projected training gradients and :math:`p_i` is the model's probability of the
training example's labels (the geometric-mean token probability for a language
model). :math:`\phi` is Bergson's per-module double-sided random projection,
concatenated over modules; the Gram :math:`\Phi^\top \Phi` is fit over that
concatenation as one matrix (the ``autocorrelation`` Hessian with
``scope: joint``, the default ``kernel: joint``) and its damped inverse is
applied to the query gradients. The
damping :math:`\lambda` is ``preprocess_cfg.inversion_cfg.damping_factor`` times
the mean eigenvalue. ``kernel: per_module`` instead uses the per-module Gram (``scope: per_module``),
a block-diagonal approximation of the same kernel.
``projection_target: global`` gives TRAK's single global sketch when its
per-module ``k x d`` projection matrices fit in memory. The concatenated sketch
must have at most 20,000 dimensions (for GPT-2's 48 modules, ``projection_dim``
16 gives 12,288). Passing several independently trained checkpoints
averages their score stores, as in the paper's ensembles.

What It Produces
----------------

A directory at ``run_path`` with the following subdirectories:

- ``train_hessian/`` — the projected-gradient Gram fit on the training set
  (``hessians.pth``, ``hessians_eigen.pth``, ``normalizers.pth``; with the joint
  kernel also ``joint_layout.json``, the module order of the concatenation).
- ``query/`` — the Gram-whitened query gradient index (same artifacts as ``build``).
- ``scores/`` — scores for the training set (same artifacts as ``score``), with
  ``trak_weights.npy`` holding the ``1 - p_i`` weights that were applied.
- ``checkpoint_<i>/`` — the same layout per ensemble member when
  ``checkpoints`` is set; ``scores/`` is then their mean.

Key Options
-----------

- ``--data.dataset``: the training dataset.
- ``--query.dataset``: the query dataset.
- ``--projection_dim``: projected gradient size per module (default 16).
- ``--trak_cfg.preprocess_cfg.inversion_cfg.damping_factor``: Gram damping
  relative to the mean eigenvalue (default 0.1).
- ``--trak_cfg.kernel``: ``joint`` (default, TRAK) or ``per_module`` (block-diagonal).
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
       --projection_dim 16

Scores are influence-signed (``higher_is_better``): a positive score marks a
proponent of the query. See :doc:`cli` for the full ``TrakConfig`` API reference.
