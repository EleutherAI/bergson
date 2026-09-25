TracIn
======

``tracin`` implements `Estimating Training Data Influence by Tracing Gradient Descent <https://arxiv.org/abs/2002.08484>`_
(2020) for models trained with SGD. It scores a training example ``z`` for a query ``z_q`` as

.. math::

   \sum_c \eta_c \, \nabla L(z_q; \theta_c)^\top \nabla L(z; \theta_c)

where :math:`\theta_c` are checkpoints from one training run and :math:`\eta_c` is the learning rate at checkpoint
:math:`c`. Gradients are uncompressed unless ``projection_dim`` is set, in which case every checkpoint uses the same
random projection.

The scores match `Captum <https://captum.ai>`_'s ``TracInCP`` on the same checkpoints (``tests/test_tracin.py``).

What It Produces
----------------

A directory at ``run_path`` with the following subdirectories:

- ``checkpoint_<i>/query/`` — the query gradient index at checkpoint ``i`` (same artifacts as ``build``).
- ``checkpoint_<i>/scores/`` — the training set's unweighted scores at checkpoint ``i`` (same artifacts as ``score``).
- ``scores/`` — the learning-rate-weighted sum over checkpoints.

Key Options
-----------

- ``--dataset``: the training dataset.
- ``--data.dataset``: the query dataset.
- ``--checkpoints``: checkpoints from the training run.
- ``--lr_list``: the learning rate at each checkpoint.
- ``--tokenizer``: needed when the checkpoint directories hold no tokenizer files.

Example
-------

.. code-block:: bash

   bergson tracin runs/my-tracin \
       --dataset NeelNanda/pile-10k \
       --split "train[20:]" \
       --truncation \
       --data.dataset NeelNanda/pile-10k \
       --data.split "train[:20]" \
       --data.truncation \
       --checkpoints runs/train/checkpoint-100 runs/train/checkpoint-200 \
       --lr_list 1e-3 5e-4

Scores are ``higher_is_better``, so a positive score indicates a query proponent.
See :doc:`cli` for the full ``TracInConfig`` API reference.
