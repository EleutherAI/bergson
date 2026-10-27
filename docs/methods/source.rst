SOURCE (Approximate Unrolling)
==============================

`SOURCE <https://arxiv.org/abs/2405.12186>`_ is an influence method midway between influence functions and direct differentiation of a model behavior with respect to its data weights. It can be thought of as roughly equivalent to averaging an influence function across several training checkpoints. For a precise view, read the `paper <https://arxiv.org/abs/2405.12186>`__. For practical purposes, note that SOURCE tends to be somewhat more accurate than single-checkpoint influence functions and somewhat less accurate than MAGIC, and that it requires several (e.g., six) training checkpoints to be available.

SOURCE is particularly suitable for attributing over multi-stage training pipelines, e.g. when models have undergone both pretraining and post-training.

See ``examples/tiny_method_demonstrations/approx_unrolling_pythia.yaml`` and the ``approxunrolling`` entry in :doc:`../cli`.
