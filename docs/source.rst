SOURCE (Approximate Unrolling)
==============================


`SOURCE <https://arxiv.org/abs/2405.12186>`_ is a halfway point between direct differentiation of a model behavior with respect to its data weights, and the more efficient approaches in influence functions. It can be thought of as roughly equivalent to averaging an influence function across several training checkpoints. For a precise view, read the `paper <https://arxiv.org/abs/2405.12186>`__. For practical purposes, note that SOURCE tends to be somewhat more accurate than single-checkpoint influence functions and somewhat less accurate than MAGIC, and that it requires several (e.g., six) training checkpoints to be available.

See ``examples/pipelines/approx_unrolling_pythia.yaml`` and the ``approxunrolling`` entry in :doc:`cli`.
