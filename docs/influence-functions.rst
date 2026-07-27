Influence Functions & Preconditioners
=====================================

Influence functions can be roughly conceptualized as a preconditioned gradient similarity search, with formula :math:`g_q H^{-1} g_t^\top` where :math:`g_q` is the query gradient, :math:`g_t` is a training gradient, and :math:`H` is a Hessian (approximation). Many Hessian approximations are used, and hyperparameters such as the inversion damping factor are also usually incorporated. In some cases, heuristics such as gradient unit normalization are also applied.

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
     - ``--optimizer_state <path>`` on ``build`` / ``reduce`` / ``score``

.. Gradient Autocorrelation
.. ------------------------

.. Gradient o

.. KFAC
.. ----

.. .. TODO

.. TKFAC
.. -----

.. .. TODO

.. EK-FAC
.. ------

.. .. TODO: hyperparameter sensitivity (damping, inversion), the LDS ≈ 0 failure
..    mode, and the ``bergson ekfac`` end-to-end pipeline.

.. Shampoo
.. -------

.. .. TODO

.. Optimizer State
.. ---------------

.. .. TODO: Adam / Adafactor second-moment normalization via ``--optimizer_state``.
