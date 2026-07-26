.. automodule:: bergson
   :members:
   :undoc-members:
   :show-inheritance:

Mixture-of-Experts
------------------

Fused-parameter MoE experts and routers are not ``nn.Linear`` layers, so
:func:`bergson.expand_moe` exposes each expert projection as a tracked module
before collection. See the *Mixture-of-Experts models* section of the README for
the CLI flags and their index-size implications.

.. automodule:: bergson.moe
   :members:
   :undoc-members:
   :show-inheritance:
