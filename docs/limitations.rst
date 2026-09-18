Limitations
===========

MoE layers that fuse their experts into one ``nn.Parameter`` are attributed only when named in ``moe_experts`` (see :doc:`gradient-collection`), and their routers not at all. Everything else must be an ``nn.Linear``, HF ``Conv1D``, or ``nn.Conv{1,2,3}d`` module to be tracked.

MAGIC is reliant on being used in a training setup with good Lipschitz smoothness with respect to the data weights. Influence functions also appear to be positively affected by high smoothness. They also seem sensitive to the Hessian inversion damping hyperparameter in some toy models, although not obviously so at GPT-2 scale and above.
