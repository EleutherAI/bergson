Known Limitations
=================

MoE fused-parameter experts are not attributed (only ``nn.Linear``, HF ``Conv1D``, and ``nn.Conv{1,2,3}d`` modules are tracked).

MAGIC is reliant on being used in a training setup with good Lipschitz smoothness with respect to the data weights. Influence functions also appear to be positively affected by high smoothness. They also seem sensitive to the Hessian inversion damping hyperparameter in some toy models, although not obviously so at GPT-2 scale and above.