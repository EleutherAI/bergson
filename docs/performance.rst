Performance
===========

If `nvidia-smi topo -m` shows SYS between GPUs, setting NCCL_P2P_LEVEL=SYS might speed up multi-GPU Bergson runs.

Influence Functions
-------------------

Hessian fitting can be batched over modules as well as documents. Set ``hessian_cfg.module_partitions`` to fit the factors in that many groups of modules, one pass over the data per group, when they don't all fit in memory at once.

Set ``hessian_cfg.checkpoint_interval`` to save the fit's running sums every that many documents. An interrupted fit then resumes with ``--resume``, repeating only the documents since the last save and skipping the phases it finished.
