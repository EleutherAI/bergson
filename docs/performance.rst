Performance
===========

If `nvidia-smi topo -m` shows SYS between GPUs, setting NCCL_P2P_LEVEL=SYS might speed up multi-GPU Bergson runs.

Influence Functions
-------------------

Hessian fitting can be batched over modules as well as documents. Set ``hessian_cfg.module_partitions`` to fit the factors in that many groups of modules, one pass over the data per group, when they don't all fit in memory at once.

Host memory
-----------

Under FSDP every rank loads the whole model on the CPU before sharding it, and ``astra.factors_on_cpu`` keeps a full copy of the Kronecker factors on every rank, with the query and the solver's momentum beside them. Budget host memory as ranks × (model + factors + two gradient-sized vectors): for a 7B model with the factors of 32 MLP modules that is about 100 GB per rank, 400 GB on a 4-GPU node.

PyTorch builds that use mimalloc for CPU allocations (the 2.15 CUDA wheels do) keep freed CPU memory committed, so the model's CPU copy stays resident after sharding. ``export MIMALLOC_PURGE_DELAY=0`` returns freed memory to the OS at once.

On Grace Hopper (GH200) nodes each GPU's memory is also a NUMA node of the host. Once the CPUs' own memory fills, the kernel places host pages in GPU memory, and CUDA runs out of memory while PyTorch reports far less allocated than the device holds. ``numastat -p <pid>`` shows host pages on the GPU nodes; ``numactl --membind=<CPU nodes> python -m bergson ...`` keeps them off (``numactl -H`` lists the nodes; the GPU nodes have no CPUs).
