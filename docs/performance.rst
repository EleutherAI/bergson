Performance
===========

If `nvidia-smi topo -m` shows SYS between GPUs, setting NCCL_P2P_LEVEL=SYS might speed up multi-GPU Bergson runs.
