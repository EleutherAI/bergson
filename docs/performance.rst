Performance
===========

Setting ``NCCL_P2P_LEVEL=SYS`` allows data to be shared directly between GPUs that don't share a PCIe switch, which can improve performance on some nodes.
