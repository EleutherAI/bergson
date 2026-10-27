Performance
===========

Setting ``NCCL_P2P_LEVEL=SYS`` lets NCCL send data directly between GPUs that don't share a PCIe switch, which made the K-FAC Hessian stage about 1.7 times as fast on an 8×A40 node.
