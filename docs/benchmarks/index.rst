Benchmarks
==========

This section provides indicative performance numbers for the Bergson benchmark suite. Performance will vary based on your hardware configuration and choice of hyperparameters. Indicative performance for dattri provided where possible.

8 GPU Configuration (CLI)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: cli_benchmark_8x_NVIDIA_H100_80GB_HBM3.png
   :alt: 8 GPU Benchmark
   :align: center

1 GPU Configuration (CLI)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: cli_benchmark_1x_NVIDIA_H100_80GB_HBM3.png
   :alt: 1 GPU Benchmark
   :align: center

1 GPU Configuration (Random Projection)
~~~~~~~~~~~~~~~~~~~

.. image:: projection_comparison_with_projection.png
   :alt: 1 GPU Comparison with Projection (16x16)
   :align: center

1 GPU Configuration (No Random Projection)
~~~~~~~~~~~~~~~~~~~

.. image:: projection_comparison_without_projection.png
   :alt: 1 GPU Comparison without Projection
   :align: center

1 GPU In-Memory Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: inmem_benchmark_1x_NVIDIA_H100_80GB_HBM3.png
   :alt: 1 GPU In-Memory Benchmark
   :align: center

Running Your Own Benchmarks
----------------------------

To generate benchmarks for your specific setup, you can use the shell scripts in `benchmarks`.
