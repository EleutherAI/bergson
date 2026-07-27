Numerical Stability
===================

Some models produce inconsistent per-example gradients when batched together. This is caused by nondeterminism in optimized SDPA attention backends (flash, memory-efficient). Bergson ships a diagnostic that tests both padding-induced and equal-length batch divergence to pinpoint the source. Use the diagnostic to check your model:

.. code-block:: bash

   bergson test_model_configuration --model <model_name>

This automatically tests escalating configurations and reports exactly which
flags (if any) you need:

.. code-block:: bash

   # If force_math_sdp alone is sufficient:
   bergson build <output_path> --model <model_name> --force_math_sdp
   # If fp32 with TF32 matmuls is sufficient (cheaper than full fp32):
   bergson build <output_path> --model <model_name> --precision fp32 --use_tf32_matmuls --force_math_sdp
   # If full fp32 precision is required:
   bergson build <output_path> --model <model_name> --precision fp32 --force_math_sdp

The same flags apply to ``score`` and ``trackstar``. The diagnostic also
detects special token (BOS/EOS) duplication for chat templates. See the
``test_model_configuration`` entry in :doc:`cli` for all options.

Performance impact
------------------

Benchmarked on A100-80GB with 500 documents from pile-10k:

.. list-table::
   :header-rows: 1
   :widths: 20 45 15 20

   * - Model
     - Settings
     - Build time
     - vs bf16 baseline
   * - Pythia-160M
     - bf16
     - 31.2s
     - —
   * - Pythia-160M
     - bf16 + ``--force_math_sdp``
     - 31.0s
     - -0.7%
   * - Pythia-160M
     - fp32 + ``--use_tf32_matmuls``
     - 26.6s
     - -14.7%
   * - Pythia-160M
     - fp32 + ``--use_tf32_matmuls`` + ``--force_math_sdp``
     - 27.5s
     - -11.9%
   * - Pythia-160M
     - fp32
     - 35.4s
     - +13.3%
   * - Pythia-160M
     - fp32 + ``--force_math_sdp``
     - 40.6s
     - +29.9%
   * - OLMo-2-1B
     - bf16
     - 45.5s
     - —
   * - OLMo-2-1B
     - bf16 + ``--force_math_sdp``
     - 53.9s
     - +18.4%
   * - OLMo-2-1B
     - fp32 + ``--use_tf32_matmuls``
     - 51.3s
     - +12.7%
   * - OLMo-2-1B
     - fp32 + ``--use_tf32_matmuls`` + ``--force_math_sdp``
     - 54.0s
     - +18.8%
   * - OLMo-2-1B
     - fp32
     - 131.8s
     - +189.8%
   * - OLMo-2-1B
     - fp32 + ``--force_math_sdp``
     - 141.2s
     - +210.5%

``--use_tf32_matmuls`` with fp32 precision is significantly cheaper than full
fp32 and may be sufficient for many models.

Not all models are affected — run ``bergson test_model_configuration`` before
enabling these flags to avoid unnecessary overhead.
