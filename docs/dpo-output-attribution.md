# DPO output-token attribution

This branch preserves the local DPO/LoRA curvature fixes on top of `dpo-test`
and adds attribution of predicted-token losses using forward-mode JVPs.

## Configuration and semantics

Set `ScoreConfig.token_influence="output"` with individual dot-product scoring,
`IndexConfig.attribute_tokens=True`, and no gradient projection. The scorer uses
independent GPU model replicas rather than FSDP. Query vectors must match the
adapter parameter layout. For DPO, supply the expanded paired dataset in chosen
rows followed by rejected rows, with frozen reference log-probabilities.

A score at position t belongs to the loss predicting label t+1. Prompt and padding
labels masked with -100 are excluded. Chosen and rejected contributions use the
signed DPO chain rule; summing both sides reconstructs the pair-level gradient
dot product. This differs from decomposing gradients by input-layer position.

`ScoreConfig.output_compute_precision` defaults to `fp32`; `bf16` enables
mixed-precision matrix/attention computation for the output JVP. Parameters,
query directions, vocabulary loss, and saved scores remain FP32. BF16 changes
numerical results and should be checked on representative data before use.

## Memory and scheduling

Reverse-mode collection can enable `stream_output_loss`, `checkpoint_mlp`, and
`checkpoint_attention` for OLMo3. The checkpointed vocabulary loss avoids storing
full sequence-by-vocabulary logits. Replayed forward hooks do not collect
curvature twice. CPU activation offload preserves padded attention-bias strides
and broadcast views. The optional `bergson_efficient_sdpa` attention implementation
supports the memory-efficient reverse-mode path.

Output scoring selects JVP-compatible chunked attention, disables reverse-mode
checkpointing, and trims sequences to their real lengths. Global paired batches
are assigned once using estimated sequence costs. Workers can take unequal
numbers of steps, so dataset size need not divide evenly by GPU count. Written-row
flags support restart; a final coverage check rejects incomplete output.

## Validation and scope

The focused tests cover CE/DPO token sums against direct autograd, attention JVPs,
streamed losses and checkpointed gradients, KFAC/EKFAC collection parity, and
weighted DPO batch invariance. During development, an eight-GPU uneven ten-pair
integration run checked coverage and direct JVP agreement. A separate real-model
runtime validation exercised eight preference pairs including long sequences.
These are validation results, not a claim that full-dataset scoring is complete.

This repository contains the Bergson library changes. Experiment-specific data
preparation, provenance checks, pipeline launchers, and filtering scripts live in
the separate Camilla workspace and are not part of this branch. No datasets,
checkpoints, reference caches, or attribution artifacts are committed here.
