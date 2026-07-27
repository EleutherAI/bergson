Training
========

We provide a scalable, distributed, deterministic, and twice-differentiable Trainer.

*Twice-differentiable* implies support for the double backward pass used in MAGIC. We're not aware of any standard alternative trainers with this feature.

*Deterministic* implies that research based on the trainer will be reproducible. We also integrate with a per-token weighted loss function so training items can be filtered or re-weighted without changing the data order.

*Scalable* and *distributed* imply that you should be able to scale this trainer to large-scale experiments without fuss. We haven't personally tested it beyond 7B, but please get in touch if you're scaling up further - for very large pre-training runs, we think it should be possible to use an optimized library like GPT-NeoX for the forward phase, and the MAGIC trainer for the backward phase.
