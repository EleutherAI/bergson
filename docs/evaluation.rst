Evaluation
==========

Bergson supports validating model attributions using the linear datamodeling score (LDS), a widely used metric introduced in `TRAK: Attributing Model Behavior at Scale <https://arxiv.org/abs/2303.14186>`_, via ``bergson validate``. This method compares summed attribution scores to ground-truth results over hundreds of re-training runs.

To evaluate attributions of large models that cannot be re-trained many times, we provide a synthetic factual dataset and its generator, and a proxy evaluation of how well attribution can be used to retrieve logically entailing data via ``bergson recall``. The evaluation reports MRR and Recall@k.


.. TODO: Evaluating attribution scores — ``bergson validate`` (leave-k-out
   retraining / linear datamodeling score), ``bergson recall`` (synthetic
   factual recall, MRR / Recall@k), and ``bergson metasmoothness``.
