LiSSA
=====

``bergson lissa`` estimates each query's :math:`(G + \lambda I)^{-1} g_q`, where :math:`G` is the Gauss-Newton Hessian approximation of the training loss, with the `LiSSA <https://arxiv.org/abs/1602.03943>`_ iterative solver's truncated series :math:`\sigma^{-1} \sum_t (I - (G + \lambda I) / \sigma)^t g_q`, each term estimating :math:`G` on a random batch of training documents, averaged over repeats.

This is the influence estimator of `Bae et al. <https://arxiv.org/abs/2209.05364>`_.

The series converges when the scale :math:`\sigma` exceeds half the largest eigenvalue of :math:`G + \lambda I`. Each term costs about five forward passes over the batch, per query. See ``examples/gpt2_wikitext_leaderboard/lissa.yaml``.
