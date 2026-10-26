ASTRA
=====

`ASTRA <https://arxiv.org/abs/2507.14740>`_ refines a Kronecker-factored approximate Hessian :math:`(H + D)^{-1} g_q` with momentum SGD on :math:`\tfrac12 x^\top (H + D) x - x^\top g_q`, preconditioned by the damped Kronecker-factored inverse, where :math:`H` is the Gauss-Newton Hessian estimated on a random batch of training documents per step.

Enable it in the CLI or in a YAML in the ``ekfac`` command, by setting ``hessian_pipeline_cfg.astra.num_steps``. Each step costs about five forward passes over the batch, per query.

``astra.solver: cg`` is an alternative to the canonical ASTRA implementation that uses the CG solver (preconditioned conjugate gradients) on one batch of ``batch_size`` documents.

See ``examples/replications/bae_approx_unrolling_source_replication/wikitext_gpt2_astra.yaml`` and ``examples/gpt2_wikitext_leaderboard/ekfac_astra_cg.yaml``.
