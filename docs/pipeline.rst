Pipeline Concepts
=================

Bergson's post-hoc attribution exposes three generic building blocks — ``build``, ``reduce``,
and ``score`` — that together implement gradient-based data attribution. This page
explains what each command does, what it produces, and when you should use each one.
It also covers the supporting command ``hessian``.

Overview
--------

The three core commands run the same underlying gradient collection pipeline:

.. code-block:: text

   raw gradient → apply normalizer → apply random projection → write or aggregate

The difference between them is **what they do with the collected gradients**:

- ``build`` writes a **per-example gradient** to an on-disk index.
- ``reduce`` **aggregates** all gradients from a dataset into a single vector and writes it to an on-disk file.
- ``score`` **computes similarity scores** by comparing gradients from one dataset against a pre-built query.

The supporting command ``hessian`` computes Hessian approximations
(``autocorrelation`` — the gradient second-moment and the default — or ``kfac``,
``tkfac``, ``shampoo``) without collecting per-example gradients. Non-autocorrelation
methods are stored as sharded covariance matrices.

.. _build-command:

``build`` — Build a Per-Example Gradient Index
-----------------------------------------------

``build`` runs every example in your dataset through the model, collects a gradient for
each one, and stores the resulting vectors in a memory-mapped index on disk.

The index is keyed by example and supports fast nearest-neighbour search via
``bergson query``.

**Typical use cases**

- You want to check training influences on ad-hoc prompts
- You want to find which training examples are most similar to a given query (e.g. an
  eval example or a generated output).
- You intend to query the index multiple times against different queries.
- You are using small datasets, or random projections (``--projection_dim > 0``) so each
  gradient is small enough to store individually.

**What it produces**

A directory at ``run_path`` containing:

- ``gradients.bin`` — a memory-mapped binary file of per-example gradients.
- ``info.json`` — metadata (num_grads, dtype structure, grad_sizes).
- ``data.hf/`` — a HuggingFace dataset with per-example metadata and losses.
- ``index_config.json`` — configuration snapshot.
- ``processor_config.json`` — gradient processor configuration.
- ``normalizers.pth`` — normalizer state dicts.
- ``hessians.pth`` — fitted hessian matrices.
- ``hessians_eigen.pth`` — eigendecompositions of hessians.

**Example**

.. code-block:: bash

   bergson build runs/my-index \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --projection_dim 16

After building, use ``bergson query`` to interactively search the index:

.. code-block:: bash

   bergson query --index runs/my-index

.. note::

   Random projections (``--projection_dim > 0``) dramatically reduce per-example
   storage. With no projection (``--projection_dim 0``), storing per-example gradients
   is only practical for small models or small datasets.

.. _reduce-command:

``reduce`` — Aggregate a Dataset into a Single Query Gradient
-------------------------------------------------------------

``reduce`` collects per-example gradients and immediately **aggregates** them into a
single representative vector (mean or sum). Only the aggregate is written to disk, not
the individual per-example gradients.

The resulting aggregate is typically used as the **query** for a subsequent ``score``
run.

**Typical use cases**

- You want to run ``score`` on an aggregated dataset query.
- You want to compute the average influence of a dataset on another dataset (e.g.
  finding which training examples are relevant to an entire eval set).

**What it produces**

A directory at ``run_path`` containing:

- ``gradients.bin`` — a single aggregated gradient vector (one row).
- ``info.json`` — metadata (num_grads=1, dtype structure, grad_sizes).
- ``data.hf/`` — a HuggingFace dataset (single row with query index).
- ``index_config.json`` — configuration snapshot.
- ``processor_config.json`` — gradient processor configuration.
- ``normalizers.pth`` —  normalizer state dicts.
- ``hessians.pth`` — fitted hessian matrices.
- ``hessians_eigen.pth`` — eigendecompositions of hessians.

**Key options**

- ``--aggregation mean`` (default) or ``--aggregation sum``: how to aggregate gradients.
- ``--unit_normalize``: unit-normalize individual gradients *before* aggregating.

**Example**

.. code-block:: bash

   bergson reduce runs/my-query \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --aggregation mean \
       --unit_normalize \
       --projection_dim 0

.. note::

   ``--unit_normalize`` in ``reduce`` applies normalization *per example before*
   aggregating, so each example contributes equally to the mean direction regardless of
   gradient magnitude. This is different from normalizing the final aggregated vector
   (which would have no effect on downstream ranking). When using hessians,
   normalization must happen after preconditioning, which is done in ``score`` not
   ``reduce``.

.. _score-command:

``score`` — Score a Dataset Against Pre-Computed Query Gradients
----------------------------------------------------------------

``score`` computes a scalar influence score for every example in a dataset by comparing
its gradient against a set of pre-computed **query gradients** loaded from disk.

The query gradients were previously produced by ``reduce`` (or ``build``). The scoring
process in ``score`` applies preconditioning and normalization to the loaded query
gradients before computing dot products.

**Typical use cases**

- You have a query index (from ``reduce`` or ``build``) and want to rank a dataset
  by influence.
- You don't need to store individual training gradients on disk — ``score`` computes
  and immediately discards each training gradient after comparing it.

**What it produces**

A directory at ``run_path`` containing:

- ``scores.bin`` — a memory-mapped structured array of scores (one entry per example,
  with per-query score fields).
- ``score_config.json`` — scoring configuration (query_path, modules, score method).
- ``info.json`` — metadata (num_items, num_scores, dtype structure).
- ``data.hf/`` — a HuggingFace dataset with per-example metadata.
- ``index_config.json`` — configuration snapshot.
- ``processor_config.json``, ``normalizers.pth``, ``hessians.pth``,
  ``hessians_eigen.pth`` — gradient processor artifacts.

**Scoring modes** (``--score``)

- ``individual`` (default): compute a separate score for every query gradient.
  Produces one score field per query in ``scores.bin``.
- ``nearest``: compare each training gradient to the *most similar* query gradient
  (max over all queries). Useful when queries represent distinct individual examples.

**Key options**

- ``--query_path``: path to the pre-computed query gradient index (required).
- ``--unit_normalize``: unit-normalize training gradients before scoring.
- ``--hessian_path``: path to a precomputed gradient processor. Set to apply a Hessian approximation.
- ``--modules``: restrict scoring to a subset of model modules.

**Example**

.. code-block:: bash

   bergson score runs/my-scores \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --query_path runs/my-query \
       --score individual \
       --unit_normalize \
       --projection_dim 0

.. _hessian-command:

``hessian`` — Compute Hessian Approximations
---------------------------------------------

``hessian`` computes Hessian approximations on a dataset without collecting or
storing per-example gradients. The estimator is selected with ``--method``:

- ``autocorrelation`` (default) — gradient second-moment, saved as a
  ``GradientProcessor`` (normalizers + per-module hessian matrices).
- ``kfac``, ``tkfac``, ``shampoo`` — factorised approximations, saved as sharded
  activation/gradient covariance matrices.

**What it produces**

A directory at ``run_path``. With ``--method autocorrelation``:

- ``index_config.json`` — configuration snapshot.
- ``processor_config.json`` — gradient processor configuration.
- ``normalizers.pth`` — normalizer state dicts.
- ``hessians.pth`` — fitted per-module hessian matrices.
- ``hessians_eigen.pth`` — eigendecompositions of hessians.

With ``--method kfac`` / ``tkfac`` / ``shampoo``:

- ``index_config.json`` — configuration snapshot.
- ``hessian_config.json`` — Hessian-specific configuration (method, dtype, ev_correction).
- ``total_processed.pt`` — total number of samples processed.
- ``activation_sharded/shard_*.safetensors`` — sharded activation covariance matrices (one per GPU).
- ``gradient_sharded/shard_*.safetensors`` — sharded gradient covariance matrices (one per GPU).
- ``eigen_activation_sharded/shard_*.safetensors`` — eigendecompositions of activation covariances (if computed).
- ``eigen_gradient_sharded/shard_*.safetensors`` — eigendecompositions of gradient covariances (if computed).

**Key options**

- ``--method autocorrelation`` (default), ``kfac``, ``tkfac``, or ``shampoo``: Hessian approximation method.
- ``--ev_correction``: additionally compute eigenvalue correction (KFAC family).
- ``--hessian_dtype``: precision for the Hessian computation.

**Example**

.. code-block:: bash

   bergson hessian runs/my-hessian \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --method kfac

.. _preconditioners:

Preconditioners: Inverting and Applying Hessians
------------------------------------------------

A fitted Hessian is applied to gradients as a *preconditioner* ``H^p``. Two
knobs control this: ``apply_power`` (how much inverse) and the
``InversionConfig`` (how eigenvalues are regularized before the power is
taken). The fit itself is identical for every setting of both — only the
apply changes.

**Apply power**

``HessianConfig.apply_power`` is the inverse power ``p`` applied to the fitted
eigenvalue grid. For a factored (Kronecker) Hessian ``H = A ⊗ S``, a grid
power ``p`` is equivalent to per-side factor powers ``p``, since
``(A ⊗ S)^p = A^p ⊗ S^p``:

- ``-1.0`` — the full inverse ``A^-1 G S^-1``, the influence-function
  preconditioner.
- ``-0.5`` — the two-sided square-root inverse ``A^-1/2 G S^-1/2``. Applied to
  *both* sides of a score inner product this recovers the full inverse
  (``⟨H^-1/2 q, H^-1/2 g⟩ = q^T H^-1 g``, the ``unit_normalize`` convention);
  applied one-sided it is a genuinely gentler preconditioner.
- ``-0.25`` — the per-side quarter roots ``A^-1/4 G S^-1/4``: classic
  Shampoo's exponent (Meta distributed_shampoo's default root ``1/(2·order)``
  for a matrix) and the Muon-orthogonalization-equivalent preconditioner.

Because one fit serves every power, the hessian pipeline tags its output
directory with non-default powers (e.g. ``shampoo_p0.25_query``).

**Inversion modes and their knobs**

Each ``InversionConfig.inversion`` mode maps eigenvalues ``λ`` to a multiplier
``m(λ)``, which is then raised to ``-apply_power``; regularization therefore
composes *inside* the power for the damped modes (``(λ + δ)^p``) and as
truncate-then-power for the pseudoinverse modes (``λ^p·𝟙[λ > tol]``).

The mean-relative modes (``damped_inverse``, ``tikhonov_filtered``,
``pseudoinverse``, ``factored_tikhonov``) are controlled by
``damping_factor``: their damping or truncation threshold is
``damping_factor · mean(λ)``.

The rank-based modes (``pseudoinverse_rank``, ``pseudoinverse_factored``)
ignore ``damping_factor`` and instead use the ``torch.linalg.matrix_rank``
convention, matching Meta distributed_shampoo's ``PseudoInverseConfig``:

- threshold = ``clamp(rank_rtol · max(λ), min=rank_atol)``;
- ``rank_rtol=None`` (the default) means ``numel(λ) · machine_eps``, the
  ``torch.linalg.matrix_rank`` / distributed_shampoo default;
- distributed_shampoo requires ``epsilon=0`` with pseudo-inverse, and
  correspondingly no damping term is added in these modes.

``pseudoinverse_rank`` thresholds the joint eigenvalue spectrum;
``pseudoinverse_factored`` computes a separate threshold per Kronecker factor,
so a grid cell ``λ_G·λ_A`` survives only if both factor eigenvalues pass their
own factor's cutoff — a cell whose *product* is large is still zeroed when one
factor is rank-deficient, reproducing distributed_shampoo's per-side truncated
roots ``L^{†p} G R^{†p}``.

**Tolerance units**

The stored factor eigenvalues are trace-normalized and scaled by
``sqrt(total_processed)`` per factor. ``rank_rtol`` is relative to
``max(λ)``, so it is invariant to both rescalings and transfers across fits —
prefer it. ``rank_atol`` is compared in the stored units, *not* in units of a
raw gradient-covariance accumulator, so an absolute floor tuned for one fit
(or borrowed from an optimizer setting such as ``eps=1e-15``) does not
transfer directly.

**Background: influence-function vs optimizer-style inversion**

The two regularization families come from different lineages, and they answer
the rank-deficiency question differently.

The influence-function literature (Koh & Liang 2017; EK-FAC influence, Grosse
et al. 2023) inverts with Tikhonov damping — ``(H + δI)^-1`` with ``δ``
proportional to the mean eigenvalue. Near-null directions receive a large but
*bounded* gain of about ``1/δ``: rare directions get extra credit, but noise
in the unexplored subspace cannot dominate a score. This is bergson's default
(``damped_inverse``).

The optimizer lineage instead takes fractional inverse roots of the gradient
second moment (Shampoo's per-side quarter roots), and in practice handles
rank deficiency by *truncation*: Meta's distributed_shampoo with
pseudo-inverse enabled zeroes the deficient part of each factor's spectrum
rather than damping it, as highlighted in Rohan Anil's NanoGPT-speedrun
retuning (https://x.com/_arohan_/status/2064631528806908134). Null directions
are ignored entirely — the preconditioner acts only on the subspace the data
actually explored.

That optimizer configuration transplants into attribution as
``apply_power: -0.25`` with ``inversion: pseudoinverse_factored`` and default
tolerances; ``examples/pipelines/rohan_shampoo_gpt2_lds.yaml`` runs exactly
this against the influence-function baseline. Whether zeroing or damping the
null space attributes better is an empirical question — the LDS comparisons
these knobs exist for.

Choosing the Right Command
--------------------------

The decision tree below covers the most common scenarios:

.. code-block:: text

   Do you want to search a gradient index interactively (e.g. per-prompt)?
   ├── Yes → use build + query
   └── No  → Do you want to search using aggregated gradients?
             ├── Yes → use reduce (for query) + score
             └── No → use build + score

**Using hessians**

When using a Hessian approximation (autocorrelation / Adam second moments,
KFAC, EK-FAC, etc.), preconditioning is applied in ``reduce`` and/or ``score``
depending on whether unit normalization is enabled. The recommended pipeline is:

.. code-block:: text

   bergson hessian → fit hessians
   bergson reduce  → aggregate query gradients (with preconditioning)
   bergson score   → score training data (sometimes with preconditioning)

Note: if you apply unit normalization, you need to apply hessians in both
reduce and score.

Worked Example: Query Influence with Hessians
-----------------------------------------------------

This example computes the influence of a training set on a small evaluation set
using preconditioned cosine similarity.

**Step 1 — Fit a hessian on training data**

.. code-block:: bash

   bergson hessian runs/hessian \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --projection_dim 16

**Step 2 — Reduce the eval set to a query gradient**

.. code-block:: bash

   bergson reduce runs/eval-query \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --hessian_path runs/hessian \
       --unit_normalize \
       --aggregation mean \
       --projection_dim 16

**Step 3 — Score training examples against the query**

.. code-block:: bash

   bergson score runs/scores \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --query_path runs/eval-query \
       --hessian_path runs/hessian \
       --unit_normalize \
       --projection_dim 16

The resulting ``runs/scores/scores.bin`` contains one score per training example.
Higher scores indicate stronger positive influence on the eval set.
