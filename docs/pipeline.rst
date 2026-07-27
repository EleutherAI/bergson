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
(``autocorrelation`` — the gradient second-moment — ``kfac``, ``tkfac``, or
``shampoo``, selected with the required ``--method`` flag) without collecting
per-example gradients. Non-autocorrelation methods are stored as sharded
covariance matrices.

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
- ``config.yaml`` — the run's config, replayable with ``bergson <path>``.
- ``processor_config.yaml`` — gradient processor configuration.
- ``normalizers.pth`` — normalizer state dicts.
- ``hessians.pth`` — fitted hessian matrices.
- ``hessians_eigen.pth`` — eigendecompositions of hessians.
- ``total_processed.pt`` — total number of samples processed.

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
- ``config.yaml`` — the run's config, replayable with ``bergson <path>``.
- ``processor_config.yaml`` — gradient processor configuration.
- ``normalizers.pth`` — normalizer state dicts.
- ``hessians.pth`` — fitted hessian matrices.
- ``hessians_eigen.pth`` — eigendecompositions of hessians.
- ``total_processed.pt`` — total number of samples processed.

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
- ``info.json`` — metadata (num_items, num_scores, dtype structure).
- ``data.hf/`` — a HuggingFace dataset with per-example metadata.
- ``config.yaml`` — the run's config (including scoring options), replayable
  with ``bergson <path>``.
- ``processor_config.yaml``, ``normalizers.pth``, ``hessians.pth``,
  ``hessians_eigen.pth`` — gradient processor artifacts.
- ``total_processed.pt`` — total number of samples processed.

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
storing per-example gradients. The estimator is selected with the required
``--method`` flag:

- ``autocorrelation`` — gradient second-moment, saved as a
  ``GradientProcessor`` (normalizers + per-module hessian matrices).
- ``kfac``, ``tkfac``, ``shampoo`` — factorised approximations, saved as sharded
  activation/gradient covariance matrices.

**What it produces**

A directory at ``run_path``. With ``--method autocorrelation``:

- ``config.yaml`` — the run's config, replayable with ``bergson <path>``.
- ``processor_config.yaml`` — gradient processor configuration.
- ``normalizers.pth`` — normalizer state dicts.
- ``hessians.pth`` — fitted per-module hessian matrices.
- ``hessians_eigen.pth`` — eigendecompositions of hessians.
- ``total_processed.pt`` — total number of samples processed.

With ``--method kfac`` / ``tkfac`` / ``shampoo``:

- ``config.yaml`` — the run's config, replayable with ``bergson <path>``.
- ``total_processed.pt`` — total number of samples processed.
- ``activation_sharded/shard_*.safetensors`` — sharded activation covariance matrices (one per GPU).
- ``gradient_sharded/shard_*.safetensors`` — sharded gradient covariance matrices (one per GPU).
- ``eigen_activation_sharded/`` and ``eigen_gradient_sharded/`` — eigendecompositions of the activation/gradient covariances.
- ``eigenvalue_sharded/``, ``factor_eig_a/``, ``factor_eig_g/`` — sharded eigenvalues and Kronecker-factor eigenvectors.

**Key options**

- ``--method autocorrelation``, ``kfac``, ``tkfac``, or ``shampoo`` (required): Hessian approximation method.
- ``--ev_correction``: additionally compute eigenvalue correction (KFAC family).
- ``--hessian_dtype``: precision for the Hessian computation.

**Example**

.. code-block:: bash

   bergson hessian runs/my-hessian \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --method kfac

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
