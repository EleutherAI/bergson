Pipeline Concepts
=================

Bergson's post-hoc attribution exposes three generic building blocks — ``build``, ``reduce``,
and ``score`` — that together implement gradient-based data attribution. This page
explains what each command does, what it produces, and when you should use each one.
It also covers the supporting commands ``hessian`` and ``preconditioners``.

Overview
--------

The three core commands run the same underlying gradient collection pipeline:

.. code-block:: text

   raw gradient → apply normalizer → apply random projection → write or aggregate

The difference between them is **what they do with the collected gradients**:

- ``build`` writes a **per-example gradient** to an on-disk index.
- ``reduce`` **aggregates** all gradients from a dataset into a single vector and writes it to an on-disk file.
- ``score`` **computes similarity scores** by comparing gradients from one dataset against a pre-built query.

The supporting commands handle preconditioner computation and end-to-end pipelines:

- ``hessian`` computes Hessian approximations (KFAC, TKFAC, Shampoo) stored as sharded covariance matrices.
- ``preconditioners`` fits normalizers and preconditioners without collecting gradients.

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
- ``normalizers.pth`` — fitted normalizer state dicts.
- ``preconditioners.pth`` — fitted preconditioner matrices.
- ``preconditioners_eigen.pth`` — eigendecompositions of preconditioners.

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
- ``normalizers.pth`` — fitted normalizer state dicts.
- ``preconditioners.pth`` — fitted preconditioner matrices.
- ``preconditioners_eigen.pth`` — eigendecompositions of preconditioners.

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
       --projection_dim 0 \
       --skip_preconditioners

.. note::

   ``--unit_normalize`` in ``reduce`` applies normalization *per example before*
   aggregating, so each example contributes equally to the mean direction regardless of
   gradient magnitude. This is different from normalizing the final aggregated vector
   (which would have no effect on downstream ranking). When using preconditioners,
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
- ``processor_config.json``, ``normalizers.pth``, ``preconditioners.pth``,
  ``preconditioners_eigen.pth`` — gradient processor artifacts.

**Scoring modes** (``--score``)

- ``individual`` (default): compute a separate score for every query gradient.
  Produces one score field per query in ``scores.bin``.
- ``nearest``: compare each training gradient to the *most similar* query gradient
  (max over all queries). Useful when queries represent distinct individual examples.

**Key options**

- ``--query_path``: path to the pre-computed query gradient index (required).
- ``--unit_normalize``: unit-normalize training gradients before scoring.
- ``--preconditioner_path``: path to a precomputed preconditioner to apply.
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
       --projection_dim 0 \
       --skip_preconditioners

.. _hessian-command:

``hessian`` — Compute Hessian Approximations
---------------------------------------------

``hessian`` computes Hessian approximations (KFAC, EK-FAC, TKFAC, or Shampoo) by collecting
activation and gradient covariance matrices across the dataset. These are used as
preconditioners in downstream ``score`` runs.


**What it produces**

A directory at ``run_path`` containing:

- ``index_config.json`` — configuration snapshot.
- ``hessian_config.json`` — Hessian-specific configuration (method, dtype, ev_correction).
- ``total_processed.pt`` — total number of samples processed.
- ``activation_sharded/shard_*.safetensors`` — sharded activation covariance matrices (one per GPU).
- ``gradient_sharded/shard_*.safetensors`` — sharded gradient covariance matrices (one per GPU).
- ``eigen_activation_sharded/shard_*.safetensors`` — eigendecompositions of activation covariances (if computed).
- ``eigen_gradient_sharded/shard_*.safetensors`` — eigendecompositions of gradient covariances (if computed).

**Key options**

- ``--method kfac`` (default), ``tkfac``, or ``shampoo``: Hessian approximation method.
- ``--ev_correction``: additionally compute eigenvalue correction.
- ``--hessian_dtype``: precision for the Hessian computation.

**Example**

.. code-block:: bash

   bergson hessian runs/my-hessian \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --method kfac

.. _preconditioners-command:

``preconditioners`` — Fit Normalizers and Preconditioners
----------------------------------------------------------

``preconditioners`` fits normalizers and preconditioners on a dataset without collecting
or storing per-example gradients. This is equivalent to running ``build`` with
``--skip_index``.

**Typical use cases**

- You only need the fitted processor (normalizers/preconditioners) for a subsequent
  ``build`` or ``score`` run via ``--processor_path``.
- You want to compute preconditioners separately from gradient collection.

**What it produces**

A directory at ``run_path`` containing:

- ``index_config.json`` — configuration snapshot.
- ``processor_config.json`` — gradient processor configuration.
- ``normalizers.pth`` — fitted normalizer state dicts.
- ``preconditioners.pth`` — fitted preconditioner matrices.
- ``preconditioners_eigen.pth`` — eigendecompositions of preconditioners.

**Example**

.. code-block:: bash

   bergson preconditioners runs/my-processor \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --normalizer adam

Choosing the Right Command
--------------------------

The decision tree below covers the most common scenarios:

.. code-block:: text

   Do you want to search a gradient index interactively (e.g. per-prompt)?
   ├── Yes → use build + query
   └── No  → Do you want to search using aggregated gradients?
             ├── Yes → use reduce (for query) + score
             └── No → use build + score

**Using preconditioners**

When using preconditioners (KFAC, EK-FAC, Adam second moments), preconditioning is
applied in ``reduce`` and/or ``score`` depending on whether unit normalization is
enabled. The recommended pipeline is:

.. code-block:: text

   bergson preconditioners → fit normalizers/preconditioners
   bergson reduce          → aggregate query gradients (with preconditioning)
   bergson score           → score training data (sometimes with preconditioning)

For Hessian-based preconditioners (KFAC, EK-FAC), use ``bergson hessian`` instead.

Note: if you apply unit normalization, you need to apply any preconditioners in both
reduce and score.

Worked Example: Query Influence with Preconditioners
-----------------------------------------------------

This example computes the influence of a training set on a small evaluation set
using preconditioned cosine similarity.

**Step 1 — Build a preconditioner on training data**

.. code-block:: bash

   bergson preconditioners runs/preconditioner \
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
       --preconditioner_path runs/preconditioner \
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
       --preconditioner_path runs/preconditioner \
       --unit_normalize \
       --projection_dim 16

The resulting ``runs/scores/scores.bin`` contains one score per training example.
Higher scores indicate stronger positive influence on the eval set.
