Building Blocks
===============

Bergson exposes generic building blocks like ``train``, ``build``, ``score`` and ``hessian`` that can be used to
produce diverse attribution pipelines. This page explains some basic commands and how to assemble them into multi-step pipelines.

Overview
--------

Two core commands run the same underlying gradient collection pipeline:

.. code-block:: text

   raw gradient → apply normalizer → apply random projection → write or aggregate

The difference between them is **what they do with the collected gradients**:

- ``build`` writes each **per-example dataset gradient** to an on-disk index (gradient store). Aggregation configuration may be used to write a single final gradient instead.
- ``score`` **computes influence scores** by comparing gradients from one dataset to a pre-built query. Aggregation configuration may be used to compress a multi-row query index into one.

The supporting command ``hessian`` computes Hessian approximations which can be passed into score or used with a gradient store at query time (``autocorrelation`` — the gradient second-moment
 — ``kfac``, ``tkfac``, or ``shampoo``, selected with the required ``--method`` flag).

Pipeline
--------

You can define a pipeline in a YAML file like so:

.. code-block:: yaml

   run_path: runs/minimal

   steps:
     - build:
         index_cfg:
           run_path: runs/minimal/query
           model: EleutherAI/pythia-14m
           data:
             dataset: NeelNanda/pile-10k
             split: "train[:64]"
             truncation: true
         preprocess_cfg:
           aggregation: mean

     - score:
         score_cfg:
           query_path: runs/minimal/query
         index_cfg:
           run_path: runs/minimal/scores
           model: EleutherAI/pythia-14m
           data:
             dataset: NeelNanda/pile-10k
             split: "train[:64]"
             truncation: true

Run it with ``bergson pipeline.yaml``. Each step takes the same options as its
CLI counterpart, but grouped into config sections rather than passed as flat
flags. A finished run can be replayed with ``bergson <run_path>/config.yaml``.

.. _build-command:

``build`` — Build a Gradient Store
-------------------------------------------

``build`` runs every example in your dataset through the model, collects a gradient for each one, and stores the resulting vectors in a memory-mapped index on disk. The index is keyed by example and supports fast nearest-neighbour search via ``bergson query``. It's good for when you have enough disk space for the gradient store and want to run many serial queries.

``build`` with ``--aggregation mean`` or ``sum`` accumulates every gradient into a **single row store**, discarding the per-example gradients. This is the usual way to produce a query for ``score``.

**Typical use cases**

Per-example (``--aggregation none``):

- You want to check training influences on ad-hoc prompts
- You want to find which training examples are most similar to a given query (e.g. an
  eval example or a generated output).
- You intend to query the index multiple times against different queries.
- You are using small datasets, or random projections (``--projection_dim > 0``) so each
  gradient is small enough to store individually.

Aggregated (``--aggregation mean`` / ``sum``):

- You want to run ``score`` against a whole dataset treated as one query.
- You want the average influence of one dataset on another (e.g. finding which training
  examples are relevant to an entire eval set).

**What it produces**

A directory at ``run_path`` containing:

- ``gradients.bin`` — a memory-mapped binary file of gradients: one row per token or sequence, or a
  single aggregated row when ``--aggregation`` is set.
- ``info.json`` — metadata (num_grads, dtype structure, grad_sizes). ``num_grads`` is 1
  when aggregating, and ``attribute_tokens`` records the store's granularity.
- ``offsets.npy`` — per-token stores only. Rows ``offsets[i]:offsets[i+1]`` are example
  ``i``'s tokens, used by downstream readers.
- ``data.hf/`` — a HuggingFace dataset with per-example metadata and losses, or a single
  row carrying the query index when aggregating.
- ``config.yaml`` — the run's config, replayable with ``bergson <path>``.
- ``processor_config.yaml`` — gradient processor configuration.
- ``normalizers.pth`` — normalizer state dicts.
- ``hessians.pth`` — fitted hessian matrices.
- ``hessians_eigen.pth`` — eigendecompositions of hessians.
- ``total_processed.pt`` — total number of samples processed.

**Key options**

- ``--aggregation none`` (default), ``mean``, or ``sum``: whether and how to aggregate
  gradients into a single row.
- ``--attribute_tokens``: store one gradient per token rather than per example.
  Requires ``--aggregation none``.
- ``--unit_normalize``: unit-normalize individual gradients, applied *before* aggregating.
- ``--projection_dim``: random-projection size; ``0`` keeps the full gradient.

**Example** — a per-example index

.. code-block:: bash

   bergson build runs/my-index \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --projection_dim 16

After building, use ``bergson query`` to interactively search the index:

.. code-block:: bash

   bergson query --index runs/my-index

**Example** — a single aggregated query gradient

.. code-block:: bash

   bergson build runs/my-query \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --aggregation mean \
       --unit_normalize \
       --projection_dim 0

.. note::

   Random projections (``--projection_dim > 0``) dramatically reduce per-example
   storage. With no projection, storing per-example gradients is only practical
   for small models or small datasets. In contrast, aggregated build disk usage
   is constant.

.. note::

   ``--unit_normalize`` applies normalization *per example before* aggregating, so each
   example contributes equally to the mean direction regardless of gradient magnitude.
   This is different from ``--normalize_aggregated_grad``, which normalizes the final
   vector and has no effect on downstream ranking. When using hessians, normalization
   must happen after preconditioning, which is done in ``score`` not ``build``.

.. _score-command:

``score`` — Score a Dataset Against Pre-Computed Query Gradients
----------------------------------------------------------------

``score`` computes a scalar influence score for every example in a dataset by comparing
its gradient against a set of pre-computed **query gradients** loaded from disk. It's good if a gradient store would exceed your available disk space, such as if you are not using random project to compress gradients.

The query gradients were previously produced by ``build``, either per-example or
aggregated into a single row. The scoring process in ``score`` applies preconditioning
and normalization to the loaded query gradients before computing dot products.

**Typical use cases**

- You have a query index (from ``build``) and want to rank a dataset by influence.
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
             ├── Yes → use build --aggregation mean (for query) + score
             └── No → use build + score

**Using hessians**

When using a Hessian approximation (autocorrelation / Adam second moments,
KFAC, EK-FAC, etc.), preconditioning is applied in ``build`` and/or ``score``
depending on whether unit normalization is enabled. The recommended pipeline is:

.. code-block:: text

   bergson hessian → fit hessians
   bergson build   → aggregate query gradients (with preconditioning)
   bergson score   → score training data (sometimes with preconditioning)

Note: if you apply unit normalization, you need to apply hessians in both
build and score.

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

**Step 2 — Aggregate the eval set into a query gradient**

.. code-block:: bash

   bergson build runs/eval-query \
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
