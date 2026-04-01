Gradient Preprocessing
======================

Bergson supports several gradient preprocessing operations that affect the quality and meaning of similarity scores. This page explains the operations available, when to apply them to query versus index gradients, and walks through concrete use cases.

Operations
----------

**Optimizer normalization** (``--normalizer``): Scales each gradient element by the inverse root-mean-square (RMS) of that parameter's gradient history — i.e., divides by :math:`\sqrt{E[g^2]} + \varepsilon`, where :math:`E[g^2]` is the mean of squared gradients across the dataset. Applied elementwise during gradient collection. Unlike the Adam optimizer used during training, this uses a simple mean over the dataset rather than an exponential moving average. This downweights parameters with large gradient magnitudes and amplifies signal in directions with consistently small gradients.

**Unit normalization** (``--unit_normalize``): Normalizes each gradient vector to unit L2 norm before similarity computation, enabling cosine similarity when used with inner product scoring.

**Preconditioning** (``--query_preconditioner_path``, ``--index_preconditioner_path``): Applies a per-module matrix transformation derived from a Hessian approximation (second moment matrix of gradients). For inner product scoring, :math:`H^{-1}` is applied to the query side. For cosine similarity scoring, :math:`H^{-1/2}` must be applied to both sides symmetrically.

Query vs Index Gradients
------------------------

Every similarity computation involves two sides:

- **Index gradients**: Gradients from the training dataset you want to search.
- **Query gradients**: Gradients from the dataset whose most similar training examples you want to find.

For a similarity score to be meaningful, preprocessing applied to query and index gradients must be consistent.

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Operation
     - Can apply one-sided?
     - Notes
   * - Optimizer normalization
     - Yes
     - Apply the same ``--normalizer`` when collecting both query and index gradients
   * - Preconditioning (inner product)
     - Yes
     - :math:`H^{-1}` applied to query only; relative score rankings are preserved
   * - Preconditioning (cosine similarity)
     - **No**
     - :math:`H^{-1/2}` must be applied to **both** sides before unit normalization
   * - Unit normalization
     - **No**
     - Must be applied consistently to both sides

**Unit normalization is a non-linear operation and does not commute with preconditioning.** When unit normalization is enabled alongside preconditioning, the preconditioner must be applied to both query and index gradients before normalization. Bergson handles this automatically: when ``unit_normalize=True``, it applies :math:`H^{-1/2}` to the query gradient upfront in the ``score`` command and applies :math:`H^{-1/2}` to each index gradient as it is collected during scoring.

Case Studies
------------

Cosine similarity with an optimizer normalizer (full gradients)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goal:** Rank training examples by cosine similarity to a query, using optimizer-normalized gradients.

Optimizer normalization scales each parameter's gradient by :math:`1/(\sqrt{v} + \varepsilon)`, where :math:`v = E[g^2]` is the mean of squared gradients across the dataset. Applied before cosine similarity, this reweights the gradient space by the inverse RMS of each parameter's gradient history, emphasizing directions with consistently small-magnitude gradients.

The normalizer is applied during gradient collection, so the same ``--normalizer`` must be set when collecting both query and index gradients. Unit normalization is then applied at scoring time to obtain cosine similarity.

.. code-block:: bash

   # Reduce query dataset to a single mean gradient with optimizer normalization
   bergson reduce runs/query \
       --model EleutherAI/pythia-160m \
       --dataset query_data \
       --projection_dim 0 \
       --normalizer adafactor \
       --aggregation mean \
       --skip_preconditioners

   # Score: collect training gradients with the same normalizer, unit normalize for cosine similarity
   bergson score runs/scores \
       --query_path runs/query \
       --model EleutherAI/pythia-160m \
       --dataset training_data \
       --projection_dim 0 \
       --normalizer adafactor \
       --unit_normalize

Both commands use ``--projection_dim 0`` to preserve the full gradient, and the same ``--normalizer`` to ensure consistent per-parameter scaling. The ``score`` command applies unit normalization to both the loaded query gradient and each training gradient, giving cosine similarity in the optimizer-normalized space.

Inner product with an optimizer normalizer (full gradients)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goal:** Rank training examples by inner product with a query gradient in optimizer-normalized space, approximating the classic influence function.

The influence function estimates the change in query loss from upweighting a training example as :math:`\partial L_q / \partial \varepsilon_t \approx -g_q H^{-1} g_t^T`, where :math:`H` is the Hessian. The optimizer normalizer provides a diagonal approximation to :math:`H^{-1/2}`, so applying it to both query and index gradients approximates the full influence inner product.

Unlike cosine similarity, inner product preserves gradient magnitude, so training examples with larger gradients contribute more to the score.

.. code-block:: bash

   # Reduce query dataset to a single mean gradient with optimizer normalization
   bergson reduce runs/query \
       --model EleutherAI/pythia-160m \
       --dataset query_data \
       --projection_dim 0 \
       --normalizer adafactor \
       --aggregation mean \
       --skip_preconditioners

   # Score: inner product (no --unit_normalize)
   bergson score runs/scores \
       --query_path runs/query \
       --model EleutherAI/pythia-160m \
       --dataset training_data \
       --projection_dim 0 \
       --normalizer adafactor

**Inner product vs cosine similarity:** Use inner product when gradient magnitude carries information (larger gradients indicate stronger relevance). Use cosine similarity to compare direction independently of magnitude, which is more robust when examples differ systematically in gradient norm (e.g., due to different sequence lengths or loss scales).

Randomly projected gradients with reduce and score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goal:** Select training examples most similar to a query set using random projection, keeping full-batch scoring tractable for large models.

Random projections approximately preserve inner products and cosine similarities (Johnson-Lindenstrauss) while reducing gradient dimensionality by orders of magnitude. For large models, full gradients may be gigabytes per example; projecting to a few thousand dimensions makes the ``reduce → score`` pipeline tractable while retaining most of the signal.

``reduce`` aggregates all query gradients into a single vector (mean or sum) without storing per-example gradients. ``score`` then collects each training gradient on-the-fly and scores it against the precomputed query vector, avoiding the need to build or store a full training gradient index.

.. code-block:: bash

   # Reduce query dataset to a single mean gradient vector
   bergson reduce runs/query \
       --model EleutherAI/pythia-160m \
       --dataset query_data \
       --projection_dim 4096 \
       --aggregation mean \
       --skip_preconditioners

   # Score training data against the reduced query
   bergson score runs/scores \
       --query_path runs/query \
       --model EleutherAI/pythia-160m \
       --dataset training_data \
       --projection_dim 4096

Both commands must use the same ``--projection_dim`` and identical model configuration so that both sides are projected into the same random subspace. The random projection matrix is derived deterministically from the model architecture and the projection dimension.

.. note::

   **Preprocessing order:** Optimizer normalization must be applied during gradient collection (set ``--normalizer`` at both ``reduce`` and ``score`` time). It cannot be applied after the mean-reduction in ``reduce`` - the normalizer is non-linear so applying it to the mean gradient is not the same as normalizing each gradient then taking the mean.

Randomly projected gradients with unit normalization, preconditioners, build, and score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goal:** Compute preconditioner-weighted cosine similarity using random projections. This is the approach used by the ``trackstar`` command.

When combining preconditioning with cosine similarity, the preconditioner must be applied before unit normalization to both query and index gradients. Bergson applies :math:`H^{-1/2}` to the query gradient at the start of ``score``, and :math:`H^{-1/2}` to each index gradient as it is collected. The resulting score is:

.. math::

   g_q^p = g_q \cdot H^{-1/2}

   g_t^p = g_t \cdot H^{-1/2}

   \text{score}(q, t) = \frac{g_q^p}{\|g_q^p\|} \cdot \frac{g_t^p}{\|g_t^p\|}

This is cosine similarity in the :math:`H^{-1}`-weighted inner product space — the same geometry used by the influence function.

.. code-block:: bash

   # Step 1: Compute normalizers and preconditioners on the query dataset
   bergson preconditioners runs/query_precond \
       --model EleutherAI/pythia-160m \
       --dataset query_data \
       --projection_dim 4096

   # Step 2: Compute normalizers and preconditioners on the training dataset
   bergson preconditioners runs/index_precond \
       --model EleutherAI/pythia-160m \
       --dataset training_data \
       --projection_dim 4096

   # Step 3: Build per-example query gradient index
   # The query normalizer (from runs/query_precond) is applied during collection
   bergson build runs/query \
       --model EleutherAI/pythia-160m \
       --dataset query_data \
       --projection_dim 4096 \
       --processor_path runs/query_precond \
       --skip_preconditioners

   # Step 4: Score training data against query
   # H^(-1/2) is applied to both query and index gradients, then unit normalized
   bergson score runs/scores \
       --query_path runs/query \
       --model EleutherAI/pythia-160m \
       --dataset training_data \
       --projection_dim 4096 \
       --processor_path runs/index_precond \
       --skip_preconditioners \
       --unit_normalize \
       --query_preconditioner_path runs/query_precond \
       --index_preconditioner_path runs/index_precond

This pipeline is also available as the ``trackstar`` command, which automates the four steps above. See ``bergson trackstar --help`` for the full argument list.

**Why** :math:`H^{-1/2}` **on both sides?** For inner product scoring, applying :math:`H^{-1}` to one side only is sufficient since the relative ordering of :math:`g_q H^{-1} g_t^T` is preserved. For cosine similarity, the unit normalization would undo a one-sided application: normalizing :math:`g_t` to unit norm discards the preconditioner's geometry. Applying :math:`H^{-1/2}` symmetrically to both sides before normalization preserves the preconditioned structure and ensures the normalization operates in the correct space.

**Mixing query and index preconditioners:** When query and index datasets come from different distributions, ``--target_downweight_components`` (default 1000) is used to compute a mixing coefficient that interpolates between their second moment matrices (i.e. the empirical Fisher information matrices):

.. math::

   H_\text{mixed} = \alpha \cdot H_\text{query} + (1 - \alpha) \cdot H_\text{index}

Values close to 1.0 weight the query distribution more heavily; values close to 0.0 weight the index distribution. Adjust this according to the guidelines in https://arxiv.org/abs/2410.17413
