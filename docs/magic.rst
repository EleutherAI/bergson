MAGIC (Unrolling)
=================

`MAGIC <https://arxiv.org/abs/2504.16430>`_ attributes evaluation loss to individual training examples by backpropagating through the entire training trajectory.

We provide a `Trainer` class that takes differentiable training steps and handles all three phases of MAGIC attribution. We support FSDP training using the `bergson.magic.dtensor_patch` runtime patch, which makes PyTorch's DTensor redistribution twice-differentiable (`pytorch/pytorch#160509 <https://github.com/pytorch/pytorch/pull/160509>`_). The patch is applied in memory, so no torch source files are modified.

How it works
------------

MAGIC attribution has three phases:

1. **Forward training with checkpoints**: Fine-tune the model, saving intermediate checkpoints at each step.
2. **Evaluate**: Compute the evaluation loss and its gradients with respect to the final model parameters.
3. **Backward through training**: Backpropagate the evaluation gradients through the checkpointed training steps using reverse-mode autodiff, accumulating attribution scores for each training example (or token).

The ``Trainer`` class handles all three phases. It uses `torchopt <https://github.com/metaopt/torchopt>`_ for functional (stateless) differentiable optimization.

Usage
-----

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="0" bergson magic runs/magic-ckpts \
       --data.dataset NeelNanda/pile-10k \
       --query.dataset NeelNanda/pile-10k \
       --query.split "train[:8]" \
       --model EleutherAI/pythia-14m

Output files
------------

After a run completes, ``run_cfg.run_path`` contains:

* ``config.yaml`` — the serialized configuration needed to replicate the run.
* ``checkpoints/`` — forward-pass trainer checkpoints, plus
  ``log_history.json`` (the per-step learning rates the schedule produced).
  With the default ``cleanup_ckpts=True`` the backward pass deletes
  checkpoints as it consumes them, but multi-query runs keep them for re-use.

* ``scores/`` — a score directory, the same self-describing format the
  scoring pipeline writes. ``info.json`` records ``attribute_tokens`` and
  ``num_scores``, so consumers never infer the layout from the shape.
  Read it with :func:`bergson.data.load_scores_loss_signed`, which
  returns ``(scores, multi_query)``:

  * Per-example: ``(num_train_docs, 1)``, indexed directly by ``doc_id``.
  * Per-token: ``(num_chunks, seq_len)``, indexed by ``(chunk_idx,
    token_idx)`` in the *post-shuffle* order used during training.
  * Per-query (``query_method: none``) adds a trailing query axis, so
    per-token per-query scores are ``(num_chunks, seq_len,
    num_query_docs)``.

  Pad rows appended to make the dataset divisible by ``batch_size`` are
  trimmed before saving. Per-token scores are stored ragged — a row holds
  ``length - 1`` values, the positions ``weighted_causal_lm_ce`` can reach
  — and are unpacked back into the dense grid on load.

* ``per_query/q{i}.pt`` — per-query runs only. The score tensor for query
  document ``i``, written as soon as that query's backward finishes so an
  interrupted run resumes without redoing completed queries. The trailing
  query axis in ``scores/`` is these tensors stacked.

* ``scores/doc_ids.npy`` — written for every per-token run, shape
  ``(num_chunks, seq_len)`` matching the loaded scores. Each
  entry is the original (pre-shuffle) document id for that token position,
  so the scores can be aggregated over chunks:

  .. code-block:: python

     from bergson.data import load_scores_loss_signed

     scores, _ = load_scores_loss_signed("runs/magic/scores")
     doc_ids = torch.from_numpy(np.load("runs/magic/scores/doc_ids.npy"))
     num_docs = int(doc_ids.max()) + 1

     # Trailing axis is the query axis on per-query runs, absent otherwise;
     # reshaping to it keeps both cases on one path.
     flat = scores.reshape(doc_ids.numel(), -1)
     per_doc = torch.zeros(num_docs, flat.shape[1], dtype=flat.dtype)
     per_doc.scatter_add_(0, doc_ids.flatten()[:, None].expand_as(flat), flat)

  ``per_doc`` comes back as ``(num_docs, num_query_docs)``, or
  ``(num_docs, 1)`` for a single-query run.

  When ``data.chunk_length > 0`` the ``doc_ids`` column comes from
  ``tokenize_and_chunk`` and chunks may pack multiple docs or split one
  across chunks. When ``chunk_length`` is 0, each row is one document
  and ``doc_ids`` is broadcast from the row's pre-shuffle index; tokens
  past the row's actual length carry zero MAGIC score and contribute
  nothing to the scatter-add.

Models and optimizer state
^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``optimizer.pt`` — second moments of the trained optimizer state, when
  ``save_optimizer_state`` is not ``"none"``.
* ``retrained/base/`` and ``retrained/subset_{i}/`` — the fully-trained
  model and each leave-subset-out retrain, with tokenizer, when
  ``save_models=True``. ``retrained/base`` is the query baseline
  ``evaluate_retrained`` reads; the ``subset_{i}`` retrains only exist for
  a run that actually validates (``bergson validate``, or ``bergson magic
  --skip_validation False``). (Commands outside the leave-k-out family —
  plain ``bergson train`` — write the trained model to ``model/`` instead.)

Score trajectory
----------------

A MAGIC run processes the shuffled training documents in batches of
``batch_size`` — batch ``s`` is optimizer step ``s`` — so the saved scores group
back into per-step buckets. ``bergson score_trajectory`` plots the per-step
median ``log10|score|`` (the score *level*) against training step:

.. code-block:: bash

   bergson score_trajectory runs/my-magic   # -> runs/my-magic/score_vs_step.png

It reads ``<run_path>/scores`` and ``<run_path>/config.yaml`` and writes
``<run_path>/score_vs_step.png``. A smooth, gently-varying band is healthy; a
level that sweeps tens of decades or oscillates ("rings") flags a run whose
attribution may not be trustworthy. It is a direct view of score behaviour and
complements the metasmoothness check (below): the two are related but distinct —
a high metasmoothness score does not guarantee a stable level.

Requires the optional plotting dependency: ``pip install 'bergson[viz]'``.

.. figure:: _static/score_trajectory_healthy.png
   :width: 100%
   :alt: Per-step attribution-score level for a healthy deep-ignorance MCQA run.

   A healthy run (deep-ignorance strong-filter on MedMCQA — the ``magic``
   example from the pipeline). The level holds a tight band near -6.5 dex for the
   whole run, dipping only in the final few hundred steps.

.. figure:: _static/score_trajectory_pythia.png
   :width: 100%
   :alt: Per-step attribution-score level for a pathological pythia-160m run.

   A pathological run (pythia-160m): the level sweeps **~37 orders of magnitude**,
   and the first ~1550 steps produce no score at all (red band) — so a raw score
   at step 500 is not comparable to one at step 6000.

.. figure:: _static/score_trajectory_r32.png
   :width: 100%
   :alt: Per-step attribution-score level for a high-metasmoothness deep-ignorance run.

   High metasmoothness does not guarantee a stable score level. This
   deep-ignorance r32 run scores **0.99** on the ``bergson metasmoothness`` test
   — about as metasmooth as a run gets — yet its level still drifts ~7 decades
   and rings periodically (the regular dips). So check the trajectory even when
   the metasmoothness score looks good.

Smoothness
----------

MAGIC is valid when the model training function you differentiate through is smooth with respect to the data weightings (metasmooth). There a few heuristics known to encourage smoothness:

* Use the Muon optimizer
* Increase batch size
* Scale model outputs down
* Clip gradients
* Pre-activation batch norm
* QK norm
* Tune weight decay

Many of these methods boil down to "Identify and manage spikes in your training loss." You can measure your smoothness with ``bergson metasmoothness``.

Core components
^^^^^^^^^^^^^^^

**Trainer**: Functional trainer that supports forward training with checkpoints and backward-through-training.

.. code-block:: python

   from bergson.magic import BackwardState, DataStream, Trainer, TrainerState
   import torchopt

   # Initialize
   opt = torchopt.adam(lr=1e-4)
   trainer, state = Trainer.initialize(model, opt)

   # Forward training with checkpoints
   stream = DataStream(dataset, batch_size=4, device="cuda")
   state = trainer.train(state, stream, save_dir="checkpoints/")

   # Compute eval gradients, then backward through training
   bwd_state = trainer.backward("checkpoints/", stream, bwd_state, state)
   scores = bwd_state.weight_grads  # attribution scores

**DataStream**: Wraps a dataset with differentiable per-example (or per-token) weights that receive gradients during the backward pass.

.. code-block:: python

   # Per-example attribution
   stream = DataStream(dataset, batch_size=4, device="cuda")

   # Per-token attribution
   stream = DataStream(dataset, batch_size=4, device="cuda", weight_shape=(len(dataset), max_length))

**DTensor patch**: For multi-GPU runs with FSDP, apply the DTensor patch before any distributed operations:

.. code-block:: python

   from bergson.magic.dtensor_patch import apply_dtensor_patch
   apply_dtensor_patch()

   # Your MAGIC worker call here

Per-token vs per-example attribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``DataStream`` creates a 1D weight tensor ``[n_examples]`` for per-example attribution. By passing a 2D tensor ``[n_examples, max_length]`` as the ``weight_shape`` parameter, each token receives its own attribution score. The ``weighted_causal_lm_ce`` loss function supports both shapes.

To use per-token attribution, set ``model.loss_function = weighted_causal_lm_ce`` so the model uses the weighted loss during training.

.. code-block:: python

   from bergson.utils.math import weighted_causal_lm_ce
   model.loss_function = weighted_causal_lm_ce

Key implementation details
--------------------------

- **Functional optimization**: ``torchopt.adam`` (or similar) provides a pure-function optimizer whose state is a pytree of tensors. This allows ``torch.autograd.grad`` to differentiate through optimizer updates.
- **Checkpoint strategy**: By default, checkpoints are saved at ``sqrt(N)`` intervals, giving ``O(sqrt(N))`` memory and ``O(N)`` recomputation cost. ``save_mode`` also supports ``all`` (``O(N)`` space) and the original MAGIC paper's ``log`` (``O(log N)`` space, ``O(N log N)`` time).
- **FSDP compatibility**: The DTensor runtime patch adds a ``NestedRedistribute`` autograd function that makes the FSDP all-gather/reduce-scatter differentiable through second-order backward passes.
- **Loss weighting**: ``weighted_causal_lm_ce`` multiplies per-token cross-entropy by the DataStream weights before averaging. During backward-through-training, autograd accumulates gradients into these weights, yielding the attribution scores.
