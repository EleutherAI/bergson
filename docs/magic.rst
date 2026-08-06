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

* ``scores/doc_ids.npy`` — written for every per-token run, shape
  ``(num_chunks, seq_len)`` matching the loaded scores row-for-row. Each
  entry is the original (pre-shuffle) document id for that token position.
  Downstream aggregation is one line:

  .. code-block:: python

     from bergson.data import load_scores_loss_signed

     scores, _ = load_scores_loss_signed("runs/magic/scores")
     doc_ids = torch.from_numpy(np.load("runs/magic/scores/doc_ids.npy"))
     num_docs = int(doc_ids.max()) + 1
     per_doc = torch.zeros(num_docs, dtype=scores.dtype)
     per_doc.scatter_add_(0, doc_ids.flatten(), scores.flatten())

  When ``data.chunk_length > 0`` the ``doc_ids`` column comes from
  ``tokenize_and_chunk`` and chunks may pack multiple docs or split one
  across chunks. When ``chunk_length`` is 0, each row is one document
  and ``doc_ids`` is broadcast from the row's pre-shuffle index; tokens
  past the row's actual length carry zero MAGIC score and contribute
  nothing to the scatter-add.

* ``config.yaml`` — serialized ``MagicConfig`` used for the run.
* ``validation.csv`` — leave-subset-out validation results (if validation
  was run).

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
