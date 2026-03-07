MAGIC Attribution
=================

`MAGIC <https://arxiv.org/abs/2504.16430>`_ (Model-Agnostic Generation-time Influence via Checkpointing) attributes evaluation loss to individual training examples by backpropagating through the entire training process. Unlike influence functions which use a local approximation, MAGIC computes exact counterfactual attribution by differentiating through checkpointed training steps.

We provide a `Trainer` class that takes differentiable training steps and handles all three phases of MAGIC attribution. We support FSDP training using the `bergson.magic_patch` runtime patch, which makes PyTorch's DTensor redistribution twice-differentiable (`pytorch/pytorch#160509 <https://github.com/pytorch/pytorch/pull/160509>`_). The patch is applied in memory, so no torch source files are modified.

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

Core components
^^^^^^^^^^^^^^^

**Trainer**: Functional trainer that supports forward training with checkpoints and backward-through-training.

.. code-block:: python

   from bergson.trainer import Trainer, DataStream, BackwardState, TrainerState
   import torchopt

   # Initialize
   opt = torchopt.adam(lr=1e-4)
   trainer, state = Trainer.initialize(model, opt)

   # Forward training with checkpoints
   stream = DataStream(dataset, tokenizer, batch_size=4, num_batches=250, device="cuda")
   state = trainer.train(state, stream, save_dir="checkpoints/")

   # Compute eval gradients, then backward through training
   bwd_state = trainer.backward("checkpoints/", stream, bwd_state)
   scores = bwd_state.weight_grads  # attribution scores

**DataStream**: Wraps a dataset with differentiable per-example weights that receive gradients during the backward pass.

.. code-block:: python

   stream = DataStream(dataset, tokenizer, batch_size=4, num_batches=250, device="cuda")

**DTensor patch**: For multi-GPU runs with FSDP, apply the DTensor patch before any distributed operations:

.. code-block:: python

   from bergson.magic_patch import apply_dtensor_patch
   apply_dtensor_patch()

Key implementation details
--------------------------

- **Functional optimization**: ``torchopt.adam`` (or similar) provides a pure-function optimizer whose state is a pytree of tensors. This allows ``torch.autograd.grad`` to differentiate through optimizer updates.
- **Checkpoint strategy**: By default, checkpoints are saved at ``sqrt(N)`` intervals, giving ``O(sqrt(N))`` memory and ``O(N * sqrt(N))`` recomputation cost.
- **FSDP compatibility**: The DTensor runtime patch adds a ``NestedRedistribute`` autograd function that makes the FSDP all-gather/reduce-scatter differentiable through second-order backward passes.
- **Loss weighting**: ``weighted_causal_lm_ce`` multiplies per-token cross-entropy by the DataStream weights before averaging. During backward-through-training, autograd accumulates gradients into these weights, yielding the attribution scores.
