Command Line Interface
======================

Bergson exposes all functionality through subcommands of the ``bergson`` CLI:

.. code-block:: bash

   bergson {build,query,score,hessian,mix,trackstar,ekfac,approxunrolling,magic,train,metasmoothness,validate,recall,test_model_configuration} [OPTIONS]

The commands fall into four groups:

**Building blocks** — ``build``, ``query``, ``score``, ``hessian``, and ``mix``. ``build`` and ``query`` are designed for working with compressed gradients stored on disk and queried multiple times. ``build`` and ``score`` are designed for working with both compressed and uncompressed gradients primarily on GPUs, with a single predetermined query set: use ``build --aggregation mean`` (or ``sum``) to accumulate a dataset into a single query gradient, and ``score`` to map over an arbitrarily large dataset, computing the gradient of each item and scoring it against precomputed query gradients. ``hessian`` computes Hessian statistics (KFAC, TKFAC, Shampoo, or gradient
autocorrelation) independently of per-example gradient collection, and ``mix`` combines two autocorrelation hessians into one. See :doc:`building_blocks`.

**Method pipelines** — ``trackstar``, ``ekfac``, and ``approxunrolling`` orchestrate
building blocks into end-to-end attribution recipes (see :doc:`trackstar`, :doc:`influence-functions`, and :doc:`source`). ``magic`` attributes by backpropagating through training (see :doc:`magic`).

**Training & evaluation** — ``train`` trains a model with the Bergson trainer; ``metasmoothness``, ``validate``, and ``recall`` evaluate training runs and attribution scores (see :doc:`training` and :doc:`evaluation`).

**Diagnostics** — ``test_model_configuration`` checks a model for batching-dependent
gradient nondeterminism (see :doc:`numerical-stability`).

Building Blocks
---------------

.. autoclass:: bergson.__main__.Build
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson build runs/my-index \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation

**Example** — aggregating a dataset into a single query gradient:

.. code-block:: bash

   bergson build runs/my-query \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --aggregation mean \
       --unit_normalize \
       --projection_dim 0

.. autoclass:: bergson.__main__.Query
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson query \
       --index runs/my-index

.. autoclass:: bergson.__main__.Score
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson score runs/my-scores \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --query_path runs/my-index \
       --projection_dim 16

.. autoclass:: bergson.__main__.Hessian
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson hessian runs/my-hessian \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --method kfac

.. autoclass:: bergson.__main__.Mix
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson mix \
       --query_path runs/query-hessian \
       --index_path runs/index-hessian \
       --output_path runs/mixed-hessian

Method Pipelines
----------------

.. autoclass:: bergson.__main__.Trackstar
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson trackstar runs/my-trackstar \
       --model EleutherAI/pythia-14m \
       --data.dataset NeelNanda/pile-10k \
       --data.truncation \
       --query.dataset NeelNanda/pile-10k \
       --query.truncation \
       --projection_dim 16

.. autoclass:: bergson.__main__.Ekfac
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson ekfac runs/my-ekfac \
       --model EleutherAI/pythia-14m \
       --data.dataset NeelNanda/pile-10k \
       --data.truncation \
       --query.dataset NeelNanda/pile-10k \
       --query.split "train[:8]" \
       --method kfac \
       --hessian_cfg.ev_correction true

See ``examples/magic/compare/q3_ekfac.yaml`` for a complete pipeline
configuration.

.. autoclass:: bergson.__main__.ApproxUnrolling
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson approxunrolling runs/my-source \
       --model EleutherAI/pythia-14m \
       --data.dataset NeelNanda/pile-10k \
       --data.truncation \
       --method kfac \
       --ev_correction true \
       --checkpoints ckpts/checkpoint-1000 ckpts/checkpoint-2000 \
       --segments 2 \
       --query.dataset NeelNanda/pile-10k \
       --query.split "train[:8]"

See ``examples/pipelines/approx_unrolling_pythia.yaml`` for a complete pipeline
configuration.

.. autoclass:: bergson.__main__.Magic
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson magic runs/my-magic \
       --model EleutherAI/pythia-14m \
       --data.dataset NeelNanda/pile-10k \
       --query.dataset NeelNanda/pile-10k \
       --query.split "train[:8]"

See ``examples/magic/gpt2_wikitext_tiny.yaml`` for a complete run
configuration.

Training & Evaluation
---------------------

.. autoclass:: bergson.__main__.Train
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson train runs/my-train \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --batch_size 32 \
       --num_epochs 1

.. autoclass:: bergson.__main__.Metasmoothness
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson metasmoothness runs/my-metasmoothness \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation

.. autoclass:: bergson.__main__.Score_Trajectory
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson score_trajectory runs/my-magic
   bergson score_trajectory runs/my-magic --window 5

.. autoclass:: bergson.__main__.Validate
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson validate runs/my-validation \
       --model EleutherAI/pythia-14m \
       --scores runs/my-scores \
       --data.dataset NeelNanda/pile-10k \
       --data.truncation \
       --query.dataset NeelNanda/pile-10k \
       --query.split "train[:8]" \
       --num_subsets 10

.. autoclass:: bergson.__main__.Recall
   :members:
   :undoc-members:
   :show-inheritance:

``recall`` evaluates attribution scores by synthetic factual recall: a model is
trained on generated fact *statements* and queried with the matching
*questions*; scores from ``score`` are ranked per question and reported as MRR
and Recall@k against the gold (entailing) statements. Datasets are generated
once per ``(num_people, seed)`` and cached (e.g.
``data/statements_1000p_seed0.hf``). See
``examples/pipelines/recall_synthetic.yaml`` for the full pipeline.

**Example:**

.. code-block:: bash

   bergson recall runs/my-recall \
       --scores runs/my-scores \
       --num_people 1000

Diagnostics
-----------

.. autoclass:: bergson.__main__.Test_Model_Configuration
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson test_model_configuration --model EleutherAI/pythia-14m
