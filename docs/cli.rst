Command Line Interface
======================

Bergson's post-hoc attribution exposes three building block commands — ``build``, ``reduce``, and
``score`` — plus supporting commands for querying, hessian computation, and
end-to-end pipelines.

``build`` and ``query`` are designed for working with compressed gradients stored on disk and queried
multiple times.

``reduce`` and ``score`` are designed for working with uncompressed gradients primarily on GPUs, with
a single predetermined query set. Use ``reduce`` to accumulate a dataset into a single query gradient (mean or sum),
and ``score`` to map over an arbitrarily large dataset, computing the gradient of each item and scoring it against
precomputed query gradients.

``hessian`` computes Hessian statistics (KFAC, TKFAC, Shampoo, or gradient
autocorrelation) independently of per-example gradient collection. ``trackstar``
runs hessian fitting, build, and score as a single pipeline (see :doc:`trackstar`).

.. code-block:: bash

   bergson {build,query,reduce,score,hessian,trackstar} [OPTIONS]

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

.. autoclass:: bergson.__main__.Query
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson query \
       --index runs/my-index

.. autoclass:: bergson.__main__.Reduce
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson reduce runs/my-index \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --aggregation mean \
       --unit_normalize \
       --projection_dim 0 \
       --skip_hessians

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

Sharded (data-parallel) runs
----------------------------

``build`` and ``score`` are embarrassingly parallel across the dataset, and
both accept ``--num_shards``/``--shard_id`` to split a run into independent
single-node jobs that share one ``run_path``:

.. code-block:: bash

   # Typically launched as a SLURM job array (sbatch --array=0-63 --requeue);
   # --shard_id is inferred from SLURM_ARRAY_TASK_ID when unset.
   bergson build runs/my-index \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --num_shards 64 --shard_id $SLURM_ARRAY_TASK_ID

Each shard processes one contiguous slice of the dataset, writes into
``run_path/shards/<id>-of-<n>.part``, and atomically renames the directory
into place when it finishes. A crashed shard is rebuilt by re-running the
same command; a finished shard is skipped, so requeued jobs are idempotent.
The first shard to arrive writes a canonical ``run_path/config.yaml`` and
every other shard verifies its configuration against it.

No stitching is needed afterwards: ``load_gradients``, ``load_scores``,
``Attributor``, and friends read the published shards as one concatenated
index. Inspect progress with:

.. code-block:: bash

   bergson status runs/my-index

See ``examples/slurm/data_parallel_score.sh`` for a complete job-array
script. Sharded runs do not support simultaneous Hessian estimation or
gradient aggregation; compute those separately.
