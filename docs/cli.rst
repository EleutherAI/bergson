Command Line Interface
======================

Bergson provides a command-line interface with four main commands: ``build``, ``query``, ``reduce``, and ``score``.

``build`` and ``query`` are designed for working with compressed gradients stored on disk and queried
multiple times.

``reduce`` and ``score`` are designed for working with uncompressed gradients primarily on GPUs, with
a single predetermined query set. Use ``reduce`` to accumulate a dataset into a single query gradient (mean or sum),
and ``score`` to map over an arbitrarily large dataset, computing the gradient of each item and scoring it against
precomputed query gradients.

.. code-block:: bash

   bergson {build,query,reduce,score} [OPTIONS]

.. autoclass:: bergson.__main__.Build
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson build runs/my-index \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \

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
       --aggregation mean \
       --unit_normalize
       --projection_dim 0

.. autoclass:: bergson.__main__.Score
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson score \
        runs/my-scores \
        --query_path /runs/my-index \
        --dataset EleutherAI/SmolLM2-135M-10B
        --projection_dim 0
