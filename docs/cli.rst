Command Line Interface
======================

Bergson's post-hoc attribution exposes three building block commands — ``build``, ``reduce``, and
``score`` — plus supporting commands for querying, preconditioner computation, and
end-to-end pipelines.

``build`` and ``query`` are designed for working with compressed gradients stored on disk and queried
multiple times.

``reduce`` and ``score`` are designed for working with uncompressed gradients primarily on GPUs, with
a single predetermined query set. Use ``reduce`` to accumulate a dataset into a single query gradient (mean or sum),
and ``score`` to map over an arbitrarily large dataset, computing the gradient of each item and scoring it against
precomputed query gradients.

``hessian`` and ``preconditioners`` compute preconditioner statistics independently of gradient
collection. ``trackstar`` runs preconditioners, build, and score as a single pipeline
(see :doc:`trackstar`).

.. code-block:: bash

   bergson {build,query,preconditioners,reduce,score,hessian,trackstar} [OPTIONS]

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
       --skip_preconditioners

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

.. autoclass:: bergson.__main__.Preconditioners
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson preconditioners runs/my-processor \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \
       --truncation \
       --normalizer adam

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
