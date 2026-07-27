Bergson Documentation
=====================

Bergson is a library for tracing the memory of deep neural nets with gradient-based data attribution techniques.

We provide options for analyzing models and datasets at any scale or level of granularity:

* Compressed or uncompressed gradients.
* Store gradients on-disk or process them in memory.
* Accumulate queries following `LESS <https://arxiv.org/pdf/2402.04333>`_ and other strategies.
* Query small gradient datasets on-GPU, and large ones using a sharded FAISS index.
* Collect gradients during or after training.
* Parallelize Bergson operations across multiple GPUs or nodes.
* Load gradients with or without their module-wise structure.
* Split attention module gradients by head.

.. TODO: Remove above, What data attribution is, cost/fidelity tradeoffs between methods,
   and a "what do you have?" decision guide:

   * one trained checkpoint → influence functions / TrackStar
   * several checkpoints from a run → SOURCE
   * ability to (re)run training → MAGIC

Installation
------------

.. code-block:: bash

   pip install bergson

Quick Start
-----------

Build an index of gradients:

.. code-block:: bash

   bergson build runs/quickstart --model EleutherAI/pythia-14m --dataset NeelNanda/pile-10k --truncation

Load the gradients:

.. code-block:: python

   from pathlib import Path
   from bergson import load_gradients

   gradients = load_gradients(Path("runs/quickstart"))

.. toctree::
   :maxdepth: 2
   :caption: Methods

   influence-functions
   trackstar
   source
   magic

.. toctree::
   :maxdepth: 2
   :caption: Pipeline & Tools

   pipeline
   gradient-collection
   preprocessing
   training
   evaluation
   numerical-stability

.. toctree::
   :maxdepth: 2
   :caption: Reference

   cli
   api
   benchmarks/index
   limitations

.. toctree::
   :maxdepth: 2
   :caption: Experiments

   experiments

Content Index
-------------

* :ref:`genindex`

If you have suggestions, questions, or would like to collaborate, please email lucia@eleuther.ai or drop us a line in the #data-attribution channel of the EleutherAI Discord!
