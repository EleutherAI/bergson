Bergson Documentation
=====================

Bergson is a library for gradient-based data attribution of transformers. Data attribution methods estimate the effect on a behavior of interest of removing data points from a model's training corpus, and enable data filtering, re-weighting, and interpretability. 

Naively computing leave-one-out effects for a corpus of N items requires 2**N retraining runs. Our most costly and powerful method, MAGIC, uses compute equivalent to 3-5 training runs to produce per-token or per-sequence scores that correlate with the effects of leave-k-out retraining at ρ>0.9 in well-behaved settings. More efficient methods like EK-FAC and TrackStar use compute equivalent to ~1-2 training runs (with more modest VRAM usage), but correlate less with leave-k-out retraining (ρ\~=0.1 to 0.5).

We provide options for analyzing models and datasets at any scale or level of granularity:

* Compressed or uncompressed gradients.
* Per-token or per-sequence attribution.
* On-disk gradient stores or on-the-fly queries.
* HuggingFace Transformers models and Datasets, including on-disk datasets in a variety of formats.
* Query aggregation following `LESS <https://arxiv.org/pdf/2402.04333>`_ and other strategies.
* On-GPU gradient store queries, or sharded FAISS indexes for fast queries at scale.
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

Quickstart
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
   reproducibility

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
