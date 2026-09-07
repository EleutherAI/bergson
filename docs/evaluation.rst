Evaluation
==========

Bergson supports validating model attributions using the linear datamodeling score (LDS), a widely used metric introduced in `TRAK: Attributing Model Behavior at Scale <https://arxiv.org/abs/2303.14186>`_, via ``bergson validate``. This method compares summed attribution scores to ground-truth results over hundreds of re-training runs.

To evaluate attributions of large models that cannot be re-trained many times, we provide a synthetic factual dataset and its generator, and a proxy evaluation of how well attribution can be used to retrieve logically entailing data via ``bergson recall``. The evaluation reports MRR and Recall@k.


.. TODO: Evaluating attribution scores — ``bergson validate`` (leave-k-out
   retraining / linear datamodeling score), ``bergson recall`` (synthetic
   factual recall, MRR / Recall@k), and ``bergson metasmoothness``.

Validation experiments
----------------------

Filtering
~~~~~~~~~

Remove a ranked fraction of the data and compare its query loss change with
random removals of the same size. The fraction defaults to 0.01; random controls
default to three retrains.

.. code-block:: bash

   bergson validate runs/filter --scores runs/magic/scores \
       --method filter --direction proponents --fraction 0.05 --count 3
   bergson validate runs/filter-only --scores runs/magic/scores \
       --method filter --fraction 0.05 --controls none
   bergson validate runs/filter-bank --scores runs/magic/scores \
       --method filter --fraction 0.05 --controls bank --paths runs/bank --count 2

``--direction detractors`` removes the opposite end of the score ranking.
Bank controls evaluate the first ``count`` entries. Multiple ``--paths`` average
corresponding bank entries.

Equivalent YAML:

.. code-block:: yaml

   steps:
     - validate:
         run_path: runs/filter
         scores: runs/magic/scores
         method:
           kind: filter
           direction: proponents
           fraction: 0.05
           controls:
             kind: retrain
             count: 3

Use ``controls: {kind: none}`` to omit the comparison, or
``controls: {kind: bank, paths: [runs/bank], count: 2}`` to reuse retrains.

LDS
~~~

LDS defaults to a random partition into 100 subsets. The following command
samples 100 independent removals, each containing five percent of the eligible
data:

.. code-block:: bash

   bergson validate runs/lds --scores runs/magic/scores \
       --method lds --fraction 0.05 --count 100
   bergson validate runs/lds-bank --scores runs/magic/scores \
       --method lds --subsets bank --paths runs/bank

.. code-block:: yaml

   steps:
     - validate:
         run_path: runs/lds
         scores: runs/magic/scores
         method:
           kind: lds
           subsets: retrain
           fraction: 0.05
           count: 100
           start: 0
           stop: null

``start`` and ``stop`` select a range for sharded retraining or evaluation.
To evaluate using an existing bank of retrained models, set ``subsets: bank``
and ``paths: [runs/bank]`` in the LDS method config.

Weight steps
~~~~~~~~~~~~

``--method weight_step --lrs 0.1 0.2``, or
``method: {kind: weight_step, lrs: [0.1, 0.2]}`` in YAML.
