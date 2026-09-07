Evaluation
==========

Bergson supports validating model attributions using the linear datamodeling score (LDS), a widely used metric introduced in `TRAK: Attributing Model Behavior at Scale <https://arxiv.org/abs/2303.14186>`_, via ``bergson validate``. This method compares summed attribution scores to ground-truth results over hundreds of re-training runs.

To evaluate attributions of large models that cannot be re-trained many times, we provide a synthetic factual dataset and its generator, and a proxy evaluation of how well attribution can be used to retrieve logically entailing data via ``bergson recall``. The evaluation reports MRR and Recall@k.


.. TODO: Evaluating attribution scores — ``bergson validate`` (leave-k-out
   retraining / linear datamodeling score), ``bergson recall`` (synthetic
   factual recall, MRR / Recall@k), and ``bergson metasmoothness``.

Validation experiments
----------------------

``validate`` and MAGIC's optional validation use one method-specific config.
Training and query settings remain shared. LDS holds its subset settings;
filtering has a ``controls`` config. CLI help shows the selected method's
arguments.

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
corresponding bank entries. Their removal sets must match across banks, and
their removal sizes and training settings must be comparable with the filtering
experiment. Bank counts larger than the available entries are rejected; YAML
``count: null`` uses all.

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
A random control is a new random removal set. The shared ``seed`` controls
subset selection and training randomness.

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

``fraction: 0`` divides the pool into ``count`` subsets. A positive ``fraction``
draws ``count`` independent random subsets of that size. Subset selection uses
the shared training ``seed``. ``manifest`` can point to an existing
``subsets.json``; otherwise an existing manifest in the run directory is reused.
``start`` and ``stop`` select a range for sharded retraining or evaluation.
To evaluate a bank, set ``subsets: bank`` and ``paths: [runs/bank]`` in the LDS
method config.

Weight steps and migration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Weight-step validation is a separate method:
``--method weight_step --lrs 0.1 0.2``, or
``method: {kind: weight_step, lrs: [0.1, 0.2]}`` in YAML.

Existing flat YAML configs remain readable with a deprecation warning. Migration
preserves their old defaults, including filtering's implicit ``1 / num_subsets``
removal fraction, training-seed-based subset sampling, and use of every bank
entry. Newly saved configs use a ``kind`` tag to identify the selected method.
Mixing legacy flat options with a nested method config is rejected.

CLI invocations and direct Python construction should use the new method configs:
``--num_subsets`` becomes ``--count``, ``--subset_fraction`` becomes
``--fraction``, and ``--retrained_dir`` becomes the selected bank source's
``--paths``. The old ``--method filter-proponents`` becomes
``--method filter --direction proponents``. The legacy global ``controls`` modes
are replaced by the filtering ``--controls retrain/bank/none`` selector.
