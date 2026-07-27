Reproducibility
===============

Every run writes a self-describing ``config.yaml`` into its output directory, enabling the command(s) to be reproduced:

.. code-block:: bash

   bergson <path_to_config.yaml>

Each config contains ``steps:`` a list of ``- command: {...}`` entries plus a ``metadata:`` block (bergson version, timestamp, git SHA). Multi-step pipelines may optionally specify a ``run_path``. See ``examples/pipelines/hessian_then_build.yaml`` for an example of a multi-step run.

Note that replaying a config whose ``run_path`` already exists raises a ``FileExistsError``. Set ``overwrite: true`` or change the ``run_path`` if you are in the original working directory.

.. code-block:: yaml

   steps:
     - build: {index_cfg: {run_path: runs/idx}, preprocess_cfg: {}}
   metadata:
     bergson_version: 0.9.1
     created: 2026-06-03T14:22:10Z
     git_sha: abc1234
