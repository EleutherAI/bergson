Data Preprocessing
==================

Every command that consumes a dataset — ``build``, ``score``, ``hessian``, ``magic``, ``validate``, and the SOURCE pipeline — shares one preprocessing pipeline that loads the dataset, tokenizes it if needed, and prepares the columns the gradient collectors read.

Loading
-------

``--dataset`` accepts a Hugging Face Hub dataset ID, a local ``.csv`` or ``.json``/``.jsonl`` file, or a directory produced by ``Dataset.save_to_disk``. Use ``--split`` and ``--subset`` to select within a Hub dataset, and ``--data_kwargs arg1=val1,arg2=val2`` to pass extra arguments to ``load_dataset``.

Text datasets
-------------

Column options determine how rows are tokenized and which token positions receive loss:

* By default each row's ``text`` column is tokenized for vanilla next-token prediction, with loss on every position. ``--prompt_column`` selects a different column.
* ``--completion_column`` treats each row as a prompt–completion pair. The pair is rendered with the tokenizer's chat template as a user/assistant exchange, and loss covers the completion tokens.
* ``--conversation_column`` selects a column of chat conversations. Each is rendered with the chat template, and loss covers the assistant turns.
* ``--format_template`` points to a YAML file containing a Jinja2 template (``doc_to_text``, optionally ``doc_to_target`` and ``doc_to_choice``) that formats rows into text before tokenization. An MCQA template ships at ``bergson/templates/mcqa.yaml``.

Long documents raise a warning when they exceed the model's context length or ``--token_batch_size``. Pass ``--truncation`` to truncate them to the smaller of the two.

Chunking
--------

``--chunk_length N`` (with ``N > 0``) concatenates all documents, separated by EOS, and slices the token stream into fixed-length chunks of ``N`` tokens — the standard setup for pretraining-style attribution. Each chunk carries a ``doc_ids`` column mapping every token back to its source document, so per-token scores can be aggregated per document. ``chunk_length`` operates on the raw text column and cannot be combined with ``--truncation`` or ``--format_template``.

Pre-tokenized datasets
----------------------

A dataset that already has an ``input_ids`` column is used as-is; the tokenizer is skipped. A ``length`` column is derived from ``input_ids`` when absent. An optional ``labels`` column restricts the loss to specific positions, e.g. assistant turns, with ``-100`` marking unsupervised positions. ``--chunk_length`` applies to raw text and should stay at its default of 0.

Rewards
-------

``--reward_column`` selects a per-example scalar reward. Gradients are then computed with the Dr. GRPO policy gradient loss (https://arxiv.org/abs/2503.20783), using advantages estimated from the rewards. Rows with NaN rewards raise an error; pass ``--skip_nan_rewards`` to filter them out instead.

Output columns
--------------

After preprocessing, each row carries ``input_ids``, ``length``, and optionally ``labels``. Other columns are dropped by default; pass ``--drop_columns False`` to keep them.
