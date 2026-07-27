Gradient Collection
===================

Most gradient collection for attribution happens on a single model checkpoint. For this use ``bergson build``:

.. code-block:: bash

   bergson build <output_path> --model <model_name> --dataset <dataset_name>

This will create a directory at ``<output_path>`` containing the gradients for each training sample in the specified dataset. The ``--model`` and ``--dataset`` arguments should be compatible with the Hugging Face ``transformers`` library. ``--dataset`` accepts a Hugging Face Hub dataset ID, a local ``.csv`` or ``.json``/``.jsonl`` file, or a directory produced by ``Dataset.save_to_disk``. By default it assumes that the dataset has a ``text`` column, but you can specify other columns using ``--prompt_column`` and optionally ``--completion_column``. The ``--help`` flag will show you all available options.

You can also use the library programmatically to build an index. The ``collect_gradients`` function is just a bit lower level than the CLI tool, and allows you to specify the model and dataset directly as arguments. The result is a HuggingFace dataset which contains a handful of new columns, including ``gradients``, which contains the gradients for each training sample. You can then use this dataset to compute attributions.

At the lowest level of abstraction, the ``GradientCollector`` context manager allows you to efficiently collect gradients for *each individual example* in a batch during a backward pass, simultaneously randomly projecting the gradients to a lower-dimensional space to save memory. If you use Adafactor normalization we will do this in a very compute-efficient way which avoids computing the full gradient for each example before projecting it to the lower dimension. There are three main ways to consume the collected gradients:

1. With a ``builder``, which streams each batch of per-example gradients to an on-disk index. This is what ``bergson build`` uses.
2. With a ``scorer``, which scores each per-example gradient against precomputed query gradients on the fly and discards it. This is what ``bergson score`` uses.
3. With ``skip_index=True``, which accumulates gradients in the collector's ``mod_grads`` dictionary (keyed by module name) for direct inspection. This is the simplest and most flexible approach but is more memory-intensive.

To instead consume each module's gradients mid-backward — e.g. for per-example summary statistics, without holding a full batch of gradients — subclass ``GradientCollector`` and override ``backward_hook``, then run your forward/backward inside the collector's context manager:

.. code-block:: python

   @dataclass(kw_only=True)
   class SummaryStatsCollector(GradientCollector):
       stats: dict = field(default_factory=dict)

       @HookCollectorBase.split_attention_heads  # keep attention_cfgs working
       def backward_hook(self, module, g):
           P = self._compute_gradient(module, g)  # normalized + projected, [N, ...]
           self.stats.setdefault(module._name, []).append(P.flatten(1).norm(dim=1).cpu())

Score a Dataset
---------------

You can score a dataset against an existing query index that is held in memory without saving its gradients to disk. Score each query index item individually, or aggregate the query index items into one using ``--aggregation mean`` or ``--aggregation sum``:

.. code-block:: bash

   bergson score <output_path> --model <model_name> --dataset <dataset_name> --query_path <existing_index_path> --score individual --aggregation mean

You can also aggregate your query dataset into a single mean or sum gradient as it's built:

.. code-block:: bash

   bergson build <output_path> --model <model_name> --dataset <dataset_name> --aggregation mean --unit_normalize --hessian_path <path_to_hessian>

Query an On-Disk Gradient Index
-------------------------------

We provide a query Attributor which supports unit normalized gradients and KNN search out of the box. Access it via CLI with

.. code-block:: bash

   bergson query --index <index_path> --model <model_name> --unit_norm

or programmatically with

.. code-block:: python

   from bergson import Attributor, FaissConfig

   attr = Attributor(args.index, device="cuda")

   ...
   query_tokens = tokenizer(query, return_tensors="pt").to("cuda:0")["input_ids"]

   # Query the index
   with attr.trace(model.base_model, 5) as result:
       model(query_tokens, labels=query_tokens).loss.backward()
       model.zero_grad()

To efficiently query on-disk indexes, perform ANN searches, and explore many other scalability features add a FAISS config:

.. code-block:: python

   attr = Attributor(args.index, device="cuda", faiss_cfg=FaissConfig("IVF1,SQfp16", mmap_index=True))

   with attr.trace(model.base_model, 5) as result:
       model(query_tokens, labels=query_tokens).loss.backward()
       model.zero_grad()

Collect Raw Training Gradients
------------------------------

For experiments using raw training gradients, use the HF Trainer callback (``GradientCollectorCallback``). Gradient collection during training is supported via an integration with HuggingFace's Trainer and SFTTrainer classes. Training gradients are saved in the original order corresponding to their dataset items, and when the ``track_order`` flag is set the training steps associated with each training item are separately saved.

.. code-block:: python

   from bergson.huggingface import GradientCollectorCallback, prepare_for_gradient_collection

   callback = GradientCollectorCallback(
       path="runs/example",
       track_order=True,
   )
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=dataset,
       eval_dataset=dataset,
       callbacks=[callback],
   )
   trainer = prepare_for_gradient_collection(trainer)
   trainer.train()

Collect Individual Attention Head Gradients
-------------------------------------------

By default Bergson collects gradients for named parameter matrices, but per-attention head gradients may be collected by configuring an ``AttentionConfig`` for each module of interest.

.. code-block:: python

   from bergson import AttentionConfig, IndexConfig, collect_gradients
   from transformers import AutoModelForCausalLM

   model = AutoModelForCausalLM.from_pretrained("RonenEldan/TinyStories-1M", trust_remote_code=True, use_safetensors=True)

   collect_gradients(
       model=model,
       data=data,
       processor=processor,
       cfg=IndexConfig(run_path="runs/split_attention"),
       attention_cfgs={
           # Head configuration for the TinyStories-1M transformer
           "h.0.attn.attention.out_proj": AttentionConfig(num_heads=16, head_size=4, head_dim=2),
       },
   )

Collect GRPO Loss Gradients
---------------------------

Where a reward signal is available we compute gradients using a weighted advantage estimate based on Dr. GRPO:

.. code-block:: bash

   bergson build <output_path> --model <model_name> --dataset <dataset_name> --reward_column <reward_column_name>
