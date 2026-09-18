Gradient Collection
===================

Most gradient collection for attribution happens on a single model checkpoint. For this use ``bergson build``:

.. code-block:: bash

   bergson build <output_path> --model <model_name> --dataset <dataset_name>

This will create a directory at ``<output_path>`` containing the gradients for each training sample in the specified dataset. The ``--model`` and ``--dataset`` arguments should be compatible with the Hugging Face ``transformers`` library. ``--dataset`` accepts a Hugging Face Hub dataset ID, a local ``.csv`` or ``.json``/``.jsonl`` file, or a directory produced by ``Dataset.save_to_disk``. By default it assumes that the dataset has a ``text`` column, but you can specify other columns using ``--prompt_column`` and optionally ``--completion_column``. See :doc:`data-preprocessing` for the full set of column options. The ``--help`` flag will show you all available options.

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

Track Mixture-of-Experts Models
-------------------------------

Bergson tracks ``nn.Linear``, HF ``Conv1D`` and ``nn.Conv{1,2,3}d`` modules. In ``transformers`` 5.x an MoE layer holds every expert in a single 3D ``nn.Parameter`` rather than an ``nn.Linear`` each, so none of its experts is a module bergson recognizes and gradient collection silently covers only attention and ``lm_head`` — 5.8% of gpt-oss-20b's parameters. Older MoE layouts with one ``nn.Linear`` per expert are tracked as they are and need nothing here.

Name the fused layers with ``--moe_experts`` and each expert projection becomes its own module:

.. code-block:: bash

   bergson build <output_path> --model openai/gpt-oss-20b --dataset <dataset_name> \
       --moe_experts "model.layers.*.mlp.experts"

The value is a comma-separated list of globs over ``model.named_modules()``, matched against the module that owns the ``gate_up_proj``/``down_proj`` parameters; pointing it at anything else is an error rather than a silent no-op. The pattern is matched at load time, against the whole model. Gradient collection then runs on the base model, so the index names the new modules ``layers.0.mlp.experts.expert_3.gate_up_proj`` -- without the ``model.`` prefix, exactly as it names every other module -- and ``filter_modules`` and ``AttentionConfig`` globs reach them like any other module. See ``examples/moe_experts.yaml`` for a runnable pipeline.

The layer's fused matmul over all experts is replaced by a loop that runs each expert over the tokens routed to it, which is what lets the hooks see one expert's activations at a time. That loop is the cost of the option: it gives up the grouped-matmul kernel, so collection is several times slower. Nothing is enabled unless you pass the flag.

``bergson build``, ``score``, ``ekfac`` and the rest all load the model through the same path, so the flag reaches every one of them. ``GradientCollectorCallback`` is the exception, since it is handed a model you loaded yourself: call ``bergson.expand_moe(model, "model.layers.*.mlp.experts")`` before you build the Trainer.

Two limits follow from how routing works. ``attribute_tokens`` is rejected, because an expert's gradient rows are the tokens routed to it and those do not line up one-to-one with token positions. The router itself is still untracked; it is a 2D parameter rather than a module, and on gpt-oss-20b it holds about 92K of the model's 21B parameters.
