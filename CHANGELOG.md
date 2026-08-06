# CHANGELOG


## v0.25.1 (2026-08-06)

### Bug Fixes

- **magic**: Reject chunked query sets in per-query MAGIC
  ([#418](https://github.com/EleutherAI/bergson/pull/418),
  [`012b54a`](https://github.com/EleutherAI/bergson/commit/012b54adec7e57d98ce4040d87bce20697328229))

Per-query MAGIC scores one column per query *document*, but a chunked query set (`query.chunk_length
  > 0`) has rows that pack several documents and documents that span several rows, so query `i` is
  not row `i`. `compute_per_query_magic_scores` selected row `qi` and sized the stream's weights by
  row, while `DataStream` indexes weights by document id whenever the batch carries `doc_ids`, so a
  run crashed with:

IndexError: index 2 is out of bounds for dimension 0 with size 2 bergson/magic/data_stream.py:132
  self.weights[indices]

Chunking exists to pack a training set efficiently; a query set is small — 50 documents in the
  compare_wikitext runs — so packing it buys nothing, and every shipped config already passes
  query.chunk_length 0. Rather than teach the per-query path to split and repack documents, require
  the query rows to be documents and say so at config time, before a run trains for hours.

Co-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>


## v0.25.0 (2026-08-06)

### Features

- **magic**: Per-token per-query MAGIC, and score-format cleanups
  ([#415](https://github.com/EleutherAI/bergson/pull/415),
  [`b333a20`](https://github.com/EleutherAI/bergson/commit/b333a20b25db4877b947306146211ec797228c15))

* feat(magic): support per-token per-query MAGIC

attribute_tokens=True with query_method="none" produced an unusable score tensor. Per-token weights
  are [rows, seq_len] and the per-query stack used dim=1, giving [rows, num_queries, seq_len] —
  query axis in the middle, which nothing downstream reads. validate_scores takes shape[-1] as the
  query count, so it compared seq_len against the query document count and died naming the wrong
  dimension:

ValueError: scores has 8 query columns but the query dataset has 2 documents

on a run with 2 queries and seq_len 8.

Stack on dim=-1 instead. The query axis then comes last in both modes — [rows, num_queries] per-doc,
  [rows, seq_len, num_queries] per-token — matching the layout Scores.to_grid already produces for
  multi-query token score directories, which load_attribution_scores already flags multi_query.
  validate_scores needed no change: shape[-1] is the query count and reshape(-1, num_queries)
  flattens the leading axes into leave-out units, documents or token positions as appropriate.
  dim=-1 is identical to dim=1 for 1-D inputs, so per-doc per-query scores are unchanged.

Fix the padding trim in the same path, which applied weight_pad_count regardless of rank while the
  main scoring path picks by rank. The two differ once doc_ids are present (pad rows route to one
  synthetic doc id), so a 5-doc dataset at batch_size 4 kept 7 of its 8 padded rows instead of
  trimming to 5, leaving pad rows in the saved scores.

Teach both .pt classifiers about 3-D: scores_are_per_token so a reloaded run sizes its weights
  per-token, and _pt_scores_are_per_query so it is recognised as multi-query. 3-D needs no config
  lookup to disambiguate, unlike 2-D.

The aggregation test is the numerical gate: per-token per-query scores summed over each document's
  tokens reproduce the per-doc per-query run. Both new end-to-end tests fail on the parent commit
  with shape (7, 2, 8) against the expected (5, 8, 2) — wrong axis order and untrimmed padding
  together.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

* refactor(magic): decide score format from the config, not the shape

Score layout was inferred from tensor rank in several places, which is guesswork: a 2-D .pt is
  [docs, seq_len] or [docs, queries] depending only on how the run was configured. #407 established
  the fix for one of those call sites — read the config.yaml that save_run_config writes next to
  scores.pt — but scores_are_per_token was left sniffing shapes, and the parsing lived inline rather
  than beside the other config readers.

Add read_first_step_config to config_io, next to read_config and load_subconfig, and route both
  classifiers through it. The flags say everything the rank could:

query_method attribute_tokens layout none yes [docs, seq_len, queries] none no [docs, queries]
  mean/sum yes [docs, seq_len] mean/sum no [docs]

so a run is per-query iff query_method is none, whatever rank results, and per-token iff
  attribute_tokens is set. Neither needs the shape. Score directories keep reading info.json, and
  load_attribution_scores now takes the query count from the store's num_scores rather than
  re-deriving it from the grid it just built.

Shape survives in exactly one place: a .pt with no config beside it, where nothing else is knowable
  and only rank 3 is unambiguous.

cfg_attributes_tokens reads attribute_tokens with the deprecated per_token as an alias, in one
  place, so a run written with the current field name is no longer missed.

test_load_attribution_scores_pt_per_query asserted that a 2-D tensor whose config said query_method:
  none and per_token: true was single-query. No run produces that pair — attributing tokens per
  query yields rank 3 — so the case was describing an unreachable artifact and pinning the
  shape-derived answer for it. Repointed at the 3-D tensor such a run does produce.

Drop six tests that asserted torch's own view()/reshape() indexing semantics. They called no bergson
  code, so they could not fail unless PyTorch itself changed, and the behaviour they stood in for is
  covered end to end by the per-query aggregation test.

* refactor(score): stop duplicating the token score store format

save_sequence_scores delegates to MemmapSequenceScoreWriter, but its token counterpart reimplemented
  MemmapTokenScoreWriter inline: the memmap creation, offsets.npy, and an info.json payload
  identical field for field. So the on-disk token score format was written from two places that had
  to be kept in step by hand.

That matters more now that scores_are_per_token reads info.json["attribute_tokens"] as
  authoritative: a drift between the two writers stops being a cosmetic inconsistency and becomes a
  misclassification.

Delegating needs the writer to accept what it actually uses. It only ever took a Dataset to call
  compute_num_token_grads on it, while save_token_scores already holds the offsets those counts came
  from, so __init__ now takes num_token_grads and a from_dataset classmethod covers the callers that
  hold a dataset.

Also gives the token writer the overwrite flag its sequence twin already had, which delegation
  needs: save_token_scores wrote with mode="w+" unconditionally, and without overwrite the writer
  would silently reuse a stale scores.bin instead of replacing it.

Net 19 lines out of score_writer.py, and one place left that knows the format.

* refactor(score): drop .npy attribution score support

bergson never writes a .npy score file — every writer emits a score directory — so .npy was an
  ingest-only path for arrays produced outside the scoring pipeline, and nothing in the repo feeds
  one: no config sets scores: to a .npy, and the examples that save scores.npy read it straight back
  with np.load rather than through load_attribution_scores.

Remove the branch from load_attribution_scores, scores_are_per_token and worker's score-path
  dispatch, and with it ArrayScores, which existed only to give a bare array the Scores interface.

It also carried its own rules. A .npy could not have a score_cfg, so it alone skipped the
  higher_is_better negation and had to be supplied in the loss-diff convention already; and it was
  the one input whose multi-query flag came from a raw column count. Score directories record
  num_scores in info.json, so the surviving formats all describe themselves.

The bank-loss-cache tests used .npy as a convenient way to hand a score matrix to
  evaluate_retrained. They now write a score directory via save_sequence_scores, which is what a
  caller would reach for, and the multi_query parametrization still passes both ways.

* refactor(magic): inline the per-token config lookup

cfg_attributes_tokens had one caller left once _pt_scores_are_per_query was inlined, and reaching
  across from magic.cli into validate for a two key dict lookup bought nothing.

Drop the scores_are_per_token docstring with it: the branches say what they read, and the function
  had none before this PR.

---------

Co-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>


## v0.24.6 (2026-08-06)

### Bug Fixes

- **magic**: Write doc_ids.pt on fresh per-token runs
  ([#412](https://github.com/EleutherAI/bergson/pull/412),
  [`80d72ba`](https://github.com/EleutherAI/bergson/commit/80d72bae9168ac9d6cd54e44f43f7e131612921b))

doc_ids.pt lets downstream code turn per-token scores into per-doc scores with one scatter_add,
  without replaying the shuffle seed or re-tokenizing the raw dataset. 25936cf7 added it directly
  after the torch.save(scores) in worker()'s compute path; a later refactor turned that region into
  an if/elif chain over score_path and the save came to rest in the trailing else — the branch that
  only runs when scores are LOADED from an existing .pt. Walking the chain as it stands:

if not score_path and query_method == 'none' doc_ids.pt: no elif not score_path doc_ids.pt: no elif
  isdir(score_path) or endswith('.npy') doc_ids.pt: no else (load existing .pt) doc_ids.pt: yes

So `bergson magic --attribute_tokens true` — the case the file exists for — never wrote it, while
  `bergson validate --scores foo.pt` did. Confirmed end to end: a per-token run produced scores.pt
  of shape (148, 64) and no doc_ids.pt.

Hoist the save out of the branch chain to a single call site keyed on per_token, the flag the run
  already computed to size the weight tensor. Sitting inside the chain is what let the save drift
  away from the path it belonged to; there is now one place to look and no per-branch condition to
  keep in sync.

per_token is the whole condition. doc_ids maps the [docs, seq_len] training weight axes back to
  documents, so it applies to any per-token scores no matter what else the tensor carries — a query
  axis rides alongside without changing what doc_ids means. Scores.to_grid() already produces
  exactly that layout, [docs, seq_len, num_scores], for multi-query token score directories, so
  conditioning on the score tensor's rank (or excluding multi_query) would suppress doc_ids in the
  case that needs it most and would need revisiting the moment per-token multi-query MAGIC lands.

Two incidental corrections fall out: the save is now rank-0 only (every rank previously raced to
  write the same path), and a loaded (n, 1) column vector no longer produces a doc_ids.pt, since
  such scores are per-doc and the mapping is meaningless.

The existing tests could not catch this: _run_magic_cli reimplements worker()'s
  pad/train/backward/trim sequence rather than calling it, so nothing asserted on worker()'s actual
  output directory. The new test calls worker() and checks the files on disk. It fails on the parent
  commit with "wrote scores.pt but no doc_ids.pt" and passes here.

Scores are byte-identical before and after (baseline loss 3.709878112140455 either way); this only
  adds the companion file.

Co-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>


## v0.24.5 (2026-08-06)

### Bug Fixes

- **magic**: Let worker() run on CPU-only machines
  ([#411](https://github.com/EleutherAI/bergson/pull/411),
  [`513954f`](https://github.com/EleutherAI/bergson/commit/513954f250fc62aad26c55800f844f2aee2f6c0f))

worker() called torch.cuda.set_device() unconditionally as its first statement, so any CPU-only
  machine died with

AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'

before a single line of work ran. Every other device lookup in the file already degrades gracefully
  — get_device() returns "cpu" when CUDA is unavailable — so this one call was the sole blocker.

Guarding it makes `bergson magic` run end to end on CPU for small models, which matters for two
  reasons: contributors without a GPU can exercise the CLI, and the MAGIC output-file behaviour
  becomes testable in CI without a GPU runner (the existing tests reimplement worker()'s logic
  rather than calling it, so they cannot catch regressions in what it writes to disk).

Verified by running a full per-token attribution on CPU:

bergson magic <run> --data.dataset NeelNanda/pile-10k \ --data.split "train[:16]"
  --data.chunk_length 64 \ --model EleutherAI/pythia-14m --batch_size 4 \ --attribute_tokens true
  --query_method mean --skip_validation true

which trains, backprops through the trajectory, and saves scores.pt.

Co-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>


## v0.24.4 (2026-08-06)

### Bug Fixes

- **metasmoothness**: Forward grad_accum_steps and max_grad_norm to train
  ([#410](https://github.com/EleutherAI/bergson/pull/410),
  [`c059b63`](https://github.com/EleutherAI/bergson/commit/c059b6377cf56541eca54d7c0bcc5b24758c02b8))

The worker's trainer.train call dropped both, so each rank ran its full batch in one graph (OOM at
  1024-ctx batch 256 on 48GB GPUs) and ignored clipping the MAGIC trajectory would apply.


## v0.24.3 (2026-08-06)

### Bug Fixes

- **trainer**: Forward save_interval when fast-forwarding resume schedule
  ([#409](https://github.com/EleutherAI/bergson/pull/409),
  [`eee6d3e`](https://github.com/EleutherAI/bergson/commit/eee6d3ef2fdfdcf1e3856a400da70bf36b13eb35))

The resume branch's schedule fast-forward called next_save_index without save_interval, so resuming
  any save_mode='interval' run raised "save_mode='interval' requires save_interval > 0".

- **validate**: Detect per-query .pt scores via the run config
  ([#407](https://github.com/EleutherAI/bergson/pull/407),
  [`94a1b8a`](https://github.com/EleutherAI/bergson/commit/94a1b8a62aa25cb07c4dfd5b1316643990ffe1f9))

A 2-D scores.pt is [docs, seq_len] for per-token runs but [docs, queries] for query_method: none,
  and load_attribution_scores always chose the per-token reading, so evaluate_retrained rejected
  per-query MAGIC scores with 'expects per-doc (1D) scores'. Disambiguate with the config.yaml
  written next to scores.pt.

### Documentation

- **magic**: Record the WikiText MAGIC LDS in gpt2_wikitext.yaml
  ([#406](https://github.com/EleutherAI/bergson/pull/406),
  [`eeb6cc5`](https://github.com/EleutherAI/bergson/commit/eeb6cc595c12e97ae7caf23198bd256c0dc31e09))

Per-query MAGIC LDS = 0.952 (95% CI [0.944, 0.959]), m=50 queries against an N=100 leave-1%-out
  retrain bank, for this config's eps_root=1e-8 / bs256 recipe.

Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>

### Testing

- Per-query MAGIC scores only real queries when query set is padded
  ([#405](https://github.com/EleutherAI/bergson/pull/405),
  [`ba22bc0`](https://github.com/EleutherAI/bergson/commit/ba22bc0b334e7f1d6294fbcfd3dd8e99cecc3c1b))


## v0.24.2 (2026-08-04)

### Bug Fixes

- **magic**: Pass grad_accum_steps to per-query query gradients
  ([#403](https://github.com/EleutherAI/bergson/pull/403),
  [`1b7cf60`](https://github.com/EleutherAI/bergson/commit/1b7cf60bf9dfd95b9272712f451616ec8bac80d3))

* fix(magic): pass grad_accum_steps to per-query query gradients

compute_per_query_magic_scores forwarded each padded query batch as a single micro-batch, so
  batch_size x seq_len logits OOMed at GPT-2 scale. The aggregate pass already splits by
  grad_accum_steps; results are exact either way since accumulate_grads rescales micro-batches by
  token count.

* fix(magic): score only real query docs, not batch padding

pad_dataset_to_batch_size returns the padded length as num_docs when the query set has no doc_ids
  column, so compute_per_query_magic_scores ran one trajectory backward per pad row (batch_size
  backwards for a single query) and produced duplicate score columns that validate_scores then
  rejects.

* fix(magic): await in-flight async checkpoint save when training crashes

An exception escaping the training loop left the DCP async_save writer thread running; an in-process
  resume() then raced its rmtree cleanup of the incomplete checkpoint against that writer (OSError:
  Directory not empty). Surfaced as test_magic_resume_preserves_checkpoint_schedule flaking under
  pytest -n 8.


## v0.24.1 (2026-08-04)

### Bug Fixes

- **magic**: Bound double-backward memory with double_backward_batch_size
  ([#402](https://github.com/EleutherAI/bergson/pull/402),
  [`d1d625c`](https://github.com/EleutherAI/bergson/commit/d1d625c1c4b953b390f38ef225487812c2cf10fa))

Stage B of microbatch_step_vjp rebuilds each micro-batch's gradient graph with create_graph=True,
  and the CE double backward keeps ~11 vocab-sized tensors alive per graph (~36 GiB for 16 GPT-2
  sequences of 1024 tokens), OOMing at the first replayed step regardless of how references are
  freed.

Re-split micro-batches down to double_backward_batch_size sequences for Stage B only. The batch
  gradient is a weighted sum over sequences, so any partition is exact; under dropout the flag is
  ignored because the replay must reuse the forward's micro-batches to draw the same masks. Stage 0
  keeps the forward's split either way.

Also thread grad_accum_steps into the per-query backward, which silently fell back to the
  single-shot traced step.

gpt2_wikitext_tiny.yaml (bs 64, ga 4, double_backward_batch_size 4): 48.5 -> 26.4 GiB peak, ~5%
  slower backward; scores match the unsplit path to fp32 associativity.


## v0.24.0 (2026-08-04)

### Features

- Add per-query MAGIC scoring (query_method="none")
  ([#401](https://github.com/EleutherAI/bergson/pull/401),
  [`3053ade`](https://github.com/EleutherAI/bergson/commit/3053adea586e1d969824c78e5db02d132538ec20))

* Add per-query MAGIC scoring (query_method="none")

MAGIC previously reduced the query set to one gradient before the backward (query_method mean/sum),
  yielding a single aggregate-query score. Aggregate-query LDS over a handful of subsets is noisy
  and not comparable to the per-query LDS the rest of the pipeline reports, so per-query is the
  right unit — but it was not available for MAGIC (EK-FAC already had query_aggregation="none").

Add query_method="none": one backward per query, sharing a single forward, into a [num_train_docs,
  num_query_docs] score matrix that the existing multi_query validate path consumes. Because the
  backward is linear in the query cotangent the result is exact; per-query scores are written
  incrementally to per_query/q{i}.pt (a crash or preemption only loses the in-flight query, and a
  rerun skips finished ones), the final state is restored before each query since the backward walks
  it back down the trajectory, and per-query GPU state copies and the backward's temp checkpoints
  are freed each iteration (an unbounded loop otherwise OOMs).

Tests (CPU): the mean over queries of the per-query scores reproduces the aggregate-query score to
  1e-6 (equal-length queries), and per-query files are written incrementally.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

* Default query_method to none; single-query distributed tests

Aggregate query_method over a 512-doc query set would run one backward per query in the distributed
  tests; point them at a single query doc instead. Per-query iterates dataset rows, so it assumes
  one chunk per query doc: multi-chunk docs score only their first chunk, and num_query_docs > chunk
  count scatters out of range.

---------

Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>


## v0.23.1 (2026-08-04)

### Bug Fixes

- **magic**: Oom with grad accumulation at large batch size
  ([#400](https://github.com/EleutherAI/bergson/pull/400),
  [`2a7022f`](https://github.com/EleutherAI/bergson/commit/2a7022ff6a5f79dc1ef303b8956da3532d360615))

The query-gradient pass always ran a full batch_size forward/backward, ignoring grad_accum_steps;
  micro-batch it with accumulate_grads.

Free Stage A's traced update graph and gradient copies before Stage B's per-micro-batch double
  backward in microbatch_step_vjp, restoring memory parity with the single-shot traced step.

- **magic**: Preserve fp64 logits in the loss
  ([#399](https://github.com/EleutherAI/bergson/pull/399),
  [`e95e478`](https://github.com/EleutherAI/bergson/commit/e95e4785485da8a909268f14960d05cf2d76e0f4))

* fix(magic): preserve fp64 logits in the loss

weighted_causal_lm_ce did `logits.float()` unconditionally, silently downcasting fp64 logits to fp32
  and capping the precision of an fp64 metagradient run. At eps_root=1e-8 (ill-conditioned,
  per-example scores ~1e4) an fp64 run with the fp32 loss leaves a ~1.3% ga1-vs-ga2 metagradient
  scale residual (forward divergence 2.9e-5); keeping fp64 in the loss closes it (scale 1.000,
  forward 9.7e-10). fp16/bf16 still promote to fp32 for cross-entropy stability.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

---------

Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>


## v0.23.0 (2026-08-04)

### Features

- **magic**: Add fp64 precision for ill-conditioned metagradients
  ([#398](https://github.com/EleutherAI/bergson/pull/398),
  [`a153f7f`](https://github.com/EleutherAI/bergson/commit/a153f7f86193afa7745bcfbcac08d1028054a9f1))

* feat(magic): add fp64 precision for ill-conditioned metagradients

At eps_root=1e-8 the MAGIC metagradient is severely ill-conditioned (per-example scores reach ~1e4).
  In fp32, grad_accum's micro-batch summation-order difference (fp non-associativity) is amplified
  chaotically by that ill-conditioning: over 288 steps the ga=1 vs ga=2 trained models diverge ~1.6%
  and their metagradients differ by a ~0.77 scale (rank-preserving, so LDS is unaffected, but
  magnitudes are not comparable across grad_accum). fp64 collapses it -- forward divergence 1.6e-2
  -> 1e-9, metagradient scale 0.77 -> 1.000 -- confirming the effect is finite-precision, not an
  algorithmic difference between the grad_accum paths.

- precision: add "fp64" (config Literal + worker_utils model-load match; the
  convert_precision_to_torch converter already handled it) - tests: ga=1 vs ga>1 weight_grads must
  match (guards microbatch_step_vjp's weight-gradient path, previously untested), fp32 and fp64 arms

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

---------

Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>


## v0.22.1 (2026-08-04)

### Bug Fixes

- Metasmoothness worker no longer re-expands epochs
  ([#397](https://github.com/EleutherAI/bergson/pull/397),
  [`10be667`](https://github.com/EleutherAI/bergson/commit/10be66785660f2515dbaaa27f3a449c9e2e7be01))

#393 moved epoch expansion into run_metasmoothness (shuffled_epochs builds num_epochs independently
  shuffled copies) but left the worker's train_dataset.repeat(num_epochs). Since training length is
  len(dataset) // batch_size, the worker then trained num_epochs**2 epochs (e.g. 250 steps instead
  of 125 for a 2-epoch 4k/bs64 config), so measured metasmoothness no longer matched the config it
  claimed to measure.

Drop the redundant repeat and add a worker-level regression test asserting the training stream is
  built from exactly num_epochs*N docs.

Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>

### Documentation

- Five-seed replication LDS results ([#395](https://github.com/EleutherAI/bergson/pull/395),
  [`b402f91`](https://github.com/EleutherAI/bergson/commit/b402f91d39e76ad9c84e0b79569e42aba27e1dc3))


## v0.22.0 (2026-08-03)

### Features

- **magic**: Replicate WikiText MAGIC at the paper's eps_root 1e-8
  ([#394](https://github.com/EleutherAI/bergson/pull/394),
  [`cc649d7`](https://github.com/EleutherAI/bergson/commit/cc649d754843bf3cd83f9feef44a2116e99161c0))

Switch the GPT-2/WikiText MAGIC replication from the eps_root 1e-6 / bs64 stand-in to the paper's
  damping (eps_root 1e-8) with the batch size recovered by our sweep. At eps_root 1e-8, per-query
  MAGIC LDS rises from ~0.17 at bs64 to 0.95 at bs256 (num_epochs fixed at 4, so the data is
  identical -- it is the larger batch, not more training). N=100 leave-1%-out bank; m=50 per-query
  evaluation.

Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>


## v0.21.2 (2026-08-03)

### Bug Fixes

- Metasmoothness trains with run_magic's epoch pipeline
  ([#393](https://github.com/EleutherAI/bergson/pull/393),
  [`0ec50f4`](https://github.com/EleutherAI/bergson/commit/0ec50f45d3aff460b8ee056d5275d40572ece403))

run_metasmoothness shuffled once and never concatenated epoch copies, so its three trainings ran a
  single epoch with a fixed order regardless of num_epochs; run_magic implements epochs as
  independently shuffled concatenated copies. The measured config now trains exactly as MAGIC would.


## v0.21.1 (2026-08-03)

### Bug Fixes

- Source resume polarity ([#390](https://github.com/EleutherAI/bergson/pull/390),
  [`19ecf86`](https://github.com/EleutherAI/bergson/commit/19ecf868649a7f99d532c4f1f1000d3c36eb066c))

* Fix inverted resume polarity in the SOURCE pipeline

Steps 2-4 passed `resume=index_cfg.overwrite`, and step 5 skipped its work when
  `index_cfg.overwrite` was set. `resume=True` means "skip if the output exists", so the flag was
  inverted everywhere it was used: with `overwrite: true` -- what
  examples/pipelines/approx_unrolling_pythia.yaml sets -- step 1 recomputed the per-checkpoint
  Hessians while steps 2-5 reused stale segment covariances, lambdas and query gradients. With
  `overwrite: false` the opposite happened. Neither setting produced "re-run == fresh run".

Step 1 already used the flag correctly, which is why the two halves of the pipeline disagreed.

The new test stubs each step and asserts its resume flag is `not overwrite`; without this change it
  fails in both directions.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

* Track step completion with the .part rename

Skipping a step because its output directory exists cannot tell a finished step from one that
  crashed halfway: a resumed run reuses the partial output as if it were complete. Steps that go
  through build/score already write to <out>.part and rename on success, and that rename is atomic
  on one filesystem, so the final name's existence is a completion record already.

step_state.py exposes that as prepare_step/promote_step, and the two segment aggregation steps now
  use it instead of writing straight to their output directory and probing for artifacts.
  _step_complete in the hessian pipeline used a bare Path.exists(); it now clears an interrupted
  .part and re-runs.

A resumed run therefore skips completed steps and restarts interrupted ones, and overwrite forces
  everything.

The covariance worker's inner "skip summing if total_processed.pt exists" check is dropped: a
  partial directory is always cleared before the step re-runs, so it described a mid-step resume
  this design does not support.

---------

Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>


## v0.21.0 (2026-08-03)

### Documentation

- 5-seed replication ground truth; validate averages over retrain runs
  ([#385](https://github.com/EleutherAI/bergson/pull/385),
  [`12966d2`](https://github.com/EleutherAI/bergson/commit/12966d296f5c517df279801b4432874a961e50a0))

validate gains a subsets path to reuse a prior draw, retrained_dir accepts a list whose query losses
  are averaged (Eq. 8's expectation over training randomness), and the retrain config runs seeds
  1004-1008. Adds the LDS table to the README.

### Features

- Save_models on every training command; one name for token attribution
  ([#387](https://github.com/EleutherAI/bergson/pull/387),
  [`e429cfc`](https://github.com/EleutherAI/bergson/commit/e429cfc0ca2d03e50c9693c4d9761e7b0d0967eb))

save_models moves to TrainingConfig: train, magic, and metasmoothness save the trained (unperturbed)
  model to <run_path>/model; validate keeps the leave-k-out family at retrained/{base,subset_<i>}.
  MagicConfig.per_token becomes attribute_tokens, the name the rest of the codebase uses for the
  same toggle. Subclass-only fields are read behind isinstance instead of getattr-with-default,
  which silently returned the default after renames.


## v0.20.0 (2026-08-03)

### Documentation

- Wikitext-2 replication configs ([#381](https://github.com/EleutherAI/bergson/pull/381),
  [`047743a`](https://github.com/EleutherAI/bergson/commit/047743abff75fb21b04718a997f335b302d1f5bd))

Train / SOURCE / EK-FAC / retrain / validate chain against the hosted
  EleutherAI/bergson-wikitext-2-4656-chunks dataset, with the script that reproduces it. Ground
  truth is regenerated by the paper's recipe.

### Features

- Expand a step's matrix mapping into one step per grid cell
  ([#386](https://github.com/EleutherAI/bergson/pull/386),
  [`1fb3f39`](https://github.com/EleutherAI/bergson/commit/1fb3f391e13c637a792a871677ef17bc235048d2))

A `matrix:` block on a pipeline step cartesian-expands it: `{key}` in string values substitutes the
  cell's value, and a value that is exactly "{key}" takes the typed grid value. Cells run
  sequentially like any other steps; colliding or missing run_paths across cells are an error.


## v0.19.0 (2026-08-02)

### Features

- Adam SOURCE preconditioner and Eq-43 hybrid
  ([#379](https://github.com/EleutherAI/bergson/pull/379),
  [`5d4b765`](https://github.com/EleutherAI/bergson/commit/5d4b76567d64e730014280f34ecd6c1f2708e5ed))

Evaluate the unrolling eigenfunctions on a diagonal approximation of the preconditioned Hessian
  P^1/2 H P^1/2, built per segment from the checkpoints' saved second moments (Bae et al. 2024, App.
  C). With adam_segment_hybrid, F_segment uses the App. D Eq-43 form: the diagonal supplies only the
  matrix exponential and the EK-FAC factors supply H^-1. Requires optimizer.pt in each checkpoint
  dir, written by the trainer's save_optimizer_state and carried across by export_checkpoints.

### Testing

- Save_optimizer_state must not hang under FSDP
  ([#380](https://github.com/EleutherAI/bergson/pull/380),
  [`054ebc7`](https://github.com/EleutherAI/bergson/commit/054ebc7c25901a0ae75a3a69cc6ba4957a2328ec))

Covers the every-rank collective fix from #376.


## v0.18.0 (2026-08-02)

### Features

- Auto-export trainer DCP checkpoints in the SOURCE pipeline
  ([#378](https://github.com/EleutherAI/bergson/pull/378),
  [`92867b2`](https://github.com/EleutherAI/bergson/commit/92867b27c993b6c1df2f184f15479bdf9671efc9))

* fix: read the training config out of the saved {steps, metadata} document

save_run_config has always wrapped the step list in a document with a metadata block, so
  load_training_config rejected every real run directory; its test hand-rolled the bare list it
  expected instead of calling save_run_config. Have the test write configs the way the writer does,
  and report an unparseable payload as ValueError so callers guessing at a run dir still degrade to
  their fallback.

(cherry picked from commit cf91c5b67892ff4e4c830ba816099d5335033962)

* feat: auto-export trainer DCP checkpoints in the SOURCE pipeline

resolve() converts raw checkpoints/step_<n>.ckpt paths to the HF exported/checkpoint-<n> dirs
  from_pretrained needs, exporting on demand and reusing existing exports, instead of rejecting
  them. load_training_config gains a fallback so a checkpoint whose sibling config.yaml belongs to
  an attribution run infers nothing rather than failing the pipeline.

* refactor: put trainer and export code with their owners

trainer_run.py becomes train_cfg_io.py: it reads the training run's config and artifacts, and the
  old name read as "run the trainer". write_lr_history moves into the magic trainer that calls it,
  ending magic's import of approx_unrolling; ensure_exported and EXPORT_DIRNAME move next to
  export_checkpoints in utils.trainer_export.


## v0.17.1 (2026-08-02)

### Bug Fixes

- Score each segment against its own checkpoints' training gradients
  ([#377](https://github.com/EleutherAI/bergson/pull/377),
  [`4d5de7e`](https://github.com/EleutherAI/bergson/commit/4d5de7e33c66eeb47408dcc00e7a641d9e4141b1))

score_per_segment_and_aggregate evaluated every segment's training gradients at the final
  checkpoint, collapsing SOURCE toward influence functions at convergence. Bae et al. 2024 Sec. 3.4
  defines g_bar_l as the expected training gradient over segment l's checkpoints; scores are linear
  in the training gradient, so averaging per-checkpoint scores within the segment is equivalent and
  reuses the existing scorer.


## v0.17.0 (2026-08-02)

### Bug Fixes

- Call save_second_moments_as_optimizer_pt from every rank
  ([#376](https://github.com/EleutherAI/bergson/pull/376),
  [`4174502`](https://github.com/EleutherAI/bergson/commit/41745028e7966538b889e9c29b60ed9ba7421347))

Gathering FSDP's sharded (DTensor) moments is a collective, so gating the call on global_rank == 0
  hangs non-zero ranks; the function already restricts the file write to rank 0.

### Features

- Interval save mode for the trainer ([#373](https://github.com/EleutherAI/bergson/pull/373),
  [`8eea5ef`](https://github.com/EleutherAI/bergson/commit/8eea5ef2d5bfe91245454f5c8018376631508a9f))

save_mode="interval" writes a checkpoint every save_interval steps, plus the final state when the
  cadence lands on it (the other modes' backward replay indexes the data stream by checkpoint, so
  only interval mode may add a trailing snapshot). save_mode and save_interval move from MagicConfig
  to TrainingConfig so `bergson train` covers trainer-only runs such as producing SOURCE
  checkpoints, and MagicSaveMode becomes SaveMode to match; the MAGIC backward rejects interval
  mode.


## v0.16.1 (2026-08-02)

### Bug Fixes

- Normalize SOURCE segment eigenvalues per document
  ([#375](https://github.com/EleutherAI/bergson/pull/375),
  [`ad939f5`](https://github.com/EleutherAI/bergson/commit/ad939f5afc69808d23c26926a5e4b712c6b9fab7))

Segment eigenvalues were unnormalized sums over documents x checkpoints. EK-FAC is unaffected --
  damping is relative to mean(lambda), so every inversion is homogeneous in the scale -- but
  SOURCE's eigenfunctions are not scale-invariant: sigma enters exp(-lr*K*sigma), and the
  unnormalized scale puts it in the function's dead range (LDS 0.125 vs 0.383). Eq. 1 of Bae et al.
  2024 defines the risk as a per-document mean, and kronfluence divides its lambda matrix by the
  processed-sample count; dividing the pooled segment sum by the pooled document count matches both.


## v0.16.0 (2026-08-02)

### Features

- **validate**: Save subsets.json by default and reuse it if present
  ([#372](https://github.com/EleutherAI/bergson/pull/372),
  [`781583b`](https://github.com/EleutherAI/bergson/commit/781583bf096f7d5e8315ac049df53de8fdfeeda4))

Write subsets.json and reload it when possible so different methods using the same re-train bank
  re-use them

### Refactoring

- Rename save_retrained_models to save_models
  ([#371](https://github.com/EleutherAI/bergson/pull/371),
  [`d6c3940`](https://github.com/EleutherAI/bergson/commit/d6c3940b0c0baf8a329cb3b75ef64f8eaeb63456))

Rename save_retrained_models -> save_models


## v0.15.1 (2026-07-29)

### Bug Fixes

- Import load_from_optimizer lazily to break a spawn-time cycle
  ([#368](https://github.com/EleutherAI/bergson/pull/368),
  [`f824782`](https://github.com/EleutherAI/bergson/commit/f8247824424db8a39216fe6b88228a7740e11536))

Importing it at module scope pulls in bergson.gradients, which re-enters the package __init__ and
  reaches magic.cli -> validate -> magic.trainer. In-process that resolves, but a spawned worker
  unpickling through magic.cli hits magic.trainer half-initialized and fails on TrainerState, taking
  out test_grad_accum_matches_full_batch.


## v0.15.0 (2026-07-29)

### Features

- Seamless Bergson Trainer -> SOURCE attribute
  ([#367](https://github.com/EleutherAI/bergson/pull/367),
  [`914c77a`](https://github.com/EleutherAI/bergson/commit/914c77a0eb208194015abc523b9f360a3d0a0bf2))

feat: run SOURCE directly off a bergson training run


## v0.14.0 (2026-07-29)

### Features

- Magic gradient accumulation ([#357](https://github.com/EleutherAI/bergson/pull/357),
  [`e7d0358`](https://github.com/EleutherAI/bergson/commit/e7d0358cd59b948a299019529f42ad174c89ce59))

feat: Add gradient accumulation (grad_accum_steps) to MAGIC trainer


## v0.13.4 (2026-07-28)

### Performance Improvements

- Re-use re-train bank losses in `bergson validate`
  ([#365](https://github.com/EleutherAI/bergson/pull/365),
  [`52a5a34`](https://github.com/EleutherAI/bergson/commit/52a5a34bf99a850cf7d502fe65708a6d5ebcf0b7))

* perf: cache method-independent bank losses in evaluate_retrained

Evaluating attribution scores against a pre-saved leave-k-out bank re-runs every banked model on the
  query set to get each subset's query-loss diff. Those per-subset losses depend only on the bank
  and the query set, never on the attribution scores, so scoring a second method against the same
  bank repeats the entire (dominant) cost for nothing.

Cache the per-subset losses (and baseline) under the bank keyed by the query set and load settings,
  and reuse them automatically whenever the key matches. The first method scored on a bank pays the
  full cost; every later method skips model evaluation entirely and only re-sums its own scores. A
  metadata guard recomputes when the key no longer matches.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

---------

Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>


## v0.13.3 (2026-07-27)

### Performance Improvements

- Remove torch.compile from normalizers (fixes import on Python 3.14)
  ([#361](https://github.com/EleutherAI/bergson/pull/361),
  [`315088c`](https://github.com/EleutherAI/bergson/commit/315088c21fd17c9d2663b5172214688030baf67d))

* fix: guard torch.compile so bergson imports on Python 3.14

torch.compile raises RuntimeError at decoration time on interpreters Dynamo doesn't support (Python
  3.14+ with torch <= 2.9). The normalizers in bergson/gradients.py apply it at import time, so
  importing bergson crashed outright on 3.14.

- Add compile_if_supported: falls back to the eager function with a warning when torch.compile is
  unavailable - Add tests covering the fallback path, including a subprocess test that simulates the
  3.14 condition on any interpreter - Add an import-py314 CI job that installs bergson on Python
  3.14 and imports it, catching future import-time regressions

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

* ci: run compile-fallback tests on the Python 3.14 job

* perf: remove torch.compile from normalizers

A100 benchmarks (torch 2.7.1, py3.11) show the compiled normalizers are net slower in the common
  case: ~50us of dynamo guard overhead per call dominates the sub-100us kernels at 768-dim shapes
  and on every bias path, and each run pays 0.3-1.5s compile latency per graph (17 graphs observed
  from bin-packing's varying batch sizes). Compiled kernels only win on very large tensors (up to
  ~2x on [4, 50304, 768] bf16), a regime worth a few percent of end-to-end build time at most, and
  only when --optimizer_state is set.

Removing the decorators also makes bergson importable on Python 3.14, where torch.compile raises at
  decoration time (torch <= 2.9). The import-py314 CI job guards against reintroducing import-time
  compiles.

---------

Co-authored-by: Claude Fable 5 <noreply@anthropic.com>


## v0.13.2 (2026-07-24)

### Bug Fixes

- Thread hessian_dtype into the EK-FAC lambda collector; add fp64
  ([#356](https://github.com/EleutherAI/bergson/pull/356),
  [`2671534`](https://github.com/EleutherAI/bergson/commit/26715342c5705782814596dde59450b98b1fb675))

hessian_dtype only reached the covariance collectors — the eigenvalue-correction pass ignored it and
  accumulated in the activations' dtype. Pass it to LambdaCollector in both the standard and
  per-checkpoint (SOURCE) paths, and add an fp64 option for kronfluence-parity precision studies.


## v0.13.1 (2026-07-22)

### Bug Fixes

- Scale random projections by 1/sqrt(p) (Johnson-Lindenstrauss)
  ([#346](https://github.com/EleutherAI/bergson/pull/346),
  [`4b9b255`](https://github.com/EleutherAI/bergson/commit/4b9b255b332d6efb15452873a41afcd57a2ca26e))

* Scale random projections by 1/sqrt(p) (Johnson-Lindenstrauss)

create_projection_matrix normalized each row over its n entries, giving entry variance 1/n. JL
  requires 1/p (p = projected dim) so that E[A^T A] = I and projected inner products are unbiased.
  The consequence is not the overall scale — a constant cancels from any ranking — but that the
  factor p/n depends on the module's SHAPE, so modules are silently reweighted against each other in
  the summed score. Attention (d x d) and MLP (d x 4d) blocks differ by 4x.

Measured E[proj]/true, 3000 draws per shape:

shape p before predicted p^2/(o*i) after 32x32 16 0.24972 0.25000 0.99889 64x64 32 0.25042 0.25000
  1.00167 128x128 32 0.06247 0.06250 0.99949 256x64 32 0.06253 0.06250 1.00045

This matches the TrackStar paper (arXiv 2410.17413 §A.1.2), which specifies the same two-sided
  per-block projection and states "Projection matrix entries are sampled i.i.d. from N(0, 1/d)" —
  variance 1/(number of rows), exactly A /= sqrt(m). It also matches docs/preprocessing.rst's claim
  that projections preserve inner products, which the old scaling did not.

Existing tests could not catch this: they compare the collector against L @ G @ R^T using the SAME
  matrices, so they are invariant to how those are scaled. test_global_projection_linearity uses one
  R sliced into blocks — the paper's construction, not the code's.
  tests/test_projection_inner_products.py now pins projected against unprojected (12 tests, all 12
  fail without this fix), including that the scale does not depend on module shape.

BREAKING: changes all stored index scales. Existing indices built with projection_dim > 0 must be
  regenerated, and any published numbers re-run. runs/test_build_cache.npy is regenerated here; the
  measured ratio is exactly 0.5000, matching theory (sqrt(o*i)/p = 8/16 for tiny-Phi3).

THREE TESTS FAIL ON THIS BRANCH AND ARE LEFT FAILING, PENDING A DECISION:

tests/test_batch_size_invariance.py::test_gradient_scale_invariance[100-100]
  tests/test_batch_size_invariance.py::test_gradient_scale_invariance[50-150] Relative error is
  UNCHANGED by this fix (6.83e-06 on main vs 7.58e-06 here) and already exceeds the assert_close
  default rtol=1.3e-06 on main. It passes on main only because the values are ~150x smaller, so the
  absolute difference stays under atol=1e-05. Restoring the correct scale trips atol. The test is
  magnitude-sensitive rather than wrong about the invariant.

tests/test_gradients.py::test_gradient_collector_proj_norm 1 of 256 elements, greatest relative
  difference 2.46e-04 against the test's explicit 1e-4 tolerance. Probably a near-cancelling element
  landing differently at the new scale — the test already carries a comment about accumulating
  numerical error — but this is NOT proven.

These are regression guards; loosening them is a call for a human, not something to do to make a
  branch green.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

* Make two projection tests independent of gradient magnitude

test_gradient_scale_invariance compared std separately-vs-combined with assert_close's defaults. The
  invariant is scale-free, but the default atol=1e-5 let a relative disagreement pass whenever the
  operands were small enough: on main the relative error is already 6.83e-06 against rtol=1.3e-06,
  and the test passes only because the values are ~150x smaller than the correctly-scaled ones. Use
  rtol=1e-4 with atol=0 so the check tracks the invariant rather than the magnitude.

test_gradient_collector_proj_norm compares A @ normalize(g) @ B.T against the collector, which
  projects activations and output grads separately. The two orderings are mathematically identical
  and differ only in fp32 rounding; at the corrected scale that noise reached 2.46e-04 relative on
  an O(1) element, over the existing 1e-4. Widen to 1e-3 and say what the tolerance is measuring.

* Add projection_scale to read indexes built with the old scaling

Indexes store their projected gradients, so changing the projection scaling makes existing ones
  unreadable at the correct weighting. projection_scale selects the convention: "jl" (default,
  variance 1/projection_dim) or "row_norm".

GradientProcessor.load resolves a missing projection_scale key to "row_norm", so existing indexes
  keep working with no user action, and new ones record "jl" in processor_config.yaml. Threaded
  through IndexConfig, create_processor and EkfacConfig, which builds matching matrices to compress
  the IVHP output.

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

---------

Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>


## v0.13.0 (2026-07-19)

### Features

- Compress K-FAC IVHP output to match compressed gradient stores
  ([`157fa1b`](https://github.com/EleutherAI/bergson/commit/157fa1b96b80b24648aabf593d3400f83a407666))

EKFAC's H^-1 must be applied to the full, unprojected gradient (the eigenbasis rotation only makes
  sense at the true parameter dimension), so apply_hessian.py's IVHP path and the query build step
  have always forced projection_dim=0. But the training-side gradient store built by `bergson
  build`/`bergson score` uses index_cfg.projection_dim's per-module Kronecker random projection by
  default, so the hessian_pipeline's scoring step had to force projection_dim=0 there too --
  silently ignoring whatever compression the user asked for and scoring against full dense
  gradients.

EkfacConfig gains projection_dim/projection_type; when set, EkfacApplicator.compute_ivhp_sharded
  compresses each module's H^-1 G output to [p, p] as a post-processing step, using the same
  create_projection_matrix identifiers (f"{name}/left" / f"{name}/right") that bergson build already
  uses on training gradients. The hessian_pipeline now threads
  index_cfg.projection_dim/projection_type into the apply step and no longer overrides the score
  step's projection_dim to 0, so a compressed query and a compressed training-side store are
  directly comparable end to end.

Supersedes EleutherAI/bergson#275, which predates the FactoredPreconditioner refactor and no longer
  applies; ports the same idea (compress the IVHP output rather than the pre-Hessian gradient) onto
  the current apply path.

Co-Authored-By: Girish Gupta <girish@girishgupta.com>

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>


## v0.12.1 (2026-07-18)

### Bug Fixes

- Correctly differentiate cross-rank gradient sync in MAGIC's backward-through-training
  ([`bfc2938`](https://github.com/EleutherAI/bergson/commit/bfc29381bc2436310ef0b554716177ee858711e4))

Trainer.step() synchronizes per-rank parameter gradients with a differentiable all-reduce so the
  resulting graph can be differentiated again during backward-through-training. Rebased onto main,
  which had independently landed a different fix for the same underlying bug (51a822dd, "Update DDP
  MAGIC all reduce") using torch.distributed.nn.functional.all_reduce plus a single `/world_size`
  correction applied once at the end of Trainer.backward().

This commit replaces that mechanism with a custom _ReplicatedAllReduceSum autograd Function,
  differentiable to arbitrary order via a recursive backward. Unlike torch.distributed.nn's
  all_reduce, this one already applies its `/world_size` correction inline (dividing before the
  reduce, once per occurrence), so the leftover unconditional `bwd_state.weight_grads /=
  dist.get_world_size()` at the end of `backward()` — carried over from main's fix, which does need
  it — was double-correcting and silently deflating scores by another 1/world_size. Confirmed via
  tests/test_ddp.py::test_ddp_matches_single_process, which should be exact with no cross-rank
  correction needed for a single step: it failed at exactly a 1/world_size ratio with the line still
  in place, and passes with it removed. Both CPU (test_ddp.py) and real multi-GPU
  (test_distributed_magic.py, FSDP-vs-DDP with and without grad clipping) suites pass after the fix.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>


## v0.12.0 (2026-07-18)

### Features

- Add mix_hessians toggle to TrackStar pipeline
  ([`08e9e23`](https://github.com/EleutherAI/bergson/commit/08e9e23876320b731c662b1466bfdbb4bdd7c617))

Lets TrackStar skip computing and mixing the query autocorrelation hessian (steps 2-3) and
  precondition with the value/train hessian alone. Useful when query and train data are IID and the
  query set is too small to estimate its own hessian reliably.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

### Refactoring

- Use invert_psd_matrix in semantic scoring examples
  ([`d005091`](https://github.com/EleutherAI/bergson/commit/d0050914b1d419c25ea768b98f78ab7e305c931d))

Updates example scripts to call bergson.hessians.inversion.invert_psd_matrix instead of the old
  bergson.utils.math.damped_psd_power, matching the already-landed inversion API. Where a custom
  regularizer was passed in, fold it into H + damping_factor*regularizer before inverting rather
  than passing it as a separate argument.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>


## v0.11.1 (2026-07-18)

### Bug Fixes

- Optimizer-state loading — orientation, HF param groups, PEFT, FSDP
  ([`342a216`](https://github.com/EleutherAI/bergson/commit/342a2160a07af76f7b5c31e6b221a9d7251f9b83))

Four fixes to the optimizer.pt loading used for attribution normalizers, plus matching export
  hardening:

- Orient second moments by module class (LayerAdapter.weight_transposed: HF Conv1D stores [in, out])
  instead of shape matching, which cannot detect transposed storage for SQUARE weights — GPT-2's
  attn.c_proj normalizers were silently transposed. - Map state indices group-aware: HF Trainer
  writes two decay/no-decay param groups whose index order differs from named_parameters(); the old
  positional mapping assigned moments to the wrong modules. Reconstructed with transformers' own
  utilities and shape-verified. - Apply the group-aware mapping to PEFT checkpoints too (HF Trainer
  splits LoRA params across the same groups). - Make save_second_moments_as_optimizer_pt
  collective-safe (FSDP DTensor moments are gathered by every rank; rank 0 writes — previously a
  rank-0 -only call would hang) and record each entry's canonical param_name, since FSDP wrapping
  renames and reorders parameters; readers prefer the recorded name. Optional standard
  step/betas/eps fields make snapshot exports self-describing. Model type widened to nn.Module.

Tests: square-Conv1D orientation, decay-split identity vs a real two-group AdamW, PEFT + two-group
  checkpoint, and the real-checkpoint round-trips now look entries up group-aware with shape
  assertions the old convention fails.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>


## v0.11.0 (2026-07-17)

### Features

- Custom gradient store class
  ([`3ae519c`](https://github.com/EleutherAI/bergson/commit/3ae519c57e45090ea153d15bc0ba24c97b8e0937))

Trim comments referencing the removed structured-dtype layout.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>


## v0.10.2 (2026-07-11)

### Bug Fixes

- Update bergson
  ([`ab4b933`](https://github.com/EleutherAI/bergson/commit/ab4b933083f52c8be144ebd25b2c7cb5c7d0a1f4))

### Documentation

- Add Known limitations section (MoE fused experts, FSDP host-RAM load)
  ([`290140a`](https://github.com/EleutherAI/bergson/commit/290140abbb605f97fddbf83d4839838199783dcd))

Document two issues we are not fixing now, each with the responsible source file:line so maintainers
  can locate them:

- MoE fused-parameter experts and router are bare nn.Parameters on custom modules, outside
  LayerAdapter.supported_modules (bergson/gradients.py:238; module walk in
  bergson/collector/collector.py:154-155), so they are silently skipped. Only attention projections
  and lm_head (~1-2% of params) are tracked. - Under FSDP the load path uses device_map="cpu" per
  rank (bergson/utils/worker_utils.py:166), so every rank replicates a full dequantized model in
  host RAM before sharding, which can OOM host memory independent of GPU VRAM.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_012NMtUnc12k72cAXd4uRMzN

- Compact MoE limitation, drop FSDP host-RAM limitation
  ([`1f573ba`](https://github.com/EleutherAI/bergson/commit/1f573ba0c9ef0ffb9f5e0b998704445b7203b8e3))

Trim the MoE fused-experts note to what users need (affected model families, consequence,
  legacy-layout caveat) and remove the FSDP host-RAM limitation, which is a straightforward fix
  pending coordination.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_0149bxnr7vz4KcR8HpT4ipgn


## v0.10.1 (2026-07-09)

### Bug Fixes

- Load mmap'd FAISS ANN indices on CPU without crashing
  ([`12e54ac`](https://github.com/EleutherAI/bergson/commit/12e54acdf8345405c06a16c5a4d52be7903642a7))

`index_to_device(index, "cpu")` unconditionally called `faiss.index_gpu_to_cpu`, which clones the
  index and raises `RuntimeError: clone not supported ... OnDiskInvertedLists` for any IVF/ANN index
  mmap'd from disk (the shipped `mmap_index=False` default). Only exact `Flat` survived. The CPU
  branch is now a no-op; the GPU->CPU conversion after a GPU build is done explicitly in
  `create_index` only when `device != "cpu"`.

Also expose the FAISS index config on the `query` CLI (nested `QueryConfig.faiss_cfg`) so ANN /
  `mmap_index=True` is actually reachable, and add a regression test for the on-disk ANN CPU load
  path.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_012NMtUnc12k72cAXd4uRMzN

### Documentation

- Drop redundant inline comments at index_to_device call sites
  ([`2ebc293`](https://github.com/EleutherAI/bergson/commit/2ebc293fb936f88c4ec7987f354df5fc7685a2d4))

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01EeaHeis3YkmXwc5emPJnXo

- Trim index_to_device docstring
  ([`5ecc0b0`](https://github.com/EleutherAI/bergson/commit/5ecc0b0f6b3531688ee060453ed258bab7f996eb))

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01EeaHeis3YkmXwc5emPJnXo

### Refactoring

- Keep symmetric index_to_device, fix the bug at the __init__ call site
  ([`50345cf`](https://github.com/EleutherAI/bergson/commit/50345cf6d09ce87fac16224ba4242b479d7c264c))

Revert the index_to_gpu rename. index_to_device stays a symmetric CPU<->GPU move; the actual bug was
  only that `FaissIndex.__init__` routed an already-CPU mmap'd shard through it with `"cpu"`,
  hitting the `index_gpu_to_cpu` clone that crashes on OnDiskInvertedLists. The loader now guards
  `device != "cpu"` so an already-CPU shard is never moved, and `create_index` uses the same helper
  for its genuine GPU->CPU conversion (also guarded). No behavioural change on any tested path; the
  crash stays fixed.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01EeaHeis3YkmXwc5emPJnXo

- Make index_to_device self-guarding (no-op when already on target)
  ([`fc43de6`](https://github.com/EleutherAI/bergson/commit/fc43de6aa02c345f0aed41459eec48f99b5a6501))

Rather than requiring call sites to know "don't ask for cpu if it's already cpu", `index_to_device`
  now detects GPU residency (`_is_gpu_resident`: a direct GpuIndex, or the IndexShards/IndexReplicas
  container a multi-GPU move returns) and treats a CPU->CPU request as a no-op, returning the index
  unchanged instead of cloning via `index_gpu_to_cpu` (which crashes on mmap'd OnDiskInvertedLists).
  Both call sites (`FaissIndex.__init__`, `create_index`) drop their `device != "cpu"` guards and
  just call the helper.

Adds unit tests for the guard: it no-ops on an in-memory CPU index and on the mmap'd OnDisk index
  that used to crash (asserting the raw `index_gpu_to_cpu` still raises, so the regression case is
  real).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01EeaHeis3YkmXwc5emPJnXo

- Rename index_to_device -> index_to_gpu, drop dead CPU branch
  ([`080bbf0`](https://github.com/EleutherAI/bergson/commit/080bbf0380e5f77ab9b62a1c5cc859a935b32e45))

The CPU destination in `index_to_device` was only ever a no-op after the prior fix:
  `FaissIndex.__init__` reads shards from disk (already CPU) and only ever needs to push *up* to a
  GPU, and the one genuine GPU->CPU move lives in `create_index`. Rename to `index_to_gpu` (no-op on
  "cpu") and guard the loader call with `device != "cpu"` so the direction is explicit and the
  OnDiskInvertedLists clone trap cannot be reintroduced. Also trims a verbose config docstring.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01EeaHeis3YkmXwc5emPJnXo

### Testing

- Cover the index_to_device op path, not just the no-op
  ([`64484d8`](https://github.com/EleutherAI/bergson/commit/64484d8e2968ad8eb7847ef35b32fa7b2ce50ba2))

Add a test that a GPU-resident index is actually converted (not returned as-is): an `IndexShards`
  container is what a multi-GPU move returns and what `_is_gpu_resident` flags for conversion, so
  bringing it to CPU must yield a new, non-container, still-searchable index. Complements the two
  no-op tests.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01EeaHeis3YkmXwc5emPJnXo


## v0.10.0 (2026-06-05)

### Features

- Consolidate per-run YAMLs into one reproducible config.yaml
  ([`d33a6bb`](https://github.com/EleutherAI/bergson/commit/d33a6bb056a617e212f2d811187eefcde36cfdad))


## v0.9.1 (2026-04-10)

### Bug Fixes

- Preconditioner bug in attributor.py
  ([`6232d1e`](https://github.com/EleutherAI/bergson/commit/6232d1e3464ec4fc9056a5c45b5efc7b4c421318))


## v0.9.0 (2026-03-18)

### Bug Fixes

- Release
  ([`dec3df9`](https://github.com/EleutherAI/bergson/commit/dec3df98a0707f0058bf193c27ef4f4e50fab6ac))

### Features

- Add flag to enable TF32
  ([`35ab164`](https://github.com/EleutherAI/bergson/commit/35ab16400afda484ccff717b7a4b48ae6f06811d))


## v0.8.1 (2026-03-18)

### Bug Fixes

- Release bergson without pinned transformers
  ([`ef9dc9a`](https://github.com/EleutherAI/bergson/commit/ef9dc9a6bd4604162fcd9c1ba5bcca18f3936455))


## v0.8.0 (2026-03-08)

### Features

- Set default precision to fp32 in IndexConfig and ScoreConfig
  ([`92d4807`](https://github.com/EleutherAI/bergson/commit/92d4807df7b73cee21c6e375c79454b021998671))

Co-authored-by: Lucia Quirke <luciaquirke@users.noreply.github.com>


## v0.7.2 (2026-03-04)


## v0.7.1 (2026-03-03)

### Bug Fixes

- Always compute mixing coefficient in Trackstar pipeline
  ([`c990375`](https://github.com/EleutherAI/bergson/commit/c990375e69d309f348c489f9bfc9cf9cddc28f6d))

Remove the conditional guard — lambda is always auto-computed from the preconditioner eigenvalues
  since the cost is negligible.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.7.0 (2026-03-03)

### Bug Fixes

- Standardize trace collector preconditioning
  ([`6a14e53`](https://github.com/EleutherAI/bergson/commit/6a14e534a403c72bae4a340009ab84d385b7928b))

### Features

- Enable trackstar
  ([`2dd26d3`](https://github.com/EleutherAI/bergson/commit/2dd26d31fe4f88d1f2d19537958208b914cec2c8))


## v0.6.2 (2026-03-02)

### Bug Fixes

- Convert PyArrow Column to list in allocate_batches
  ([`7fe4dd3`](https://github.com/EleutherAI/bergson/commit/7fe4dd32181c5bc7ce5684e452bc442862e22e7f))

HuggingFace Dataset column access (ds["length"]) returns a PyArrow Column, not a Python list.
  Iterating over it element-by-element (via sorted(), random indexing) is ~1000x slower than on a
  native list. For 10M items this caused allocate_batches to hang for 13+ hours instead of
  completing in ~17 seconds.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Convert PyArrow columns to list at callsites of allocate_batches
  ([`5d734dc`](https://github.com/EleutherAI/bergson/commit/5d734dc23bb083819890ca17d1b44f377ae35d69))

Move the list conversion out of allocate_batches (which types doc_lengths as list[int]) to the
  callsites that pass HF Dataset columns. Use ds["length"][:] which returns a plain list[int].

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Remove redundant zero-fill loop in MemmapSequenceScoreWriter
  ([`558829f`](https://github.com/EleutherAI/bergson/commit/558829f717f8679d517765d5c3d9beac2f2249b2))

np.memmap w+ mode already creates a zero-filled file, making the per-field written flag
  initialization loop unnecessary. For large datasets (10M+ items) with many query scores, the
  strided writes through the structured dtype caused multi-hour hangs.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Use [:] instead of list() for consistency
  ([`c76d131`](https://github.com/EleutherAI/bergson/commit/c76d131c357b6b8e7880da48b4640510ffe5a654))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.6.1 (2026-03-02)

### Bug Fixes

- Unpin transformers by explicitly setting float32 dtype in tests
  ([`0b6c226`](https://github.com/EleutherAI/bergson/commit/0b6c22615b7cce4ca62f71cb93847e3027fa68ba))

Transformers 4.56+ changed from_config() to honor the config's torch_dtype field, causing test
  models (tiny-GPTNeoX, tiny-Phi3) to be created in float16 instead of float32. This caused gradient
  comparison tests to fail from reduced precision, not from any actual change in gradient collection
  logic.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.6.0 (2026-02-17)

### Bug Fixes

- Use _csv._writer type for csv_recorder annotation
  ([`6e6289c`](https://github.com/EleutherAI/bergson/commit/6e6289c266b36304a6d79a35bb6b9fe3c35fa95a))

csv.writer is a function, not a class, so it cannot be used as a type annotation. Import the private
  _writer type from _csv and use it for the Generator yield type. Also fix the None check to use `if
  not path` since QueryConfig.record uses empty string as the sentinel value.

Co-authored-by: Lucia Quirke <luciaquirke@users.noreply.github.com>

### Continuous Integration

- Pin pyright version and fix faiss type error
  ([`b9f54cf`](https://github.com/EleutherAI/bergson/commit/b9f54cf9e7caf3c13af78f1a2d3d766f2055c3da))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Use Python 3.11 for typechecking
  ([`9ef4122`](https://github.com/EleutherAI/bergson/commit/9ef4122903eed2ecf496f803c5d1aba4c62295cb))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Use Python 3.11 for typechecking
  ([`ea50dd8`](https://github.com/EleutherAI/bergson/commit/ea50dd8ed9dc02b0f21ce7621f7d0ff53622ea87))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- Add --record flag to query CLI for saving results to CSV
  ([`59770ff`](https://github.com/EleutherAI/bergson/commit/59770ff88c5dbfffabd6ce0f51e5a56edbae2c0b))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- Replace try/finally CSV block with context manager
  ([`6431320`](https://github.com/EleutherAI/bergson/commit/6431320b7c167191b157b3fc53013818ecdd5135))

Co-authored-by: Lucia Quirke <luciaquirke@users.noreply.github.com>


## v0.5.2 (2026-02-17)

### Bug Fixes

- Pass batches to CollectorComputer in fit_normalizers
  ([`c95d5d4`](https://github.com/EleutherAI/bergson/commit/c95d5d498ad900af8a95902535fdfe740696088f))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Continuous Integration

- Improve Claude workflows (fetch-depth, timeout, max-turns, pip install)
  ([`7a315e5`](https://github.com/EleutherAI/bergson/commit/7a315e58758fac24f76400043eeac559380a2952))

- Run tests and typechecking in parallel
  ([`e690fc0`](https://github.com/EleutherAI/bergson/commit/e690fc0bed99ff5e705e8e82d790e961f3ceba33))


## v0.5.1 (2026-01-30)

### Bug Fixes

- Release
  ([`f0ad2be`](https://github.com/EleutherAI/bergson/commit/f0ad2bee12b0eb16f1c211a891b8bd78e89ea45e))


## v0.5.0 (2026-01-08)

### Features

- Add optimizer-aware gradients
  ([`497edab`](https://github.com/EleutherAI/bergson/commit/497edab8f2ca19d8fcb1d409fbd99452a929584e))


## v0.4.6 (2026-01-06)

### Bug Fixes

- Update build.yml
  ([`ba4cd5a`](https://github.com/EleutherAI/bergson/commit/ba4cd5ad49d36595c5ea063037eb832aa3a1a3b4))


## v0.4.5 (2026-01-06)

### Bug Fixes

- Always use unstructured gradients in score
  ([`595ed92`](https://github.com/EleutherAI/bergson/commit/595ed92deb06278f343a489f782e318916036eb2))


## v0.4.4 (2026-01-05)

### Bug Fixes

- Release bergson
  ([`c9040a6`](https://github.com/EleutherAI/bergson/commit/c9040a6dc12bea49b8f3e4bf8efbe82c92022bca))


## v0.4.3 (2026-01-05)

### Bug Fixes

- Release bergson
  ([`350dafe`](https://github.com/EleutherAI/bergson/commit/350dafe9c419ac3a874848a9d355af52de2407bb))


## v0.4.2 (2025-12-22)

### Bug Fixes

- Unit normalize in float32
  ([`cae8352`](https://github.com/EleutherAI/bergson/commit/cae8352c783cd68516ccab18a6746ba974455043))


## v0.4.1 (2025-12-20)

### Bug Fixes

- Pin transformers to avoid fp error bug
  ([`9feac20`](https://github.com/EleutherAI/bergson/commit/9feac20e237d66825a5d16c385e4174bb02f4705))


## v0.4.0 (2025-12-03)

### Features

- Enable specifying a custom tokenizer
  ([`9781a55`](https://github.com/EleutherAI/bergson/commit/9781a5538491aae3bf53af8247ae2509fe801b59))


## v0.3.0 (2025-12-03)

### Features

- Release bergson
  ([`64b5baf`](https://github.com/EleutherAI/bergson/commit/64b5baf4aa998c4e7573e24dcda939e74185c5f4))


## v0.2.0 (2025-11-13)

### Features

- Add on-the-fly queries
  ([`0ce0ee2`](https://github.com/EleutherAI/bergson/commit/0ce0ee2a0ec151f3fa0e6ee1eef3810408a54128))


## v0.1.1 (2025-10-16)

### Bug Fixes

- Simplify query
  ([`fd37173`](https://github.com/EleutherAI/bergson/commit/fd37173bf7c3d25daa6af065e7f261f2b774ce69))


## v0.1.0 (2025-10-16)

### Features

- Add on-the-fly queries
  ([`294661e`](https://github.com/EleutherAI/bergson/commit/294661e1d7ad7220917562991a1c7582b6181632))


## v0.0.0 (2025-10-07)
