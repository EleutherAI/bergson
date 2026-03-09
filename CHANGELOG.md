# CHANGELOG


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
