# CHANGELOG


## v0.26.2 (2026-09-01)

### Bug Fixes

- Give every process group one collective timeout that survives rank-0-only work
  ([#444](https://github.com/EleutherAI/bergson/pull/444),
  [`c563fbd`](https://github.com/EleutherAI/bergson/commit/c563fbde1b506783ee51fb92f4303f561c78d30a))


## v0.26.1 (2026-08-18)

### Bug Fixes

- Stray barrier in one-shot score savers deadlocks aggregate-query MAGIC
  ([#424](https://github.com/EleutherAI/bergson/pull/424),
  [`3d7ff2a`](https://github.com/EleutherAI/bergson/commit/3d7ff2ac2ef518a3ba6231ddefc8c6f592456c35))


## v0.26.0 (2026-08-07)

### Features

- **validate**: Retrain a slice of the subsets
  ([`ab11f99`](https://github.com/EleutherAI/bergson/commit/ab11f9906df0e2ddc040df628a3e1fcd0b502972))


## v0.25.1 (2026-08-06)

### Bug Fixes

- **magic**: Reject chunked query sets in per-query MAGIC
  ([#418](https://github.com/EleutherAI/bergson/pull/418),
  [`012b54a`](https://github.com/EleutherAI/bergson/commit/012b54adec7e57d98ce4040d87bce20697328229))


## v0.25.0 (2026-08-06)

### Features

- **magic**: Per-token per-query MAGIC, and score-format cleanups
  ([#415](https://github.com/EleutherAI/bergson/pull/415),
  [`b333a20`](https://github.com/EleutherAI/bergson/commit/b333a20b25db4877b947306146211ec797228c15))


## v0.24.6 (2026-08-06)

### Bug Fixes

- **magic**: Write doc_ids.pt on fresh per-token runs
  ([#412](https://github.com/EleutherAI/bergson/pull/412),
  [`80d72ba`](https://github.com/EleutherAI/bergson/commit/80d72bae9168ac9d6cd54e44f43f7e131612921b))


## v0.24.5 (2026-08-06)

### Bug Fixes

- **magic**: Let worker() run on CPU-only machines
  ([#411](https://github.com/EleutherAI/bergson/pull/411),
  [`513954f`](https://github.com/EleutherAI/bergson/commit/513954f250fc62aad26c55800f844f2aee2f6c0f))


## v0.24.4 (2026-08-06)

### Bug Fixes

- **metasmoothness**: Forward grad_accum_steps and max_grad_norm to train
  ([#410](https://github.com/EleutherAI/bergson/pull/410),
  [`c059b63`](https://github.com/EleutherAI/bergson/commit/c059b6377cf56541eca54d7c0bcc5b24758c02b8))


## v0.24.3 (2026-08-06)

### Bug Fixes

- **trainer**: Forward save_interval when fast-forwarding resume schedule
  ([#409](https://github.com/EleutherAI/bergson/pull/409),
  [`eee6d3e`](https://github.com/EleutherAI/bergson/commit/eee6d3ef2fdfdcf1e3856a400da70bf36b13eb35))

- **validate**: Detect per-query .pt scores via the run config
  ([#407](https://github.com/EleutherAI/bergson/pull/407),
  [`94a1b8a`](https://github.com/EleutherAI/bergson/commit/94a1b8a62aa25cb07c4dfd5b1316643990ffe1f9))

### Documentation

- **magic**: Record the WikiText MAGIC LDS in gpt2_wikitext.yaml
  ([#406](https://github.com/EleutherAI/bergson/pull/406),
  [`eeb6cc5`](https://github.com/EleutherAI/bergson/commit/eeb6cc595c12e97ae7caf23198bd256c0dc31e09))

### Testing

- Per-query MAGIC scores only real queries when query set is padded
  ([#405](https://github.com/EleutherAI/bergson/pull/405),
  [`ba22bc0`](https://github.com/EleutherAI/bergson/commit/ba22bc0b334e7f1d6294fbcfd3dd8e99cecc3c1b))


## v0.24.2 (2026-08-04)

### Bug Fixes

- **magic**: Pass grad_accum_steps to per-query query gradients
  ([#403](https://github.com/EleutherAI/bergson/pull/403),
  [`1b7cf60`](https://github.com/EleutherAI/bergson/commit/1b7cf60bf9dfd95b9272712f451616ec8bac80d3))


## v0.24.1 (2026-08-04)

### Bug Fixes

- **magic**: Bound double-backward memory with double_backward_batch_size
  ([#402](https://github.com/EleutherAI/bergson/pull/402),
  [`d1d625c`](https://github.com/EleutherAI/bergson/commit/d1d625c1c4b953b390f38ef225487812c2cf10fa))


## v0.24.0 (2026-08-04)

### Features

- Add per-query MAGIC scoring (query_method="none")
  ([#401](https://github.com/EleutherAI/bergson/pull/401),
  [`3053ade`](https://github.com/EleutherAI/bergson/commit/3053adea586e1d969824c78e5db02d132538ec20))


## v0.23.1 (2026-08-04)

### Bug Fixes

- **magic**: OOM with grad accumulation at large batch size
  ([#400](https://github.com/EleutherAI/bergson/pull/400),
  [`2a7022f`](https://github.com/EleutherAI/bergson/commit/2a7022ff6a5f79dc1ef303b8956da3532d360615))

- **magic**: Preserve fp64 logits in the loss
  ([#399](https://github.com/EleutherAI/bergson/pull/399),
  [`e95e478`](https://github.com/EleutherAI/bergson/commit/e95e4785485da8a909268f14960d05cf2d76e0f4))


## v0.23.0 (2026-08-04)

### Features

- **magic**: Add fp64 precision for ill-conditioned metagradients
  ([#398](https://github.com/EleutherAI/bergson/pull/398),
  [`a153f7f`](https://github.com/EleutherAI/bergson/commit/a153f7f86193afa7745bcfbcac08d1028054a9f1))


## v0.22.1 (2026-08-04)

### Bug Fixes

- Metasmoothness worker no longer re-expands epochs
  ([#397](https://github.com/EleutherAI/bergson/pull/397),
  [`10be667`](https://github.com/EleutherAI/bergson/commit/10be66785660f2515dbaaa27f3a449c9e2e7be01))

### Documentation

- Five-seed replication LDS results ([#395](https://github.com/EleutherAI/bergson/pull/395),
  [`b402f91`](https://github.com/EleutherAI/bergson/commit/b402f91d39e76ad9c84e0b79569e42aba27e1dc3))


## v0.22.0 (2026-08-03)

### Features

- **magic**: Replicate WikiText MAGIC at the paper's eps_root 1e-8
  ([#394](https://github.com/EleutherAI/bergson/pull/394),
  [`cc649d7`](https://github.com/EleutherAI/bergson/commit/cc649d754843bf3cd83f9feef44a2116e99161c0))


## v0.21.2 (2026-08-03)

### Bug Fixes

- Metasmoothness trains with run_magic's epoch pipeline
  ([#393](https://github.com/EleutherAI/bergson/pull/393),
  [`0ec50f4`](https://github.com/EleutherAI/bergson/commit/0ec50f45d3aff460b8ee056d5275d40572ece403))


## v0.21.1 (2026-08-03)

### Bug Fixes

- Source resume polarity ([#390](https://github.com/EleutherAI/bergson/pull/390),
  [`19ecf86`](https://github.com/EleutherAI/bergson/commit/19ecf868649a7f99d532c4f1f1000d3c36eb066c))


## v0.21.0 (2026-08-03)

### Documentation

- 5-seed replication ground truth; validate averages over retrain runs
  ([#385](https://github.com/EleutherAI/bergson/pull/385),
  [`12966d2`](https://github.com/EleutherAI/bergson/commit/12966d296f5c517df279801b4432874a961e50a0))

### Features

- Save_models on every training command; one name for token attribution
  ([#387](https://github.com/EleutherAI/bergson/pull/387),
  [`e429cfc`](https://github.com/EleutherAI/bergson/commit/e429cfc0ca2d03e50c9693c4d9761e7b0d0967eb))


## v0.20.0 (2026-08-03)

### Documentation

- WikiText-2 replication configs ([#381](https://github.com/EleutherAI/bergson/pull/381),
  [`047743a`](https://github.com/EleutherAI/bergson/commit/047743abff75fb21b04718a997f335b302d1f5bd))

### Features

- Expand a step's matrix mapping into one step per grid cell
  ([#386](https://github.com/EleutherAI/bergson/pull/386),
  [`1fb3f39`](https://github.com/EleutherAI/bergson/commit/1fb3f391e13c637a792a871677ef17bc235048d2))


## v0.19.0 (2026-08-02)

### Features

- Adam SOURCE preconditioner and Eq-43 hybrid
  ([#379](https://github.com/EleutherAI/bergson/pull/379),
  [`5d4b765`](https://github.com/EleutherAI/bergson/commit/5d4b76567d64e730014280f34ecd6c1f2708e5ed))

### Testing

- Save_optimizer_state must not hang under FSDP
  ([#380](https://github.com/EleutherAI/bergson/pull/380),
  [`054ebc7`](https://github.com/EleutherAI/bergson/commit/054ebc7c25901a0ae75a3a69cc6ba4957a2328ec))


## v0.18.0 (2026-08-02)

### Features

- Auto-export trainer DCP checkpoints in the SOURCE pipeline
  ([#378](https://github.com/EleutherAI/bergson/pull/378),
  [`92867b2`](https://github.com/EleutherAI/bergson/commit/92867b27c993b6c1df2f184f15479bdf9671efc9))


## v0.17.1 (2026-08-02)

### Bug Fixes

- Score each segment against its own checkpoints' training gradients
  ([#377](https://github.com/EleutherAI/bergson/pull/377),
  [`4d5de7e`](https://github.com/EleutherAI/bergson/commit/4d5de7e33c66eeb47408dcc00e7a641d9e4141b1))


## v0.17.0 (2026-08-02)

### Bug Fixes

- Call save_second_moments_as_optimizer_pt from every rank
  ([#376](https://github.com/EleutherAI/bergson/pull/376),
  [`4174502`](https://github.com/EleutherAI/bergson/commit/41745028e7966538b889e9c29b60ed9ba7421347))

### Features

- Interval save mode for the trainer ([#373](https://github.com/EleutherAI/bergson/pull/373),
  [`8eea5ef`](https://github.com/EleutherAI/bergson/commit/8eea5ef2d5bfe91245454f5c8018376631508a9f))


## v0.16.1 (2026-08-02)

### Bug Fixes

- Normalize SOURCE segment eigenvalues per document
  ([#375](https://github.com/EleutherAI/bergson/pull/375),
  [`ad939f5`](https://github.com/EleutherAI/bergson/commit/ad939f5afc69808d23c26926a5e4b712c6b9fab7))


## v0.16.0 (2026-08-02)

### Features

- **validate**: Save subsets.json by default and reuse it if present
  ([#372](https://github.com/EleutherAI/bergson/pull/372),
  [`781583b`](https://github.com/EleutherAI/bergson/commit/781583bf096f7d5e8315ac049df53de8fdfeeda4))

### Refactoring

- Rename save_retrained_models to save_models
  ([#371](https://github.com/EleutherAI/bergson/pull/371),
  [`d6c3940`](https://github.com/EleutherAI/bergson/commit/d6c3940b0c0baf8a329cb3b75ef64f8eaeb63456))


## v0.15.1 (2026-07-29)

### Bug Fixes

- Import load_from_optimizer lazily to break a spawn-time cycle
  ([#368](https://github.com/EleutherAI/bergson/pull/368),
  [`f824782`](https://github.com/EleutherAI/bergson/commit/f8247824424db8a39216fe6b88228a7740e11536))


## v0.15.0 (2026-07-29)

### Features

- Seamless Bergson Trainer -> SOURCE attribute
  ([#367](https://github.com/EleutherAI/bergson/pull/367),
  [`914c77a`](https://github.com/EleutherAI/bergson/commit/914c77a0eb208194015abc523b9f360a3d0a0bf2))


## v0.14.0 (2026-07-29)

### Features

- MAGIC gradient accumulation ([#357](https://github.com/EleutherAI/bergson/pull/357),
  [`e7d0358`](https://github.com/EleutherAI/bergson/commit/e7d0358cd59b948a299019529f42ad174c89ce59))


## v0.13.4 (2026-07-28)

### Performance Improvements

- Re-use re-train bank losses in `bergson validate`
  ([#365](https://github.com/EleutherAI/bergson/pull/365),
  [`52a5a34`](https://github.com/EleutherAI/bergson/commit/52a5a34bf99a850cf7d502fe65708a6d5ebcf0b7))


## v0.13.3 (2026-07-27)

### Performance Improvements

- Remove torch.compile from normalizers (fixes import on Python 3.14)
  ([#361](https://github.com/EleutherAI/bergson/pull/361),
  [`315088c`](https://github.com/EleutherAI/bergson/commit/315088c21fd17c9d2663b5172214688030baf67d))


## v0.13.2 (2026-07-24)

### Bug Fixes

- Thread hessian_dtype into the EK-FAC lambda collector; add fp64
  ([#356](https://github.com/EleutherAI/bergson/pull/356),
  [`2671534`](https://github.com/EleutherAI/bergson/commit/26715342c5705782814596dde59450b98b1fb675))


## v0.13.1 (2026-07-22)

### Bug Fixes

- Scale random projections by 1/sqrt(p) (Johnson-Lindenstrauss)
  ([#346](https://github.com/EleutherAI/bergson/pull/346),
  [`4b9b255`](https://github.com/EleutherAI/bergson/commit/4b9b255b332d6efb15452873a41afcd57a2ca26e))


## v0.13.0 (2026-07-19)

### Features

- Compress K-FAC IVHP output to match compressed gradient stores
  ([`157fa1b`](https://github.com/EleutherAI/bergson/commit/157fa1b96b80b24648aabf593d3400f83a407666))


## v0.12.1 (2026-07-18)

### Bug Fixes

- Correctly differentiate cross-rank gradient sync in MAGIC's backward-through-training
  ([`bfc2938`](https://github.com/EleutherAI/bergson/commit/bfc29381bc2436310ef0b554716177ee858711e4))


## v0.12.0 (2026-07-18)

### Features

- Add mix_hessians toggle to TrackStar pipeline
  ([`08e9e23`](https://github.com/EleutherAI/bergson/commit/08e9e23876320b731c662b1466bfdbb4bdd7c617))

### Refactoring

- Use invert_psd_matrix in semantic scoring examples
  ([`d005091`](https://github.com/EleutherAI/bergson/commit/d0050914b1d419c25ea768b98f78ab7e305c931d))


## v0.11.1 (2026-07-18)

### Bug Fixes

- Optimizer-state loading — orientation, HF param groups, PEFT, FSDP
  ([`342a216`](https://github.com/EleutherAI/bergson/commit/342a2160a07af76f7b5c31e6b221a9d7251f9b83))


## v0.11.0 (2026-07-17)

### Features

- Custom gradient store class
  ([`3ae519c`](https://github.com/EleutherAI/bergson/commit/3ae519c57e45090ea153d15bc0ba24c97b8e0937))


## v0.10.2 (2026-07-11)

### Bug Fixes

- Update bergson
  ([`ab4b933`](https://github.com/EleutherAI/bergson/commit/ab4b933083f52c8be144ebd25b2c7cb5c7d0a1f4))

### Documentation

- Add Known limitations section (MoE fused experts, FSDP host-RAM load)
  ([`290140a`](https://github.com/EleutherAI/bergson/commit/290140abbb605f97fddbf83d4839838199783dcd))

- Compact MoE limitation, drop FSDP host-RAM limitation
  ([`1f573ba`](https://github.com/EleutherAI/bergson/commit/1f573ba0c9ef0ffb9f5e0b998704445b7203b8e3))


## v0.10.1 (2026-07-09)

### Bug Fixes

- Load mmap'd FAISS ANN indices on CPU without crashing
  ([`12e54ac`](https://github.com/EleutherAI/bergson/commit/12e54acdf8345405c06a16c5a4d52be7903642a7))

### Documentation

- Drop redundant inline comments at index_to_device call sites
  ([`2ebc293`](https://github.com/EleutherAI/bergson/commit/2ebc293fb936f88c4ec7987f354df5fc7685a2d4))

- Trim index_to_device docstring
  ([`5ecc0b0`](https://github.com/EleutherAI/bergson/commit/5ecc0b0f6b3531688ee060453ed258bab7f996eb))

### Refactoring

- Keep symmetric index_to_device, fix the bug at the __init__ call site
  ([`50345cf`](https://github.com/EleutherAI/bergson/commit/50345cf6d09ce87fac16224ba4242b479d7c264c))

- Make index_to_device self-guarding (no-op when already on target)
  ([`fc43de6`](https://github.com/EleutherAI/bergson/commit/fc43de6aa02c345f0aed41459eec48f99b5a6501))

- Rename index_to_device -> index_to_gpu, drop dead CPU branch
  ([`080bbf0`](https://github.com/EleutherAI/bergson/commit/080bbf0380e5f77ab9b62a1c5cc859a935b32e45))

### Testing

- Cover the index_to_device op path, not just the no-op
  ([`64484d8`](https://github.com/EleutherAI/bergson/commit/64484d8e2968ad8eb7847ef35b32fa7b2ce50ba2))


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


## v0.7.2 (2026-03-04)


## v0.7.1 (2026-03-03)

### Bug Fixes

- Always compute mixing coefficient in Trackstar pipeline
  ([`c990375`](https://github.com/EleutherAI/bergson/commit/c990375e69d309f348c489f9bfc9cf9cddc28f6d))


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

- Convert PyArrow columns to list at callsites of allocate_batches
  ([`5d734dc`](https://github.com/EleutherAI/bergson/commit/5d734dc23bb083819890ca17d1b44f377ae35d69))

- Remove redundant zero-fill loop in MemmapSequenceScoreWriter
  ([`558829f`](https://github.com/EleutherAI/bergson/commit/558829f717f8679d517765d5c3d9beac2f2249b2))

- Use [:] instead of list() for consistency
  ([`c76d131`](https://github.com/EleutherAI/bergson/commit/c76d131c357b6b8e7880da48b4640510ffe5a654))


## v0.6.1 (2026-03-02)

### Bug Fixes

- Unpin transformers by explicitly setting float32 dtype in tests
  ([`0b6c226`](https://github.com/EleutherAI/bergson/commit/0b6c22615b7cce4ca62f71cb93847e3027fa68ba))


## v0.6.0 (2026-02-17)

### Bug Fixes

- Use _csv._writer type for csv_recorder annotation
  ([`6e6289c`](https://github.com/EleutherAI/bergson/commit/6e6289c266b36304a6d79a35bb6b9fe3c35fa95a))

### Continuous Integration

- Pin pyright version and fix faiss type error
  ([`b9f54cf`](https://github.com/EleutherAI/bergson/commit/b9f54cf9e7caf3c13af78f1a2d3d766f2055c3da))

- Use Python 3.11 for typechecking
  ([`9ef4122`](https://github.com/EleutherAI/bergson/commit/9ef4122903eed2ecf496f803c5d1aba4c62295cb))

- Use Python 3.11 for typechecking
  ([`ea50dd8`](https://github.com/EleutherAI/bergson/commit/ea50dd8ed9dc02b0f21ce7621f7d0ff53622ea87))

### Features

- Add --record flag to query CLI for saving results to CSV
  ([`59770ff`](https://github.com/EleutherAI/bergson/commit/59770ff88c5dbfffabd6ce0f51e5a56edbae2c0b))

### Refactoring

- Replace try/finally CSV block with context manager
  ([`6431320`](https://github.com/EleutherAI/bergson/commit/6431320b7c167191b157b3fc53013818ecdd5135))


## v0.5.2 (2026-02-17)

### Bug Fixes

- Pass batches to CollectorComputer in fit_normalizers
  ([`c95d5d4`](https://github.com/EleutherAI/bergson/commit/c95d5d498ad900af8a95902535fdfe740696088f))

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
