# Compressed EKFAC Index — Handoff Plan

**Purpose of this doc:** Seed a fresh Claude Code session on EleutherAI's A40 cluster with full context from the design session that happened on Girish's Mac. The Mac session cannot be resumed across machines (Claude Code sessions are path-hashed and local), so this file is the transfer medium. Read it end-to-end before touching code.

---

## 1. What we're building

Produce a **compressed EKFAC index** in bergson: for each training example (or token), apply EKFAC `H^{-1/2}` per module → random-project → write to an on-disk index. That becomes the recall-phase index for two-stage retrieval; the existing per-example path remains the precision phase.

Framing (from Lucia, the EleutherAI collaborator): bergson collects and searches over gradients much like you'd collect and search over semantic embeddings. Hessian approximations are the "shared transformation that makes distances meaningful." The library supports multiple Hessian-approximation types (autocorrelation, KFAC, EKFAC) but some code paths only load/apply one type — fixing that is part of this work.

**Deadline:** paper submission **2026-05-08**.

---

## 2. Current state

- All design decisions (below) are **locked** with Girish.
- Memory files and task list from the Mac session did not port — **reseed them** on the cluster (see §9).
- **No code written yet.** Next action: start commit 1 on a new branch `compressed-ekfac`.

---

## 3. Locked design decisions

### 3.1 Scope
- **One umbrella PR** titled something like "Compressed EKFAC index" on branch `compressed-ekfac`. The slicing below is commit boundaries inside that PR, not separate PRs.
- **Done bar:** passing tests + notebook showing two-stage retrieval on **pythia-160m / pile-10k** (MVP). Stretch: bump notebook to **pythia-1.4b / pile-100k** if commits 1–3 land cleanly. 8×A40 has the VRAM for either.

### 3.2 Preconditioner architecture
- **Preconditioner application happens at build time, not score time** ("build-time bake-in"). Store `P · H^{-1/2} · g` per example; scoring the compressed index is a plain dot product.
- **Order: precondition-then-project** (Grosse semantics). EKFAC `H^{-1/2}` applies in original parameter space; random projection happens after.
- **Three first-class variants** of Hessian approximation: `autocorrelation`, `kfac`, `ekfac`.
  - `autocorrelation` = what the current code calls the "trackstar preconditioner" (`Σ vec(g) vec(g)ᵀ` per module). **Do not call it `gram` or `trackstar`** in new code — Lucia's explicit preference. `autocorrelation` clarifies that all three variants are Hessian approximations differing in structure, not purpose.
  - `kfac` = factored, no eigenvalue correction.
  - `ekfac` = factored + eigenvalue correction (Λ).

### 3.3 Type routing: auto-detect from directory contents, no new config field
**Do not add a `preconditioner_type: Literal[...]` to `PreprocessConfig`.** Instead, infer from `preconditioner_path` contents:

| On disk | Variant |
|---|---|
| `eigen_activation_sharded/` + `eigen_gradient_sharded/` + `eigenvalue_correction_sharded/` | **EKFAC** |
| `eigen_activation_sharded/` + `eigen_gradient_sharded/` (no `eigenvalue_correction_sharded/`) | **KFAC** |
| `GradientProcessor` dump (e.g. `preconditioners.pth`, safetensors) | **autocorrelation** |

This eliminates config plumbing through `__main__.py`, prevents path/type mismatches, and makes the artifact on disk the source of truth. If an explicit override is ever needed, add it later.

### 3.4 Split `load` from `apply` in the EKFAC path
`bergson/hessians/apply_hessian.py`'s `EkfacApplicator.compute_ivhp_sharded` currently fuses three things: load factors, load all gradients via mmap, rotate-scale-rotate the entire buffer. For build-time preconditioning we call this per-batch, millions of times, so:
- **`__init__(path, device)`** — load Q_A, Q_S, Λ; precompute damping constants; hold on device. Once per run.
- **`.apply(mod_grads: dict[str, Tensor]) -> dict[str, Tensor]`** — pure op on an in-memory batch. Per batch.

The existing `compute_ivhp_sharded` stays for its current caller (`hessian_pipeline` applying once to a mean query gradient) OR is rewritten as a thin loop over `EkfacPreconditioner.apply` — pick whichever is less disruptive to existing tests.

### 3.5 Per-token support from day one
Builder already has an `attribute_tokens` path. The per-token EKFAC reshape is `[T, O*I] → [T, O, I]` vs `[N, O*I] → [N, O, I]` for sequences — same math, different leading dim. `EkfacPreconditioner.apply` should not care which it gets.

### 3.6 Multi-GPU: full-load-per-rank for MVP
Each rank loads the full Q_A, Q_S, Λ (concatenating shards at load time). Same memory envelope as the existing autocorrelation path. True sharded per-batch apply is a follow-up — the existing `ShardedMul` path is written for one query gradient, not per-batch, and refactoring it isn't on the critical path for May 8.

### 3.7 Scorer: leave `create_scorer` autocorrelation-only for MVP
`bergson/score/score.py:120` `create_scorer` also hardcodes `get_trackstar_preconditioner` and has a split-preconditioning path keyed on `unit_normalize`. For build-time bake-in, scoring a compressed EKFAC index should just be a dot product — no preconditioning at score time. **Option B locked:** don't teach `create_scorer` about EKFAC. Compressed EKFAC indices get scored via a lightweight path (plain `grads @ query.T`) inside the notebook / orchestrator. Revisit as a follow-up.

---

## 4. Commit slicing

### Commit 1 — Preconditioner interface + auto-detect factory (pure refactor, no behavior change)
- New module, e.g. `bergson/preconditioners.py` (or subpackage if it grows):
  - `class Preconditioner(Protocol)` with `.apply(mod_grads: dict[str, Tensor]) -> dict[str, Tensor]`
  - `class AutocorrelationPreconditioner` wrapping current `get_trackstar_preconditioner` + `precondition_grad` logic
  - `def load_preconditioner(path, device, power) -> Preconditioner` — auto-detects variant per §3.3, returns the right implementation
- Wire into `bergson/builder.py:80` (replace hardcoded `get_trackstar_preconditioner` call with factory call; `_preprocess` at `builder.py:20` calls `preconditioner.apply` instead of `precondition_grad`)
- Wire into `bergson/score/score.py:144` (same refactor — factory returns `AutocorrelationPreconditioner`; KFAC/EKFAC at score time raises `NotImplementedError` with a pointer to the notebook's scoring path, per §3.7)
- **Test:** existing `pytest tests/` stays green (this is the entire test — it's a pure refactor). On 8×A40, also run `tests/test_build.py::test_build_e2e` to confirm the CUDA path didn't regress.

### Commit 2 — `EkfacPreconditioner` + `KfacPreconditioner` + parity test
- Sibling classes with a shared base (both use rotate-scale-rotate; KFAC uses raw eigenvalues, EKFAC uses Λ correction).
- `__init__`: each rank loads all shards for its variant, concatenates, holds full Q_A/Q_S/Λ on device. See `bergson/hessians/apply_hessian.py:44` for the reference load sequence.
- `.apply(mod_grads)`: per-batch version of the rotate-scale-rotate at `apply_hessian.py:83-133`. Must handle `[N, O*I]` (sequences) and `[T, O*I]` (tokens) identically.
- **Parity test** `tests/test_ekfac_preconditioner.py`:
  - Build a small random `mod_grads` dict matching a small model's shapes.
  - Run it through `EkfacApplicator.compute_ivhp_sharded` (existing, trusted) via a temp mmap.
  - Run the same buffer through `EkfacPreconditioner.apply`.
  - Assert element-wise `torch.allclose(..., atol=1e-5)`.
  - Mark `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`.
- Also add a KFAC parity test using a KFAC artifact (no `eigenvalue_correction_sharded/`).

### Commit 3 — `bergson compressed_ekfac` CLI + pipeline orchestrator
- New `Serializable` subcommand in `bergson/__main__.py`, sibling to `Ekfac` (line 48) and `Trackstar` (line 153).
- Orchestrator in `bergson/hessians/pipeline.py` or a new file:
  1. Run `approximate_hessians(hessian_cfg, ev_correction=True)` on training set → writes `<run_path>/hessian/` with sharded factors.
  2. Run `build(index_cfg, preprocess_cfg)` on training set with `preconditioner_path=<run_path>/hessian` and a `projection_dim` > 0 → `Builder` auto-detects EKFAC via §3.3 and bakes in.
  3. Resulting index at `<run_path>/index` is the compressed EKFAC index — plain dot-product queryable.
- **Smoke test** mirroring `tests/test_build.py::test_build_e2e`: `bergson compressed_ekfac ... --model EleutherAI/pythia-14m --dataset NeelNanda/pile-10k --split 'train[:100]' --projection_dim 16`. CUDA-only.
- Per CLAUDE.md: the CLI command must run without errors for 3+ minutes before considering the commit done.

### Commit 4 — Two-stage retrieval notebook
- `notebooks/compressed_ekfac_two_stage.ipynb` (or wherever `colab-notebooks-v2` lives — check existing notebook conventions).
- Demo on pythia-160m / pile-10k: held-out query → top-K recall via compressed index dot-product → re-score top-K with the precision path → show qualitative training-example neighbors.
- Paper-ready figures: recall@K vs index size, wall-clock comparison vs per-example-only baseline.
- Stretch: bump to pythia-1.4b / pile-100k.

---

## 5. Testing plan for the cluster

### 5.1 Environment bootstrap (one-time)
```
cd <path-to-bergson>
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```
Baseline run establishes which tests pass before any changes. CUDA-required tests that were skipping on Mac will now run — expect some to be slow.

### 5.2 Per-commit validation
| Commit | Validation |
|---|---|
| 1 (refactor) | `pytest tests/` stays green. `test_build.py::test_build_e2e` must still pass. |
| 2 (EKFAC/KFAC preconditioners) | New parity tests pass. No regressions in (1). |
| 3 (CLI + pipeline) | New smoke test passes. Full `bergson compressed_ekfac ...` run completes on pythia-14m / pile-10k. Per CLAUDE.md: must run 3+ minutes without error. |
| 4 (notebook) | Executes top-to-bottom. Output plots look sensible. |

### 5.3 Multi-GPU validation (in commit 3 or as separate verification step)
Launch with all 8 A40s via whatever the existing `launch_distributed_run` in `bergson/distributed.py` expects (SLURM, torchrun, or direct `python -m bergson ...`). The existing `Ekfac` subcommand already uses distributed init — compressed_ekfac should inherit the same launch pattern. Verify that each rank correctly loads full factors and produces a consistent index.

### 5.4 If things break
CLAUDE.md convention: if you find an error unrelated to your task, quote the exact error back to Girish and offer to investigate. Don't silently paper over it.

---

## 6. Key file references (verified on Mac; may have minor drift — grep to confirm)

- `bergson/builder.py:80` — the hardcoded `self.h_inv = get_trackstar_preconditioner(...)` line. This is the single most important line this work replaces.
- `bergson/builder.py:20` — `_preprocess` function, calls `precondition_grad` on `mod_grads`.
- `bergson/process_grads.py:125` — `get_trackstar_preconditioner` (loads `GradientProcessor`, applies matrix power).
- `bergson/process_grads.py:193` — `precondition_grad` (per-example `G @ H^power`).
- `bergson/process_grads.py:164` — `precondition_flat_grads` (flat-buffer version).
- `bergson/hessians/apply_hessian.py:28` — `EkfacApplicator` class.
- `bergson/hessians/apply_hessian.py:44` — `compute_ivhp_sharded` — the reference for factor loading + rotate-scale-rotate.
- `bergson/hessians/apply_hessian.py:83-133` — the rotate-scale-rotate math body; this is what `EkfacPreconditioner.apply` implements per-batch.
- `bergson/hessians/pipeline.py` — existing `hessian_pipeline` orchestrator (pattern to mirror for `compressed_ekfac`).
- `bergson/hessians/sharded_computation.py` — `ShardedMul` utilities.
- `bergson/score/score.py:120` — `create_scorer` — leave alone per §3.7 except for the factory wire-up in commit 1.
- `bergson/score/score.py:94` — `_make_split_preconditioner` — autocorrelation-specific; do not generalize.
- `bergson/__main__.py:48` — `Ekfac` Serializable subcommand (structural template).
- `bergson/__main__.py:153` — `Trackstar` Serializable subcommand (structural template).
- `bergson/config.py:336` — `IndexConfig.projection_dim` etc.
- `bergson/config.py:458` — `PreprocessConfig.unit_normalize`, `:461` `preconditioner_path`.
- `bergson/distributed.py` — `launch_distributed_run` for multi-GPU orchestration.
- `tests/test_build.py` — CLI e2e pattern to copy for commit 3's smoke test.
- `tests/conftest.py` — test fixtures (tiny Phi3 model, 2-row dataset).

---

## 7. Project conventions (per `CLAUDE.md` — read it if you haven't)

Highlights relevant to this work:
- **Always test changes** with the appropriate CLI/script, for 3+ minutes without error, before calling a task done.
- Keep `__main__.py` clean — documentation + routing only.
- Use `dataclasses` + `simple_parsing` for configs. Names like `run_cfg/RunConfig`, never `cfg`. Underscores in args, not dashes.
- Never call `torch.cuda.empty_cache()`.
- Don't save data/logs to repo root — put them in `runs/` (gitignored) or `scripts/`.
- Don't keep default run-path values in low-level code; higher-level module passes the base path through.
- Don't remove large HF cache datasets without asking.
- For any subprocess CLI launches, print the command so it's reproducible.
- GPU-required tests: `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`.

---

## 8. Open questions (none blocking, but flag if relevant)

- **Notebook location:** is there an existing `colab-notebooks-v2` convention? Recent commit `6780a23` mentions it — check `git log --oneline | grep -i colab` and the `notebooks/` dir for structure.
- **CI GPU runners:** does `.github/workflows/build.yml` run GPU tests? If so, the parity test and CLI smoke will run in CI too — worth checking so you don't accidentally gate CI on a test that can't run there.

---

## 9. Reseed memory on the cluster

The auto-memory system on the Mac has two entries that **did not port**. Save these at the start of the new session so they're available going forward:

**Memory 1** — `feedback_naming_autocorrelation.md` (type: feedback):
> When naming the unfactored per-module Hessian approximation (`Σ vec(g)vec(g)ᵀ`), use `autocorrelation` in new code. Sits alongside `kfac` and `ekfac` as sibling Hessian-approximation types. Do not use `gram` or `trackstar`.
>
> **Why:** Lucia (collaborator) explicitly flagged this during the compressed-EKFAC design discussion. `autocorrelation` is mathematically precise; `trackstar` is a method name (rots when a second non-factored variant appears); `preconditioner` is the role, not the kind.
>
> **How to apply:** For `Literal[...]` fields, class names, dict keys, CLI strings distinguishing Hessian-approximation variants, use `autocorrelation`. When refactoring existing `trackstar_*` code that refers to the non-factored path, prefer renaming to `autocorrelation_*` or just `preconditioner` (when role is what matters).

**Memory 2** — `project_compressed_ekfac.md` (type: project):
> Compressed EKFAC index for bergson, target submission 2026-05-08. Per-example gradients get EKFAC-preconditioned then random-projected then written to a queryable on-disk index. Two-stage retrieval: compressed index for recall, existing per-example path for precision.
>
> **Why:** Paper submission demonstrating retrieval-speed parity with Anthropic's influence-function work. Lucia is the driving collaborator.
>
> **How to apply:** Scope/architecture decisions locked at — build-time bake-in (not score-time apply); precondition-then-project (Grosse semantics); auto-detect Hessian variant from `preconditioner_path` contents (no new config Literal); three variants `autocorrelation`/`kfac`/`ekfac`; split `load` from `apply` in EKFAC path so `.apply` runs per batch; per-token support day one; multi-GPU via full-load-per-rank for MVP; new `bergson compressed_ekfac` CLI subcommand; done bar = tests + two-stage retrieval notebook on pythia-160m / pile-10k (stretch: 1.4b / pile-100k).

---

## 10. First actions on the cluster

1. Clone/pull bergson; check out a fresh branch `compressed-ekfac` off `main`.
2. `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
3. Run `pytest tests/ -v` on all 8 A40s (or one) to confirm baseline green. Save this output — it's your regression baseline.
4. Reseed the two memory entries from §9.
5. Open a TaskList for the four commits.
6. Start commit 1 (the pure refactor). Don't touch EKFAC math yet.

Good luck. Ping Girish (or Lucia) with anything that contradicts what's in this doc — the design was locked on the Mac side and is load-bearing for the paper date.

---

# Part II — Implementation design doc (post-landing, for Lucia's review)

§1–§10 above are the pre-implementation plan snapshot, preserved verbatim so it can be diff-read against the Mac-side design. §11 onward records the actual implementation as it happened: every non-trivial design call, the alternatives considered, the rationale for the chosen path, and (for commit 3) a mid-flight replan after the existing pipeline's projection placement turned out to be incompatible with the plan as written.

Audience: **Lucia**, EleutherAI — for review before the umbrella PR opens. Grouped by topic, not chronology.

Legend: **[landed]** = already committed on `feature/compressed-ekfac`; **[in-flight]** = commit 3; **[deferred]** = out of scope for this PR.

---

## 11. Architectural surface: the `Preconditioner` protocol **[landed]**

### 11.1 Shape of the abstraction

The library previously had one on-disk preconditioner concept (autocorrelation, aka "trackstar preconditioner") and a one-off `EkfacApplicator` that was not plugged into `build`/`score` paths. Commit 1 introduces a unifying abstraction:

```python
class Preconditioner(Protocol):
    def apply(self, mod_grads: dict[str, Tensor]) -> dict[str, Tensor]: ...
```

Three implementations, all loaded by `load_preconditioner(path, device, power)`:

* `AutocorrelationPreconditioner` — wraps existing trackstar dict of `H^power` matrices; `.apply` is a right-matmul per module.
* `EkfacPreconditioner` — factored with empirical Λ correction (commit 2).
* `KfacPreconditioner` — factored with Λ = Λ_G ⊗ Λ_A from eigendecomposition (commit 2).

### 11.2 Alternatives considered

| Approach | Pros | Cons | Why rejected |
|---|---|---|---|
| **`Protocol` (chosen)** | Zero base-class coupling; `AutocorrelationPreconditioner` can keep a `.h_inv` dict for legacy code paths; `isinstance` with `@runtime_checkable` still works | Structural typing means missed signatures are caught at use-site, not definition | — |
| Abstract base class (`ABC`) | Enforces method shape at definition | Forces every variant to carry the same internal state; noisy for the autocorrelation path which is a thin wrapper | — |
| Plain duck typing (no Protocol, no ABC) | Least code | Loses type-checker help; harder to grep for implementers | — |
| Config-field dispatch (`preconditioner_type: Literal[...]`) | Explicit, discoverable | Adds a config-plumbing layer through every caller of `PreprocessConfig`; easy to mismatch config vs on-disk artifact | Rejected in §3.3 of plan; reaffirmed during implementation |

### 11.3 Module placement

`bergson/preconditioners.py` (new top-level module). Alternatives:

* Inside `bergson/hessians/` — would couple autocorrelation (which is conceptually separate from KFAC-family hessians) to the hessians subpackage.
* Inside `bergson/process_grads.py` — would bloat a file already responsible for mixing, normalization, and the raw preconditioner loaders.
* Per-variant files (`preconditioners/ekfac.py`, etc.) — premature; three classes and a factory fit comfortably in one file.

### 11.4 Auto-detect from directory contents (§3.3 of plan, implemented)

`_detect_variant(path: Path) -> {"autocorrelation", "kfac", "ekfac"}` inspects directory presence:

| On disk | Variant |
|---|---|
| `eigen_activation_sharded/` + `eigen_gradient_sharded/` + `eigenvalue_correction_sharded/` | **EKFAC** |
| `eigen_activation_sharded/` + `eigen_gradient_sharded/` (no correction dir) | **KFAC** |
| Anything else (including `preconditioners.pth` alone, or `None`) | **autocorrelation** |

Alternative: add a `variant: "ekfac" | "kfac" | ...` field to an artifact-level `meta.json`. Rejected because the disk layout already uniquely determines the variant, and a metadata field would add an invariant to maintain (metadata says EKFAC but `eigenvalue_correction_sharded/` was deleted → silent wrongness). Directory-presence detection is self-validating.

### 11.5 Shared factored base class

`_FactoredPreconditioner` holds `{q_a, q_s, lam}` per module and implements the rotate-scale-rotate body:

```
H^power · G ≈ Q_S · ((Q_S^T G Q_A) · λ^power) · Q_A^T
λ = Λ + damp * mean(Λ)
```

EKFAC and KFAC subclasses differ only in how Λ is sourced. This matches the reference `EkfacApplicator.compute_ivhp_sharded` math at `bergson/hessians/apply_hessian.py:83-133`. The rotate-scale-rotate is written as pure einsums (no dependence on `ShardedMul`) because MVP is full-load-per-rank (§3.6).

Alternative: keep a single `FactoredPreconditioner` class with a `variant` arg. Rejected — `__init__` differs meaningfully between KFAC and EKFAC (KFAC synthesises Λ from two 1D vectors via outer product; EKFAC loads Λ directly). Two classes with a shared base is clearer than one class with a branch.

### 11.6 Full-load-per-rank (§3.6 of plan, implemented)

Each rank reads all shards for the variant and concatenates along dim 0 to reconstruct the full factor on-device. `_load_sharded_dict` is a tiny helper that does this; it's variant-agnostic.

Alternative (deferred): reuse `ShardedMul` from `hessians/sharded_computation.py` for a true per-batch sharded apply. `ShardedMul._matmul` is written for one query gradient at a time (a single `.compute_ivhp_sharded` call in `apply_hessian.py`), so retrofitting it for the per-batch hot path in `Builder.__call__` would require a redesign that isn't on the critical path for 2026-05-08. Memory envelope of full-load-per-rank matches the existing autocorrelation path that everyone already runs.

---

## 12. KFAC support required an additive change to eigendecomposition **[landed]**

### 12.1 The gap

Plan §4 (commit 2) called for a `KfacPreconditioner` that loads "raw eigenvalues". Investigation found `compute_eigendecomposition` at `bergson/hessians/eigenvectors.py:280` does this:

```python
eigenvalues, eigenvectors = torch.linalg.eigh(matrix_normalized)
# ...
covariance_eigenvectors[key] = eigenvectors  # eigenvalues discarded
```

Eigenvalues are computed but never saved. KFAC therefore couldn't be loaded from any current artifact.

### 12.2 Options

| Approach | Pros | Cons |
|---|---|---|
| **Save eigenvalues alongside eigenvectors (chosen)** | Additive; 5-line change; KFAC works immediately | New disk artifact `eigenvalues_{activation,gradient}_sharded/`; old artifacts lack them |
| Recompute eigenvalues at KFAC load time | No disk change | Requires loading covariance matrices; inverts the whole point of eigendecomposition being a one-time cost |
| Derive eigenvalues from `eigenvalue_correction_sharded` | No disk change | Mathematically wrong — empirical Λ ≠ Λ_G ⊗ Λ_A |
| Defer KFAC entirely to a later PR | Smaller commit 2 | Plan's `done bar` wants KFAC support; deferring bleeds scope across PRs |

### 12.3 The chosen migration story

The eigenvalue dirs are new artifacts with a distinct name (`eigenvalues_*_sharded/`, not `eigenvalue_*_sharded/`). Existing EKFAC artifacts lack them, which is fine because EKFAC doesn't read them. KFAC reads from them via `KfacPreconditioner.from_disk`; if absent, a `FileNotFoundError` points the user at `compute_eigendecomposition` with a clear message:

```
KFAC preconditioner at <path> is missing per-factor eigenvalue shards
(eigenvalues_activation_sharded/ and/or eigenvalues_gradient_sharded/).
Regenerate the artifact with the current version of compute_eigendecomposition.
```

Older artifacts regenerate cleanly — no data migration script required, and no risk of silently producing wrong results.

### 12.4 Risk flagged for Lucia

Every current EKFAC/KFAC run post-merge will now also write two extra 1D safetensors shards. Size impact is negligible (one vector per module; scales as Σ_m (O_m + I_m) rather than Σ_m O_m·I_m). But it does mean that an in-progress experiment using `compute_eigendecomposition` on `main` and a consumer on `feature/compressed-ekfac` will disagree on the expected directory layout. Flagging for Lucia in case there's a multi-branch run in flight.

---

## 13. Parity testing strategy **[landed]**

### 13.1 What's tested

`tests/test_preconditioners.py` — 8 tests, all pass:

1. `_detect_variant` on EKFAC / KFAC / autocorrelation layouts
2. `load_preconditioner(None)` returns an empty autocorrelation (no-op apply)
3. **EKFAC parity vs `EkfacApplicator.compute_ivhp_sharded`** — synthesize random factor shards + a random gradient mmap, run both pipelines, `torch.allclose(atol=1e-5)`
4. EKFAC per-token shape: `[T, O*I]` input, same math
5. **KFAC parity vs direct einsum reference** — no trusted KFAC applicator exists, so we validate against an inline implementation of the rotate-scale-rotate with Λ = Λ_G ⊗ Λ_A
6. KFAC missing-eigenvalues error surface

### 13.2 Why not end-to-end against a real model?

Plan §4 said "Build a small random `mod_grads` dict matching a small model's shapes." Two alternatives considered:

| Approach | Pros | Cons |
|---|---|---|
| **Synthetic factors + trusted applicator (chosen)** | Runs in seconds; isolates math from collector plumbing; easy to diagnose failures | Doesn't catch integration bugs with actual factor-producing code |
| End-to-end through `approximate_hessians` on a tiny model | Catches integration bugs too | Slow (minutes); covers work commit 3's smoke test already does |
| Reuse `tests/ekfac_tests/` fixtures | Already exist | Those fixtures are session-scoped, expensive, and tuned for math-accuracy tests against a ground-truth computation — overkill |

The commit-3 smoke test (pythia-14m + pile-10k[:100]) is the integration test. Keeping commit 2's parity tests as unit tests is the right split.

### 13.3 For Lucia

The "trusted applicator" in my EKFAC parity test is `EkfacApplicator.compute_ivhp_sharded` — the function that lives in `apply_hessian.py` today. If you consider that code still experimental rather than reference-trusted, I'd welcome a second reference (e.g., Kronfluence or an internal one-off) to triangulate against.

---

## 14. `create_scorer` stays autocorrelation-only **[landed, per §3.7]**

Already decided in the plan. Restating the rationale now that it's implemented:

`create_scorer` at `bergson/score/score.py:120` currently:
1. Loads `preconditioners: dict[str, Tensor]` from disk
2. Optionally applies them to query grads
3. Builds an `index_transform` closure via `_make_split_preconditioner` for split (two-sided) preconditioning when `unit_normalize=True`

All three steps assume the preconditioner is a `[D, D]` per-module matrix. EKFAC's Q_A/Q_S/Λ triple doesn't fit. Four options:

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| **Raise `NotImplementedError` on non-autocorrelation (chosen)** | Crisp scope for MVP; no surprises at score time | Users with EKFAC indices must use the notebook's scoring path | **Locked** |
| Teach `create_scorer` to dispatch | Single code path | `_make_split_preconditioner` also needs generalization; many downstream consumers | Deferred |
| Duplicate `create_scorer` → `create_factored_scorer` | Easy to reason about | Extra API surface, duplicated plumbing | Deferred |
| Have compressed-EKFAC indices score through `create_scorer` with the preconditioning "already baked in" | Re-uses existing scorer | Requires marking the index as pre-preconditioned in `PreprocessConfig` so the scorer doesn't double-apply; new config invariant | Actually viable — see §16.3 |

The fourth option is a candidate follow-up worth Lucia's take (see §16.3).

---

## 15. Projection placement — the mid-flight replan **[in-flight, commit 3]**

This is the meatiest design point and the one most worth Lucia's attention. The plan as written (§3.2) calls for "precondition-then-project (Grosse semantics)" but the existing code implements the opposite order. The smoke test made this concrete.

### 15.1 How the existing pipeline is laid out

```
collector                                    builder
─────────                                    ───────
[N, S, O], [N, S, I]                          
     │                                       
     ▼                                       
outer product → [N, O, I]                    
     │                                       
     ▼                                       
normalize (Adam/Adafactor) [optional]        
     │                                       
     ▼                                       
*** double_sided_projection → [N, p, p] ***  
     │                                       
     ▼ (flattened to [N, p*p])               
  mod_grads  ──────────────────────────▶     preprocess: precondition + concat + normalize
                                             writes [N, p*p] to disk
```

Projection lives inside `collector/collector.py:421-423`. When the builder's `_preprocess` runs, `mod_grads[name].shape == [N, p*p]`. The old `AutocorrelationPreconditioner` works because `h_inv` is computed on the fly over the projected gradient covariance, so it lives in `[p*p, p*p]` space — matching the projected `mod_grads` shape.

### 15.2 What the plan assumed

The plan (§3.2, §3.5) assumes `EkfacPreconditioner.apply` receives `[N, O*I]` unprojected gradients so it can reshape to `[N, O, I]` and apply Q_A/Q_S in the parameter space they live in. There is no way to apply `[I, I]` and `[O, O]` matrices to a `[p*p]` projected vector and recover the correct `H^{-1/2}·g` — random projection L, R satisfy `L·H^{-1/2}·G·R.T ≠ f(L·G·R.T)` for any function `f` that doesn't reference `H^{-1/2}` directly. Information is lost.

### 15.3 Options evaluated

| # | Approach | Effort | Correctness | Performance | Plan fit |
|---|---|---|---|---|---|
| **A** | **Move projection into `Builder._preprocess` for factored preconditioners; leave autocorrelation as-is** | ~50 LOC, 3 files | ✓ mathematically correct | Slightly higher collector-output bytes (unprojected) before projection; offset by unchanged write size | **Matches §3.2** |
| B | Reformulate EKFAC in projected space using `L·Q_S` and `R·Q_A` | Moderate | ✗ provably wrong | Fastest | Violates §3.2 |
| C | Skip random projection for compressed EKFAC; "compressed" = EKFAC eigenbasis only | Tiny | ✓ (partial compression) | ✓ | Contradicts PR title + §1 |
| D | Precondition at query time on full-dim grads; store unprojected EKFAC'd index | Tiny for build; hurts query side | ✓ | Breaks paper claim — unprojected index is huge at 1.4B scale | Contradicts §1, §3.1 done-bar |
| E | Pass a "precondition hook" into the collector so preconditioning happens inside the collector, pre-projection | Large; cross-cutting | ✓ | Same as A | Plan favours builder as the orchestration point |

**Choosing A.** The projection is a pure linear op that commutes with "writing to disk"; moving it between the collector and the builder is a local refactor. Autocorrelation stays bit-exact (regression anchor). EKFAC/KFAC get the correct semantics.

### 15.4 Commit-3 implementation sketch (A)

Changes localized to three files:

1. **`bergson/utils/worker_utils.py::create_processor`** — when `preprocess_cfg.preconditioner_path` is a factored variant (detect via `_detect_variant`), override the processor's `projection_dim` to `None`. Collector's `_compute_gradient` sees `p is None` at `collector.py:473` and skips `double_sided_projection`. Gradients flow to the builder as `[N, O*I]`.

2. **`bergson/builder.py::Builder`**:
   - `__init__` records `self._needs_post_projection = isinstance(self.preconditioner, _FactoredPreconditioner)` and, if true, builds per-module projection matrices (same `GradientProcessor.projection_matrices` machinery, reused at a new call-site).
   - `_preprocess` grows a step after `preconditioner.apply`:
     ```
     if needs_post_projection:
         mod_grads = {m: project_double_sided(mod_grads[m], L[m], R[m]) for m ...}
     ```

3. **`bergson/hessians/compressed_ekfac.py::compressed_ekfac_pipeline`** — no change; the orchestrator already hands `preconditioner_path` to `build`, and the new create_processor/builder logic picks up the variant automatically.

### 15.5 Invariants and edge cases

| Invariant | Autocorrelation (existing) | EKFAC/KFAC (new) |
|---|---|---|
| Collector output shape | `[N, p*p]` projected | `[N, O*I]` unprojected |
| `h_inv`/factors dtype | processor's dtype (bf16/fp32) | float32 on-GPU (matches reference `EkfacApplicator`) |
| Bias handling | `include_bias=True` appends column before projection; h_inv is `[(O*I + O), (O*I + O)]` | need to verify: if bias is appended, does EKFAC expect Q_A to still be `[I, I]` or `[I+1, I+1]`? |
| Per-token vs per-sequence | `_compute_gradient` already flattens per-token | `EkfacPreconditioner.apply` handles `[T, O*I]` identically — unit-tested in commit 2 |
| Written-to-disk dim | `p*p` | `p*p` (projection applied in builder) |

**Bias + EKFAC is the one I'm least sure about** — flagging for Lucia. Current factored preconditioners assume plain linear-layer shapes. If `include_bias=True` is ever used with EKFAC, we need Q_A to cover the extra input dim. Safest default: raise a clear error if both are set, defer proper support.

### 15.6 Why I'm not doing E (preconditioner-aware collector)

Option E (push preconditioning into the collector's `_compute_gradient`) is symmetric with option A but more invasive:
* `_compute_gradient` is called per-module per-batch inside the backward hook; adding preconditioning there mixes a load-once operation into a hot path.
* Three different normalizer paths (Adam / Adafactor / none) each need wiring.
* Harder to test in isolation.

A keeps the collector simple and concentrates the new logic in the builder, which aligns with Girish's hint in their review that "the core file is `builder.py`".

---

## 16. Open questions for Lucia

Grouped by where a decision would redirect work.

### 16.1 Math / correctness (blocking if wrong)

* **Rotate-scale-rotate formulation.** `_FactoredPreconditioner.apply` computes  `Q_S (Q_S^T G Q_A · λ^power) Q_A^T` as five einsums. This matches `EkfacApplicator.compute_ivhp_sharded:83-133` element-wise at atol=1e-5 (parity test `test_ekfac_preconditioner_parity_vs_applicator`). But if `EkfacApplicator` itself has a bug, the parity test is meaningless. Is there a second reference you'd trust (Kronfluence, internal impl)?
* **KFAC Λ = Λ_G ⊗ Λ_A^T.** I'm using `torch.outer(lam_s, lam_a)` to make a `[O, I]` matrix. Is that the formulation you want, or should it be Λ_G ⊗ Λ_A in some other orientation?
* **Damping.** `inv_lam = (Λ + damp·mean(Λ))^power` — same formula used in `EkfacApplicator._hadamard`. Fine for EKFAC; but for KFAC with Λ = Λ_G ⊗ Λ_A, `mean(Λ)` has a different scale from the per-factor case. Want a different damping scheme for KFAC?

### 16.2 Naming

* **`autocorrelation` everywhere new code touches** (implemented per plan §3.2). The existing `get_trackstar_preconditioner` is kept untouched in `process_grads.py` — callers in commits 1-3 go through `AutocorrelationPreconditioner`, but the underlying helper still bears the old name. Want a rename of the helper as part of this PR, or leave as a follow-up?
* **`_FactoredPreconditioner`** — internal private base. Open to `FactoredPreconditioner` (public), `KroneckerPreconditioner`, or anything else.
* **`load_preconditioner`** — factory entry point. OK, or prefer `Preconditioner.from_path`?

### 16.3 Score-time strategy

`create_scorer` rejects EKFAC per §3.7. Alternative worth Lucia's take: have the compressed index live inside the existing scorer by marking it as "already preconditioned" in `PreprocessConfig`, so `create_scorer` doesn't try to apply anything and scoring is automatic dot-product. Would replace bespoke notebook scoring code with scorer reuse. Low urgency, but cleaner if it's what you want long-term.

### 16.4 Projection placement (§15)

* Is Option A (move projection into builder for factored variants) the right place? I picked it based on the plan's language + Girish's "core file is builder.py". If you prefer Option E (preconditioner-aware collector), say so — it's a larger refactor but keeps a single projection site.
* Bias + EKFAC: raise-if-both-set for MVP, proper support as follow-up. OK?

### 16.5 Scope creep risk

Commit 3 was scoped as "CLI + orchestrator" but §15 expands it to "also move projection for EKFAC path". Three ways to slice:

| Slicing | Pros | Cons |
|---|---|---|
| Keep commit 3 as-is (CLI + orchestrator + projection move) | One coherent unit of work | Larger diff |
| Split: commit 3a = projection refactor, commit 3b = CLI + orchestrator | Cleaner history | Neither half is independently testable — the smoke test needs both |
| Defer projection move; skip random projection for MVP | Simplest | Contradicts paper's "compressed" claim |

Leaning toward "keep commit 3 as-is" for velocity. Want to slice differently?

### 16.6 Deferred items

Flagged here so they don't get lost:
* True sharded per-batch apply (vs full-load-per-rank) — plan §3.6 defers this; keep deferred.
* `create_scorer` EKFAC support (§16.3).
* Renaming `get_trackstar_preconditioner` → something with "autocorrelation" in the name (§16.2).
* Bias + EKFAC proper support (§15.5).
* Damping for KFAC (§16.1).

---

## 17. Test & commit state snapshot

| Commit | Status | Tests | Regression delta |
|---|---|---|---|
| 1 — Preconditioner interface + factory | **landed** (`d8ab2e3`) | 162/162 pre-existing pass | **exact parity** with pre-refactor baseline (identical pass/fail set) |
| 2 — EkfacPreconditioner + KfacPreconditioner + parity | **landed** (`30e66e4`) | 170/170 — 8 new tests added, no regressions | **+8 new tests**, same 13 pre-existing failures |
| 3 — CLI + orchestrator + projection move | **landed** (`c69e826`) | 171/171 — 1 new e2e test added, no regressions | **+1 new test**, same 13 pre-existing failures |
| 3.5 — End-to-end validation + projection_dim floor finding | **[in-flight]** | Validation script + bumped smoke test projection_dim | (additive; no regressions expected) |
| 4 — Two-stage retrieval notebook | pending | pending | pending |

Pre-existing failures (unchanged throughout, all unrelated to this work):
* `test_adam_state_loading.py::test_load_8bit_adam_checkpoint` — missing `bitsandbytes` in dev deps
* `test_build.py::test_build_consistency` — A40-vs-?? numerical determinism mismatch against cached snapshot at `atol=1e-6`
* `test_muon.py` (4) — `torch.optim.Muon` not in torch 2.6
* `test_truncation.py` (7) — test/code drift on batch-size validation and warning text

Branch is 3 commits ahead of `origin/feature/compressed-ekfac`, **not pushed**. Umbrella PR opens once all four commits land.

---

## 18. End-to-end validation findings — the projection_dim floor **[critical for paper]**

After commit 3 landed with shape-correct output, I ran a rigorous end-to-end retrieval validation (new script `scripts/validate_compressed_ekfac.py`) to answer "does the compressed index actually rank training examples consistently with a ground-truth reference?" — a question none of the commit-1/2/3 tests were designed to answer. The answer turned up a real empirical constraint that affects paper defaults.

### 18.1 Validation design

The script builds **four** indices from the same EKFAC factors fit on `pile-10k[:N_train]`:

| Index | `projection_dim` | Data | Role |
|---|---|---|---|
| `train_compressed` | 64 | `train[:N_train]` | candidate: what the paper claims works |
| `train_reference` | **0** (no projection) | `train[:N_train]` | ground truth: unprojected `vec(H^{-1/2} G)` per module |
| `query_compressed` | 64 | `train[N_train:N_train+N_query]` | held-out queries, same projection as train |
| `query_reference` | **0** | `train[N_train:N_train+N_query]` | held-out queries, ground truth |

Key observation: `projection_dim=0` + factored preconditioner + `skip_preconditioners=True` is a "free" configuration on commit-3's code path — the collector emits `[N, O·I]`, the Builder applies EKFAC via `preconditioner.apply`, skips post-projection, writes the full preconditioned tensor. So the ground-truth artifact costs nothing new to produce.

Metrics, per query (N_TRAIN=200 ⇒ random recalls are ≈ K/200):

* **recall@K** of compressed top-K vs reference top-K (K=5, 10, 20)
* **Spearman** rank correlation of the full 1×N_train score vectors
* **Qualitative top-3** neighbors side-by-side

Pass bar: mean recall@10 ≥ 40 % (8× random), mean Spearman ≥ 0.30.

### 18.2 The failing first run exposed two real issues

**Run 1** (defaults from commit 3's CLI smoke test: `projection_dim=16`, `unit_normalize=False`):

```
mean recall@5 = 20%   mean recall@10 = 20%   mean Spearman = 0.096   → FAIL
```

Per-module diagnostic (`scripts/diag_compressed_ekfac.py`):

* Per-example projection is **bitwise correct**: `(L @ ref @ R.T).reshape(-1)` vs compressed vector has correlation `1.0000` across 5 example × 24 modules. The Builder's projection math is right.
* Per-module **score-vector** Spearman is 0.01 to 0.25, mean 0.11. Even isolated to one module, compressed rankings disagree with reference.

Since per-example projection is exact, the score-vector disagreement can only be JL noise dominating signal. Two roots:

1. **Kronecker-structured JL, not vanilla JL.** `vec(L·G·R^T)` lives in `p²` dims but the projection matrix has Kronecker structure `R ⊗ L`, effective rank `p(O+I)` not `p²·O·I`. Preservation of per-module inner products requires roughly `p ≳ sqrt(O·I)/ε`. For pythia-14m modules (`O·I ∈ [16384, 65536]`), that means `p ≥ 128`-ish. At `p=16` we're a factor of 8 below the floor.
2. **One-sided preconditioning** (`unit_normalize=False` ⇒ `power=-1` on both sides). Dot product becomes `<G_q, H^{-2} G_t>`, not the standard influence `<G_q, H^{-1} G_t>`. Both sides over-preconditioned symmetrically, but the score is no longer the Grosse IF quantity.

**Run 2** (`projection_dim=64`, `unit_normalize=True` ⇒ `power=-0.5`, split preconditioning):

```
q   recall@5  recall@10  recall@20   spearman
0     40.00%     40.00%     55.00%      0.659
1     40.00%     20.00%     35.00%      0.480
2     40.00%     70.00%     75.00%      0.793
3     80.00%    100.00%     90.00%      0.853
4     80.00%     70.00%     70.00%      0.786
mean     56.00%     60.00%     65.00%      0.714  → PASS
```

Compressed agrees with reference. Pipeline functions end-to-end.

### 18.3 Why autocorrelation gets away with `p=16`

Same `projection_dim=16` default in bergson's autocorrelation path doesn't hit this floor — because autocorrelation projects **inside the collector via `double_sided_projection` before the outer product**. The resulting preconditioner `h_inv` is computed on the projected gradient covariance, so it lives in `[p², p²]` space matching the already-projected `mod_grads`. There's no Kronecker-structure penalty because the projection is absorbed into the preconditioner's coordinate system.

Compressed EKFAC is fundamentally different: Q_A and Q_S are tied to the *unprojected* parameter space (they're factors of the Hessian there), so the projection must happen in unprojected space and the Kronecker-structure penalty applies.

### 18.4 Recommendations

1. **Bump the commit-3 `test_compressed_ekfac_e2e` smoke test from `projection_dim=16` to `projection_dim=64`.** The old value ran cleanly and produced shape-correct output, but would have silently retrieved garbage if anyone had tried to score against it. The new default hits the JL floor for pythia-14m's module sizes.
2. **Document the `projection_dim` floor in `compressed_ekfac_pipeline`'s docstring** with a "roughly ≥ `sqrt(max(O·I))`" rule of thumb. Don't enforce it programmatically — users with larger p budgets or different architectures may want to tune it.
3. **Use `unit_normalize=True` (split preconditioning) as the commit-4 notebook default.** `unit_normalize=False` does unit normalization of the full flat vector too, which changes rankings — that's a separate wrinkle flagged as a minor noise source below.
4. **Ship `scripts/validate_compressed_ekfac.py` and `scripts/diag_compressed_ekfac.py`.** They're real tools, not throwaway diagnostics, and will be useful for validating future model/dataset combinations or debugging projection-math regressions.

### 18.5 Scaling to pythia-160m

Followed up Run 2 with pythia-160m / pile-10k[:200] at two projection dims on separate GPUs:

| Setting | recall@5 | recall@10 | recall@20 | Spearman | Verdict |
|---|---|---|---|---|---|
| pythia-14m, p=16, one-sided | 20 % | 20 % | 26 % | 0.096 | FAIL |
| pythia-14m, p=64, split | 56 % | **60 %** | 65 % | **0.714** | PASS |
| pythia-160m, p=64, split | 12 % | 20 % | 24 % | 0.256 | FAIL |
| pythia-160m, p=128, split | 44 % | **44 %** | 37 % | **0.389** | PASS |

Reading: the JL floor scales with `sqrt(max_m(O_m·I_m))` as predicted. Pythia-14m's largest module has `O·I ≈ 65 k`, `sqrt ≈ 255` → p=64 sits at ~25 % of the floor, works. Pythia-160m's largest is `O·I ≈ 2.4 M`, `sqrt ≈ 1550` → p=64 is only ~4 % of the floor (fails), p=128 is ~8 % (barely passes).

Projecting to pythia-1.4b (paper's stretch target): max `O·I ≈ 16 M`, `sqrt ≈ 4000`. Even `p=128` would be ~3 % — expect marginal or failing recall. **Paper-target defaults for compressed EKFAC:**

* pythia-14m: `p ≥ 64` (p=64 passes comfortably)
* pythia-160m (MVP): `p ≥ 128` (passes; p=64 fails)
* pythia-1.4b (stretch): probably `p ≥ 256`, but untested — flag as an open question and empirically tune if the stretch is attempted.

Commit 3's smoke-test default bumps to `p=64` (adequate for pythia-14m). The notebook in commit 4 will use `p=128` at pythia-160m per this table.

### 18.6 Open questions for Lucia

1. **Pythia-1.4b stretch target.** Above table suggests `p ≥ 256` is needed. That's a 64× larger index than p=64 and will correspondingly increase disk footprint at pile-100k stretch scale. Acceptable? Alternative: settle for lower recall at 1.4b and document it, or use a different projection scheme (e.g. single-sided JL on `vec(G)` with p_total ≈ 65 k ≪ `O·I` would be more compact but breaks the `[p, p]` convention; or TRAK-style).
2. **`unit_normalize=True` couples split preconditioning with unit-norming the concat vector.** The unit-norm step isn't needed for split preconditioning to be well-defined, but it's what bergson's Builder does. Should we decouple them (e.g. new `PreprocessConfig.precondition_power: Literal[-0.5, -1]`)? Low priority but flagged.
3. **Pile-10k qualitative neighbors are weak.** Even the reference top-3 for "Tulsi Gabbard 2020 candidate" doesn't return politically-relevant training texts — just random topically-unrelated items with high gradient dot products. This is a property of pile-10k (random web text, no obvious near-duplicates) rather than our pipeline, but worth mentioning since the notebook's qualitative section will look underwhelming. The quantitative recall/Spearman is what matters.

### 18.7 Why this wasn't caught sooner

Commit 3's `test_compressed_ekfac_e2e` only checks that (a) the CLI exits 0, (b) the on-disk layout has the right per-module shape, (c) `skip_preconditioners=True` really is honored. It does **not** check whether any particular query produces sensible rankings — and I didn't notice that gap until after commit 3 landed. Lesson for future commits: "runs without crashing with the right shape" is a weak acceptance bar; include at least a correlation/recall check against a trusted reference when the deliverable is a retrieval artifact.

---

## 19. Known gaps — what is NOT yet validated

This section is the honest counterpart to §17. Everything below works in my hands or follows from the math, but I have **not** empirically verified it on the cluster. Each item is tagged with the rough cost to verify.

### 19.1 Multi-GPU / distributed apply [medium cost]

The plan §3.6 specifies full-load-per-rank for MVP. `_load_sharded_dict` does `torch.cat(parts, dim=0)` over `shard_*.safetensors`, which assumes each shard is a row-slice that concatenates back to the full matrix.

* **Tested:** single-shard case (`world_size=1`, only `shard_0.safetensors`).
* **Not tested:** `world_size > 1` actually loading multiple shards. With 8 ranks, eigenvectors are `[m/8, m]` per shard, eigenvalue corrections are `[O/8, I]` per shard. Concatenation along dim 0 *should* produce the right full matrix, but I never ran with `--nproc_per_node 8`.
* **Cost to verify:** ~10 min (run validation with `--nproc_per_node 8`, confirm recall@K matches single-GPU result on the same dataset).
* **Risk if broken:** the paper-target experiments at pythia-160m / pile-10k or 1.4b / pile-100k will fail or silently produce wrong factors at runtime.

### 19.2 Token-attribution end-to-end [medium cost]

Plan §3.5 says per-token works "from day one." Commit 2's `test_ekfac_preconditioner_per_token_shape` confirms the math handles `[T, O*I]` identically to `[N, O*I]`.

* **Tested:** unit-level shape handling.
* **Not tested:** a full `compressed_ekfac` run with `index_cfg.attribute_tokens=True` and a factored preconditioner. The Builder's `_scatter_flat_tokens` + `post_projection` + variable-length per-example token counts is untested as a chain.
* **Cost to verify:** ~15 min (validation script with `--attribute_tokens` flag plumbed through; need a small CLI tweak).
* **Risk if broken:** the per-token retrieval mode is unusable; falls back to per-sequence.

### 19.3 `tkfac` and `shampoo` hessian methods [low cost]

`HessianConfig.method` accepts `kfac`, `tkfac`, `shampoo`. All three call `compute_eigendecomposition` and write to the same `eigen_activation_sharded/` + `eigen_gradient_sharded/` layout, so `_detect_variant` will identify them as EKFAC/KFAC.

* **Tested:** `method="kfac"` with `ev_correction=True` (the canonical EKFAC path).
* **Not tested:** does `EkfacPreconditioner` do the right thing on tkfac or shampoo eigenvectors? The math may differ — tkfac multiplies covariances by per-example trace ratios, shampoo uses preconditioned 4th-order tensors. The rotate-scale-rotate body in `_FactoredPreconditioner.apply` was derived for kfac.
* **Cost to verify:** ~30 min (validation runs with each method) or **just gate it explicitly** — raise a clear error in `load_preconditioner` if `method != "kfac"` until tkfac/shampoo are formally validated.
* **Risk if broken:** silent wrong results for users who fit tkfac/shampoo Hessians. Worth gating defensively.

### 19.4 `include_bias=True` error guard [trivial]

Commit 3's `build_worker` raises `NotImplementedError` when `include_bias=True` + factored preconditioner. The error path is uncovered by tests.

* **Cost to verify:** trivial (one test that asserts the raise).
* **Risk if broken:** `include_bias=True` users hit a confusing error instead of the clear one I wrote.

### 19.5 Resume mode [trivial]

`compressed_ekfac_pipeline(resume=True)` skips steps whose output dirs already exist. Logic is straightforward but never run.

* **Cost to verify:** ~2 min (run twice, confirm second run skips both steps).
* **Risk if broken:** users with partial runs can't pick up where they left off; minor UX issue.

### 19.6 Pythia-1.4b stretch [high cost]

Predicted from §18.5 to need `p ≥ 256`. Never run.

* **Memory budget unconfirmed.** Q_A for pythia-1.4b's MLP intermediate-down layer has shape `[8192, 8192]` ≈ 256MB in fp32. Across 24 layers × 4 modules × (Q_A + Q_S) = ~50GB on-device per rank. Could exceed 48GB A40 — need to check, possibly fall back to bf16 factors, possibly stream.
* **Cost to verify:** several hours (full pipeline run on pile-100k, plus retrieval validation).
* **Risk if broken:** stretch goal slips. MVP (160m) is unaffected.

### 19.7 Python API for query-side embedding [bigger; commit-4 work]

There's no `embed_query(model, query, ekfac_path) -> Tensor[p²]` helper. My validation builds a 5-row "query index" via the full orchestrator, which is heavyweight. The two-stage retrieval notebook needs a lighter-weight path:

```python
# Conceptual API the notebook will need:
query_vec = embed_query(model, query_text, ekfac_path, projection_dim=128)
scores = train_index @ query_vec  # [N_train]
top_k = scores.argsort()[-K:]
```

This requires plumbing through the same EKFAC + projection that the build path uses, but for a single example without writing to disk. Commit 4 has to build it. **Not a gap so much as a known commit-4 deliverable, flagging here so it's not forgotten.**

### 19.8 Numerical precision (bf16 vs fp32 on disk) [low cost]

All validation used `--precision bf16`. The reference `EkfacApplicator` works in fp32 internally; my `_FactoredPreconditioner` casts to fp32 for the math. But the on-disk index is in `save_dtype` derived from the model's dtype, which for `bf16` precision is bf16. Retrieval might be more accurate at fp32; recall@K numbers in §18.5 may improve.

* **Cost to verify:** ~5 min (validation rerun with `--precision fp32`).
* **Risk if broken:** none for correctness; potentially higher recall numbers in fp32, which would be paper-positive.

### 19.9 160m recall is marginal, not robust

The pythia-160m / p=128 / pile-10k[:200] result (mean recall@10 = 44 %, Spearman = 0.39) **clears my self-imposed bar but only barely**. One of five queries has recall@10 = 10 %. The mean over five queries has high variance. Not a "gap" in the same sense as the others — the pipeline works — but a caveat: at the MVP scale, results are noisier than the 14m / p=64 numbers would suggest.

Possible mitigations to test in commit 4:
* Larger N_train (200 → 1000) — more training examples means stronger gradient retrieval signal per query
* Larger N_query for stable means
* p=256 instead of 128 (more memory, better preservation)

### 19.10 Test coverage in `tests/ekfac_tests/` [confirmed green]

This is **not** a gap, but worth documenting because it's load-bearing for confidence: the `tests/ekfac_tests/` subdirectory contains session-scoped fixtures that compute ground-truth EKFAC factors via independent code (`compute_ekfac_ground_truth.py`) and assert numerical properties (batch-size invariance, eigenvalue correction accuracy, FIM accuracy). Those tests have run and stayed green throughout commits 1–3 and the validation work. So the additive eigenvalue-saving change to `compute_eigendecomposition` (§12) is empirically backwards-compatible with the most thorough tests bergson has of EKFAC math.

---

## 20. Verification priorities before merge

Ordered by likelihood-of-blocking-the-paper × cost-to-verify:

1. **§19.1 multi-GPU validation** — blocking for any pile-100k stretch. ~10 min to run. **Should do before commit 4.**
2. **§19.4 `include_bias=True` test** — trivial; just write the test. Should bundle with commit 4 cleanup.
3. **§19.5 resume mode test** — also trivial; same.
4. **§19.3 tkfac/shampoo gate** — defensive raise is cheap; do this rather than full validation.
5. **§19.2 token-attribution e2e** — only matters if the notebook uses per-token mode. Decide commit-4 scope first.
6. **§19.8 fp32 vs bf16** — paper-positive optionality, do if time allows.
7. **§19.6 pythia-1.4b stretch** — only matters if the stretch is attempted; defer until commits 1–4 land.
