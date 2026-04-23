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
