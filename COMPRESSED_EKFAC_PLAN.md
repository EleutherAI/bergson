# Compressed EKFAC Index — Implementation Report

**For Lucia (EleutherAI), pre-PR review.** Branch: `feature/compressed-ekfac`, target submission **2026-05-08**.

This doc replaces an earlier handoff/cluster-bootstrap version. It's grouped by topic, not chronology: what shipped, why each decision was made (with alternatives considered), and questions where your call could redirect work.

---

## 1. What shipped

A `bergson compressed_ekfac` CLI subcommand that produces a per-example (or per-token) gradient index where each row is `vec(P · H^{-1/2} · g)` — EKFAC preconditioning baked in at build time, double-sided random projection applied after preconditioning. The resulting index is queryable by plain dot product; it's the recall-phase artifact of a two-stage retrieval setup, with the existing per-example path supplying the precision phase.

**Done bar (met):**
- 179 tests pass (162 pre-existing + 17 added by this work; pre-existing failures unchanged — see commit log §10).
- End-to-end retrieval validation against an unprojected ground-truth reference passes on pythia-14m and pythia-160m (§6).
- Two-stage retrieval notebook executes top-to-bottom on pythia-160m / pile-10k showing 108× compression and 145× faster Stage-1 retrieval (commit `93cd8f7`).
- 8-GPU multi-rank result matches 1-GPU within JL noise (§6.3).

**Deferred:**
- pythia-1.4b stretch (§7.1) — predicted to need `p ≥ 256` and may exceed 48 GB on-device factor budget per A40.
- True sharded per-batch apply (vs. full-load-per-rank) — full-load matches the existing autocorrelation memory envelope.
- `create_scorer` EKFAC support (§2.7) — compressed-EKFAC scoring goes through a notebook dot-product path for now.
- Bias + EKFAC support (§3.5).
- `tkfac` / `shampoo` factored-preconditioner math validation — gated with a clear `NotImplementedError` for now; tracked as [issue #244](https://github.com/EleutherAI/bergson/issues/244).

---

## 2. Architecture

### 2.1 The `Preconditioner` protocol

Before this work, bergson had one on-disk preconditioner (the "trackstar" autocorrelation dump) plus a one-off `EkfacApplicator` not plugged into `build`/`score`. We unified them:

```python
class Preconditioner(Protocol):
    def apply(self, mod_grads: dict[str, Tensor]) -> dict[str, Tensor]: ...
```

Three implementations, all loaded via `load_preconditioner(path, device, power)`:

* `AutocorrelationPreconditioner` — wraps the existing trackstar dict of `H^power` matrices; `.apply` is a per-module right-matmul.
* `EkfacPreconditioner` — factored with empirical Λ correction.
* `KfacPreconditioner` — factored with `Λ = Λ_G ⊗ Λ_A`.

**Alternatives considered:**

| Approach | Why rejected |
|---|---|
| **`Protocol` (chosen)** | — |
| Abstract base class | Forces every variant to carry the same internal state; noisy for autocorrelation, which is a thin wrapper |
| Plain duck typing | Loses type-checker help; harder to grep for implementers |
| Config-field dispatch (`preconditioner_type: Literal[...]`) | Adds a config-plumbing layer through every caller of `PreprocessConfig`; easy to mismatch config vs on-disk artifact (§2.2) |

Module placement: `bergson/preconditioners.py` (new top-level). Inside `bergson/hessians/` would couple autocorrelation (conceptually separate) to the hessians subpackage; inside `process_grads.py` would bloat a file already mixing normalization and preconditioner loaders.

### 2.2 Auto-detect variant from directory contents

**Direct answer to your question.** You asked whether to (a) read variant info from metadata inside `preconditioner_path`, or (b) route the variant explicitly through CLI / `PreprocessConfig`. We picked a third option that's effectively (a) but cheaper: **infer the variant from which sub-directories are present inside `preconditioner_path`** — no separate metadata file, no config plumbing, no API change. `load_preconditioner(path, ...)` is the only entry point; consumers still pass just `preconditioner_path`. The two alternatives we considered and rejected:

* **Routing through `PreprocessConfig` (option b)** — adds plumbing through every caller of the dataclass and opens a config-vs-artifact mismatch surface (user points at an EKFAC dir but ticked `kfac` on the CLI). Listed in §2.1's alternatives table.
* **A `meta.json` `variant` field inside the dir (a literal reading of option a)** — would create an invariant to keep in sync (metadata says EKFAC but `eigenvalue_correction_sharded/` was deleted → silent wrongness). Directory presence dodges this: if `eigenvalue_correction_sharded/` exists, EKFAC apply has the data it needs; if not, the variant is correctly identified as KFAC.

The detection itself, in `_detect_variant(path) -> {"autocorrelation", "kfac", "ekfac"}`:

| On disk | Variant |
|---|---|
| `eigen_activation_sharded/` + `eigen_gradient_sharded/` + `eigenvalue_correction_sharded/` | EKFAC |
| `eigen_activation_sharded/` + `eigen_gradient_sharded/` (no correction dir) | KFAC |
| Anything else (incl. `preconditioners.pth`, or `None`) | autocorrelation |

### 2.3 Shared factored base class

`_FactoredPreconditioner` holds `{q_a, q_s, lam}` per module and implements rotate-scale-rotate:

```
H^power · G ≈ Q_S · ((Q_S^T G Q_A) · λ^power) · Q_A^T
λ = Λ + damp · mean(Λ)
```

EKFAC and KFAC subclasses differ only in how Λ is sourced (EKFAC loads it directly; KFAC synthesises it from `torch.outer(lam_s, lam_a)`).

The math matches the reference `EkfacApplicator.compute_ivhp_sharded` body at `bergson/hessians/apply_hessian.py:83-133`. We chose two classes with a shared base over a single class with a `variant` arg because `__init__` differs meaningfully between the two — splitting reads cleaner than branching.

The body is written as five einsums (no dependence on `ShardedMul`) because MVP is full-load-per-rank (§2.6).

### 2.4 Build-time bake-in, precondition-then-project

Two locked design decisions:

* **Apply the preconditioner at build time, not score time.** Store `vec(P · H^{-1/2} · g)` per example so scoring is a plain dot product.
* **Order: precondition then project (Grosse semantics).** EKFAC `H^{-1/2}` applies in original parameter space; random projection happens after.

The "after" part required moving the projection step from the collector into the builder for the factored path — the meatiest design call in this branch. Detail in §3.

### 2.5 Per-token support from day one

Builder already had an `attribute_tokens` path. The factored apply handles `[N, O*I]` (sequences) and `[T, O*I]` (tokens) identically — same reshape to `[*, O, I]`, different leading dim. Validated end-to-end (`test_compressed_ekfac_token_attribution`).

### 2.6 Full-load-per-rank for MVP

Each rank reads all shards for the variant and concatenates along dim 0 to reconstruct full factors on-device. `_load_sharded_dict` is variant-agnostic. Memory envelope matches the existing autocorrelation path everyone already runs.

**Alternative deferred:** reuse `ShardedMul` from `hessians/sharded_computation.py` for true per-batch sharded apply. `ShardedMul._matmul` is written for one query gradient at a time, so retrofitting it for the per-batch hot path would require a redesign that wasn't on the critical path for May 8.

### 2.7 `create_scorer` stays autocorrelation-only

`create_scorer` at `bergson/score/score.py:120` loads a `dict[str, Tensor]` of per-module `[D, D]` matrices, optionally applies them to query grads, and builds an `index_transform` closure for split preconditioning when `unit_normalize=True`. EKFAC's Q_A/Q_S/Λ triple doesn't fit any of those steps.

**Decision:** raise `NotImplementedError` for non-autocorrelation paths at score time. Compressed-EKFAC indices score via plain dot product in the notebook / orchestrator.

| Alternative | Why deferred |
|---|---|
| Teach `create_scorer` to dispatch | `_make_split_preconditioner` also needs generalization; many downstream consumers |
| Duplicate to `create_factored_scorer` | Extra API surface, duplicated plumbing |
| Mark compressed indices as "already preconditioned" in `PreprocessConfig` | Viable; new config invariant — see Q in §9.3 |

---

## 3. The projection-placement replan

This is the most significant deviation from the originally scoped commit boundaries.

### 3.1 The conflict

The existing pipeline projects inside the collector, before the outer product:

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

Autocorrelation works under this layout because its `h_inv` is computed on the projected gradient covariance, so it lives in `[p², p²]` space matching `mod_grads`.

EKFAC doesn't: Q_A and Q_S are tied to the *unprojected* parameter space (they're factors of the Hessian there). There is no function `f` such that `L · H^{-1/2} · G · R^T = f(L · G · R^T)` without `f` referencing `H^{-1/2}` directly — projection erases the structure the factored apply needs.

### 3.2 Options evaluated

| # | Approach | Effort | Correctness | Plan fit |
|---|---|---|---|---|
| **A** | **Move projection into `Builder._preprocess` for factored preconditioners; leave autocorrelation as-is** | ~50 LOC, 3 files | ✓ correct | Matches "precondition-then-project" |
| B | Reformulate EKFAC in projected space using `L·Q_S` and `R·Q_A` | Moderate | ✗ provably wrong | — |
| C | Skip random projection for compressed EKFAC; "compressed" = EKFAC eigenbasis only | Tiny | ✓ partial | Contradicts paper claim |
| D | Precondition at query time on full-dim grads; store unprojected EKFAC'd index | Tiny build, hurts query | ✓ | Unprojected index is huge at 1.4B |
| E | Push preconditioning into the collector | Large; cross-cutting | ✓ | Mixes a load-once op into a hot path; harder to test |

**Chose A.** Projection is a pure linear op that commutes with "writing to disk"; moving it between collector and builder is a local refactor. Autocorrelation stays bit-exact (regression anchor); EKFAC/KFAC get the correct semantics.

### 3.3 What changed

Three files:

1. `bergson/build.py::build_worker` — when the preconditioner_path points at a factored variant, set `processor.projection_dim = None` so the collector hands off unprojected `[N, O*I]` grads.
2. `bergson/builder.py::Builder` — when `post_projection_dim` is set, build per-module L (`[p, O]`) and R (`[p, I]`) matrices via the existing `create_projection_matrix` helper and apply `vec(L · G · R^T)` after `preconditioner.apply`.
3. `bergson/collector/gradient_collectors.py` — derives whether to enable `post_projection_dim` from `processor.projection_dim is None and cfg.projection_dim > 0` (single decision point — `build_worker` is the only file that decides; collector reads the resulting state).

### 3.4 Invariants

| Invariant | Autocorrelation (existing) | EKFAC/KFAC (new) |
|---|---|---|
| Collector output shape | `[N, p*p]` projected | `[N, O*I]` unprojected |
| `h_inv` / factors dtype | processor's dtype (bf16/fp32) | float32 on-GPU |
| Bias handling | `include_bias=True` appends column before projection | **Refused** (raises `NotImplementedError` early in `compressed_ekfac_pipeline`) |
| Per-token vs per-sequence | already flattened per-token by collector | `apply` handles `[T, O*I]` identically — unit + e2e tested |
| Written-to-disk dim | `p*p` | `p*p` (projection applied in builder) |

### 3.5 Bias + EKFAC

Refused for MVP. Q_A is sized `[I, I]` but `include_bias=True` makes the gradient `[O, I+1]`. Safest default: raise. The orchestrator (`compressed_ekfac_pipeline`) raises *before* step 1 burns the Hessian fit; `build_worker` keeps a defensive backstop for direct `build()` callers. Question for you in §9.4.

---

## 4. KFAC support required saving eigenvalues

### 4.1 The gap

The plan called for `KfacPreconditioner` to load "raw eigenvalues". Investigation found `compute_eigendecomposition` at `bergson/hessians/eigenvectors.py:280` was discarding them:

```python
eigenvalues, eigenvectors = torch.linalg.eigh(matrix_normalized)
# ...
covariance_eigenvectors[key] = eigenvectors  # eigenvalues discarded
```

KFAC therefore couldn't be loaded from any current artifact.

### 4.2 Options

| Approach | Why rejected |
|---|---|
| **Save eigenvalues alongside eigenvectors (chosen)** | Additive; ~5-line change |
| Recompute at KFAC load time | Inverts the point of eigendecomposition being a one-time cost |
| Derive from `eigenvalue_correction_sharded` | Mathematically wrong — empirical Λ ≠ Λ_G ⊗ Λ_A |
| Defer KFAC entirely | Bleeds scope across PRs |

### 4.3 Migration

The eigenvalue dirs use a distinct name (`eigenvalues_*_sharded/`, not `eigenvalue_*_sharded/`) so they don't collide with EKFAC's correction dir. Existing EKFAC artifacts lack them, which is fine — EKFAC doesn't read them. KFAC reads via `KfacPreconditioner.from_disk`; if absent, a `FileNotFoundError` points the user at `compute_eigendecomposition` with a clear message:

```
KFAC preconditioner at <path> is missing per-factor eigenvalue shards
(eigenvalues_activation_sharded/ and/or eigenvalues_gradient_sharded/).
Regenerate the artifact with the current version of compute_eigendecomposition.
```

Older artifacts regenerate cleanly. No data migration script. No silent-wrong-result risk.

### 4.4 Risk

Every current EKFAC/KFAC run post-merge will write two extra 1D safetensors shards. Size impact is negligible (`Σ_m (O_m + I_m)` rather than `Σ_m O_m·I_m`), but a multi-branch experiment (eigendecomposition on `main`, consumer on this branch) will see different layouts. Flagging in case there's an in-flight run.

---

## 5. Test strategy

### 5.1 Unit-level: parity tests

`tests/test_preconditioners.py` (12 tests, all pass):

| Coverage | Mechanism |
|---|---|
| `_detect_variant` on EKFAC / KFAC / autocorrelation | Synthetic dirs |
| `is_factored_preconditioner` (used by `build_worker`) | Synthetic dirs + `PreprocessConfig` |
| `load_preconditioner(None)` no-op | Direct |
| `_load_sharded_dict` round-trip across two shards | Synthetic shards |
| `_load_sharded_dict` rejects mismatched trailing dims | Synthetic shards |
| **EKFAC parity vs `EkfacApplicator.compute_ivhp_sharded`** | Synthesize Q_A/Q_S/Λ + gradient mmap, run both, `torch.allclose(atol=1e-5)` |
| EKFAC per-token shape `[T, O*I]` | Direct |
| **KFAC parity vs direct einsum reference** | No trusted KFAC applicator existed; validated against an inline rotate-scale-rotate with `Λ = Λ_G ⊗ Λ_A` |
| KFAC missing-eigenvalues error surface | Synthesize incomplete artifact, expect `FileNotFoundError` |

We chose synthetic factors over end-to-end through `approximate_hessians` because:
* It runs in seconds and isolates the math from collector plumbing.
* The commit-3 smoke test (`test_compressed_ekfac_e2e`) is the integration test.
* `tests/ekfac_tests/` fixtures (which compute ground-truth EKFAC factors via independent code) are session-scoped, expensive, and overkill for this purpose.

### 5.2 Integration: e2e smoke + CLI

`tests/test_compressed_ekfac.py` runs `python -m bergson compressed_ekfac` on pythia-14m / pile-10k[:100] / `projection_dim=64` and asserts the on-disk layout matches what downstream code expects (each module row = `p² = 4096`). Plus tests for resume mode, `include_bias` early-rejection, and token-attribution end-to-end.

### 5.3 Empirical: retrieval-correctness validation

This was added after commit 3 landed and is described in §6.

### 5.4 What the parity test trusts

The "trusted applicator" in EKFAC parity is `EkfacApplicator.compute_ivhp_sharded` itself. If you consider that code experimental rather than reference-trusted, a second reference (Kronfluence, internal one-off) would let us triangulate. **Question in §9.1.**

---

## 6. Empirical validation — the projection_dim floor

This is the most paper-relevant finding, and it surfaced only after I ran a retrieval-correctness validation that none of the commits-1/2/3 tests had been designed to answer. Lesson: "runs without crashing with the right shape" is a weak acceptance bar for a retrieval artifact.

### 6.1 Validation design

`scripts/validate_compressed_ekfac.py` builds four indices from the same EKFAC factors fit on `pile-10k[:N_train]`:

| Index | `projection_dim` | Data | Role |
|---|---|---|---|
| `train_compressed` | varied | `train[:N_train]` | candidate |
| `train_reference` | **0** (no projection) | `train[:N_train]` | ground truth: full `vec(H^{-1/2} G)` per module |
| `query_compressed` | varied | `train[N_train:N_train+N_query]` | held-out queries |
| `query_reference` | **0** | same | held-out ground truth |

Key trick: `projection_dim=0` + factored preconditioner + `skip_preconditioners=True` is "free" on the post-replan code path — collector emits `[N, O·I]`, builder applies EKFAC and skips the post-projection step. Zero extra code to produce ground truth.

Metrics per query: recall@K (K=5,10,20) of compressed top-K vs reference top-K, plus Spearman over the full score vector. Pass bar: mean recall@10 ≥ 40 % (8× random at N_train=200), mean Spearman ≥ 0.30.

### 6.2 What the failing first run revealed

**Run 1** (`projection_dim=16`, `unit_normalize=False` — the originally scoped smoke-test default):

```
mean recall@5 = 20%   mean recall@10 = 20%   mean Spearman = 0.096   → FAIL
```

`scripts/diag_compressed_ekfac.py` confirmed per-example projection is bitwise correct (`(L @ ref @ R^T).reshape(-1)` vs the compressed vector correlates 1.0000 across 5 examples × 24 modules). Per-module score-vector Spearman is 0.01 to 0.25, mean 0.11. Even isolated to one module, rankings disagree.

Two roots:

1. **Kronecker-structured JL, not vanilla JL.** `vec(L · G · R^T)` lives in `p²` dims but the projection matrix has Kronecker structure `R ⊗ L`, effective rank `p(O+I)` not `p²·O·I`. Per-module inner-product preservation needs `p ≳ √(O·I) / ε`. Pythia-14m's largest module has `O·I ≈ 65k → √ ≈ 256`. At `p=16` we're a factor of 16 below the floor.

2. **One-sided preconditioning.** `unit_normalize=False ⇒ power=-1` on both sides → score is `<G_q, H^{-2} G_t>`, not the standard `<G_q, H^{-1} G_t>` influence quantity.

**Run 2** (`projection_dim=64`, `unit_normalize=True ⇒ power=-0.5`, split):

```
q   recall@5  recall@10  recall@20   spearman
0     40.00%     40.00%     55.00%      0.659
1     40.00%     20.00%     35.00%      0.480
2     40.00%     70.00%     75.00%      0.793
3     80.00%    100.00%     90.00%      0.853
4     80.00%     70.00%     70.00%      0.786
mean  56.00%     60.00%     65.00%      0.714  → PASS
```

### 6.3 Why autocorrelation tolerates `p=16` and EKFAC doesn't

Autocorrelation projects *inside the collector before the outer product*; its `h_inv` is computed over the projected gradient covariance, so the preconditioner lives in `[p², p²]` space matching the projected `mod_grads`. The Kronecker-structure penalty is absorbed.

Compressed EKFAC is structurally different — Q_A/Q_S are tied to unprojected parameter space, so the projection has to happen there and the Kronecker penalty is exposed.

### 6.4 Scaling matrix

| Setting | n_train | n_query | recall@5 | recall@10 | recall@20 | Spearman | Verdict |
|---|---|---|---|---|---|---|---|
| pythia-14m, p=16, one-sided | 200 | 5 | 20 % | 20 % | 26 % | 0.096 | FAIL |
| pythia-14m, p=64, split, 1 GPU | 200 | 5 | 56 % | **60 %** | 65 % | **0.714** | PASS |
| pythia-14m, p=64, split, **8 GPUs** | 200 | 5 | 56 % | 56 % | 61 % | 0.705 | PASS — within JL noise of 1-GPU |
| pythia-160m, p=64, split | 200 | 5 | 12 % | 20 % | 24 % | 0.256 | FAIL |
| pythia-160m, p=128, split, **bf16** | 200 | 5 | 44 % | **44 %** | 37 % | 0.389 | PASS |
| pythia-160m, p=128, split, **fp32** | 200 | 5 | 44 % | 46 % | 52 % | 0.477 | PASS — fp32 better |
| pythia-160m, p=128, split, **n_train=1000** | 1000 | 20 | 47 % | **40 %** | 40 % | 0.396 | PASS — Spearman holds |

**Reading:**

* JL floor scales with `√(maxₘ O_m·I_m)` as predicted. 14m largest module ~65k → √~255 → p=64 sits at ~25 % of floor and works. 160m largest ~2.4M → √~1550 → p=64 is ~4 % (fails); p=128 is ~8 % (passes but marginal).
* Multi-GPU and 1-GPU rankings match. The shard-concat in `_load_sharded_dict` is verified at world_size=8.
* fp32 marginally better than bf16 on Spearman (0.48 vs 0.39); recall@10 essentially unchanged. Use bf16 by default to halve disk size.
* Recall@K does NOT improve with more training data at fixed p — retrieving the *exact* top-K out of a larger pool is harder by exactly the right amount to keep recall flat. **Spearman is the stable signal across n_train**.

---

## 7. Defaults, recommendations, deferred work

### 7.1 Per-target defaults

| Target | `projection_dim` | Notes |
|---|---|---|
| pythia-14m | ≥ 64 | p=64 passes comfortably |
| pythia-160m (MVP) | ≥ 128 | p=64 fails; p=128 passes but is marginal |
| pythia-1.4b (stretch) | predicted ≥ 256 | **untested** — flagged as open; max `O·I ≈ 16M`, √ ≈ 4000 |

Pythia-1.4b also has a memory concern: Q_A for the MLP intermediate-down layer is `[8192, 8192]` ≈ 256 MB in fp32. Across 24 layers × 4 modules × (Q_A + Q_S) ≈ 50 GB on-device per rank — could exceed 48 GB A40, possibly forcing bf16 factors or streaming. **Question in §9.5.**

### 7.2 `unit_normalize=True` is the right default

`unit_normalize=False` collapses to one-sided `H^{-2}` preconditioning *and* doesn't unit-norm the concat vector — both wrong for influence-functions retrieval. `unit_normalize=True` gives split preconditioning (`H^{-1/2}` on both sides → standard `<q, H^{-1} t>`). The notebook and the orchestrator docstring both flag this.

### 7.3 Other defaults

* `precision=bf16` is sufficient for retrieval; fp32 marginally improves Spearman.
* The smoke-test `projection_dim` was bumped from 16 to 64 after §6.2 — at 16 the test passed shape checks but would have silently retrieved garbage.

### 7.4 Deferred items (carrying forward)

* True sharded per-batch apply (vs full-load-per-rank) — §2.6.
* `create_scorer` EKFAC support — §2.7 + §9.3.
* Bias + EKFAC proper support — §3.5 + §9.4.
* Damping scheme review for KFAC (mean of `Λ_G ⊗ Λ_A` has a different scale than per-factor) — §9.1.
* `tkfac` / `shampoo` factored-preconditioner math validation — gated with a clear `NotImplementedError` until each method's rotate-scale-rotate body is independently checked. Tracked as [issue #244](https://github.com/EleutherAI/bergson/issues/244).
* Pythia-1.4b stretch — §7.1.

---

## 8. Tools shipped alongside the index code

* `bergson compressed_ekfac` CLI subcommand + `bergson.hessians.compressed_ekfac.compressed_ekfac_pipeline` — orchestrator.
* `bergson.hessians.compressed_ekfac.embed_query` — light-weight helper for one-off query-side embeddings (writes a temp jsonl, runs a tiny build, reads the result back). The two-stage notebook uses it.
* `scripts/validate_compressed_ekfac.py` — produces the four-index validation matrix in §6.
* `scripts/diag_compressed_ekfac.py` — per-module/per-example diagnostics (used to disambiguate "is the projection broken?" from "is it the JL floor?").
* `notebooks/compressed_ekfac_two_stage.ipynb` — paper figures (recall@K vs index size, wall-clock vs per-example baseline) on pythia-160m.

---

## 9. Questions for Lucia

Grouped by where a decision could redirect work.

### 9.1 Math / correctness (blocking if wrong)

* **Rotate-scale-rotate formulation.** `_FactoredPreconditioner.apply` computes `Q_S (Q_S^T G Q_A · λ^power) Q_A^T` as five einsums. Matches `EkfacApplicator.compute_ivhp_sharded` element-wise at `atol=1e-5` (parity test). But if `EkfacApplicator` itself has a bug, the parity test is meaningless. Is there a second reference you'd trust (Kronfluence, internal impl)?
* **KFAC `Λ = Λ_G ⊗ Λ_A`.** I'm using `torch.outer(lam_s, lam_a)` for a `[O, I]` matrix. Right orientation?
* **Damping.** `inv_lam = (Λ + damp · mean(Λ))^power`, same formula `EkfacApplicator._hadamard` uses. Fine for EKFAC; for KFAC's outer-product Λ, `mean(Λ)` has a different scale than the per-factor case. Want a different scheme?

### 9.2 Naming

* `autocorrelation` is used everywhere new code touches (your preference). `get_trackstar_preconditioner` has been renamed to `get_autocorrelation_preconditioner`; the old name is preserved as a backwards-compatible alias in `bergson/process_grads.py` for external users who import it directly. Happy to drop the alias if you'd rather force a clean break.
* `_FactoredPreconditioner` is internal/private. Open to `FactoredPreconditioner` (public), `KroneckerPreconditioner`, etc.
* `load_preconditioner` factory entry point — OK, or prefer `Preconditioner.from_path`?

### 9.3 Score-time strategy

`create_scorer` rejects EKFAC. Worth your take: have the compressed index live inside the existing scorer by marking it as "already preconditioned" in `PreprocessConfig`, so `create_scorer` doesn't try to apply anything and scoring is automatic dot-product. Would replace bespoke notebook scoring with scorer reuse. Low urgency, but cleaner long-term.

### 9.4 Projection placement and bias

* Option A (move projection into builder for factored variants) — right call? Option E (preconditioner-aware collector) is symmetric but more invasive; happy to switch if you'd rather keep a single projection site.
* Bias + EKFAC: refuse-if-both-set for MVP, proper support as follow-up. OK?

### 9.5 Pythia-1.4b stretch

§7.1 predicts `p ≥ 256`. That's a 64× larger index than `p=64` and ~50 GB factor budget per rank. Acceptable? Alternatives: settle for lower recall and document; switch projection scheme (single-sided JL on `vec(G)` with `p_total ≈ 65k ≪ O·I` is more compact but breaks the `[p, p]` convention; or TRAK-style).

### 9.6 Decoupling preconditioning power from unit-norm

`unit_normalize=True` couples split preconditioning (`power=-0.5`) with unit-norming the concat vector. The unit-norm step isn't required for split preconditioning — they happen to share a flag in bergson's `Builder`. Want a new `PreprocessConfig.precondition_power: Literal[-0.5, -1]` to decouple them? Low priority.

### 9.7 Pile-10k qualitative neighbors

Even unprojected reference top-3 neighbors for a "Tulsi Gabbard 2020 candidate" query don't return politically relevant training text — pile-10k is random web with no obvious near-duplicates. The notebook's qualitative section will look underwhelming, but the quantitative recall/Spearman is what matters. Flagging in case you want a different demo dataset for the paper.

### 9.8 KFAC artifact-version skew

§4.4: in-flight runs that fit eigendecomposition on `main` won't have the new `eigenvalues_*_sharded/` dirs and will fail on `feature/compressed-ekfac` consumers. Worth a heads-up to anyone running multi-branch experiments.

---

## 10. Commit log

| Commit | What it does |
|---|---|
| `d8ab2e3` | Preconditioner protocol + `AutocorrelationPreconditioner` + `load_preconditioner` factory + auto-detect (pure refactor, 162/162 tests stay green) |
| `30e66e4` | `EkfacPreconditioner` + `KfacPreconditioner` + parity tests (+8 tests) |
| `c69e826` | `bergson compressed_ekfac` CLI + orchestrator + projection-placement replan into builder (+1 e2e test) |
| `fd235dc` | End-to-end retrieval validation; bumped smoke-test `projection_dim` 16 → 64 after §6 finding (additive) |
| `ae5e3e3` | Documented known gaps (§19/§20 of prior plan iteration) — doc-only |
| `62f86b9` | Phase A: defensive gates (`tkfac`/`shampoo` rejection in `_check_validated_hessian_method`) + trivial-gap tests for `include_bias` + resume (+3 tests) |
| `9358a17` | Phase B+C: empirical validation across 14m / 160m / multi-GPU / bf16-vs-fp32; closed §19/§20 gaps — doc-only |
| `48aa144` | Doc-only fixups for stale references |
| `93cd8f7` | Two-stage retrieval notebook (108× compression, 145× faster Stage-1) |
| `b256644` | Notebook stabilization (`N_QUERY=20`, `debug=True`); recall@10 35.5 % compressed → 48.5 % two-stage; Spearman 0.397 |
| `c021999` | Token-attribution end-to-end test (+1 test) |
| `2f10d6a` | `embed_query` helper + test (+1 test); cluster regression at this point: **176 passed**, 13 same pre-existing failures, 4 skipped |
| `4a42e2e` | **Post-review polish #1.** Move `is_factored_preconditioner` to `bergson/preconditioners.py`; collector derives `post_projection_dim` from `processor.projection_dim is None` (single decision point at `build_worker`) (+1 CPU test) |
| `bb4dad6` | **Post-review polish #2.** Fail-fast on `include_bias` in `compressed_ekfac_pipeline` so step 1's Hessian fit doesn't burn before the guard fires; defensive backstop in `build_worker` retained. Test no longer needs CUDA; gains assertions that neither `hessian/` nor `index/` dirs exist after the raise |
| `78800ff` | **Post-review polish #3.** `_load_sharded_dict` validates trailing-dim equality across shards per key, raises with file context if the on-disk shard split axis ever changes (+2 CPU tests) |
| `03105a4` | **Post-pull fix.** Polish #1 had an over-broad gate (`processor.projection_dim is None`) that swept up 7 autocorrelation tests calling `collect_gradients` directly with `GradientProcessor()`. Switched the trigger to `is_factored_preconditioner(preprocess_cfg)` — gates on the on-disk preconditioner_path, the actual source of truth. All 7 regressions restored |
| `db99c9e` | Renamed `get_trackstar_preconditioner` → `get_autocorrelation_preconditioner` in `bergson/process_grads.py` to align the helper name with the `autocorrelation` Hessian-approximation type used everywhere else. Backwards-compatible alias retained for external users; internal callers + tests updated. tkfac/shampoo follow-up: [issue #244](https://github.com/EleutherAI/bergson/issues/244). |

Pre-existing failures (unchanged throughout, all unrelated to this work):
* `test_adam_state_loading.py::test_load_8bit_adam_checkpoint` — missing `bitsandbytes` in dev deps
* `test_build.py::test_build_consistency` — A40-vs-? numerical determinism mismatch against cached snapshot at `atol=1e-6`
* `test_muon.py` (4) — `torch.optim.Muon` not in torch 2.6
* `test_truncation.py` (7) — test/code drift on batch-size validation and warning text

**Verified at HEAD: 179 passed, 13 same pre-existing failures, 4 skipped.**
