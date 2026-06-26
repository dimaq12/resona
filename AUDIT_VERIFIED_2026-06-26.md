# Resona — verified audit (2026-06-26)

Every item below was checked against the **actual code** and, for behavioral
claims, **reproduced with a runnable `python3` script** (repros in the session
scratchpad). Default stance was skeptical: a claim was treated as false until
reproduced. This supersedes the unverified report it was triaged from.

Legend: 🔴 REAL · 🟠 PARTIAL (real mechanism, narrow/unreachable trigger or
overstated) · ⚪ FALSE / non-issue · ✨ NEW (not in the original report).

---

## A. Confirmed real bugs — fix these

### A1. Wrong math (silent wrong numbers)

| ID | Location | Bug | Repro outcome |
|----|----------|-----|---------------|
| 🔴 | `defect.py:59` `richardson_limit` | Exponent `(n_k/n_{k-col})**(p0*col)` double-counts `col`; correct is `**p0`. | Known 3-term expansion: shipped → err −2e-4; corrected → machine zero. Fix: drop `* col`. |
| 🔴 | `lift.py:173` `carleman_scalar` | Basis `(x¹…x^order)` has no `z₀=1`; constant term `coeffs[0]` is dropped from the `ż₁` row. | `f=0.3+x+0.5x²`: `ż₁` off by exactly 0.300 (the constant). Breaks Riccati `a+bx+cx²`. Fix: augment basis with `z₀`. |
| 🔴 | `lift.py:78` `s_transform` | Bracket `hi=inv−1e-12` inverts (`hi<lo`) once `1/λmax ≤ 1e-12`. | relerr 0 at λmax=1e11 → **exactly 2× wrong** at 1e12+, silently. Fix: scale bracket to the pole + residual check. |
| 🔴✨ | `thermal.py:82` `correlator` | Uniform fast-path starts stepping from the **unevolved** β-state, so it returns `C(t − ts[0])` when the grid starts at `t₀≠0`. | grid `ts=2+…`: fast branch `max err 27.7` vs slow branch `0.04`. `C_fast[0]` equals ground truth at t=0, not t=2. Fix: pre-evolve to `ts[0]`, or assert `ts[0]==0`. |
| 🔴✨ | `spectral.py:1260` `apply` | Real Lanczos branch does `np.asarray(f(theta), float)`: complex `f` (e.g. `e^{-iHt}`) on **real-symmetric A + real v** → imaginary part silently dropped (the report said "crash"; reality is a silent wrong **real** result). | `apply(A, exp(-i·), ones, hermitian=True)` returns float, ≠ `exp(-iA)v`. Fix: don't cast to float when `f(theta)` is complex. |
| 🔴✨ | `defect.py:248` `defect_jump` | `np.asarray(D_n, float)` drops the imaginary part of a complex defect (module advertises complex-Hermitian / Jordan families). | `D=[1+2j,3+4j]` → returns float `[-3,1]` instead of `[-3-4j,1+2j]`. Fix: `np.asarray(D_n)`. |
| 🔴✨ | `cost.py:103` `rmt_class` | Index `len(s)//4 − 1` is **−1** for n≤4 → wraps to `s[-1]` → `R4=−0.75` → **always classifies GSE**. | n=3,4 both → GSE. Fix: `idx=max(0,…)` + guard `len≥5`. |
| 🔴 | `cloud_flow.py:287` `exceptional_point` | Golden-section refinement matches eigenvalues against the **stale** `target_pair` (closure), not the continued pair. Dead `cur = last` (line 314) is the abandoned fix. | 4×4 with drift: `refine=True` → `rig_min=1.0` (garbage), worse than `refine=False`. Fix: match against the scan's pair at the coarse minimum. |

### A2. Silently wrong operator / model

| ID | Location | Bug | Repro outcome |
|----|----------|-----|---------------|
| 🔴 | `brown.py:129` | Non-Hermitian callable given with `N` but no `rmatvec` → `rmv=A` (self-adjoint assumed), wrong `(A−zI)*(A−zI)`, no warning. Adjoint can't be inferred from a black box. | Non-normal op → realified eigs unrelated to σ²; end-to-end `nan`. Fix: require `rmatvec` (or explicit `assume_self_adjoint`). |
| 🔴 | `solve.py:113` `rayleigh_polish` | Matrix-free path uses MINRES (symmetric-only) on any callable, no symmetry check. | Non-symmetric 40×40: dense → 6.597 (right), MINRES → 0.276 (garbage), passes `info` guard. Fix: use `lgmres`/`gmres` on this path. |

### A3. Crashes / silent NaN on reachable inputs

| ID | Location | Bug | Repro |
|----|----------|-----|-------|
| 🔴 | `cloud_flow.py:131` | `nev=min(nev, N−2)` → 0 at N=2, −1 at N=1 → `eigs` `ValueError: k=0`. ARPACK shift-invert is impossible at N≤2 anyway. | `matrix_free=True`, N=2 crashes. Fix: route N<3 to dense `eig`. |
| 🔴 | `spectral.py` `of`/`apply`/`local_spectrum` | No `k≥1` guard → `k=0` raises `IndexError` (eigh of 0×0). | confirmed crash. Fix: `if k<1: raise ValueError`. |
| 🔴 | `spectral.py:~120` `_lanczos_core` | No `‖v0‖>1e-15` guard → `local_spectrum(A, zeros)` / `quadform(A,"inv",zeros)` return all-NaN silently. | confirmed. Fix: guard the norm. |
| 🔴 | `spectral.py:1080` `zoom` | Empty window → `gain=mean(empty)=nan` → all weights NaN, no warn. | `s.zoom(5,6)` off-spectrum → NaN. Fix: `if not in_win.any(): raise`. |
| 🔴 | `spectral.py:1171` `__repr__` | `repr(Spectral([],[]))` → `ValueError: zero-size array to reduction`. | crash in debugging. Fix: guard empty. |
| 🔴✨ | `defect.py:53` `richardson_limit` | `np.asarray(x,float)` → `TypeError` on a complex value sequence (e.g. extrapolating complex eigenvalues). | crash. Fix: drop the `float`. |
| 🔴 | `defect.py:42` `richardson` | `p=0` → `r=1` → `/(r−1)` div-by-zero → `[inf,inf]`. | confirmed. Fix: `if p<=0: raise`. |
| 🔴 | `defect.py:280` `generator_read` | `t=0` → `4n/t**2` `ZeroDivisionError`. | confirmed. Fix: `if t==0: raise`. |
| 🔴 | `defect.py:38` `defect` | No shape check → `(3,3)−(3,)` silently broadcasts (wrong defect) instead of erroring. | confirmed footgun. Fix: `if a.shape!=b.shape: raise`. |
| 🔴 | `cost.py:74` `level_spacing_ratio` | `len(s)<2` → `nan` (Mean of empty slice), silent. | confirmed. Fix: guard length. |
| 🔴 | `cost.py:31` `lift_rank` | Zero/constant signal → `s2.sum()=0` → `0/0 nan`; `is_extractable(zeros)` → confident `(False,[nan…])`. | confirmed. Fix: guard denominator. |
| 🔴 | `brown.py:196` `brown_measure` | Grid `n<3` → empty interior slice → all-zero `mu`, `mass=0`, no error (n=1 → `IndexError`). | confirmed. Fix: require ≥3 pts/axis. |
| 🔴 | `thermal.py:39` `state` | Large β underflows `w→0` → `psi/sqrt(w)` div-by-zero, norm `inf`, `Z=0`. | β=1000 confirmed. Fix: floor `w`. |
| 🔴 | `thermal.py:52` `expect` | `probes=0` → `0/0` → `(nan,nan)` silent. | confirmed. Fix: `if probes<1: raise`. |
| 🔴 | `subordination.py:25` `pastur` | Fixed point returns after `iters` with no convergence flag → silently inaccurate `g(z)` near hard edges. | two-atom near edge: residual stalls 3.5e-4, 0.27% ρ error, silent. Fix: return/warn on non-convergence. |

### A4. New correctness/robustness findings (✨, not in original report)

| ID | Location | Finding |
|----|----------|---------|
| 🟠✨ | `spectral.py:1105` `zoom` | Sets `_tridiags` on the filtered object while `_slq_scale=1`, `_n_atoms=0`, so `trace_certified` runs on a zoom object and returns a bogus degenerate "certificate" (`lo==hi`). Fix: set `_tridiags=None` in zoom, or reject filtered objects in `trace_certified`. |
| 🟠✨ | `cost.py:29` `lift_rank` | Short signal (`len < 2k−1`, default k=60) → empty/degenerate Hankel → silent `nan`/`1.0`; `is_extractable` (windows to k=120) silently runs truncated Hankels yet returns confident verdict. Fix: require `len ≥ 2k−1`. |
| 🟠✨ | `brown.py:252` | `mass` is summed from the **clipped** `mu≥0`, not `mu_raw` (~25% interior cells negative under noise) → mass biased >1 even when support is fully captured → unreliable "inside-the-box" diagnostic. Fix: integrate `mu_raw` for mass. |
| 🟠✨ | `__init__.py:26` | `from .cloud import cloud` rebinds `resona.cloud` to the **function**, shadowing the submodule (`resona.cloud.Cloud` → `AttributeError`). Same for `flow`/`solve`. Fix: rename or `__all__` discipline. |
| 🟠✨ | `wkernel.py:110` `track` | Per-mode `argmax` overlap matching is not one-to-one → two modes can double-assign to one column at a near-degenerate crossing, silently corrupting continuation (only the matrix-free block-boundary case warns). Fix: linear-assignment match. |
| 🟠✨ | `cost.py:84` `rmt_class` | Docstring reference vector (Poisson +0.21 / GOE +0.05 / GUE −0.05 / GSE −0.20) disagrees with the code's thresholds (0.199 / 0.009 / −0.080 / −0.230) → shifts borderline GOE/GUE calls. Fix: reconcile. |
| 🟠✨ | `brown.py:309` `_free_add_dos` | BB-subordination fixed point inside `brown_boxplus` shares the F2 silent-non-convergence flaw. Same fix pattern. |
| 🟠✨ | `solve.py:123` `rayleigh_polish` | One-sided Rayleigh quotient `vᵀAv` only gives cubic convergence for symmetric A; compounds A2's MINRES issue (routine is symmetric-only by design but accepts any callable). |
| 🟠✨ | `flow.py:41` `shock_time` | Assumes the band gap is at the arithmetic midpoint `0.5(lo+hi)`; valid for two equal-weight atoms, wrong location for 3+/unequal bands. Latent. |
| 🟢✨ | `spectral.py:1157` `effective_rank(with_err=True)` | Per-probe scatter omits the deflate atom contribution (unlike `trace`/`density`) → biased Φ₁ error bars when `deflate>0`. |
| 🟢✨ | `lift.py` `cauchy(s, z)` | Scalar-only; array `z` raises broadcast error (contradicts the vectorized style elsewhere). Errors loudly, low severity. |

---

## B. Strengthening / hygiene (real but low-stakes)

| Location | Item |
|----------|------|
| `spectral.py:868` `trace` | `trace(np.log)` on negative nodes → NaN with only a numpy warning; add a resona-level warn when `f∈{log,sqrt}` and `min node ≤ 0`. |
| `lift.py:61,80` | Bisection runs 110 iters; 55 is bit-identical (float64 below eps by then) — 2× cheaper. |
| `lift.py:53,78` | No residual/bracket validation after bisection → out-of-range target returns a bracket **endpoint** silently (root cause shared with the s_transform 2× bug). Add a residual check. |
| `free.py:164` `rie_clean_additive` | Cleaned eigenvalues can go negative silently (model breakdown); add a warn. (`rie_clean` proper is structurally ≥0 — claim N/A there.) |
| `cloud.py:47` Arnoldi early-exit | `h<1e-12` is absolute → premature truncation of tiny-norm operators (the "never breaks on huge ops" half did **not** reproduce). Use a relative tolerance. |
| `brown.py:395` `Z0≤0` | Unnormalized-DOS branch is silent but practically unreachable (`eta_free>0` keeps `Z0≈1`). Return a NaN sentinel for safety. |
| `spectral.py` `_supports_block` | Rejects `(N,1)` probes (block path silently disabled → slower, still correct); `atol=1e-300` is meaningless. Benign. |
| `beta.py` `beta_spectrum` | GIGO: inconsistent `μ1/μ2` (out of `[E0,Emax]`, over-dispersed, NaN) → silent edge-pinned/U-shaped levels. Only reachable by calling `beta_spectrum` directly — **not** via `beta_from` (Lanczos `extreme()` keeps moments consistent across 60 spiked seeds). Add input guards. |

---

## C. Corrections to the original report (do NOT act on these)

| Original claim | Verdict | Why |
|----------------|---------|-----|
| `cloud.py:35` division by zero on zero `v0` | ⚪ **FALSE** | 2,000,000 seeds at N=2,3: min `‖v0‖≈3.6e-4`. A standard-normal vector is never (near-)zero. Unreachable. |
| `brown.py:179` "use `seed+i` to decorrelate per-z bias" | ⚪ **BACKWARDS** | The shared seed is **correct**: correlated S-errors **cancel** in the 2nd difference (μ-noise amp 8.5). `seed+i` makes it **13× worse** (amp 49). Do not change. |
| `thermal.py:55` "error bars use √probes, should use n_eff" | ⚪ **FALSE** | Weights are already folded in; `code/sandwich = √(8/7)` exactly — it **is** the effective-sample-size SE. |
| `thermal.py:67` "`np.allclose(..., atol=0)`" | ⚪ **FALSE** | Call uses numpy **defaults** (`rtol=1e-5,atol=1e-8`), not `atol=0`. Premise wrong. |
| `cloud.py:96` "real probes bias the cloud for complex A" | ⚪ **FALSE** | Krylov space goes complex after the first matvec; real start is unbiased (mean-dist 0.565 vs 0.660 for complex). |
| `spectral.py:174` KPM Chebyshev blowup if eig ∉ `[lo,hi]` | 🟠 **unreachable** | Mechanism real, but 12-step×2-probe Ritz + 6% pad always brackets the true edge on every adversarial spectrum tried. Latent only. |
| `spectral.py:585` `_radau_value` singular at Radau≈Ritz | 🟠 **unreachable** | Interlacing forces `end` strictly outside the (k−1) Ritz spectrum via the public guard. Latent only. |
| `spectral.py:207` `_kpm_tridiags` `w.sum()<1e-300` | 🟠 **latent** | Guard missing but Jackson-damped density never renormalized to zero in practice. |
| `defect.py` `richardson_limit` "geometric doubling" docstring | 🟠 **benign** | Code is *more general* than the docstring (any ratio), not wrong. (The exponent bug A1 is the real issue.) |
| `wkernel.py` `kappa_w` "curvature" | 🟠 **wording** | Computes a first-difference Lipschitz slope of W (= curvature of λ, since dW=d²λ). Docstring is internally inconsistent (line 192 "curvature" vs 216 "Lipschitz"). Reword, no math bug. |
| `free.py` moment↔cumulant round-trip | ✅ **NO BUG** | `free_cumulants`↔`moments_from_cumulants` are exact inverses to ~1e-13 up to N=11; match the free `2κ₂²` signature symbolically. |

---

## Tally

- **Real bugs:** 8 wrong-math/silent-wrong (4 of them ✨ new) + 2 silently-wrong-operator + 16 crash/NaN-on-reachable-input + 4 new robustness = **~30 actionable**.
- **Corrections:** 6 claims **false/backwards** (notably the `seed+i` "fix" would regress accuracy 13×), 4 more overstated/unreachable.
- **No new API needed** for any fix — all are hardening of existing surface.

### Suggested fix order (impact × independence, each via the `library-change` gate)
1. `thermal.correlator` `ts[0]≠0` — major silent correctness bug, brand new.
2. `defect.richardson_limit` exponent — one line, wrong math in a public read.
3. `lift.carleman` constant term — breaks the documented Riccati use.
4. `cost.rmt_class` n≤4 → always-GSE — one-line clamp.
5. `apply` / `defect_jump` complex-cast (drop imaginary) — pair them.
6. `brown.rmatvec` + `solve.rayleigh_polish` non-symmetric — require/guard.
7. The reachable crash/NaN guards (batch: `k≥1`, `‖v0‖`, `zoom`, `richardson p`, `generator_read t`, `defect` shape, `lift_rank`, `level_spacing_ratio`, `brown_measure n`, `state`/`expect`).
8. `s_transform` bracket + residual check; `cloud_flow` N≤2 and EP refinement.
