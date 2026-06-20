# resona — complexity of each method

Cost of every public method, with the architectural point made explicit: the
**forward** transforms are *matrix-free* (cost set by the matvec, not by `N`), the
**inverses** are polynomial in `N`, and the difficulty mirrors how much spectral
data each consumes.  Notation: `N` = dimension, `C` = cost of one matvec (`O(N)`
dense, `O(nnz)` sparse), `k` = Lanczos/Arnoldi steps, `p` = probes.  Empirical
slopes (measured `time ∝ N^s` on tridiagonal operators, N=64…512) are in **bold**.

## Forward / read — MATRIX-FREE (the point of the library)

| method | time | data | matrix-free | notes |
|--------|------|------|:-----------:|-------|
| `of(matvec,N,k,p)` | `O(p·k·C + p·k²)` — **N^0.2 (flat)** | matvec | ✓ | SLQ; cost is the matvec, not N.  If the matvec maps blocks (`A@X`, verified automatically, N≥1000), all p probes ride ONE block matvec per step — BLAS-3, ~2–4× measured |
| `of(..., deflate=K)` | + one padded Lanczos `O((K+16)·C)` | matvec | ✓ | Hutch++ at the measure level: top-K exact atoms + complement probes.  Measured variance drop on spiked operators: **63× (Tr A²), 724× (Tr e^A)**; ~1× for log-flattened f — `effective_rank` predicts the gain |
| `of(..., engine="kpm")` | `O(p·4k·C)` matvecs, **no reorth** (the `p·k²·N` term vanishes) | matvec | ✓ | Chebyshev/Jackson harvest → same `Spectral` via the measure's recurrence.  Wins at high resolution: **2.7× at k=256** with equal-or-better moments; edges smoothed ~span/(4k); no certificates |
| `apply(matvec,f,v,k)` | `O(k·C + k²·N)` (+`k³` Arnoldi) — **N^0.2** | matvec | ✓ | f(A)·v; solve/evolve |
| `of`/`apply` with torch/cupy/array-API matvecs | unchanged, on the user's device | matvec | ✓ | invisible dispatch on the array namespace of the matvec's output (`of`) / the input vector (`apply`); vectors never leave the device — only (α,β) and the k×k eigh visit the host.  COVERAGE: `of(engine="lanczos", deflate=0)` (real & complex-Hermitian; certificates work — the harvested (α,β) are host-side) and `apply` (incl. the complex-Hermitian branch; a backend that rejects complex vectors on a real operator is handled by linearity, 2 matvecs/step); `engine="kpm"` / `deflate` **raise** on device operators, `zoom` / `quadform` are host-only for now.  The numpy path is bit-identical with the same matvec count (measured; the dispatch check is ~0.4 µs against ~170 ms per `of`).  fp32 HONESTY: torch's default float32 caps SLQ at single precision — measured 3e-8 rel. err on Tr A², expect 2–3 digits for stiff f (log-det near a small λ_min); build the operator on float64 tensors for ~1e-12 agreement with numpy (measured, torch CPU).  GPU speedup not yet measured (this dev box has no CUDA); torch **CPU** float64 is ~2.8× slower than numpy BLAS at N=2000 — the device path pays off only when the operator already lives on an accelerator |
| `local_spectrum / local_density` | `O(k·C + k²·N)` | matvec, one probe | ✓ | one Lanczos from a chosen v |
| `moment / trace / extreme / effective_rank / density` | `O(p·k)` | the nodes/weights | ✓ | reads of's output |
| `trace_certified(...)` / `quadform(certified=True)` | `O(p·k³)` (Radau eigh per probe) | stored (α,β) | ✓ | rigorous Gauss–Radau brackets, zero extra matvecs |
| `Spectral.zoom(a,b,degree)` | `O(p·(degree+k)·C)` | matvec | ✓ | Chebyshev slicing; transition band ~span·π/degree |
| `free.freeness_defect / cross_moment` | `O(p·|word|·C)` | matvec | ✓ | Hutchinson |
| `subordination.pastur` | `O(iters·N)` per z | spectrum (nodes,w) | ✓* | scalar fixed point |
| `subordination.pastur_grid / averaged_dos / flow.burgers_density` | `O(iters·n_x·p·k)` vectorized | spectrum | ✓* | whole x-grid in ONE damped iteration with an active mask (~10× the scalar loop), same fixed point & tol |
| `flow.shock_time` | `O(n_t · averaged_dos)` | spectrum | ✓* | scans t (measured 31 s → 2.2 s on the vectorized grid) |
| `defect.sigma_min(matvec,z)` | `O(k·C)` (2N realified Lanczos) | matvec (+rmatvec) | ✓ | dense path: one SVD `O(N³)`, exact |
| `defect.pseudospectrum_radius` | `O(iters·sigma_min)` | matvec (+rmatvec) | ✓ | log-bisection on the bloom, ~60 σ_min calls |
| `defect.normality(matvec,N,rmatvec)` | `O(p·C)` (p probes × matvec+rmatvec) | matvec (+rmatvec) | ✓ | global `‖[A,A*]‖²_F` by Hutchinson; the non-normality flag, no `eig` |
| `defect.hard_points(family,ks,B)` | `O(\|ks\|·p·k_cg·C)` | matvec (+B matvec) | ✓ | EP / avoided-crossing locator: `argmax_k Φ_η` via Hutchinson+CG, never diagonalizes |
| `cost.level_spacing_ratio(λ)` | `O(N log N)` | eigenvalues | — | 3 lines; resolve symmetry sectors first |
| `defect.generator_read` | `O(N)` (a scaled difference) | two solver runs | ✓ | BE-specific constant; refuses CN (measured O(1) deviation) |
| `defect.defect_barycentres` | `O(Σ\|band\|)` | the defect power in the caller's basis | ✓ | barycentre read; ±1-bin rounding; ~5× the ratio method's matvecs upstream |
| `wkernel.track(modes='all')` | `O(steps·N³)` (one eigh per midpoint) | dense family (A0, B_j) | ✗ | crossing-safe; ~100× frozen-W accuracy per eigh; 44–302× fewer eigh than FD continuation |
| `wkernel.track(modes=k)` | `O(steps·N·k)` (one `eigsh` block per midpoint) | matvec family | ✓ | selected bottom/top-k modes, MATRIX-FREE — closes the W-side dense `O(N³)`; ~294× at N=4000, rel.err 1e-10; `guard` warns on a mode leaving the block |
| `wkernel.kappa_w(modes='all')` | `O(probes·N³)` | dense family | ✗ | frozen-W ACCURACY dial (ρ=0.93); NOT a cost dial (blind ρ≈0.05) |
| `wkernel.kappa_w(modes=k)` | `O(probes·N·k)` (`eigsh` sub-Jacobian) | matvec family | ✓ | curvature of the selected-mode sub-Jacobian, MATRIX-FREE — the conjugate W-side completed |
| `subordination.contraction` | one `pastur_grid` + one G′ eval | spectrum | ✓* | pure read; critical window narrows with σ² (measured) |
| `cloud(mv,N,k,p)` | `O(p·k·C + p·k²·N)` (Arnoldi+DGKS) | matvec | ✓ | complex Ritz cloud ⊂ numerical range; reads are transient-growth dials |
| `brown.brown_measure(A)` | `O(grid·p·k·C)` — one SLQ log-det per grid point | matvec (+rmatvec) | ✓ | Brown measure via hermitization: `S(z)=(1/2N)Tr log((A−z)*(A−z))`, `μ=(1/2π)ΔS`; exact-SVD path is `O(grid·N³)` |
| `brown.brown_boxplus(A,B)` | `O(grid·(per-z free conv.))` | spectra | ✓* | free additive sum of two Brown measures (per-z Hermitian free convolution of the hermitizations) |
| `cloud_flow.cloud_wkernel / cloud_track` | `O(m·k_arn·C)` shift-invert Arnoldi (matrix-free) / `O(N³)` dense eig | matvec family | ✓ | biorthogonal `∂λ=(u*Bv)/(u*v)` from left+right Ritz pairs; EP locator (denom→0) |
| `cost.rmt_class(λ)` | `O(N log N)` | eigenvalues | — | R4 / spacing universality class (Poisson / GOE / GUE) |
| `lift.koopman(X)` | one thin SVD `O(n·T·min)` | snapshot matrix | — | returns the r×r reduced action; r = reported data rank |
| `thermal.expect/correlator` | `O(probes·k·C)` / `·len(ts)·2` | matvec | ✓ | typicality error ~1/√D_eff, stderr reported |

`✓*` = matrix-free given the spectrum, which `of` reads matrix-free.

## Inverse — recover the operator (the DATA ↔ CONDITIONING hierarchy)

| method | time | data needed | matrix-free | conditioning |
|--------|------|-------------|:-----------:|--------------|
| `from_eigenbasis(λ,V)` | `O(N²)` — **N^1.9** | full eigenbasis `V` (`O(N²)`) | ✗ (needs eig) | EXACT, every operator (~1e-14) |
| `from_measure(λ,w)` | `O(N²)`–`O(N³)` — **N^1.5** | one boundary measure (`O(N)`) | ✓* | exact (smooth) / blows up (sharp) |
| `wkernel.design(W,y,reg)` | `O(N³)` SVD (reg) / `O(m·M·minⁿ)` lstsq | response `W` | depends | bounded (reg>0), smoothed |
| `wkernel.wkernel(V,B)` | `O(m·M·C)` | m eigenvectors + M perturbations | ✗ (needs eigvecs) | — |

The more data you keep, the better conditioned the inverse: full `V` (`O(N²)` data)
→ exact; boundary measure (`O(N)` data) → ill-posed for sharp operators; eigenvalues
only → needs regularization (bounded but biased).  *Difficulty = data discarded.*

## Theory transforms

| method | time | notes |
|--------|------|-------|
| `lift.r_transform / s_transform` | `O(110·n_w·p·k)` | vectorized bisection over `G(z)=Σw/(z−λ)`, ALL w at once (~4× the per-point root-find, tighter than brentq) |
| `lift.conserved_charge(H, basis)` | `O(|basis|·D²·ω)` commutators + `O(|basis|³)` eigh | dense H; the honest heavy tool — it FORMS `[H,O_a]` |
| `solve.catastrophe_solve` | float64 `np.roots` + `mp.polyroots` at `dps=q·target` | budget auto-set by the detected cluster order q |
| `solve.rayleigh_polish` | `O(iters·N³)` dense solve / `O(iters·k·C)` minres | cubic convergence: iters ≈ 4–6 |
| `lift.free_convolution` | `O(order³)` | cumulants ↔ moments, small `order` |
| `lift.carleman_scalar` | `O(order²)` build + `apply` | polynomial-ODE lift |
| `lift.carleman_gf(p,n,·)` | `O(M³)`, `M=pⁿ` (numpy-vectorized, ~20× the Python loop) | exact GF(p) logic lift; the one genuinely heavy kernel |
| `beta.beta_spectrum / beta_from` | `O(N)` | inverse-CDF |
| `defect.richardson` | `O(N)`; `richardson_limit` `O(L²·N)` | extrapolation |
| `free.free_cumulants / moments_from_cumulants` | `O(n³)` (truncated poly powers) | small n (moments) |
| `cost.lift_rank` | `O(k³)` (SVD of a k×k Hankel); `is_extractable` `O(#windows·k³)` | the dial |
| `cost.extraction_cost / fit_law` | `O(#data)` | the law |

## The headline

- **Forward = matrix-free.** `of`/`apply` cost the matvec, not `O(N³)` — flat in the
  measurement. That is the whole library: never form the matrix, never call `eig`.
- **Inverse = polynomial, and conditioning ∝ data kept.** `from_eigenbasis` is exact
  (`O(N²)`, full eigenbasis); `from_measure` uses `O(N)` boundary data and is
  ill-posed for sharp operators; regularization buys robustness at the price of bias.
- **One heavy kernel:** `lift.carleman_gf` is `O(pⁿ·³)` — exponential in the number of
  variables (it builds the full GF(p) monomial table); fine for `n ≲ 10`, vectorized.

*Measured with the benchmark in the commit that added this file; reproduce by timing
the methods across `N` and fitting `log(time)` vs `log(N)`.*
