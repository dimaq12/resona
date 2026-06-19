# Changelog

All notable changes to resona.  The discipline throughout: every number below
is printed by a test or a gallery stand, not asserted by hand.

## [3.0.0] — 2026-06-19
Matrix-free epic — kill resona's last dense O(N³), add three matrix-free spectral
reads.  All additions are OPT-IN (the old defaults are bit-for-bit unchanged: the
gallery diffs IDENTICAL against pre-epic `main` save timing/RNG columns; 126 tests
green; absolute-arm certificates 4/4).  Numbers below are printed by the stand
`examples/spectral_phenomena/matrix_free_kernel.py` and asserted in `tests/test_theory.py`.

- **The cube-killer — `wkernel.kappa_w(modes=k)` / `track(modes=k)`.**  κ_W and the
  spectral flow only need the eigenvectors of the modes you track.  `modes=k` routes
  the block to shift-invert Lanczos (`eigsh`) instead of a full `eigh` — matrix-free
  (only matvecs / sparse solves), O(N·k) instead of O(N³) for sparse/structured A.
  - `kappa_w(modes=6)` vs dense κ_W on the same block: **rel.err ~1e-10**, speedup
    **9.9× (N=800) → 114.7× (N=3000)**, growing as N³ (dense 18.2 s vs 0.16 s at N=3000).
  - `track(modes=4)` vs dense `track` (same method): **max|Δλ| = 2.8e-15** (identical);
    a `guard` warns when a tracked mode is about to leave the selected block.
  - `modes='all'` (default) is the unchanged dense path (`kf == ka`, verified).
- **`defect.normality`** — departure-from-normality energy ‖[A,A*]‖²_F, matrix-free
  Hutchinson with **Rademacher probes + median-of-means** (the diagonal of [A,A*]ᵀ[A,A*]
  is captured with zero variance).  Dense random N=400: **rel.err 0.06%** (a Gaussian-probe
  seed that read 49.8% off); **= 0 exactly** for normal (Hermitian/symmetric) operators.
- **`defect.hard_points`** — matrix-free avoided-crossing / exceptional-point locator.
  Φ_η(k)=(η/π)²Tr[B R B R], R=((H(k)−E)²+η²)⁻¹ via Hutchinson + CG (no eig) spikes where
  the gap closes: **argmax_k Φ_η = the avoided crossing (k=0)**, verified against the true
  min-gap.  CG convergence flags are captured and a warning is raised on non-convergence.
- **`cost.rmt_class`** — random-matrix universality class (Poisson/GOE/GUE/GSE) from the
  rigidity meter R4 = ω₄(sorted unfolded spacings); the companion to `level_spacing_ratio`.
  Strict rigidity ordering **Poisson +0.207 > GOE +0.046 > GUE −0.064** (ensemble-averaged).
- **`beta.beta_from(robust=True)`** — optional Rademacher–Hutchinson moments for the Beta-law
  spectrum reconstruction; a GOE seed whose SLQ-quadrature moments read **5.84% off drops to
  0.41%**.  Default (`robust=False`) reproduces `s.levels()` exactly.

## [2.0.1] — 2026-06-13
Bugfix — `examples/quantum/phase_transition.py` had TWO stacked bugs that the
gallery ratchet could not see (it diffs output *stability*, not *correctness*).
Both confirmed against dense exact and the free-fermion analytic TFIM ground
energy; found by an independent multi-agent correctness audit + dense arbitration.
- **Root bug — the Hamiltonian was built wrong.**  `build_tfim_parts` used
  `rows = np.repeat(states, n)`, which is **misaligned** with the bit-major
  `cols = concatenate([states ^ (1<<i) ...])`, so the −ΣXᵢ flip operator (and
  hence H) was **non-symmetric and incorrect** (`allclose(H,Hᵀ)=False`).  resona's
  Hermitian Lanczos then returned a wrong E₀ (−14.31 vs true −15.32).  **Fix**:
  `rows = np.tile(states, n)`.  H is now symmetric and its ground energy matches
  the free-fermion exact value to 1e-14 (n=12,h=1: −15.3226), and resona's
  matrix-free E₀ matches dense `eigvalsh` to 1e-14.
- **Solver bug — the resolvent never converged.**  Φ_η solved `((H−E₀)²+η²)⁻¹`
  with plain CG on the SQUARED operator (condition κ²): `info=5000` every call,
  returning garbage (‖Rz−exact‖/‖·‖ ≈ 16–40).  **Fix**: factor
  `R = ((H−E₀)−iη)⁻¹·((H−E₀)+iη)⁻¹` — two complex-shifted `bicgstab` solves
  (condition κ), `info=0`, ‖Rz−exact‖/‖·‖ ≈ 1e-5.  **48× faster** (191s → 4.0s).
- With both fixes the susceptibility now matches the dense-exact `Tr[B R B R]`
  curve (up to Hutchinson noise) — it rises through the transition and plateaus,
  and `LOCATED ✓` is now a real detection, not a heuristic landing on garbage.
  Baseline snapshot re-ratified; 117 tests green.
- **Ratchet note**: this exposed that the gallery ratchet guards output
  *stability* but not *physical correctness*, and that ~15 stochastic stands ship
  without a fixed RNG seed (so they can't be reproducibly verified).  Follow-ups:
  dense/analytic certificate anchors for the numeric stands, and seeded RNG.

## [2.0.0] — 2026-06-13
EPIC3 Phase 5 — THE BREAK (owner-approved).  The deprecation cycle closes:
old names removed, one consolidation, zero math changes.
- **Removed**: `trace(certified=True)` (→ `trace_certified`),
  `defect.spectroscopy` (→ `defect_barycentres`), `cost.phi1`
  (→ `effective_rank`), `shock_time`'s grid/threshold knobs (now internal;
  dials: `t_max`, `eta`).  See MIGRATION.md — every removal has a 1.4+ name
  that has been live since 2026-06-12.
- **The Lanczos consolidation**: `apply`'s two inline Hermitian Krylov loops
  and `_lanczos`/`_lanczos_herm` now share ONE `_lanczos_core` (callers own
  the start-vector normalization, which keeps every call site bit-identical
  to its pre-merge loop).  Parity certificate: the full example gallery
  diffs clean against the 1.5 baseline (timing lines only).
- CI fix: the version test now asserts `__version__` == pyproject (the real
  invariant) instead of a hardcoded literal that broke on every release.
- Gallery and tests migrated to the 2.0 names; 117 tests.

## [1.5.0] — 2026-06-12
EPIC3 Phase 3 "the doors" — three interop/capability sockets, no new math
surface beyond them, boundary contract intact.
- **`resona.grad_trace`** — DIFFERENTIABLE spectral reads: ∂/∂θ_j Tr f(A(θ))
  = Tr(f′(A)·∂A/∂θ_j), Hutchinson probes SHARED across all parameters (one
  Krylov chain per probe + one matvec per parameter).  `with_err=True` gives
  the per-component probe scatter.  Verified against dense ground truth
  (log-det gradient Tr(A⁻¹B); multi-parameter sharpness 2·Tr(AB_j)).  This
  is the socket that makes spectral regularizers trainable.
- **The interop shim**: everywhere a matvec is taken — `of`, `apply`,
  `quadform`, `local_spectrum`/`local_density`, `cloud`, `grad_trace` — a
  scipy `LinearOperator`, sparse matrix, or dense array now works directly,
  with N read off `.shape` (`resona.of(L)`).  Bare callables are untouched
  bit-identically; shape/N contradictions raise.
- **`defect.generator_read_converged`** — the Richardson line, written:
  three resolutions → two independent generator reads → (extrapolated G,
  rel_dev convergence certificate); measured in tests: the extrapolated
  read beats the plain one with rel_dev < 0.2 as the gate.
- 117 tests; full-gallery ratchet: zero metric diffs (additive only).

## [1.4.0] — 2026-06-12
EPIC3 "the deprecation release" — honesty completion, zero breaks (every old
name still works; removals happen in 2.0 only).  See MIGRATION.md.
- **EPISTEMIC PARITY** — the 2.0 north star lands early: every stochastic
  read now offers its scatter.  `density(xs, with_err=True)`,
  `cumulants(order, with_err=True)` (nonlinearity propagated by per-probe
  recomputation, not linearization), `extreme(with_err=True)` (a
  reproducibility bar, stated as such), `wkernel.kappa_w(..., full=True)`
  (the whole per-direction distribution — curvature anisotropy).
- **The trace split**: `s.trace_certified(f, support=)` — one name per
  return shape; `trace(certified=True)` keeps working through 1.x.
- **`defect.defect_barycentres`** — the honest canonical name
  (`spectroscopy` promised a spectrum it does not deliver; it remains as a
  legacy alias until 2.0).
- **`resona.synthesize`** — the discoverable verb for `from_measure`
  (operator construction; both names stay).
- **`lift.r_inverse` / `lift.s_inverse`** — the missing duals of the R/S
  transforms for spectral design: monotone-window check + bisection to
  machine tightness (round-trip < 1e-10 in tests).
- **The measured retrofits** (EPIC3 Phase 0/1 — every delta measured in a
  scratch BEFORE the gallery; falsified predictions recorded in
  .audit/EPIC3_phase0.md): Chern marker via a precomputed Jackson/Chebyshev
  projector (C = +0.985 at L=20 in seconds; 7× faster at L=12);
  Heisenberg thermodynamics by deflating −H (ground states become exact
  atoms: max Z err 19.4% → 0.09% at L=8, 16.3% → 0.67% at L=12); GP log-det
  deflate=64 (0.84% → 0.53%); Koopman VdP at 4× data (|Δf| = 3e-4, finer
  than the FFT reference's own 1e-3 bin).
- **Persona landing pages** (docs/personas/): five half-page doors — quants,
  physicists, ML, dynamics, numerical analysts — each mapping the persona's
  first five tasks to five verified calls.
- 108 tests; full-gallery ratchet: only the four retrofitted stands moved,
  all by improvement; default paths bit-identical.

## [1.3.0] — 2026-06-12
EPIC2 "the defect listens back" — every port stress-tested in the source
corpus FIRST (FA/revise_stress/STRESS_REPORT.md: 4 of 5 gold survived; the
fifth was demoted by its own table and NOT shipped).  Boundary contract held:
+5 callables, +0 modules, no god-methods.
- **defect.generator_read** — the solver's defect IS the Koopman generator
  (BE: D_n = (t²/4n)·A²e^{−tA}u₀, O(n⁻²) exact incl. defective Grcar κ=∞);
  Crank–Nicolson refused with the measured O(1) deviation.
- **defect.spectroscopy** — per-band barycentre of the defect power (BDS):
  35/35 PDE suite reproduced THROUGH the library function; stable at 5%
  snapshot noise where the classical ratio estimator dies at 1e-5.
- **wkernel.track** — crossing-safe spectral-flow integration (exact at a
  crossing where sorted eigenvalues break by O(1); ≥100× frozen-W);
  **wkernel.kappa_w** — the trust-region dial, documented by its negative
  result first (accuracy ρ=0.93; cost ρ≈0.05 — NOT a difficulty oracle).
- **subordination.contraction** — |T′| of the Pastur fixed point: the
  edge-of-chaos read as a pure observable (the critical-window caveat from
  the stress campaign is in the docstring).
- **THE API BEAUTY PASS** (API_REVIEW.md): three persona reviews + taxonomy
  audit; newcomer guessed 8/10 calls blind.  14 polish items applied (leaked
  imports unlisted, trace certified+with_err guard, effective_rank with_err,
  rie_clean/zoom/track domain guards, honest docstring reframes); 9 written
  up for 2.0, not applied; 9 examined and kept.
- Stand: examples/defect_spectroscopy.py (generator to 4.2e-5
  Richardson-checked; the noise table live).
- 96 tests; the ratchet stayed clean throughout.

## [1.2.0] — 2026-06-12
The epic (see [EPIC.md](EPIC.md)) — three contracts held throughout: zero new
top-level verbs, the metric ratchet (full-gallery diff at every phase, zero
regressions), pre-registered honest limits.
- **Certificates**: `s.trace(f, certified=True, support=)` → Gauss–Radau
  brackets of the k-truncation (Golub–Meurant); `resona.quadform(...,
  certified=True)` — quadratic forms with UNCONDITIONAL brackets (GP
  posterior variance with a proof; width 1.5e-1 → 3.8e-4 as k = 8 → 24).
- **`s.zoom(a, b)`** — Chebyshev spectrum slicing: interior eigenvalues with
  no dense seed (4e-16 of the span after polish).
- **`of(deflate=K)`** — Hutch++ at the measure level: exact top-K atoms +
  complement probes; variance −63× (Tr A²), −724× (Tr eᴬ), honest ~1× on log.
- **`of(engine="kpm")`** — Chebyshev/Jackson harvest, no reorthogonalization;
  2.7× at k=256, same `Spectral` out; no certificates (smoothed measure).
- **Invisible GPU**: array-API dispatch in the Krylov cores — torch/cupy
  matvecs run on-device with zero new API; numpy path bit-identical.
- **`resona.cloud`** — the honest non-Hermitian sibling (Arnoldi+DGKS;
  `abscissa()` reads the NUMERICAL abscissa — the transient-growth dial).
- **`lift.koopman`** — data → the Koopman/DMD propagator's action (one SVD).
- **`resona.thermal`** — typicality: ⟨O⟩_β, C(t), S(ω); incremental stepping
  (O(T) evolutions; 29m48s → 2m30s on the L=13 stand).
- **`apply`** gains the complex-Hermitian Lanczos path (vs expm to 1e-9).
- Jaw gallery: `chern_from_noise` (a topological INTEGER from Krylov chains;
  trivial phase 0.0000 exact), `isospectral_drums` (Kac executable),
  `koopman_dynamics` (odd-harmonic ladder to 7e-4 vs FFT; the chaos wall),
  `thermal_response`, `certified_logdet`.
- Honest parking: the SLQ-measure spectral form factor failed its honesty
  probe (corr 0.45) and was NOT shipped — recorded in EPIC.md.
- 75 tests; CI gallery smoke; CHANGELOG started.

## [1.1.1] — 2026-06-12
- Fix: the 1.1.0 wheel shipped with a stale `__version__ = "1.0.0"` string
  (a disk-full shell interruption skipped one sed; PyPI files are immutable,
  hence the post-release).  Metadata and string now agree.

## [1.1.0] — 2026-06-12
- `of()` / `local_spectrum` probe **complex Hermitian** operators natively
  (complex Gaussian probes, complex Lanczos; the Jacobi tridiagonal and the
  whole read side stay real).  Auto-detected; the real path is bit-untouched.
- `trace` / `moment` gain `with_err=True` → (value, stderr) from the scatter
  of the independent probes — error bars at zero extra matvecs.
- `free.rie_clean` / `free.rie_clean_additive` — free deconvolution
  (Ledoit–Péché / Bun–Bouchaud–Potters RIE): un-add Marchenko–Pastur or
  semicircle noise at the eigenvalue level.  Measured: 1.81× closer to the
  true covariance, 95% of the oracle; the min-variance portfolio's risk
  self-deception drops 4.15× → 0.94× (`examples/covariance_cleaning.py`).
- GitHub Actions CI (py3.10 / py3.12).
- Gallery: `science/hilbert_polya.py` (the zeta-zero operator: realized to
  2.8e-13, Weyl law corr 0.9996, β-fluctuations 1.42× smoother than GUE
  controls), `science/zeta_ascent.py` (Odlyzko tables: ⟨r⟩ = 0.6000 at zero
  #10^12; the rigidity excess decays 1.42× → 1.06× up the critical line).
- `theory/zeta_saturation.py`: the Berry saturation law measured —
  L_sat ≈ ln(t/2π) over eighteen decades of height (0.91/0.96/0.97 at the
  10% criterion), plateau follows (1/π²)·ln ln t to ±0.02.

## [1.0.0] — 2026-06-12
- The journey docs: README with three doors (plain hook / theorem dictionary /
  cookbook), `docs/tour.md` (ten stops, plain words → one-line theorem →
  runnable code, all snippets executed with warnings-as-errors), hero banner
  + metro-map SVG.
- `examples/spectral_phenomena/operator_synthesis.py`: design a measure
  (bands, ⊞, flow), realize a local tridiagonal matvec — eigenvalues match
  the order to 5.6e-15; gap closed live via the free-heat knob.
- README images/door links absolute so the PyPI page renders the journey.

## [0.4.0] — 2026-06-12
- **The Spectral hub**: `boxplus` (+ `as_spectral` via Golub–Welsch),
  `cauchy`, `r`/`s` (+ full-name aliases), `cumulants`, `flow`, `shock_time`,
  `levels`, `condition` (a LOWER bound; `polish=True` refines edges) — all
  reading off the one object.  `apply`-name collision resolved
  (method `reprobe`, `apply` kept as alias).
- **Speed, accuracy bit-preserved** (full-gallery baseline diff): block-probe
  Lanczos (BLAS-3, auto-verified, 2.6× dense / 1.4× sparse), vectorized
  Pastur grid (`shock_time` 31 s → 2.2 s), broadcast `density` (75×),
  vectorized-bisection R/S transforms (~4×, tighter than brentq).
- **Ported from the spectral_hardness program** (verified): `solve`
  (`catastrophe_solve` — cluster order q auto-detected, spends exactly
  q×target digits; `rayleigh_polish`), `defect.pseudospectrum_radius` /
  `sigma_min` (the ε^{1/q} bloom law, dense + matrix-free),
  `lift.conserved_charge`, `cost.level_spacing_ratio`.
- Gallery: 21 stands densified onto the hub with zero metric drift (the only
  changed numbers were improvements: machine-zero rate 99% → 100%);
  `nonnormal_convergence.py`, `integrability_detector.py` added.
- 28 → 47 tests; `pip install resona` (first PyPI release).

## [0.1.0 – 0.3.0] — 2026-06-10 (pre-PyPI)
The research program condensed into a library: `Spectral.of` (SLQ), compose
(`+`, `@`), read (`trace`/`density`/`extreme`/`moment`), `apply` (f(A)·v),
`local_spectrum`/`local_density`, inverses (`from_measure`, `from_eigenbasis`),
theory modules (`wkernel`, `lift`, `beta`, `defect`, `free`, `subordination`,
`cost`, `flow`), the examples gallery, THEORY/NOVELTY/FRONTIER/COMPLEXITY.
