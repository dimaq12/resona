# Changelog

All notable changes to resona.  The discipline throughout: every number below
is printed by a test or a gallery stand, not asserted by hand.

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
