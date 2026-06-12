# Changelog

All notable changes to resona.  The discipline throughout: every number below
is printed by a test or a gallery stand, not asserted by hand.

## [Unreleased — 1.2.0 epic]
See [EPIC.md](EPIC.md): certified Gauss–Radau trace brackets, spectral zoom,
deflated probing (Hutch++ at the measure level), KPM engine, invisible GPU via
array-API, `cloud` (non-Hermitian), `lift.koopman`, `thermal`, the jaw gallery.

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
