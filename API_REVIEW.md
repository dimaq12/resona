# API REVIEW — the beauty pass (EPIC2 Phase C, 2026-06-12)

Protocol: three independent persona reviews (newcomer / mathematician /
maintainer) over the full public surface + a mechanical taxonomy audit.
Verdicts in three bins per the pre-commitment: **POLISH** (applied now,
back-compatible), **PROPOSE-2.0** (written up, NOT applied — breaking changes
are the owner's decision), **LEAVE** (justified as-is).  The reviews were
allowed to find nothing; they found real things instead.

The headline: the newcomer guessed 8/10 task-calls on the first try; the
mathematician's verdict is "honest when docstrings are read"; the maintainer
found two pieces of true debt and certified the rest of the complexity as
principled.  The surface holds — what follows is sanding, not surgery.

---

## POLISH — applied in 1.3.0 (back-compatible)

| # | finding (source) | fix |
|---|---|---|
| P1 | `subordination.cauchy` and `thermal.apply` are LEAKED internal imports — accidental public names duplicating lift.cauchy / resona.apply (taxonomy audit) | import as private; surface shrinks by two accidents |
| P2 | `trace(certified=True, with_err=True)` silently ignores with_err (newcomer) | ValueError: the two ask for different error sources |
| P3 | `spectroscopy` name promises a spectrum, delivers per-band barycentres (mathematician: semantic drift) | docstring now LEADS with "per-band compression to one coordinate"; rename → 2.0 bin |
| P4 | `rie_clean(q)`: q unexplained, domain unvalidated (newcomer + mathematician) | docstring "q = N/T"; ValueError outside (0, 1.1] |
| P5 | `rayleigh_polish(sigma=…)`: name reads as std-dev (newcomer) | docstring first line: "sigma = the SHIFT (a Ritz seed near the target)"; renaming the kwarg would break callers → 2.0 |
| P6 | `lift.koopman` family membership reads awkwardly (newcomer + maintainer) | lift.py module docstring gains the bridge sentence: "and one BRIDGE: koopman — data → operator, the data-driven Carleman" |
| P7 | `contraction` is a solver-stability diagnostic, not a spectral property (maintainer) | docstring line added |
| P8 | `zoom(a, b)`: a < b never validated (mathematician) | ValueError |
| P9 | `wkernel.track` silently accepts non-symmetric A0 → garbage (mathematician, MED) | symmetry check → ValueError naming the verified domain |
| P10 | `effective_rank()` is stochastic but offers no error bar — the one true with_err coherence gap (mathematician) | `effective_rank(with_err=True)` → (value, stderr) from per-probe Φ₁ scatter; same pattern as trace/moment |
| P11 | `cost.phi1` is a dead wrapper over `effective_rank` (maintainer, HIGH) | docstring marks it a thin alias, "prefer s.effective_rank()"; removal → 2.0 |
| P12 | `defect.defect` has a one-line docstring for a boundary-declaration function (maintainer) | docstring explains WHY it exists (the module's owned object), not just what it subtracts |
| P13 | Cloud-vs-Spectral never contrasted at entry (newcomer) | one line in `__init__` docstring: Hermitian → Spectral; non-Hermitian → cloud |
| P14 | `f` accepts string-or-callable invisibly in trace/quadform (newcomer) | docstrings show both forms in the first example |

## PROPOSE-2.0 — written up, NOT applied (breaking or surface-growing)

1. **Split `trace`**: `trace(f, with_err=)` + new `trace_certified(f, support)`
   (maintainer's CRITICAL: three return shapes under one name).  Clean, but
   breaking; the 1.3 guard (P2) removes the sharpest edge.
2. **Rename `spectroscopy` → `defect_barycentres`** (mathematician).  More
   honest name; rename = breaking; P3's docstring reframe carries 1.3.
3. **`synthesize` alias for `from_measure`** (newcomer's task-mapping gap).
   Would exceed this epic's +5-callable cap; also arguably a docs problem —
   the cookbook row exists.
4. **Remove `cost.phi1`** (after one deprecation cycle).
5. **`r_inverse` / `s_inverse`** — the missing duals for spectral design
   (mathematician: monotone, bisectable; elegant closure of the pair).
6. **Error-bar parity across the surface**: density bounds, cumulants
   with_err, kappa_w distribution, generator_read convergence helper.
7. **`shock_time` knob lock** (6 params → 2) — breaking signature diet.
8. **Consolidate apply's inline Lanczos** with `_lanczos_herm` — internal,
   but touches the bit-parity ratchet; do it with a dedicated parity CI run.
9. **Multiplicative deconvolution beyond MP** — mathematically open; lives
   in FRONTIER, not in code.

## LEAVE — examined and kept, with reasons

- **`of(engine=, deflate=)` branching** (maintainer): three co-equal engines
  answering different questions, explicit guards, bit-frozen default. Earned.
- **`apply`'s Lanczos/Arnoldi/complex-Hermitian branches**: mathematical
  dispatch, not domain creep.
- **The five Lanczos variants** (+device twins): each preserves a contract
  (realness / Hermiticity / BLAS-3 / device residency).  Principled.
- **`boxplus` returning moments**: the honest content of the measure-level
  operation; `as_spectral=True` exists and states its own limits.
- **`local_spectrum`/`local_density` vs `density`**: different observables
  (vector-resolved vs trace-averaged); cross-referenced, kept.
- **Module jargon** (`subordination`, `wkernel`): mathematically correct
  names; the cookbook is the newcomer's path, and it works (8/10 guessed).
- **`defect.defect`**: a boundary declaration, kept with a better docstring.
- **`thermal.correlator`'s 8 parameters**: 5 are the physics (H, O, β, ts, N);
  3 are harvest knobs with safe defaults documented.  At the line, not over.
- **`resona.cloud` (function) shadowing the submodule** (taxonomy): the
  function IS the intended public face; `import resona.cloud` still works.

## THE 2.0 RE-PASS (EPIC3 Phase 5, 2026-06-13) — fresh personas, post-break

- **Newcomer: 10/10** task-guess rate (was 8/10 in 1.3) — all three findings
  were doc nits, fixed same-day (synthesize recipe now copy-paste-safe with
  the (α, β) return shape stated; quadform row labelled as the quadratic form).
- **Mathematician:** "mathematical hygiene unusually high"; fixed same-day:
  the false `of ∘ from_measure = identity` claim (the true inverse is
  `local_spectrum`'s e₀-measure), r_transform's small-w cancellation note,
  extreme()'s Rayleigh–Ritz inner-bias statement, freeness_defect's
  resolution floor, averaged_dos's branch note.
- **Maintainer:** all 2.0 removals verified; COMPLEXITY.md stale row and
  cost.py leaked scipy names fixed same-day.

**PROPOSE-2.1 bin** (structural, deferred deliberately):
1. `with_err` parity for the LIFTED reads — boxplus / cauchy / r / s /
   free_convolution (the per-probe recipe already exists in `cumulants`).
2. `freeness_defect` / `cross_moment` → (value, stderr) returns.
3. Pastur wrong-branch guard (warn or heavier damping instead of the
   silent ρ=0 clip) — behavior change, wants its own tests.
4. `r_transform` shifted-variable bisection (u = z − 1/w) to kill the
   small-w cancellation rather than document it.
5. The `resona.cloud` function-shadows-module accident: decide (rename the
   module to `_cloud` or live with it) — works today only because `Cloud`
   is re-exported at top level.
6. `__all__` coverage for the 8 modules that define their surface implicitly.

## THE 3.0/3.1 PASS (non-Hermitian corner CLOSED, 2026-06)

The 1.3/2.0 reviews left one structural asymmetry standing: the **five verbs**
(READ / TRUST / MOVE / COMPOSE / plane-READ) were complete for Hermitian
operators (`Spectral`) but the non-Hermitian face (`cloud`) had only **READ**
(the Ritz cloud) and **TRUST** (`sigma_min` / pseudospectrum brackets). 3.0/3.1
close it:

- **MOVE** — `cloud_flow.cloud_wkernel` / `cloud_track`: the biorthogonal
  `∂λ=(u*Bv)/(u*v)` (left+right eigvecs via Arnoldi / shift-invert), the
  non-Hermitian twin of `wkernel.wkernel` / `track`; reduces to them exactly
  when A is Hermitian. EP locator falls out (denom→0).
- **COMPOSE / plane-READ** — `resona.brown`: the Brown measure by hermitization
  (`S(z)=(1/2N)Tr log((A−z)*(A−z))`, `μ=(1/2π)ΔS`, matrix-free SLQ log-det per
  grid point), with `brown_boxplus` the per-z free additive sum. The first
  read that returns a measure on the **complex plane**, and the compose verb on
  it.

All five verbs now work for BOTH Hermitian and non-Hermitian operators —
the `Spectral`/`cloud` split is a representation choice, no longer a
capability gap. The W-side dense `O(N³)` (`kappa_w`/`track`) is also closed
for selected modes (`modes=k`, `eigsh` on the tracked block), so the
matrix-free contract now holds across the whole verb table, Hermitian or not.

## Scoreboard

- Newcomer task-guess rate: **8/10** before docs.
- Surface after polish: 54 module-level callables (−2 leaked accidents),
  20 hub methods, 2 Cloud reads.  EPIC2 cap respected: +5 intentional.
- Findings: 14 polished, 9 proposed for 2.0, 9 examined-and-kept.
- The pre-committed "nothing found" outcome did not occur — and notably,
  every CRITICAL/HIGH finding had a back-compatible 1.3 form.
