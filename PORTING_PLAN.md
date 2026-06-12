# Porting plan — from the `spectral_hardness` research into resona

A research program of **44 stands / 33 findings** (in `../FA/spectral_hardness/`)
explored the structure of computational hardness — the *defect*.  Most of it is
**research or conceptual** and stays where it is.  This plan documents the few pieces
that are **genuine, general, reusable tools** missing from resona, where each goes,
how it is used, where it applies in real workflows, and exactly what it is honest about.

> **Selection rule.** Port only what is (1) general and reusable, (2) verified against
> dense ground truth, (3) *not already in resona*, (4) *not* a research essay or a
> negative result.  By that rule, **2 strong + 2 optional** of 44 stands qualify.

---

## Scoreboard

| tool | from | → module | status | value |
|------|------|----------|:------:|:-----:|
| `catastrophe_solve` — precision auto-allocation | finding #5 | **`resona/solve.py`** | **PORTED ✓** (v0.4.0, + `rayleigh_polish`) | ★★★ |
| `pseudospectrum` — the truth about non-normal spectra | finding #32 | **`resona/defect.py`** | **PORTED ✓** (matrix-free σ_min; bloom law tested q=2…5) | ★★★ |
| `conserved_charge` — blind integrals-of-motion search | finding #21 | `resona/lift.py` | **PORTED ✓** (generalized Gram, non-orthonormal bases) | ★★ |
| `level_spacing_ratio` — integrable-vs-chaos ⟨r⟩ | finding #12 | `resona/cost.py` | **PORTED ✓** (sector caveat in the docstring) | ★★ |

> Executed 2026-06-12: tests in `tests/test_solve.py` + `tests/test_hardness_tools.py`;
> examples `spectra_to_machine_precision.py` (ACT 2), `spectral_phenomena/nonnormal_convergence.py`
> (new), `quantum/integrability_detector.py` (new), `defect_sort.py` (pseudospectrum act);
> guide `docs/precision-and-defects.md`; cookbook rows in `docs/README.md`.

Everything else (38 stands) is **research** → stays in `spectral_hardness/` + `RECON.md`.
The explicit not-to-port list is at the bottom, with reasons.

---

## 1. `catastrophe_solve` — ★★★ STRONG PORT  → new `resona/solve.py`

### What it is
Near an order-`q` eigenvalue/root cluster (an Arnold A_{q-1} stratum), double precision
**silently** returns only `~16/q` correct digits.  This tool auto-detects the cluster
order `q` from the cheap float64 solve, spends a precision **budget `q × target`** (the
exact amount the catastrophe predicts — no guessing), and recomputes in `mpmath`.

It is the operational partner of the existing `theory/catastrophe_hardness.py` (which
gives the *exponent* `1-1/q`) and `theory/hardness_exponents.py`: theory says *how many
digits you lose*, this tool *spends exactly enough precision to get them back*.

### API
```python
resona.solve.catastrophe_solve(coeffs_exact, target_digits=15, cluster_tol=0.1)
    # coeffs_exact : highest-first EXACT coefficients (fractions.Fraction)
    # → (roots, q, dps, naive)   roots = full-precision, q = detected cluster order,
    #   dps = precision spent, naive = the float64 solve (for comparison)

resona.solve.exact_poly(roots)   # exact (Fraction) coefficients of ∏(x − root)
```

### Recipe
```python
import resona
from fractions import Fraction as Fr
coeffs = resona.solve.exact_poly([Fr(1)-Fr(1,10**5), Fr(1), Fr(1)+Fr(1,10**5), 5, -3])
roots, q, dps, naive = resona.solve.catastrophe_solve(coeffs)
# q=3 detected → spends ~3×15 dps → recovers 15 digits where numpy returns ~5
```

### The practical payoff — `~1000×` speedup by effort allocation
You do **not** run high precision on the whole problem.  Detect the `q`-dim core,
solve the bulk in float64, spend extended precision **only on the q-core**.  Measured
(finding #33): a degree-44 polynomial with a `q=3` cluster — naive `dps=60` on all 44
roots = **7.60 s**; defect-aware (float64 bulk + `dps=60` on `q=3`) = **0.006 s** →
**1202×**.  Speed comes from *paying the expensive resource only on the defect's
minimal support*.

### Where it applies (examples to add / extend)
- **`examples/spectra_to_machine_precision.py`** — this existing example is the perfect
  home: add a clustered-spectrum case where naive loses digits and `catastrophe_solve`
  recovers them.
- **`examples/inverse_spectral.py`** — inverse problems with near-degenerate spectra.
- Any companion-matrix / characteristic-polynomial root solve with clustering.

### Honest limit (must stay in the docstring)
Fixes **algorithm** loss on an **exact** problem.  If the coefficients are only known
to float64, the cluster's conditioning `a^{-(q-1)}` has already destroyed the
information — *no method* recovers it, and the tool returns only a partial result
(reported honestly).  It does **not** beat Abel–Ruffini.

### Test plan (`tests/test_solve.py`, new — convention: `test_spectral.py` core, `test_theory.py` modules)
- exact `q=3` cluster at `a=1e-5` → tool gives ≥15 digits, naive gives ≤6.
- `q=5` cluster → tool ≥15 digits.
- float64-coefficient version → tool reports *partial* (caps at coeff accuracy), no crash.

---

## 2. `pseudospectrum` — ★★★ STRONG PORT  → extend `resona/defect.py`

### What it is
For a **normal** operator the spectrum is the whole story.  For a **non-normal /
defective** one (a Jordan block — the conditioning defect), the spectrum **lies**: it
is a set of points, but the operator behaves as if its spectrum were a whole **region**.
The ε-pseudospectrum `Λ_ε(A) = { z : ‖(A−zI)⁻¹‖ > 1/ε }` is that region.  A Jordan block
of size `q` blooms the point `{0}` into a **disk of radius ε^{1/q}** (verified exactly,
finding #32) — the same catastrophe exponent as `theory/hardness_exponents.py`.

resona already has `defect.defect` / `richardson` (Richardson extrapolation of the
spectral defect); the pseudospectrum is the missing **geometric** half: *where* the
defect lives in the complex plane, and *how big* it is.

### API
```python
resona.defect.pseudospectrum_radius(matvec, N, z0, eps, hermitian=False)
    # largest |z−z0| with σ_min(A − zI) < eps  — the local pseudospectrum radius
    # → float  (≈ eps^{1/q} for an order-q defect at z0)

resona.defect.pseudospectrum(matvec, N, zs, eps)   # boolean mask of Λ_ε over a grid zs
```
Matrix-free: `σ_min(A − zI)` is read via the existing `resona.of` / `apply` machinery
(smallest singular value through the matvec), so no matrix is formed.

### Recipe
```python
import resona, numpy as np
# is my eigenvalue estimate trustworthy, or is the operator secretly defective?
r = resona.defect.pseudospectrum_radius(matvec, N, z0=lam_est, eps=1e-10)
# r ~ eps  → benign (normal-like).   r ~ eps^{1/q} ≫ eps → an order-q defect: the
# eigenvalue is uncertain to radius r, and an iterative solver will converge SLOWLY.
```

### Where it applies (examples to add)
- **`examples/spectral_phenomena/defect_sort.py`** — already about defects; add the
  pseudospectrum bloom and the `eps^{1/q}` law.
- **non-normal solver convergence**: GMRES/Arnoldi convergence is governed by the
  pseudospectrum, **not** the spectrum.  A short example under `examples/` (e.g. a
  convection-diffusion / non-self-adjoint operator) showing "spectrum predicts fast,
  pseudospectrum predicts slow, GMRES follows the pseudospectrum" — the practical
  payoff (correct stopping criterion & preconditioning).
- pairs naturally with `catastrophe_solve`: pseudospectrum *detects* the order-`q`
  defect and its size, `catastrophe_solve` *spends the precision* to resolve it.

### Honest limit
The matrix-free `σ_min` is itself estimated (via `resona.of`); near a deep defect it is
ill-conditioned (that *is* the phenomenon).  Report the resolution, never a headline.

### Test plan (`tests/test_theory.py`, extend — defect is a theory module)
- Jordan block `J_q`, `q=2..5`, `eps=1e-6` → `pseudospectrum_radius ≈ eps^{1/q}` (rel < 0.1).
- normal (diagonal) matrix → radius ≈ `eps` (no bloom).

---

## 3. `conserved_charge` — ★★ OPTIONAL  → extend `resona/lift.py`

### What it is
A **blind** search for (quasi-)conserved quantities: over a basis of `k`-local operators
`{O_a}`, find the `Q` that most nearly commutes with `H`, via the commutator-Gram
eigenproblem `G_ab = Tr([H,O_a]†[H,O_b])`.  Near-zero eigenvalues **are** the integrals
of motion — found with no prior knowledge (finding #21).  This is the constructive side
of resona's lift principle: *a lift exists ⟺ enough conserved charges ⟺ integrability*.

### API
```python
resona.lift.conserved_charge(H, basis)
    # H : operator (matrix or matvec); basis : list of candidate operators {O_a}
    # → (charges, comm_norms)   charges sorted by ‖[H,Q]‖/‖Q‖ ascending;
    #   comm_norms ≈ 0 ⇒ a genuine conserved charge
```

### Where it applies
- **`examples/quantum/heisenberg_gap.py` / `entanglement_transition.py`** — integrable
  XX chain vs chaotic mixed-field Ising: the search *finds* the charges (energy, total Z,
  free-fermion bilinears) for the integrable case, *reports none* for the chaotic.
- a "is this Hamiltonian integrable?" utility for the quantum examples.

### Honest limit / why optional
Needs a user-supplied operator basis and is `O(|basis| · dim²)` (it forms commutators) —
more specialized than the core verbs.  Genuine and reusable, but not everyday.

---

## 4. `level_spacing_ratio` — ★★ OPTIONAL  → extend `resona/cost.py`

### What it is
The 3-line **integrability detector**: the mean consecutive level-spacing ratio
`⟨r⟩` of a spectrum — `≈0.39` (Poisson → integrable, a lift exists) vs `≈0.53` (GOE →
chaotic, no lift).  A clean diagnostic that complements `theory/subordination_chaos.py`.

### API
```python
resona.cost.level_spacing_ratio(eigenvalues)   # → float ⟨r⟩  (0.39 Poisson / 0.53 GOE)
```

### Where it applies
- alongside `effective_rank` / `is_extractable` in **`measuring-difficulty.md`** as the
  "is there a lift?" dial.
- **`examples/quantum/`** chaos-vs-integrable demos.

### Honest limit
Must resolve symmetry sectors first (mixing sectors fakes Poisson — this was a real
numerics catch, see `../FA/spectral_hardness/DEBUG_LOG.md`).  Document it loudly.

---

## Cookbook integration (`docs/README.md` — new "I want to…" rows)

| I want to… | recipe | guide | full example |
|------------|--------|:-----:|--------------|
| solve a CLUSTERED / near-degenerate spectrum to full precision | `resona.solve.catastrophe_solve(coeffs)` | measuring-difficulty | `spectra_to_machine_precision.py` |
| know if my eigenvalue is trustworthy (non-normal defect) | `resona.defect.pseudospectrum_radius(mv,N,z0,eps)` | measuring-difficulty | `spectral_phenomena/defect_sort.py` |
| spend extended precision ONLY where it's needed (~n/q faster) | detect `q` → solve core in mpmath | measuring-difficulty | `spectra_to_machine_precision.py` |
| find conserved charges / test integrability | `resona.lift.conserved_charge(H,basis)` | lifting-nonlinear | `quantum/heisenberg_gap.py` |
| tell integrable from chaotic from the spectrum | `resona.cost.level_spacing_ratio(λ)` | measuring-difficulty | `quantum/entanglement_transition.py` |

A new short guide `docs/precision-and-defects.md` can collect (1)+(2)+(3): *"the spectrum
can lie — here is how to detect a defect, size it, and spend precision only on it."*

---

## What is explicitly NOT ported (and why) — the honest boundary

| stands | why they stay research |
|--------|------------------------|
| `recon_*`, `the_floor`, `bypass_attempts`, `avoided_crossing_period`, `phi1_not_universal`, `advantage_check` | **negative results** — pinned refutations; they belong in `RECON.md`, not a library API |
| `log_is_the_lift`, `add_mult_defect`, `break_arithmetic`, `analytic_continuation`, `math_in_the_defect`, `quantum_core_permanent`, `chaos_as_defect`, `purify_the_chaos`, `compress_the_random`, `hardness_without_core`, `deep_*`, `zeta_period`, `arithmetic_zeta`, `log_operator_defect`, `defect_decomposition`, `through_the_core` | **research essays in code** — conceptual capstones (the conserved-defect law, the +/× → quantum deformation). Profound, but not reusable functions |
| `shor_honest`, `shor_class`, `quantum_niche`, `dequant_map`, `moment_probe`, `from_the_phases`, `feynman_dequant` | **demonstrations** of where quantum is/ isn't needed — narratives, not tools |
| `chaos_via_resona`, `defect_speedup`, `bridge_spectral_projection` | **usage examples** of resona itself → fold the best into `examples/`, not the core |
| `two_axes`, `catastrophe_core`, `galois_lift_tower`, `meta_lift` | **theory** already covered by `theory/{monodromy_galois,hardness_exponents,subordination_chaos}` |

> The discipline is the point: a 44-stand research program yields **2 clean tools + 2
> optional** for a published library.  Resona stays small; the research stays in
> `spectral_hardness/`.  That ratio is healthy, not disappointing.

---

## Suggested order of work

1. **`resona/solve.py`** + `tests/test_solve.py` + extend `spectra_to_machine_precision.py`
   (the highest-value, cleanest, self-contained port).
2. **`defect.pseudospectrum*`** + extend `tests/test_theory.py` + `spectral_phenomena/defect_sort.py`
   + a non-normal-solver example (the convergence payoff).
3. New guide `docs/precision-and-defects.md` + the 5 cookbook rows.
4. *(optional)* `lift.conserved_charge`, `cost.level_spacing_ratio` + the quantum examples.

Each step is independent and verified against dense ground truth before merge — the
resona discipline (`COMPLEXITY.md`: forward matrix-free; honest limits stated; the real
number reported, never a headline).
