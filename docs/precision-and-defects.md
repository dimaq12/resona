# Precision & defects — the spectrum can lie

Three tools, one discipline: **detect** the defect, **size** it, then spend the
expensive resource (precision, iterations) **only on its minimal support**.

```python
import numpy as np, resona
```

## 1. Is my eigenvalue trustworthy? — the pseudospectrum

For a **normal** operator the spectrum is the whole story.  For a **non-normal**
one (a Jordan-type defect) the spectrum *lies*: an order-`q` defect blooms a
point eigenvalue into a **disk of radius ε^{1/q}**, and iterative solvers see
the bloom, not the points.

```python
r = resona.defect.pseudospectrum_radius(matvec, eps=1e-10, z0=lam_est,
                                        N=N, rmatvec=rmatvec)
# r ≈ eps        → benign (normal-like): trust the eigenvalue
# r ≈ eps^(1/q)  → an order-q defect: λ is uncertain to radius r,
#                  and GMRES/Arnoldi will converge SLOWLY
q_eff = np.log(eps) / np.log(r)        # read the defect order
```

The law is exact on Jordan blocks (verified q=2…5 in `tests/test_hardness_tools.py`);
`sigma_min(A, 0.0)` < ε with all eigenvalues ≥ 1 is the one-number GMRES-stall
predictor — see
[`spectral_phenomena/nonnormal_convergence.py`](../examples/spectral_phenomena/nonnormal_convergence.py):
same spectrum `[1,2]`, normal converges in 14 iterations, non-normal stalls for
thousands.

**Honest limit.** With a matvec, `σ_min` is itself a Lanczos estimate; near a
deep defect it is ill-conditioned — that *is* the phenomenon.  Report the
resolution, never a headline.

## 2. Clustered roots / eigenvalues — spend exactly the predicted budget

Near an order-`q` cluster (an Arnold A_{q−1} stratum) double precision
**silently** keeps only `~16/q` correct digits.  The catastrophe predicts the
exact budget to recover them: `dps = q × target`.

```python
from fractions import Fraction as Fr
coeffs = resona.solve.exact_poly([Fr(1)-Fr(1,10**5), 1, Fr(1)+Fr(1,10**5), 5, -3])
roots, q, dps, naive = resona.solve.catastrophe_solve(coeffs)
# q=3 detected → 60 dps spent → 16 digits where np.roots returns ~5
```

**Honest limit.** This fixes **algorithm** loss on an **exact** problem.  If the
coefficients are only known to float64, the cluster's conditioning `a^{-(q-1)}`
has already destroyed the information — *no method* recovers it (the tool then
returns a partial result; compare against `naive`).  It does not beat
Abel–Ruffini.

## 3. One eigenvalue to machine precision — pay only where it matters

The effort-allocation principle (measured ~1100×: float64 on the bulk, the
expensive tool on the q-core only):

```python
s = resona.of(matvec, N)                       # cheap matrix-free seed (Ritz)
seed = s.nodes[np.argmin(np.abs(s.nodes - target))]
lam = resona.solve.rayleigh_polish(matvec, seed, N=N)   # cubic → ~1e-16
```

[`spectra_to_machine_precision.py`](../examples/spectra_to_machine_precision.py):
35 operators × 6 eigenvalues across the spectrum — seed 1e-4 → polish 1e-16,
100% machine zero.

## The map

| symptom | detect | spend |
|---|---|---|
| eigenvalue uncertain, solver slow | `pseudospectrum_radius` (bloom ≫ ε) | preconditioning / deflation of the q-core |
| digits silently lost near a cluster | `catastrophe_solve` detects `q` | `dps = q × target`, exact coefficients |
| Ritz seed not precise enough | `s.nodes` rel.err ~1e-4 | `rayleigh_polish` on the targeted λ only |

All three are the same move: *the defect names its own small support; pay there.*
