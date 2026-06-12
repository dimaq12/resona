"""critical_window.py — HOW the Pastur fixed point's critical window scales.

THE QUESTION (parked in the EPIC2 stress campaign, answered here).  The
subordination iteration g ← G_A(z − σ²g) contracts at rate |T′| =
σ²·|G_A′(z − σ²g)|; near a spectral edge of μ_A ⊞ sc(σ²) it approaches 1
(critical slowing — `resona.subordination.contraction` is the dial).  The
stress campaign recorded the honest limit "the visible window is NOT
universal — it narrows with σ²" but never measured the LAW.  Here it is.

THE MEASUREMENT.  Base μ_A = uniform[−1, 1] (4000 atoms — soft edges,
analytic control).  For each σ² ∈ [0.01, 1]: locate the contraction peak
x_e outside the band, read the distance d(ε) at which |T′| has fallen to
1 − ε, fit  d(ε; σ²) = C(ε)·(σ²)^α(ε).

THE RESULT (printed live):

    ε = 0.5 :  α ≈ 0.88   R² ≈ 0.997
    ε = 0.3 :  α ≈ 0.82   R² ≈ 0.99
    ε = 0.1 :  α ≈ 0.6    R² ≈ 0.8   (peak-adjacent: η- and grid-limited)

  i.e. the critical window scales SUB-LINEARLY in σ² — at σ² = 0.01 the
  region where the iteration visibly slows (|T′| > 0.5) is ~2·10⁻³ wide,
  at σ² = 1 it is ~10⁻¹.  Practical reading: with weak disorder you will
  NOT see the slowdown coming — the window is narrower than typical grid
  spacings; check `contraction` explicitly near edges instead of waiting
  for the iteration count to spike.

HONEST LIMITS.  The exponent is measured for THIS base (uniform: log-
singular G_A′ at its own edges); a different edge curvature gives a
different α — the non-universality is the stress campaign's point, kept.
ε = 0.1 sits too close to the peak for the 4000-atom discretization, and
its fit quality says so.  This is an empirical scaling with R², not a
derived exponent.

Run:  python3 theory/critical_window.py     (~10 s)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from resona import Spectral
from resona.subordination import contraction

M = 4000
EPS = (0.1, 0.3, 0.5)

if __name__ == "__main__":
    print("=" * 74)
    print("THE CRITICAL WINDOW of the Pastur fixed point — width vs disorder σ²")
    print("=" * 74)
    s = Spectral(np.linspace(-1, 1, M), np.full(M, 1.0 / M))
    s2s = np.geomspace(0.01, 1.0, 9)
    print(f"\n  base μ_A = uniform[−1,1] ({M} atoms);  d(ε) = distance outside the")
    print(f"  contraction peak where |T′| = 1 − ε:\n")
    print(f"  {'σ²':>7} {'x_peak':>8} {'max|T′|':>8} " +
          " ".join(f"d({e})".rjust(9) for e in EPS))
    D = {e: [] for e in EPS}
    for s2 in s2s:
        sig = np.sqrt(s2)
        xs0 = np.linspace(1.0, 1 + 2 * sig + 0.3, 800)
        c0 = contraction(s, xs0, s2)
        x_e = xs0[np.argmax(c0)]
        ds = np.geomspace(1e-4, 2.0, 80)
        c = contraction(s, x_e + ds, s2)
        row = []
        for e in EPS:
            j = np.where(c <= 1 - e)[0]
            if len(j) == 0 or c0.max() < 1 - e:
                row.append(np.nan); D[e].append(np.nan); continue
            d = float(np.interp(-(1 - e), -c[:j[0] + 1], ds[:j[0] + 1]))
            row.append(d); D[e].append(d)
        print(f"  {s2:7.3f} {x_e:8.4f} {c0.max():8.4f} " +
              " ".join(f"{d:9.2e}" for d in row))

    print(f"\n  THE LAW  d(ε; σ²) = C·(σ²)^α  (log–log fit across the decade):\n")
    for e in EPS:
        d = np.array(D[e]); ok = ~np.isnan(d)
        a, b = np.polyfit(np.log(s2s[ok]), np.log(d[ok]), 1)
        pred = a * np.log(s2s[ok]) + b
        r2 = 1 - (np.sum((np.log(d[ok]) - pred) ** 2)
                  / np.sum((np.log(d[ok]) - np.log(d[ok]).mean()) ** 2))
        note = "   (peak-adjacent: grid/η-limited)" if e == 0.1 else ""
        print(f"    ε = {e}:  α = {a:.3f}   R² = {r2:.4f}{note}")

    print("\n" + "=" * 74)
    print("  Sub-linear in σ²: weak disorder hides its own critical slowing in a")
    print("  window narrower than your grid — read `contraction` near edges, do")
    print("  not wait for the iteration counter.  Exponent is base-dependent")
    print("  (non-universal), measured here with its R², not derived.")
    print("=" * 74)
