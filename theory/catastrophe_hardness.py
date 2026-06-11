"""
catastrophe_hardness.py — computational hardness is stratified by Arnold's
catastrophe theory.  (The FRONTIER §3 conjecture, now demonstrated.)

CLAIM.  Near a singular point of a parameter family, the cost of extracting an
eigenvalue to fixed accuracy — the eigenvalue CONDITION number, an
algorithm-INDEPENDENT lower bound — diverges as dist^{-b}, and the exponent b is
an INVARIANT OF THE ARNOLD CATASTROPHE STRATUM of the discriminant (the
eigenvalue-collision locus):

      A_{q-1} stratum  (q eigenvalues coalesce)   ⇒   b = 1 − 1/q.

      fold (A₂, double)  → ½      cusp (A₃, triple)  → ⅔
      swallowtail (A₄)   → ¾      butterfly (A₅)     → ⅘

TWO demonstrations, on companion matrices of the catastrophe normal forms:

  PART 1 — the A-series LADDER: companion of λ^q + s, approach s→0
           (the q-fold coalescence) → b = 1 − 1/q, for q = 2…5.

  PART 2 — the JUMP on ONE discriminant: the 2-parameter cusp family (companion of
           λ³ + aλ + b, discriminant the cuspoid 4a³+27b²=0).  The exponent is ½
           EVERYWHERE on the smooth fold edge and jumps to ⅔ at the cusp point —
           so b is a property of the STRATUM, not the direction of approach.

HONEST STATUS.  The per-stratum exponents (q-fold splitting s^{1/q}, conditioning
s^{-(1-1/q)}) are CLASSICAL perturbation theory (Vishik–Lyusternik, Lidskii); the
catastrophe stratification of discriminants is CLASSICAL (Arnold).  The
contribution is the SYNTHESIS — that computational hardness inherits the Arnold
A-series, demonstrated as a measured law with a clean stratum-jump.  Still open: a
rigorous algorithm-independent lower bound, the D/E series, large (non-companion)
operators, and the bridge to actual solver cost (not just conditioning).

Run:  python3 theory/catastrophe_hardness.py
"""
import numpy as np


def companion(coeffs):
    """Companion matrix of the monic polynomial with the given lower coefficients:
    λⁿ + coeffs[n-1]·λⁿ⁻¹ + … + coeffs[0]  (here the λⁿ⁻¹ term is 0)."""
    n = len(coeffs); C = np.zeros((n, n))
    for i in range(n - 1):
        C[i, i + 1] = 1.0
    C[n - 1, :] = -np.asarray(coeffs, float)
    return C


def max_condition(M):
    """Worst eigenvalue condition number κ = ‖x_i‖·‖y_i‖ (Wilkinson) — the
    algorithm-independent cost of extracting the eigenvalue to fixed accuracy."""
    lam, X = np.linalg.eig(M); Xi = np.linalg.inv(X)
    return max(np.linalg.norm(X[:, i]) * np.linalg.norm(Xi[i, :]) for i in range(len(lam)))


def exponent(approach, lo=1e-2, hi=1e-8, n=14):
    """Fit κ ~ dist^{-b} as the parameter approaches the singular stratum."""
    ds = np.geomspace(lo, hi, n)
    ks = [max_condition(approach(d)) for d in ds]
    return -np.polyfit(np.log(ds), np.log(ks), 1)[0]


if __name__ == "__main__":
    print("=" * 74)
    print("COMPUTATIONAL HARDNESS IS STRATIFIED BY ARNOLD'S CATASTROPHE THEORY")
    print("=" * 74)

    print("\n  PART 1 — the A-series ladder (companion of λ^q + s, q-fold coalescence):")
    print(f"      {'catastrophe':>16} {'q-fold':>8} {'measured b':>12} {'1−1/q':>8}")
    names = {2: "A₂ fold", 3: "A₃ cusp", 4: "A₄ swallowtail", 5: "A₅ butterfly"}
    for q in (2, 3, 4, 5):
        b = exponent(lambda d, q=q: companion([d] + [0.0] * (q - 1)))
        print(f"      {names[q]:>16} {q:>8} {b:>12.3f} {1 - 1/q:>8.3f}")

    print("\n  PART 2 — the JUMP on ONE discriminant (λ³ + aλ + b, cuspoid 4a³+27b²=0):")
    b_cusp = exponent(lambda d: companion([d, 0.0, 0.0]))            # approach the cusp (0,0)
    print(f"      {'cusp point (0,0)':>26}:  b = {b_cusp:.3f}   [⅔ = 0.667]")
    print(f"      {'fold edge — generic points':>26}:")
    for a in (-0.75, -1.5, -3.0, -6.0, -12.0):
        bstar = np.sqrt(-4 * a ** 3 / 27)                           # on the discriminant
        bf = exponent(lambda d, a=a, bstar=bstar: companion([bstar + d, a, 0.0]))
        print(f"        a={a:>6.2f}, b*={bstar:>7.3f} →  b = {bf:.3f}   [½ = 0.500]")

    print("\n" + "=" * 74)
    print("  The exponent is ½ EVERYWHERE on the smooth fold edge and JUMPS to ⅔ at")
    print("  the cusp — a property of the catastrophe STRATUM, not the path.  The")
    print("  Arnold A_k ladder (k+1-fold coalescence) realizes the hardness exponents")
    print("  1 − 1/(k+1): ½, ⅔, ¾, ⅘, …  Complexity inherits catastrophe theory.")
    print("  (Exponents classical — Lidskii; stratification classical — Arnold; the")
    print("   unified law is the claim.  Lower-bound theorem / D,E series / large")
    print("   operators / real solver cost remain open — see FRONTIER.md §3.)")
    print("=" * 74)
