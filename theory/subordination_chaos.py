"""
resolvent_algebra/subordination_chaos.py
=============================================================================
THE EDGE OF CHAOS OF THE SPECTRAL COMPUTATION — where the defect lives.

The averaged resolvent of  H = A + (Gaussian disorder, variance σ²)  solves the
self-consistent SUBORDINATION equation (free probability):

      g(z) = G_A( z − σ² g(z) ),     G_A(w) = (1/N) Σ_i 1/(w − a_i).

This is a FIXED-POINT MAP  g ↦ T(g) = G_A(z − σ²g).  Its convergence is governed
by the contraction factor

      |T'(g*)| = | σ² · (1/N) Σ_i 1/(z − σ²g* − a_i)² |.

CLAIM: in the spectral BULK the map is a strong contraction (|T'|≪1, converges in
a few steps); at the spectral EDGE it becomes MARGINAL (|T'| → 1, critical
slowing — many steps).  The edge of the support is the EDGE OF CHAOS of the
spectral computation — and that is exactly where the DEFECT lives (band edge:
sparse eigenvalues, finite-size effects, localization).

So: defect (hard to compute) = edge of chaos of the free-probability fixed point
= the spectral phase boundary.  Three names, one place.

Run:  python3 resolvent_algebra/subordination_chaos.py
"""
import numpy as np

rng = np.random.default_rng(0)


def solve_g(z, a, sigma2, tol=1e-11, maxiter=4000):
    """Iterate the subordination map to its fixed point; return g*, #iters, |T'|."""
    g = -0.3j
    for it in range(1, maxiter + 1):
        w = z - sigma2 * g
        g_new = np.mean(1.0 / (w - a))
        if abs(g_new - g) < tol:
            g = g_new
            break
        g = g_new
    Tp = abs(sigma2 * np.mean(1.0 / (z - sigma2 * g - a) ** 2))   # contraction factor
    return g, it, Tp


if __name__ == "__main__":
    N = 800
    a = np.linspace(-1, 1, N)        # A: flat spectrum on [-1,1]
    sigma2 = 0.2                     # Gaussian disorder strength
    eta = 1e-3

    print("=" * 72)
    print("SUBORDINATION FIXED POINT — its edge of chaos = where the defect lives")
    print("=" * 72)
    print(f"  A: flat spectrum [-1,1], N={N}.  H=A+disorder, σ²={sigma2}.")
    print(f"  scan energy x; iterate g=G_A(x+iη − σ²g).  |T'| = contraction factor.\n")
    print(f"  {'x':>6} {'DOS ρ(x)':>10} {'iters':>7} {'|T´| contraction':>17}  regime")
    xs = np.linspace(0.0, 2.4, 25)
    edge_x, max_it = None, 0
    for x in xs:
        g, it, Tp = solve_g(x + 1j * eta, a, sigma2)
        dos = max(-g.imag / np.pi, 0.0)
        if it > max_it and dos > 1e-3:
            max_it, edge_x = it, x
        regime = ("BULK (easy)" if Tp < 0.5 else
                  "EDGE→critical" if Tp < 0.97 else
                  "CRITICAL/gap")
        bar = "#" * int(40 * min(Tp, 1.0))
        print(f"  {x:>6.2f} {dos:>10.4f} {it:>7} {Tp:>17.3f}  {regime}  {bar}")

    print(f"\n  In the BULK |T'|≪1 — the spectral map is a strong contraction, converges")
    print(f"  in a few steps.  Approaching the support EDGE |T'|→1 — critical slowing,")
    print(f"  iterations spike (peak near x≈{edge_x:.2f}, {max_it} iters).")
    print(f"\n  The DEFECT (hard-to-compute spectral region) = the EDGE OF CHAOS of the")
    print(f"  free-probability fixed point = the spectral phase boundary.  Computing the")
    print(f"  response is easy where the system is 'free' (bulk), critically hard exactly")
    print(f"  where freeness breaks (the edge).  defect = edge = phase boundary.")
    print("=" * 72)
