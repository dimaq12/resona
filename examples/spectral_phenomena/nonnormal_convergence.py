"""
THE SPECTRUM LIES — GMRES follows the PSEUDOSPECTRUM, not the spectrum.

Two operators with the IDENTICAL spectrum λ ∈ [1, 2] (both triangular, same
diagonal).  Spectral convergence theory ("κ = λmax/λmin = 2 → fast") predicts
the same fast GMRES for both.  Reality:

    normal      → converges in ~14 iterations,
    non-normal  → stalls for thousands,

because convergence is governed by the ε-PSEUDOSPECTRUM Λ_ε = {z : σ_min(A−zI)<ε}.
For the non-normal operator the defect blooms Λ_ε from the points [1,2] into a
region that swallows the ORIGIN: the operator is numerically singular even
though every eigenvalue is ≥ 1.  `resona.defect.pseudospectrum_radius` reads
this before you waste the iterations — the practical payoff is a correct
stopping criterion / preconditioning decision from one number.

Run:  python3 examples/spectral_phenomena/nonnormal_convergence.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
from scipy.sparse.linalg import gmres
from resona.defect import pseudospectrum_radius, sigma_min, normality

N = 200
lam = np.linspace(1.0, 2.0, N)                       # the SHARED spectrum
A_normal = np.diag(lam)
A_nonnorm = np.diag(lam) + np.diag(np.full(N - 1, 1.5), 1)   # same eigenvalues!
b = np.ones(N) / np.sqrt(N)


def run_gmres(A, maxiter=500):
    its = [0]
    x, _ = gmres(A, b, rtol=1e-10, restart=None, maxiter=maxiter,
                 callback=lambda r: its.__setitem__(0, its[0] + 1),
                 callback_type='pr_norm')
    return its[0], float(np.linalg.norm(A @ x - b))


if __name__ == "__main__":
    print("=" * 72)
    print("GMRES FOLLOWS THE PSEUDOSPECTRUM, NOT THE SPECTRUM")
    print("=" * 72)
    print(f"\n  Two triangular operators, N={N}, the SAME spectrum λ ∈ [1, 2]")
    print(f"  (so the classical dial κ = λmax/λmin = 2 predicts FAST for both).\n")

    print(f"  {'operator':>12} {'eig range':>12} {'‖[A,A*]‖²':>11} {'σ_min(A)':>10} {'GMRES iters':>12} {'residual':>10}")
    for name, A in [("normal", A_normal), ("non-normal", A_nonnorm)]:
        it, res = run_gmres(A)
        sm = sigma_min(A, 0.0)
        nn, _ = normality(A)                          # GLOBAL departure-from-normality
        evr = f"[{lam[0]:.0f}, {lam[-1]:.0f}]"
        itxt = str(it) if res < 1e-8 else f"{it} (STALLED)"
        print(f"  {name:>12} {evr:>12} {nn:>11.2e} {sm:>10.1e} {itxt:>12} {res:>10.1e}")
    print("  ‖[A,A*]‖²_F (defect.normality) = 0 ⇔ normal: the cheap GLOBAL scalar that")
    print("  flags 'the spectrum will lie' BEFORE the local σ_min / pseudospectrum reads.")

    eps = 1e-6
    print(f"\n  The explanation is GEOMETRIC — the ε-pseudospectrum (ε={eps:.0e}):")
    for name, A in [("normal", A_normal), ("non-normal", A_nonnorm)]:
        rad = pseudospectrum_radius(A, eps, z0=1.0, direction=-1.0, r_max=1.5)
        verdict = ("trivial ε-fattening → the spectrum is the truth" if rad < 10 * eps
                   else f"bloom covers the ORIGIN → numerically singular, GMRES must stall")
        print(f"  {name:>12}: bloom radius from λ_min toward 0 = {rad:>8.4f}   {verdict}")

    print("\n" + "=" * 72)
    print("  Same spectrum, opposite fates.  σ_min(A) and one pseudospectrum_radius")
    print("  call read the true difficulty BEFORE iterating: for a non-normal")
    print("  operator the spectrum is a set of points, but the operator behaves as")
    print("  if its spectrum were the whole bloom region — Λ_ε is what GMRES sees.")
    print("=" * 72)
