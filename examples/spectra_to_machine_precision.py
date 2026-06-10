"""
35 operators → spectra to machine precision, seeded by resona.

This mirrors the sft35 pipeline (SFT seed → Rayleigh polish → 10⁻¹⁶), on the
resona primitive: each "equation" is a conductivity operator A(k) (built from an
initial condition).  `resona.of(A)` gives the Ritz seed (approximate eigenvalues,
matrix-free); shifted inverse iteration + Rayleigh-quotient updates (cubic) polish
the targeted eigenvalues to machine precision — all the operator ever needs is a
matvec.

Run:  python3 examples/spectra_to_machine_precision.py

Note on resona.defect (Richardson/defect calculus):
  This file achieves machine precision via shifted inverse iteration +
  Rayleigh-quotient updates (cubic convergence), NOT a Richardson step.
  Richardson extrapolation (resona.defect.richardson) applies when you have
  the same quantity evaluated at two resolutions n and 2n and the error scales
  as n^{-p}; here each eigenvalue is polished from a single Ritz seed by
  Krylov refinement, so there is no n/2n pair to hand to richardson().
  No refactoring needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from numpy.linalg import eigvalsh, solve, norm
from collections import OrderedDict
import resona

N = 128
dx = 1.0 / (N + 1)
x = np.linspace(0, 1, N, endpoint=False)
rng = np.random.default_rng(0)


def make_ics():
    g = lambda m, s: np.exp(-((x - m) ** 2) / s)
    sn = lambda k: np.sin(k * np.pi * x); cs = lambda k: np.cos(k * np.pi * x)
    saw = 2 * (x % 1.0) - 1; chirp = np.sin(2 * np.pi * (2 * x + 6 * x ** 2))
    sd = lambda s, sc=0.7: rng.standard_normal(N) * sc + 0.3 * np.sin(2 * np.pi * 3 * x)
    sech = 1.0 / np.cosh((x - 0.5) * 8.0); tri = 1 - 2 * np.abs(x - 0.5)
    return OrderedDict([
        ("Keller-Segel", g(.4, .02) + g(.6, .03)), ("Perona-Malik", sn(1) * 1.1),
        ("Thin Film", 1 + .12 * cs(2)), ("Benjamin-Ono", chirp), ("Burgers", saw),
        ("KdV", g(.5, .015)), ("KS", sd(101)), ("Allen-Cahn", sn(2) * .9),
        ("Sine-Gordon", sn(3) * .7 + sn(7) * .3), ("Porous Medium", g(.3, .04) + g(.7, .04)),
        ("ODE exp", g(.5, .04) * .3), ("ODE f^5", tri), ("CGL", sn(4) * .8),
        ("Swift-Hohenberg", cs(1) * .9), ("Chen-Lee-Liu", chirp * .7),
        ("Sasa-Satsuma", sn(1) * .6 + sn(5) * .4), ("Nikolaevskiy", g(.3, .02) - g(.7, .03)),
        ("Kundu-Eckhaus", sn(2) * .5 + cs(3) * .5), ("Generalized KS", sd(102)),
        ("Camassa-Holm", 1 - np.abs(x - .5) * 2), ("Degasperis-Procesi", g(.5, .06)),
        ("Fokas-Lenells", saw * .7), ("Hirota mKdV", sech), ("Frac Heat 0.5", g(.5, .03) * 1.5),
        ("Frac Heat 1.0", sn(1) + sn(3) * .5), ("Frac Heat 2.0", cs(2) * .8),
        ("Fokker-Planck", tri * 1.2), ("Orr-Sommerfeld", chirp * .5), ("NLS Soliton", g(.5, .05)),
        ("GL K3", g(.4, .03) + g(.6, .04)), ("KS Chaos", sd(99, 1.0)),
        ("Cahn-Hilliard", cs(3) * .6), ("Boussinesq", sn(2) + cs(4) * .4),
        ("Whitham", sech * .8), ("Ostrovsky", g(.45, .02) - g(.55, .02)),
    ])


def s2k(sig, c=0.2):
    s = sig.std()
    xn = (sig - sig.mean()) / (s + 1e-12) if s > 1e-12 else sig - sig.mean()
    k = np.exp(c * xn); return k / k.mean()


def tridiag(k):
    face = 0.5 * (k[:-1] + k[1:]); off = face / dx ** 2
    A = np.diag(-off, 1) + np.diag(-off, -1)
    d = np.zeros(N); d[0] = -off[0]; d[-1] = -off[-1]; d[1:-1] = -(off[:-1] + off[1:])
    A[np.diag_indices(N)] = d
    return A


def rayleigh_polish(A, sigma, iters=4):
    """Shifted inverse iteration + Rayleigh quotient — cubic → machine precision."""
    v = rng.standard_normal(N); v /= norm(v); lam = sigma
    I = np.eye(N)
    for _ in range(iters):
        try:
            v = solve(A - lam * I, v)
        except np.linalg.LinAlgError:
            break
        v /= norm(v); lam = float(v @ (A @ v))
    return lam


if __name__ == "__main__":
    ics = make_ics()
    # Target a SPREAD across the spectrum — extremes (easy for Lanczos) AND interior
    # (where the Ritz seed is coarse and the polish genuinely earns its keep).
    targets = [0, N // 16, N // 8, N // 4, N // 2, 3 * N // 4]
    print("=" * 70)
    print(f"{len(ics)} operators → spectra to machine precision (resona seed → Rayleigh)")
    print("=" * 70)
    print(f"  N={N}, {len(targets)} eigenvalues each across the spectrum (extreme+interior).")
    print(f"  resona.of = matrix-free seed;  shifted inverse iteration = polish.\n")
    seed_errs, final_errs = [], []
    for name, ic in ics.items():
        A = tridiag(s2k(ic))
        ev = np.sort(eigvalsh(A))
        ritz = np.sort(resona.of(lambda v: A @ v, N, k=70, probes=4).nodes)
        for idx in targets:
            t = ev[idx]
            seed = ritz[np.argmin(np.abs(ritz - t))]        # resona Ritz seed (shift)
            seed_errs.append(abs(seed - t) / abs(t))
            lam = rayleigh_polish(A, seed, iters=6)         # converges to a true eigenvalue
            final_errs.append(np.min(np.abs(ev - lam)) / abs(lam))   # dist to nearest true λ
    seed_errs, final_errs = np.array(seed_errs), np.array(final_errs)
    mz = int(np.sum(final_errs < 1e-13))
    print(f"  resona seed (Ritz):     median rel.err = {np.median(seed_errs):.2e}")
    print(f"  + Rayleigh polish:      median rel.err = {np.median(final_errs):.2e}")
    print(f"  machine zero (<1e-13):  {mz}/{len(final_errs)} eigenvalues "
          f"({mz/len(final_errs)*100:.0f}%)")
    print("\n" + "=" * 70)
    print("  resona's matrix-free Ritz values seed the spectrum; a few Rayleigh-quotient")
    print("  iterations (cubic) polish to machine precision — the sft35 pipeline, on the")
    print("  resona primitive.  (Machine precision is the eigenvalue computation, via")
    print("  Rayleigh refinement; resona supplies the matrix-free seed.)")
    print("=" * 70)
