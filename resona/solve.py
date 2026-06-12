"""
resona.solve — precision where the defect lives: clustered roots & eigenvalue polish.

Near an order-q root/eigenvalue cluster (an Arnold A_{q-1} stratum) double
precision SILENTLY keeps only ~16/q correct digits (the catastrophe law of
`theory/hardness_exponents.py`: error ~ ε^{1/q}).  This module is the
operational partner of that theory:

• `catastrophe_solve` — auto-detect the cluster order q from the cheap float64
  solve, spend the precision budget the catastrophe predicts (dps = q × target),
  recompute in mpmath.  Fixes ALGORITHM loss on an EXACT problem.

• `rayleigh_polish` — shifted inverse iteration + Rayleigh quotient (cubic):
  polish one eigenvalue from a Ritz seed (`resona.of`) to machine precision,
  dense or matrix-free.

The effort-allocation principle (measured ~1100× in `theory`-side experiments):
detect the q-core cheaply, pay the expensive resource (precision / inverse
iterations) only on the defect's minimal support — never on the whole problem.

HONEST LIMIT (catastrophe_solve): it recovers digits the float64 *algorithm*
lost on an *exact* problem.  If the coefficients are themselves only known to
float64, the cluster's conditioning a^{-(q-1)} has already destroyed the
information — no method recovers it; the result is then capped by the data
(returned honestly, compare against `naive`).  It does not beat Abel–Ruffini.
"""
from fractions import Fraction
import numpy as np

__all__ = ["exact_poly", "catastrophe_solve", "rayleigh_polish"]


def exact_poly(roots):
    """Exact (Fraction) coefficients, highest-first, of ∏(x − root).

    Accepts ints / Fractions (exact in → exact out) or floats (exact in the
    float's binary value).
    """
    c = [Fraction(1)]
    for r in roots:
        new = [Fraction(0)] * (len(c) + 1)
        for i in range(len(c)):
            new[i] += c[i]                   # c · x   (raise degree)
            new[i + 1] -= Fraction(r) * c[i]  # − r · c
        c = new
    return c


def catastrophe_solve(coeffs_exact, target_digits=15, cluster_tol=0.1):
    """Solve a polynomial with a root CLUSTER to full precision, auto-budgeted.

    coeffs_exact : highest-first EXACT coefficients (fractions.Fraction / int).
    Detects the largest cluster order q from the cheap float64 solve, then
    spends the precision budget the catastrophe predicts — dps = q·target + 15 —
    and recomputes all roots in mpmath.

    Returns (roots, q, dps, naive):
        roots : complex ndarray, full-precision roots (cast to complex128)
        q     : detected cluster order (the A_{q-1} stratum)
        dps   : decimal digits spent (the budget)
        naive : the float64 np.roots solve, for honest comparison.
    """
    import mpmath as mp
    coeffs_exact = [Fraction(c) for c in coeffs_exact]
    naive = np.roots([float(c) for c in coeffs_exact])     # the cheap float64 solve
    q = max(int(np.sum(np.abs(naive - r) < cluster_tol)) for r in naive)
    dps = max(q, 1) * target_digits + 15                   # budget = q × digits
    old = mp.mp.dps
    try:
        mp.mp.dps = dps
        roots = mp.polyroots([mp.mpf(c.numerator) / mp.mpf(c.denominator)
                              for c in coeffs_exact], maxsteps=500, extraprec=2 * dps)
        return np.array([complex(r) for r in roots]), q, dps, naive
    finally:
        mp.mp.dps = old


def rayleigh_polish(A, sigma, N=None, iters=6, v0=None, seed=0, tol=0.0):
    """Polish ONE eigenvalue near the shift `sigma` to machine precision.

    Shifted inverse iteration with Rayleigh-quotient updates — cubic convergence
    for symmetric operators.  `A` is a dense/sparse matrix OR a matvec callable
    (then pass N).  Seed `sigma` from `resona.of(...).nodes` (the Ritz values).

    Returns the polished eigenvalue λ (float).  This is the sft35 pipeline's
    polish step as a primitive: resona supplies the matrix-free seed, this
    spends the refinement only on the targeted eigenvalue.
    """
    dense = not callable(A)
    if dense:
        A = np.asarray(A, float)
        N = A.shape[0]
        mv = lambda x: A @ x
    else:
        if N is None:
            raise ValueError("rayleigh_polish(matvec, ...) needs N")
        mv = A
    rng = np.random.default_rng(seed)
    v = np.asarray(v0, float) if v0 is not None else rng.standard_normal(N)
    v = v / np.linalg.norm(v)
    lam = float(sigma)
    I = np.eye(N) if dense else None
    for _ in range(iters):
        if dense:
            try:
                v = np.linalg.solve(A - lam * I, v)
            except np.linalg.LinAlgError:
                break                                       # exactly singular: converged
        else:
            from scipy.sparse.linalg import LinearOperator, minres
            op = LinearOperator((N, N), matvec=lambda x: mv(x) - lam * x)
            v_new, _ = minres(op, v, rtol=1e-12, maxiter=4 * N)
            if not np.any(v_new):
                break
            v = v_new
        v = v / np.linalg.norm(v)
        lam_new = float(v @ mv(v))
        if tol and abs(lam_new - lam) <= tol * max(1.0, abs(lam_new)):
            lam = lam_new
            break
        lam = lam_new
    return lam
