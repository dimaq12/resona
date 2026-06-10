"""
resona.defect — the defect calculus: error-as-information.

The root idea of the whole program.  Run a solver / discretization at resolution
n and get P_n; the DEFECT

        D_n = P_n − P_{2n}

is not waste — it is the dominant error mode, and it CARRIES the operator's
spectrum.  Two consequences are exposed here:

• RICHARDSON annihilation: if the error scales as n^{-p}, the combination
  (2^p P_{2n} − P_n)/(2^p − 1) cancels the leading defect → a higher-order answer
  for free.  Iterated, it is a convergence-acceleration tower.

• The DEFECT-JUMP law: for a linear stationary iteration x ← J x + c (gradient
  descent, power iteration, Gauss–Seidel, PageRank), the defect of a doubling is
  EXACTLY D_{2n} = J^n · D_n — so one cheap defect predicts the converged answer
  with no extra iterations.
"""
import numpy as np


def defect(P_n, P_2n):
    """The defect D_n = P_n − P_{2n} (the harvested error signal)."""
    return np.asarray(P_n) - np.asarray(P_2n)


def richardson(P_n, P_2n, p=1):
    """One Richardson step for error ~ n^{-p}: (2^p P_{2n} − P_n)/(2^p − 1)."""
    f = 2.0 ** p
    return (f * np.asarray(P_2n) - np.asarray(P_n)) / (f - 1.0)


def richardson_limit(values, ns, p0=1.0):
    """Extrapolate a sequence values[k] sampled at resolutions ns[k] (assumed a
    geometric doubling) to n→∞ via a Neville/Richardson tableau on error ~ n^{-p}.

    Returns the top-of-tableau estimate of the limit.
    """
    v = [np.asarray(x, float) for x in values]
    ns = list(ns); L = len(v)
    T = [v[:]]                                              # T[0] = raw column
    for col in range(1, L):
        prev = T[col - 1]; newcol = [None] * L
        for k in range(col, L):
            r = (ns[k] / ns[k - col]) ** (p0 * col)         # ratio of resolutions^p
            newcol[k] = (r * prev[k] - prev[k - 1]) / (r - 1.0)
        T.append(newcol)
    return T[L - 1][L - 1]


def defect_jump(D_n, J, n):
    """The exact defect-jump  D_{2n} = J^n · D_n  for a linear iteration matrix J
    (or matvec).  Predicts the next-doubling defect without running the iteration.
    """
    mv = J if callable(J) else (lambda x, J=J: J @ x)
    x = np.asarray(D_n, float)
    for _ in range(n):
        x = mv(x)
    return x
