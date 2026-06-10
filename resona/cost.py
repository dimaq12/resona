"""
resona.cost — the Extraction Law: how hard is the answer to extract?

The answer pre-exists in the resolvent field G(z)=(z−A)⁻¹; a solver only EXTRACTS
from it.  The cost of extracting to accuracy ε near a query z follows

        Cost(ε, z)  ~  ε^{-a} · dist(z, Σ*)^{-b},

where Σ* is the NON-removable singular set (edges / branch points / shocks /
continua — NOT isolated poles, which deflate away).  Φ₁ = Tr(A)²/Tr(A²) is the
GLOBAL summary (low ⇒ structured/cheap, high ⇒ genuine frontier).

The operational test — REMOVABLE vs GENUINE — is lift-rank SATURATION: lift a
signal to its trajectory (Hankel) operator at growing window k.  If the effective
rank SATURATES, the singularity is removable → a finite chart linearizes it →
EXTRACTABLE.  If the rank keeps GROWING with k, there is no finite chart →
a genuine wall (structureless, e.g. aˣ mod N).  This is the dial that says
dequantizable-vs-quantum, easy-vs-hard, from the signal itself.
"""
import numpy as np
from scipy.linalg import hankel, svdvals


def phi1(spectral):
    """Φ₁ = Tr(A)²/Tr(A²) — the global cost summary (participation ratio)."""
    return spectral.effective_rank()


def lift_rank(signal, k=60):
    """Effective rank of the trajectory (Hankel) operator of a signal at window k.
    = (Σσ²)²/Σσ⁴ of the Hankel singular values — the size of the linearizing chart."""
    s = np.asarray(signal, float)
    s = (s - s.mean()) / (s.std() + 1e-12)
    H = hankel(s[:k], s[k - 1:2 * k - 1])
    s2 = svdvals(H) ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def is_extractable(signal, windows=(20, 40, 80, 120), grow=0.25):
    """Removable-vs-genuine test by lift-rank SATURATION.

    Returns (extractable: bool, ranks: list).  Saturates (rank flattens, stays a
    small fraction of the window) ⇒ removable ⇒ extractable.  Keeps growing ~k ⇒
    no finite chart ⇒ genuine wall.
    """
    ranks = [lift_rank(signal, k) for k in windows]
    # genuine if the rank is still climbing with the window (chart never closes)
    growth = (ranks[-1] - ranks[-2]) / (windows[-1] - windows[-2])
    extractable = growth < grow and ranks[-1] < 0.4 * windows[-1]
    return bool(extractable), ranks


def extraction_cost(eps, dist, a=1.0, b=1.0, c=1.0):
    """The law value  Cost = c · ε^{-a} · dist^{-b}."""
    return c * np.asarray(eps, float) ** (-a) * np.asarray(dist, float) ** (-b)


def fit_law(costs, eps, dist):
    """Fit (a, b, c) of Cost ~ c·ε^{-a}·dist^{-b} by log-log least squares."""
    L = np.log(np.asarray(costs, float))
    A = np.column_stack([-np.log(eps), -np.log(dist), np.ones_like(L)])
    a, b, logc = np.linalg.lstsq(A, L, rcond=None)[0]
    return float(a), float(b), float(np.exp(logc))
