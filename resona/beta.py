"""
resona.beta — the Beta-law spectral closure: support + 2 moments → full spectrum.

A local / structured operator's eigenvalue density is SMOOTH (a central-limit
effect over its many local terms).  On a bounded support [E0, Emax] the
MAXIMUM-ENTROPY density with a prescribed mean and variance is a BETA
distribution.  So the entire spectrum is fixed by FOUR numbers — the support
endpoints E0, Emax and the first two moments μ1 = Tr(A)/N, μ2 = Tr(A²)/N — every
one of them read matrix-free by resona.  The inverse-CDF unfolds all N levels in
O(N), one and two factors of N below dense eig (O(N²)) and diagonalization
(O(N³)).
"""
import numpy as np
from scipy.stats import beta as _beta


def beta_spectrum(E0, Emax, mu1, mu2, N, return_params=False):
    """Support [E0,Emax] + per-dimension moments (μ1,μ2) → the N Beta levels.

    μ1 = Tr(A)/N, μ2 = Tr(A²)/N.  Maximum-entropy closure on a bounded support.
    """
    span = Emax - E0
    if span <= 0:
        levels = np.full(N, float(E0))
        return (levels, (1.0, 1.0)) if return_params else levels
    m1 = (mu1 - E0) / span                                  # mean on [0,1]
    m2 = (mu2 - 2 * E0 * mu1 + E0 ** 2) / span ** 2          # 2nd moment on [0,1]
    var = max(m2 - m1 ** 2, 1e-12)
    c = m1 * (1 - m1) / var - 1                              # Beta concentration
    a, b = max(m1 * c, 1e-3), max((1 - m1) * c, 1e-3)
    levels = E0 + span * _beta.ppf((np.arange(N) + 0.5) / N, a, b)
    return (levels, (a, b)) if return_params else levels


def beta_from(spectral, N=None, return_params=False):
    """Beta spectrum straight from a Spectral object: pulls extreme() + moment(1,2)."""
    N = N or spectral.N
    E0, Emax = spectral.extreme()
    mu1, mu2 = spectral.moment(1) / spectral.N, spectral.moment(2) / spectral.N
    return beta_spectrum(E0, Emax, mu1, mu2, N, return_params)
