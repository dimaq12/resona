"""
resona.flow — free convolution as the complex Burgers flow; defect = shock = edge.

Free heat flow  μ_t = μ_0 ⊞ semicircle(variance t)  is the inviscid complex
BURGERS equation for the Cauchy transform,  ∂_t G + G ∂_z G = 0.  Its
characteristics are the subordination map; where they cross, a SHOCK forms — and
that shock is exactly a spectral EDGE / band merger.  So the program's central
object has one more face: the DEFECT is a literal PDE shock.

The flow is LINEAR in the lifted coordinate: R_{μ_t}(w) = R_{μ_0}(w) + t·w — the
shock lives only in the density G, never in the R-transform ("a shock is a sum of
linearities").  Two atoms ±1 merge into one band at the critical time t_c = 1.
"""
import numpy as np
from . import subordination as _sub


def burgers_density(spectral, t, xs, eta=1e-3):
    """Density of the free-heat-flowed measure μ_t = μ_0 ⊞ semicircle(t), on xs.

    (t is the semicircle VARIANCE; equivalently A + √t·GOE.)  A shock = band edge.
    """
    return _sub.averaged_dos(spectral, np.sqrt(max(t, 0.0)), xs, eta=eta)


def shock_time(spectral, t_max=4.0, n_t=160, n_x=400, eta=2e-3, thresh=0.02):
    """The band-merger / shock time t_c: the smallest t at which the spectral gap
    fills in under the free heat flow (two bands → one).  Returns t_c or None."""
    nodes, _ = _sub._nw(spectral)
    lo, hi = float(nodes.min()), float(nodes.max())
    mid = 0.5 * (lo + hi)
    pad = 0.5 * (hi - lo) + 1.0
    xs = np.linspace(lo - pad, hi + pad, n_x)
    for t in np.linspace(1e-3, t_max, n_t):
        rho = burgers_density(spectral, t, xs, eta=eta)
        # gap is filled when the density near the centre exceeds a fraction of its peak
        centre = rho[np.abs(xs - mid) < 0.05 * (hi - lo + 1e-9)]
        if centre.size and centre.min() > thresh * rho.max():
            return float(t)
    return None
