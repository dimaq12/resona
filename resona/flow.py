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
from __future__ import annotations

import numpy as np
from . import subordination as _sub


def burgers_density(spectral, t: float, xs, eta: float = 1e-3, g0=None) -> np.ndarray:
    """Density of the free-heat-flowed measure μ_t = μ_0 ⊞ semicircle(t), on xs.

    (t is the semicircle VARIANCE; equivalently A + √t·GOE.)  A shock = band edge.
    Thin alias of `subordination.averaged_dos(spectral, √t, xs)` — the flow face
    of the same fixed point.  `g0` warm-starts a t-sweep (see `shock_time`).
    """
    return _sub.averaged_dos(spectral, np.sqrt(max(t, 0.0)), xs, eta=eta, g0=g0)


def shock_time(spectral, t_max: float = 4.0, eta: float = 2e-3) -> float | None:
    """The band-merger / shock time t_c: the smallest t at which the spectral gap
    fills in under the free heat flow (two bands → one).  Returns t_c or None.

    Two dials (the 2.0 diet): `t_max` — how far to flow; `eta` — resolvent
    broadening (smaller = sharper gap detection, slower fixed point).  Grid
    resolution and the fill threshold are internal (160 × 400, 2% of peak):
    knobs nobody should have had to turn.

    MIDPOINT LIMITATION: the gap is probed at the ARITHMETIC midpoint
    mid = ½(min+max) of the support.  This is exact only when the gap sits there
    — i.e. for two equal-weight atoms (the symmetric ±a case).  For asymmetric /
    unequal-weight spectra the true gap is off-centre and this probe can miss it
    (read None when a merger did occur, or detect the wrong band).
    """
    n_t, n_x, thresh = 160, 400, 0.02
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
