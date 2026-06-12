"""
resona.subordination — the resolvent-branch engine: disorder averaging & free
addition with a semicircle, in CLOSED FORM (no realization loop, no eig).

The disorder-averaged resolvent g(z) = ⟨Tr (z − H)⁻¹⟩/N of  H = A + σ·W  (W a
GOE-like noise / free semicircular element of variance σ²) solves the PASTUR
self-consistent (subordination / matrix-Dyson) fixed point

        g(z) = G_A( z − σ²·g(z) ),      G_A(ζ) = ∫ dμ_A(λ)/(ζ − λ),

and the averaged density of states is  ρ(x) = −Im g(x + i0)/π.  This is free
ADDITIVE convolution with a semicircle — μ_A ⊞ semicircle(σ²) — computed from the
spectrum of A alone.  The contraction of the fixed point is fast in the bulk and
slows to a crawl at the spectral EDGE (critical slowing) — the defect / edge of
chaos of free probability's own computation.
"""
import numpy as np
from .lift import _nw, cauchy            # cauchy lives in lift (the transform module)


def pastur(GA, z, sigma2, iters=2000, tol=1e-13, damp=0.5):
    """Solve the subordination fixed point g = G_A(z − σ²·g) for complex z."""
    g = GA(z)
    for _ in range(iters):
        gn = GA(z - sigma2 * g)
        if abs(gn - g) < tol:
            return gn
        g = (1 - damp) * g + damp * gn
    return g


def pastur_grid(spectral, zs, sigma2, iters=2000, tol=1e-13, damp=0.5, g0=None):
    """The subordination fixed point g = G_A(z − σ²·g) solved on a whole grid of
    complex z AT ONCE — one vectorized damped iteration with an active mask
    (converged points freeze), instead of len(zs) scalar solves.  Same fixed
    point, same tolerance, ~10× faster.  `g0` warm-starts (e.g. from the
    previous step of a t-sweep).  Returns g(zs)."""
    nodes, w = _nw(spectral)
    w = w / w.sum()
    Z = np.asarray(zs, complex)
    GA = lambda zz: (w[None, :] / (zz[:, None] - nodes[None, :])).sum(1)
    g = GA(Z) if g0 is None else np.array(g0, complex)
    active = np.ones(len(Z), bool)
    for _ in range(iters):
        gn = GA(Z[active] - sigma2 * g[active])
        done = np.abs(gn - g[active]) < tol
        upd = (1 - damp) * g[active] + damp * gn
        upd[done] = gn[done]                       # parity with the scalar `pastur`
        g[active] = upd
        idx = np.where(active)[0]
        active[idx[done]] = False
        if not active.any():
            break
    return g


def averaged_dos(spectral, sigma, xs, eta=1e-3, g0=None):
    """Density of  μ_A ⊞ semicircle(σ²)  (= A + σ·GOE, disorder-averaged), on xs.

    Closed form via the Pastur fixed point — no disorder realizations, no eig.
    Vectorized over the whole grid (see `pastur_grid`).
    """
    g = pastur_grid(spectral, np.asarray(xs, float) + 1j * eta, sigma ** 2, g0=g0)
    return np.maximum(-g.imag / np.pi, 0.0)
    # (the moment version μ_A ⊞ semicircle is just resona.lift.free_convolution
    #  with a semicircle, or read off as m₂ = m₂(A) + σ² — no separate function.)
