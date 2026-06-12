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
from .lift import _nw, cauchy as _cauchy   # internal; the public read is lift.cauchy / s.cauchy


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

    BRANCH NOTE: the physical solution has Im g ≤ 0 in the upper half-plane;
    the ρ = max(−Im g/π, 0) clip below assumes the damped iteration stayed on
    that branch.  Near edges (where `contraction` → 1) verify convergence —
    a wrong-branch g would read as silent ρ = 0, not as an error.
    """
    g = pastur_grid(spectral, np.asarray(xs, float) + 1j * eta, sigma ** 2, g0=g0)
    return np.maximum(-g.imag / np.pi, 0.0)
    # (the moment version μ_A ⊞ semicircle is just resona.lift.free_convolution
    #  with a semicircle, or read off as m₂ = m₂(A) + σ² — no separate function.)


def contraction(spectral, xs, sigma2, eta=1e-9):
    """|T′(g*)| — the stability of the Pastur fixed point at each x.

    The subordination iteration g ← G_A(z − σ²g) contracts at rate
    |T′(g*)| = σ²·|G_A′(z − σ²g*)|.  Near a spectral EDGE of μ_A ⊞ sc(σ²)
    this number approaches 1 and the computation critically slows — the
    defect / edge-of-chaos of free probability's own fixed point.  This is
    a PURE READ: it returns the measured contraction, nothing else; compare
    to 1 yourself (≳0.95 ⇒ expect slow convergence, you are near an edge).
    Note this is a SOLVER-STABILITY diagnostic (of the fixed-point iteration
    itself), not a property of the spectral measure.

    HONEST LIMIT (measured, FA/revise_stress): the critical window where
    |T′| visibly rises is NOT universal — it narrows with σ² and the edge's
    curvature (a soft edge at σ²=0.1 reads only ~0.3 at distance 1e-3).
    The observable is exact regardless; the window is physics.
    """
    nodes, w = _nw(spectral)
    w = w / w.sum()
    Z = np.atleast_1d(np.asarray(xs, float)) + 1j * eta
    g = pastur_grid(spectral, Z, sigma2)
    zeff = Z - sigma2 * g
    Gp = -(w[None, :] / (zeff[:, None] - nodes[None, :]) ** 2).sum(1)
    out = sigma2 * np.abs(Gp)
    return out if out.size > 1 else float(out[0])
