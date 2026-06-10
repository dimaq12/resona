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
from .lift import _nw


def cauchy(spectral, z):
    """Cauchy/Stieltjes transform G(z) = Σ w_i/(z − λ_i) (complex z allowed)."""
    nodes, w = _nw(spectral); w = w / w.sum()
    return np.sum(w / (z - nodes))


def pastur(GA, z, sigma2, iters=2000, tol=1e-13, damp=0.5):
    """Solve the subordination fixed point g = G_A(z − σ²·g) for complex z."""
    g = GA(z)
    for _ in range(iters):
        gn = GA(z - sigma2 * g)
        if abs(gn - g) < tol:
            return gn
        g = (1 - damp) * g + damp * gn
    return g


def averaged_dos(spectral, sigma, xs, eta=1e-3):
    """Density of  μ_A ⊞ semicircle(σ²)  (= A + σ·GOE, disorder-averaged), on xs.

    Closed form via the Pastur fixed point — no disorder realizations, no eig.
    """
    GA = lambda z: cauchy(spectral, z)
    s2 = sigma ** 2
    out = np.empty(len(xs))
    for i, x in enumerate(xs):
        g = pastur(GA, x + 1j * eta, s2)
        out[i] = max(-g.imag / np.pi, 0.0)
    return out


def free_add_semicircle_moments(spectral, sigma, order=4):
    """Moments m_n of μ_A ⊞ semicircle(σ²).  m1 = m1(A); m2 = m2(A) + σ²; …
    (the semicircle of variance σ² contributes only κ₂ = σ²)."""
    from .lift import _nw
    nodes, w = _nw(spectral); w = w / w.sum()
    mA = [float(np.sum(w * nodes ** n)) for n in range(1, order + 1)]
    from .free import free_cumulants
    from .lift import carleman_scalar  # noqa: F401  (kept import graph explicit)
    kap = np.array(free_cumulants(mA))
    if len(kap) >= 2:
        kap[1] += sigma ** 2                       # ⊞ semicircle adds κ₂ = σ²
    from .free import _trunc_pow
    # moments from cumulants: M(z) = 1 + Σ κ_n zⁿ M(z)ⁿ
    m = [1.0]
    N = len(kap)
    for j in range(1, N + 1):
        s = 0.0
        mm = m + [0.0] * (N - len(m) + 1)
        for n in range(1, j + 1):
            s += kap[n - 1] * _trunc_pow(mm, n, N)[j - n]
        m.append(s)
    return m[1:]
