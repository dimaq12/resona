"""
resona — the resonance of an operator.  Probe it, hear its spectrum.

The FFT of operators: `fft(x)` takes a signal to the basis where convolution
becomes pointwise multiply.  `resona.of(A)` takes an operator (black-box matvec)
to the representation where composition becomes addition (free convolution) —
matrix-free — and from which every spectral functional is read.

    import resona
    s = resona.of(matvec, N)       # PROBE   — ring the operator, read its modes
    s + t ; s @ t                  # COMPOSE — A+B, A·B  (never formed, never eig'd)
    s.trace(f) ; s.density(x) ; s.extreme() ; s.moment(p)   # READ
    s.boxplus(t) ; s.cauchy(z) ; s.r(w) ; s.s(w)   # free convolution & lifted coords
    s.flow(t, xs) ; s.levels(N)    # heat flow / disorder; Beta spectrum closure
    s.effective_rank() ; s.condition()             # the honest cost dials (Φ₁, κ)

    resona.apply(matvec, f, v)     # APPLY   — f(A)·v: solve, evolve, filter
    resona.cloud(matvec, N)        # non-Hermitian? the Ritz CLOUD (Spectral is
                                   # for self-adjoint operators; cloud for general)
    resona.solve.rayleigh_polish   # PRECISION — spend effort only on the defect
"""
from .spectral import Spectral, apply, quadform, local_spectrum, local_density, from_measure, from_eigenbasis
from . import wkernel, lift, beta, defect, free, subordination, cost, flow, solve, thermal
from .cloud import cloud, Cloud

#: convenience: ``resona.of(matvec, N)`` == ``Spectral.of(matvec, N)``
of = Spectral.of

__version__ = "1.2.0"
__all__ = ["Spectral", "of", "apply", "quadform", "local_spectrum", "local_density",
           "from_measure", "from_eigenbasis", "cloud", "Cloud", "wkernel", "lift", "beta",
           "defect", "free", "subordination", "cost", "flow", "solve", "thermal"]
