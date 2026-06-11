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
    s.effective_rank()             # the honest cost dial (Φ₁)
"""
from .spectral import Spectral, apply, local_spectrum, local_density, from_measure, from_eigenbasis
from . import wkernel, lift, beta, defect, free, subordination, cost, flow

#: convenience: ``resona.of(matvec, N)`` == ``Spectral.of(matvec, N)``
of = Spectral.of

__version__ = "0.3.0"
__all__ = ["Spectral", "of", "apply", "local_spectrum", "local_density",
           "from_measure", "from_eigenbasis", "wkernel", "lift", "beta", "defect", "free",
           "subordination", "cost", "flow"]
