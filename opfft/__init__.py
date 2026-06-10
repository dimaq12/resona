"""
opfft — the FFT of operators.

`fft(x)` takes a signal to the basis where convolution becomes pointwise multiply.
`Spectral.of(A)` takes an operator (black-box matvec) to the representation where
composition becomes addition (free convolution) — matrix-free — and from which
every spectral functional is read.

    from opfft import Spectral
    s = Spectral.of(matvec, N)     # PROBE
    s + t ; s @ t                  # COMPOSE  (A+B, A·B — never formed, never eig'd)
    s.trace(f) ; s.density(x) ; s.extreme() ; s.moment(p)   # READ
    s.effective_rank()             # the honest cost dial (Φ₁)
"""
from .spectral import Spectral

__version__ = "0.1.0"
__all__ = ["Spectral"]
