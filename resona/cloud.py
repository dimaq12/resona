"""
resona.cloud — the non-Hermitian probe: a complex Ritz CLOUD, honestly.

THE WARNING COMES FIRST.  For a non-normal operator, Ritz values are NOT
eigenvalue estimates in the Hermitian sense: they live in the numerical range
and can sit far from any eigenvalue — that distance IS the pseudospectrum
phenomenon (see `resona.defect.pseudospectrum_radius`, the ε^{1/q} bloom).
This module therefore ships a deliberately SMALL object: probe + read only.
There is no compose — free probability for non-normal operators without
extra structure does not exist, and we do not fake it.

What the cloud is good for, with the right epistemics:
- `abscissa()`  — max Re of the cloud: an estimate (from below) of the
  NUMERICAL abscissa ω(A) = max Re W(A) — the dial of TRANSIENT growth
  (d/dt ‖e^{tA}‖ at t=0⁺).  For non-normal A it can sit far ABOVE the
  spectral abscissa: that gap is the pseudospectrum phenomenon, and it is
  physical (transients grow even when every eigenvalue decays).
- `radius()`    — max |cloud|: between the spectral and the numerical radius.
  For row-stochastic/Markov operators the leading mode is normal-like and
  this reads the mixing structure reliably.
- `nodes`       — the raw cloud: Koopman/DMD frequencies, Markov spectra,
  damping rates; for NORMAL operators these are ordinary Ritz values and
  converge to eigenvalues.
"""
import numpy as np

__all__ = ["Cloud", "cloud"]


def _arnoldi_ritz(matvec, v0, k):
    """k-step Arnoldi (modified Gram–Schmidt) → complex Ritz values."""
    N = len(v0)
    Q = np.zeros((N, k + 1), complex)
    H = np.zeros((k + 1, k), complex)
    Q[:, 0] = v0 / np.linalg.norm(v0)
    m = k
    for j in range(k):
        w = np.asarray(matvec(Q[:, j]), complex)
        for i in range(j + 1):
            H[i, j] = np.vdot(Q[:, i], w)
            w -= H[i, j] * Q[:, i]
        corr = Q[:, :j + 1].conj().T @ w          # DGKS re-orthogonalization:
        H[:j + 1, j] += corr                      # twice is enough — keeps the
        w -= Q[:, :j + 1] @ corr                  # Ritz cloud inside W(A)
        h = np.linalg.norm(w)
        H[j + 1, j] = h
        if h < 1e-12:
            m = j + 1
            break
        Q[:, j + 1] = w / h
    return np.linalg.eigvals(H[:m, :m])


class Cloud:
    """Complex Ritz cloud of a (possibly non-normal) operator.

    `nodes` are Arnoldi Ritz values pooled over probes.  Reads are LOWER
    bounds by construction (the cloud sits inside the numerical range)."""

    def __init__(self, nodes, matvec=None, N=None, probe_sizes=None):
        self.nodes = np.asarray(nodes, complex)
        self.matvec = matvec
        self.N = N
        self.probe_sizes = probe_sizes

    def radius(self) -> float:
        """max |cloud| — between the spectral radius ρ(A) and the numerical
        radius r(A); equal to ρ(A) for normal operators."""
        return float(np.max(np.abs(self.nodes)))

    def abscissa(self) -> float:
        """max Re of the cloud ≈ the NUMERICAL abscissa ω(A), from below —
        the transient-growth dial.  ω(A) ≥ spectral abscissa, with a gap
        exactly when the operator is non-normal (then transients grow even
        though every eigenvalue decays).  For eigenvalue-level stability of a
        non-normal operator use `defect.pseudospectrum_radius`."""
        return float(np.max(self.nodes.real))

    def __repr__(self):
        return (f"Cloud(N={self.N}, nodes={len(self.nodes)}, "
                f"radius≥{self.radius():.3g}, abscissa≥{self.abscissa():.3g})")


def cloud(matvec, N=None, k=48, probes=4, seed=0):
    """Probe a NON-HERMITIAN operator: Arnoldi from `probes` random vectors,
    pooled complex Ritz values.  See the module docstring for what these
    values are — and are not.

    `matvec` may be a callable (then N is required), or a scipy
    LinearOperator / sparse matrix / dense array (then N is read off .shape)."""
    from .spectral import _as_operator
    matvec, N = _as_operator(matvec, N)
    rng = np.random.default_rng(seed)
    nodes, sizes = [], []
    for _ in range(probes):
        th = _arnoldi_ritz(matvec, rng.standard_normal(N), k)
        nodes.append(th)
        sizes.append(len(th))
    return Cloud(np.concatenate(nodes), matvec, N, probe_sizes=sizes)
