"""
resona.spectral — the Response transform: the "FFT of operators".

`fft(x)` takes a signal to the basis where convolution becomes pointwise multiply.
`Spectral.of(A)` takes an operator (black-box matvec) to the representation where
COMPOSITION becomes ADDITION (free convolution) — matrix-free — and from which
every spectral functional is read.

One object.  Three verbs:
    s = Spectral.of(matvec, N)     # PROBE   — harvest the field   (like fft)
    s + t   ;   s @ t              # COMPOSE — A+B, A·B  (never formed, never eig'd)
    s.trace(f) ; s.density(x) ; s.extreme() ; s.moment(p)   # READ   (like ifft)

Plus the honest cost dial:
    s.effective_rank()             # Φ₁ — harvestable structure (low = cheap)
"""
from __future__ import annotations
import numpy as np

__all__ = ["Spectral"]


def _lanczos(matvec, v0, k):
    """k-step Lanczos with full reorthogonalization. Returns (alpha, beta)."""
    N = len(v0)
    V = np.zeros((N, k)); alpha = np.zeros(k); beta = np.zeros(max(k - 1, 0))
    q = v0 / np.linalg.norm(v0); V[:, 0] = q; qprev = np.zeros(N); b = 0.0
    for j in range(k):
        w = matvec(q) - b * qprev
        alpha[j] = float(q @ w)
        w = w - alpha[j] * q
        w -= V[:, :j + 1] @ (V[:, :j + 1].T @ w)
        if j < k - 1:
            b = float(np.linalg.norm(w))
            beta[j] = b
            if b < 1e-12:
                return alpha[:j + 1], beta[:j]
            qprev, q = q, w / b
            V[:, j + 1] = q
    return alpha, beta


class Spectral:
    """The response of an operator: Ritz nodes (the 'frequencies') and quadrature
    weights (the 'amplitudes'), harvested matrix-free by stochastic Lanczos
    quadrature.  Carries an optional matvec so compositions stay matrix-free.
    """

    def __init__(self, nodes, weights, matvec=None, N=None):
        self.nodes = np.asarray(nodes, float)
        self.weights = np.asarray(weights, float)
        self.matvec = matvec
        self.N = N

    # ── PROBE (forward transform) ────────────────────────────────────────────
    @classmethod
    def of(cls, matvec, N, k=48, probes=8, seed=0):
        """Harvest the response of the operator given by `matvec` (acts on R^N).

        Stochastic Lanczos quadrature: `probes` random vectors, `k` Lanczos steps
        each.  Cost O(probes * k) matvecs.  No matrix is formed, no eig is called.
        """
        rng = np.random.default_rng(seed)
        nodes, weights = [], []
        for _ in range(probes):
            al, be = _lanczos(matvec, rng.standard_normal(N), k)
            kk = len(al)
            T = np.diag(al) + np.diag(be[:kk - 1], 1) + np.diag(be[:kk - 1], -1)
            theta, S = np.linalg.eigh(T)
            nodes.append(theta)
            weights.append(S[0, :] ** 2 / probes)
        return cls(np.concatenate(nodes), np.concatenate(weights), matvec, N)

    # ── COMPOSE (the free-convolution theorem: hard op → linear, matrix-free) ──
    def __add__(self, other: "Spectral") -> "Spectral":
        """A + B — never forms A+B, never diagonalizes; uses (A+B)x = Ax + Bx."""
        self._require_matvec(other)
        return Spectral.of(lambda x: self.matvec(x) + other.matvec(x), self.N)

    def __matmul__(self, other: "Spectral") -> "Spectral":
        """A · B — composition of the actions."""
        self._require_matvec(other)
        return Spectral.of(lambda x: self.matvec(other.matvec(x)), self.N)

    def apply(self, g_matvec) -> "Spectral":
        """f(A): supply a matvec for f(A) (shift, power, etc.) — the pushforward."""
        return Spectral.of(g_matvec, self.N)

    # ── READ (inverse transform: any spectral functional) ─────────────────────
    def trace(self, f) -> float:
        """Tr f(A) = N · E[f(λ)] = N · Σ w·f(node)."""
        return float(self.N) * float(np.sum(self.weights * f(self.nodes)))

    def moment(self, p: int) -> float:
        """Tr(A^p)."""
        return self.trace(lambda x: x ** p)

    def density(self, xs, eta: float = 0.1) -> np.ndarray:
        """Density of states ρ(x), Lorentzian-broadened by `eta`."""
        xs = np.atleast_1d(np.asarray(xs, float))
        return np.array([float(np.sum(self.weights * (eta / np.pi)
                                      / ((x - self.nodes) ** 2 + eta ** 2))) for x in xs])

    def extreme(self) -> tuple[float, float]:
        """Extreme eigenvalues (Lanczos resolves these first / most reliably)."""
        return float(self.nodes.min()), float(self.nodes.max())

    # ── COST dial (the harvestable structure) ─────────────────────────────────
    def effective_rank(self) -> float:
        """Φ₁ = Tr(A)² / Tr(A²) — participation ratio / effective number of modes,
        for POSITIVE-SEMIDEFINITE operators (covariance, kernel, Hessian).  Low Φ₁
        ⇒ structured/cheap; high ⇒ near the genuine frontier.  (Computed from the
        trace moments, which SLQ estimates robustly.)"""
        m1, m2 = self.moment(1), self.moment(2)
        return float(m1 * m1 / m2) if m2 > 0 else 1.0

    # ── internals ─────────────────────────────────────────────────────────────
    def _require_matvec(self, other):
        if self.matvec is None or other.matvec is None or self.N != other.N:
            raise ValueError("composition needs both operands' matvecs on the same R^N")

    def __repr__(self):
        lo, hi = self.extreme()
        return f"Spectral(N={self.N}, support=[{lo:.3g}, {hi:.3g}], eff_rank={self.effective_rank():.1f})"
