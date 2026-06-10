"""Tests for opfft.Spectral — verified against dense ground truth (fixed seeds)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy import linalg
from opfft import Spectral

rng = np.random.default_rng(7)


def _sym(N):
    M = rng.standard_normal((N, N))
    return (M + M.T) / 2


def test_trace_of_constant_is_N():
    N = 300
    A = _sym(N)
    s = Spectral.of(lambda v: A @ v, N, probes=8)
    assert abs(s.trace(lambda x: np.ones_like(x)) - N) < 1e-6 * N


def test_moment_matches_trace():
    # PD operator with O(N) trace (kernel/Hessian-like) — low relative SLQ variance.
    N, n = 300, 500
    B = rng.standard_normal((n, N))
    A = (B.T @ B) / n                                       # eigenvalues O(1), Tr = O(N) > 0
    s = Spectral.of(lambda v: A @ v, N, k=70, probes=30)
    for p in (1, 2):
        true = float(np.trace(np.linalg.matrix_power(A, p)))
        assert abs(s.moment(p) - true) <= 0.15 * abs(true)    # SLQ noise band


def test_extreme_eigenvalues():
    N = 400
    A = _sym(N)
    s = Spectral.of(lambda v: A @ v, N, k=80, probes=4)
    ev = np.sort(linalg.eigvalsh(A))
    lo, hi = s.extreme()
    assert abs(hi - ev[-1]) < 0.02 * (ev[-1] - ev[0])
    assert abs(lo - ev[0]) < 0.02 * (ev[-1] - ev[0])


def test_compose_sum_matrix_free():
    N = 500
    A, B = _sym(N), _sym(N)
    sA = Spectral.of(lambda v: A @ v, N, k=80, probes=4)
    sB = Spectral.of(lambda v: B @ v, N, k=80, probes=4)
    lo, hi = (sA + sB).extreme()                 # A+B never formed
    ev = np.sort(linalg.eigvalsh(A + B))
    assert abs(hi - ev[-1]) < 0.02 * (ev[-1] - ev[0])
    assert abs(lo - ev[0]) < 0.02 * (ev[-1] - ev[0])


def test_effective_rank_low_vs_full():
    N = 400
    U = rng.standard_normal((N, 6)); low = U @ U.T
    full = _sym(N)
    s_low = Spectral.of(lambda v: low @ v, N, k=60, probes=12)
    s_full = Spectral.of(lambda v: full @ v, N, k=60, probes=12)
    assert s_low.effective_rank() < 0.2 * N
    assert s_full.effective_rank() > 3 * s_low.effective_rank()
