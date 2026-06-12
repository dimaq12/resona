"""resona.solve — catastrophe-aware root solving & eigenvalue polish."""
import numpy as np
from fractions import Fraction as Fr
import pytest

from resona.solve import exact_poly, catastrophe_solve, rayleigh_polish


def _digits(roots, truth):
    e = max(min(abs(complex(t) - r) for r in roots) for t in truth)
    return -np.log10(e) if e > 0 else 16.0


def test_q3_cluster_exact_recovers_full_precision():
    a = Fr(1, 10 ** 5)
    truth = [1 - a, 1, 1 + a, 5, -3]                  # q=3 cluster + 2 simple, EXACT
    roots, q, dps, naive = catastrophe_solve(exact_poly(truth))
    assert q == 3
    assert _digits(naive, truth) <= 6.5               # float64 silently loses ~16/q
    assert _digits(roots, truth) >= 15.0              # the tool recovers them


def test_q5_cluster_exact():
    a = Fr(1, 1000)
    truth = [1 + a * j for j in (-2, -1, 0, 1, 2)]    # q=5 cluster
    roots, q, dps, naive = catastrophe_solve(exact_poly(truth))
    assert q == 5
    assert dps >= 5 * 15
    assert _digits(roots, truth) >= 15.0


def test_float64_coeffs_partial_and_honest():
    # information already destroyed by rounding: tool must not crash, must not
    # be worse than naive, and CANNOT reach full precision (the honest cap)
    a = Fr(1, 10 ** 5)
    truth = [1 - a, 1, 1 + a, 5, -3]
    f64 = [Fr(float(c)) for c in exact_poly(truth)]
    roots, q, dps, naive = catastrophe_solve(f64)
    d_naive, d_tool = _digits(naive, truth), _digits(roots, truth)
    assert d_tool >= d_naive - 0.1                    # never worse than naive
    assert d_tool < 15.0                              # and honestly capped by the data


def test_simple_roots_budget_stays_small():
    truth = [1, 2, 5, -3]
    roots, q, dps, naive = catastrophe_solve(exact_poly(truth))
    assert q == 1
    assert dps == 30                                  # 1×15 + 15: no wasted budget
    assert _digits(roots, truth) >= 14.5


def _tridiag(N, seed=3):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(N)
    e = 0.5 * rng.standard_normal(N - 1)
    A = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    return A


def test_rayleigh_polish_dense():
    N = 80
    A = _tridiag(N)
    ev = np.sort(np.linalg.eigvalsh(A))
    t = ev[N // 3]
    lam = rayleigh_polish(A, t * (1 + 1e-3))          # coarse seed, 3 digits off
    assert np.min(np.abs(ev - lam)) < 1e-12 * max(1.0, abs(lam))


def test_rayleigh_polish_matvec():
    N = 80
    A = _tridiag(N)
    ev = np.sort(np.linalg.eigvalsh(A))
    t = ev[N // 2]
    lam = rayleigh_polish(lambda x: A @ x, t + 1e-3, N=N)
    assert np.min(np.abs(ev - lam)) < 1e-10 * max(1.0, abs(lam))
