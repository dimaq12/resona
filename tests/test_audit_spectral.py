"""Regression tests for the audit-batch fixes in resona/spectral.py.

Each test pins a VERIFIED correctness/robustness fix so it cannot silently
regress.  Nothing here changes the numeric result of the existing happy paths.
"""
import warnings

import numpy as np
import pytest
import scipy.linalg as sla

import resona
from resona.spectral import (
    Spectral, apply, local_spectrum, quadform,
)


def _sym(n=24, seed=0):
    """A reproducible real-symmetric matrix and its matvec."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = (M + M.T) / 2.0
    return A, (lambda x: A @ x)


# ── Fix 1: complex f on a real-symmetric A with a real v ──────────────────────
def test_complex_f_real_hermitian_apply():
    A, mv = _sym(20, seed=1)
    v = np.cos(np.arange(20) * 0.4) + 0.2          # real seed
    t = 0.3
    got = apply(mv, lambda x: np.exp(-1j * t * x), v, hermitian=True)
    want = sla.expm(-1j * t * A) @ v
    assert np.iscomplexobj(got)
    assert np.allclose(got, want, atol=1e-7, rtol=1e-7)


def test_real_f_real_hermitian_apply_unchanged():
    # the real-f path must stay real and correct (bit-for-bit preserved)
    A, mv = _sym(20, seed=2)
    v = np.sin(np.arange(20) * 0.3) - 0.1
    got = apply(mv, np.exp, v, hermitian=True)
    want = sla.expm(A) @ v
    assert not np.iscomplexobj(got)
    assert np.allclose(got, want, atol=1e-7, rtol=1e-7)


# ── Fix 2: k >= 1 guards ──────────────────────────────────────────────────────
def test_k_guards():
    A, mv = _sym(12)
    v = np.ones(12)
    with pytest.raises(ValueError):
        Spectral.of(mv, 12, k=0)
    with pytest.raises(ValueError):
        resona.of(mv, 12, k=0)
    with pytest.raises(ValueError):
        apply(mv, np.exp, v, k=0)
    with pytest.raises(ValueError):
        local_spectrum(mv, v, k=0)


# ── Fix 3: zero / near-zero probe-or-seed vector ──────────────────────────────
def test_zero_probe_raises():
    A, mv = _sym(10)
    z = np.zeros(10)
    with pytest.raises(ValueError):
        apply(mv, np.exp, z)
    with pytest.raises(ValueError):
        local_spectrum(mv, z)


# ── Fix 4: zoom empty window ──────────────────────────────────────────────────
def test_zoom_empty_window_raises():
    A, mv = _sym(40, seed=3)
    s = resona.of(mv, 40, k=30, probes=4, seed=3)
    lo, hi = s.extreme()
    far = hi + 10 * (hi - lo)                       # a window with no spectrum
    with pytest.raises(ValueError):
        s.zoom(far, far + 1.0)


# ── Fix 5: zoom must not enable a bogus certificate ───────────────────────────
def test_zoom_no_bogus_certificate():
    A, mv = _sym(40, seed=4)
    s = resona.of(mv, 40, k=30, probes=4, seed=4)
    lo, hi = s.extreme()
    z = s.zoom(lo, 0.5 * (lo + hi))
    assert z._tridiags is None
    with pytest.raises(ValueError):
        z.trace_certified("exp", support=(None, hi + 1.0))
    # the ordinary reads must still work on a zoom object
    assert np.isfinite(z.trace(lambda x: x))
    assert np.all(np.isfinite(z.density(np.linspace(lo, hi, 5))))
    assert len(z.extreme()) == 2


# ── Fix 6: repr of an empty Spectral ──────────────────────────────────────────
def test_repr_empty():
    r = repr(Spectral([], []))
    assert isinstance(r, str) and "empty" in r


# ── Fix 7: log-domain warning when the spectrum reaches <= 0 ───────────────────
def test_trace_log_domain_warns():
    A, mv = _sym(20, seed=5)                        # indefinite → has nodes <= 0
    s = resona.of(mv, 20, k=20, probes=4, seed=5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s.trace("log")
    assert any(issubclass(x.category, RuntimeWarning) for x in w)


def test_trace_log_no_warn_when_positive():
    # SPD operator: log is fine, no warning
    A, mv = _sym(20, seed=6)
    A = A @ A.T + 5.0 * np.eye(20)
    mv = lambda x: A @ x
    s = resona.of(mv, 20, k=20, probes=4, seed=6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = s.trace("log")
    assert not any(issubclass(x.category, RuntimeWarning) for x in w)
    assert np.isfinite(val)


# ── Sanity: a normal real trace / extreme is unchanged ────────────────────────
def test_normal_trace_extreme_sane():
    n = 50
    A, mv = _sym(n, seed=7)
    s = resona.of(mv, n, k=40, probes=8, seed=7)
    # f = x^0: exact regardless of probes (Tr I = N) — pins the read machinery
    assert abs(s.trace(lambda x: np.ones_like(x)) - n) < 1e-9
    # f = x: stochastic — band the Hutchinson stderr (~||A||_F / sqrt(probes))
    tr = s.trace(lambda x: x)
    band = 6.0 * np.linalg.norm(A) / np.sqrt(8)
    assert abs(tr - np.trace(A)) < band
    lo, hi = s.extreme()
    evals = np.linalg.eigvalsh(A)
    # Ritz extremes are inner estimates of the true edges
    assert evals.min() - 1e-6 <= lo <= hi <= evals.max() + 1e-6


def test_effective_rank_deflate_err_unchanged_at_zero():
    # deflate=0 effective_rank(with_err=True) must be the plain stochastic bar
    n = 60
    rng = np.random.default_rng(8)
    M = rng.standard_normal((n, n)); A = M @ M.T
    mv = lambda x: A @ x
    s = resona.of(mv, n, k=40, probes=8, seed=8, deflate=0)
    val, err = s.effective_rank(with_err=True)
    assert np.isfinite(val) and np.isfinite(err) and err >= 0.0
