"""Regression tests for the audit-batch fixes in solve / flow / free / beta.

Covers:
  • rayleigh_polish: symmetric path unchanged vs dense; non-symmetric path
    (symmetric=False, LGMRES) lands near a true eigenvalue where the old
    MINRES-only path returned garbage.
  • beta_spectrum input guards (non-finite / mu1 out of support) raise, while
    a normal beta_from still returns sensible levels.
  • rie_clean_additive emits a warning when cleaning drives eigenvalues negative.
  • shock_time on a 2-atom spectrum is finite and pinned.
"""
import warnings

import numpy as np
import pytest

import resona
from resona.solve import rayleigh_polish
from resona.beta import beta_spectrum, beta_from
from resona.free import rie_clean_additive
from resona.flow import shock_time


# --------------------------------------------------------------------------
# rayleigh_polish
# --------------------------------------------------------------------------
def test_rayleigh_polish_symmetric_dense_matrixfree_agree():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((40, 40))
    A = M + M.T                                   # symmetric
    evals = np.linalg.eigvalsh(A)
    target = evals[-1]
    seed = target + 0.05

    lam_dense = rayleigh_polish(A, seed, iters=8)
    lam_mf = rayleigh_polish(lambda x: A @ x, seed, N=40, iters=8,
                             symmetric=True)
    # both must converge onto a genuine eigenvalue
    assert np.min(np.abs(evals - lam_dense)) < 1e-8
    assert np.min(np.abs(evals - lam_mf)) < 1e-7
    # symmetric matrix-free path must reproduce its legacy default
    assert lam_mf == rayleigh_polish(lambda x: A @ x, seed, N=40, iters=8)


def test_rayleigh_polish_nonsymmetric_lands_near_true_eigenvalue():
    # A strongly NON-NORMAL operator (upper-triangular -> eigenvalues = diagonal).
    # MINRES assumes symmetry: its inner shifted solve is invalid here, so the
    # legacy matrix-free path (symmetric=True) returns garbage.  LGMRES
    # (symmetric=False) solves the non-symmetric shift correctly and the polish
    # lands near a true eigenvalue.  Deterministic (no RNG in the result).
    d = np.array([1.0, 3.0, 6.0, 10.0, 15.0, 21.0])
    n = len(d)
    A = np.diag(d) + np.triu(np.full((n, n), 4.0), 1)   # large off-diag -> non-normal
    assert np.linalg.norm(A - A.T) > 1.0                # genuinely non-symmetric
    mv = lambda x: A @ x
    seed = 15.3                                         # seed near the eigenvalue 15

    lam_ns = rayleigh_polish(mv, seed, N=n, iters=30, symmetric=False)
    lam_mr = rayleigh_polish(mv, seed, N=n, iters=30, symmetric=True)

    err_ns = np.min(np.abs(d - lam_ns))
    err_mr = np.min(np.abs(d - lam_mr))
    assert err_ns < 1e-4                                # LGMRES: near a true eigenvalue
    assert err_mr > 1e-2                                # MINRES: off the spectrum (garbage)
    assert err_ns < err_mr                              # the fix is strictly better here


# --------------------------------------------------------------------------
# beta guards
# --------------------------------------------------------------------------
def test_beta_spectrum_raises_on_mu1_outside_support():
    with pytest.raises(ValueError):
        beta_spectrum(0.0, 1.0, 1.5, 2.3, 32)     # mu1 > Emax


def test_beta_spectrum_raises_on_nonfinite():
    with pytest.raises(ValueError):
        beta_spectrum(0.0, 1.0, np.nan, 0.3, 32)
    with pytest.raises(ValueError):
        beta_spectrum(0.0, 1.0, 0.5, np.inf, 32)


def test_beta_spectrum_overdispersed_variance_clamped_not_ushaped():
    # mu2 chosen so var > Bernoulli bound: must NOT silently invert to U-shape.
    lev = beta_spectrum(0.0, 1.0, 0.5, 0.9, 64)   # var=0.65 > 0.25 bound
    assert np.all(np.isfinite(lev))
    assert lev.min() >= 0.0 and lev.max() <= 1.0


def test_beta_from_normal_spectral_sensible():
    rng = np.random.default_rng(3)
    M = rng.standard_normal((60, 60))
    A = (M + M.T) / np.sqrt(2 * 60)
    s = resona.of(A)
    lev = beta_from(s, N=60)
    E0, Emax = s.extreme()
    assert np.all(np.isfinite(lev))
    assert lev.min() >= E0 - 1e-9 and lev.max() <= Emax + 1e-9
    # monotone (ppf of a quantile grid)
    assert np.all(np.diff(lev) >= -1e-9)


# --------------------------------------------------------------------------
# rie_clean_additive negative warning
# --------------------------------------------------------------------------
def test_rie_clean_additive_warns_on_negative():
    lam = np.array([0.01, 0.02, 0.03, 1.0, 1.0, 1.0])
    with pytest.warns(UserWarning):
        xi = rie_clean_additive(lam, sigma=5.0)   # huge sigma -> drives negative
    assert np.any(xi < 0.0)


def test_rie_clean_additive_no_warn_when_clean_positive():
    lam = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # any warning -> failure
        xi = rie_clean_additive(lam, sigma=0.05)
    assert np.all(xi > 0.0)


# --------------------------------------------------------------------------
# shock_time pin
# --------------------------------------------------------------------------
def test_shock_time_two_atoms_finite_and_pinned():
    A = np.diag([-1.0] * 50 + [1.0] * 50)          # two equal-weight atoms ±1
    s = resona.of(A)
    tc = shock_time(s)
    assert tc is not None
    assert np.isfinite(tc)
    assert 0.0 < tc < 4.0
    # hard pin: the converged value must be byte-identical (the g0-threading
    # speed optimisation was deliberately SKIPPED to keep this exact).
    assert tc == pytest.approx(0.8812830188679245, abs=0, rel=0)
    assert tc == shock_time(s)                          # deterministic across runs
