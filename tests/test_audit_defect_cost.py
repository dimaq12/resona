"""Regression tests for the audit-batch fixes in resona.defect / cost / wkernel.

Each test pins one corrected behaviour; together they guard the batch against
silent re-regression.  The headline is `test_richardson_limit_known_expansion`,
which the pre-fix exponent (`(n_k/n_{k-col})**(p0*col)`) does NOT pass.
"""
import warnings

import numpy as np
import pytest

import resona
from resona.defect import (defect, defect_jump, generator_read, richardson,
                           richardson_limit)
from resona.cost import (is_extractable, level_spacing_ratio, lift_rank,
                         rmt_class)
from resona.wkernel import track


# ── THE key correctness test: Richardson/Neville annihilates a known expansion ──

def test_richardson_limit_known_expansion():
    """P_n = L + c1 n^-p + c2 n^-2p + c3 n^-3p sampled at a geometric doubling.
    With 4 samples the Neville tableau eliminates all three error terms exactly,
    so richardson_limit must return L to ~machine precision.  The pre-fix
    exponent double-counts the column and misses L by orders of magnitude."""
    L, p = 3.7, 2.0
    c1, c2, c3 = 1.3, -0.8, 2.1
    ns = [10, 20, 40, 80]                       # geometric doubling
    vals = [L + c1 * n ** -p + c2 * n ** (-2 * p) + c3 * n ** (-3 * p) for n in ns]
    est = richardson_limit(vals, ns, p0=p)
    assert abs(est - L) < 1e-10, f"expected {L}, got {est}"


def test_richardson_limit_beats_raw_and_single_step():
    """Sanity: the tableau limit is far closer to L than the finest raw sample."""
    L, p = -2.0, 1.0
    ns = [8, 16, 32, 64, 128]
    vals = [L + 5.0 * n ** -p - 3.0 * n ** (-2 * p) + 1.0 * n ** (-3 * p) for n in ns]
    est = richardson_limit(vals, ns, p0=p)
    raw_err = abs(vals[-1] - L)
    assert abs(est - L) < 1e-9
    assert abs(est - L) < 1e-6 * raw_err


def test_richardson_limit_complex_no_crash():
    """A complex-valued sequence must not be float-coerced; it extrapolates."""
    L = 1.5 - 0.5j
    ns = [10, 20, 40]
    vals = [L + (2 + 1j) / n + (1 - 1j) / n ** 2 for n in ns]
    est = richardson_limit(vals, ns, p0=1.0)
    assert np.iscomplexobj(np.asarray(est))
    assert abs(est - L) < 1e-9


# ── defect_jump preserves a complex defect ──

def test_defect_jump_preserves_complex():
    D_n = np.array([1 + 2j, 3 - 1j])
    J = np.array([[0.0, 1.0], [-1.0, 0.0]])
    out = defect_jump(D_n, J, 1)
    assert np.iscomplexobj(out)
    assert np.allclose(out, J @ D_n)
    assert np.any(out.imag != 0)


# ── guards that should now raise ──

def test_richardson_p_zero_raises():
    with pytest.raises(ValueError):
        richardson(1.0, 1.0, p=0)


def test_generator_read_t_zero_raises():
    with pytest.raises(ValueError):
        generator_read(np.ones(3), np.zeros(3), t=0, n=4)


def test_defect_shape_mismatch_raises():
    with pytest.raises(ValueError):
        defect(np.ones(4), np.ones(5))


# ── cost.rmt_class: short spectra must not fake a Dyson class ──

@pytest.mark.parametrize("n", [3, 4])
def test_rmt_class_short_spectrum_not_spurious_gse(n):
    cls, r4 = rmt_class(np.linspace(0.0, 1.0, n))
    assert cls == "undetermined"
    assert np.isnan(r4)


def test_rmt_class_large_spectrum_still_classifies():
    """Regression guard: a real (large) spectrum must still get a Dyson class."""
    rng = np.random.default_rng(0)
    H = rng.standard_normal((400, 400)); H = (H + H.T) / 2
    cls, r4 = rmt_class(np.linalg.eigvalsh(H))
    assert cls in {"Poisson", "GOE", "GUE", "GSE"}
    assert np.isfinite(r4)


# ── cost.level_spacing_ratio: too few spacings → NaN, not a crash ──

@pytest.mark.parametrize("ev", [[5.0], [1.0, 2.0]])
def test_level_spacing_ratio_too_short_is_nan(ev):
    assert np.isnan(level_spacing_ratio(np.array(ev)))


# ── cost.lift_rank: degenerate signals ──

def test_lift_rank_zero_signal_is_nan():
    assert np.isnan(lift_rank(np.zeros(40), k=10))


def test_lift_rank_short_signal_raises():
    with pytest.raises(ValueError):
        lift_rank(np.arange(10.0), k=20)          # needs 2*20-1 = 39 samples


def test_is_extractable_normal_signal_unbroken():
    """Regression guard: a normal-length signal still returns (bool, ranks)."""
    x = np.arange(300)
    ok, ranks = is_extractable(np.sin(2 * np.pi * x / 7))
    assert isinstance(ok, bool)
    assert len(ranks) == 4


# ── wkernel.track: small-N selected-block must fall back to dense eigh ──

def test_track_small_N_dense_fallback():
    """track(modes=k) with k+1 >= N once crashed in eigsh (k must be < N); now it
    falls back to dense eigh.  The selected-block flow must (a) not crash and
    (b) agree with the modes='all' flow on the bottom-2 modes (same integrator,
    same midpoint eigenvectors), and stay near the true endpoint spectrum."""
    rng = np.random.default_rng(1)
    N = 4
    A0 = rng.standard_normal((N, N)); A0 = (A0 + A0.T) / 2
    B1 = rng.standard_normal((N, N)); B1 = (B1 + B1.T) / 2
    path = np.array([[0.0], [0.3]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lams, V = track(A0, [B1], path, steps=8, modes=2, guard=True)     # kg=3<4 ok
        lams3, _ = track(A0, [B1], path, steps=8, modes=3, guard=True)    # kg=4>=4 → dense
        lams_all, _ = track(A0, [B1], path, steps=8, modes="all")
    assert lams.shape == (2, 2)
    # selected bottom-2 flow tracks the 'all' bottom-2 flow to machine precision
    assert np.allclose(np.sort(lams[-1]), np.sort(lams_all[-1])[:2], atol=1e-9)
    # and stays close to the true endpoint spectrum (frozen-W integration error)
    truth = np.sort(np.linalg.eigvalsh(A0 + 0.3 * B1))
    assert np.allclose(np.sort(lams[-1]), truth[:2], atol=1e-3)
    assert np.allclose(np.sort(lams3[-1]), truth[:3], atol=1e-3)
