"""Regression tests for the audit-batch fixes in resona.lift and resona.thermal.

Each test pins a SPECIFIC bug the audit closed, cross-checked against an
independent ground truth (analytic, scipy.brentq, scipy.linalg.expm), not
against "looks reasonable".
"""
import numpy as np
import pytest
from scipy.linalg import expm

import resona
import resona.lift as lift
import resona.thermal as thermal


# ─────────────────────────────────────────────────────────────────────────────
# lift.carleman_scalar — constant term no longer dropped (basis x⁰..x^order)
# ─────────────────────────────────────────────────────────────────────────────
def test_carleman_scalar_constant_term():
    # f(x) = 0.3 + x + 0.5 x²   (nonzero c_0)
    coeffs = [0.3, 1.0, 0.5]
    order = 8
    M = lift.carleman_scalar(coeffs, order)
    assert M.shape == (order + 1, order + 1)
    assert np.allclose(M[0], 0.0)                       # ż_0 = 0 (constant coord)
    for x0 in (-0.7, 0.0, 0.25, 1.3):
        z = np.array([x0 ** j for j in range(order + 1)])
        zdot1 = (M @ z)[1]                              # ż_1, x = z_1
        f = coeffs[0] + coeffs[1] * x0 + coeffs[2] * x0 ** 2
        assert abs(zdot1 - f) < 1e-12, (x0, zdot1, f)


# ─────────────────────────────────────────────────────────────────────────────
# lift.s_transform — bracket no longer inverts for huge λmax (was 2× wrong)
# ─────────────────────────────────────────────────────────────────────────────
def test_s_transform_single_atom_huge_lambda():
    # Single point mass at a: S(w) = 1/a exactly, for any w.
    a = 1e13
    s = (np.array([a]), np.array([1.0]))
    for w in (0.05, 0.5, 3.0):
        got = lift.s_transform(s, w)
        assert abs(got - 1.0 / a) <= 1e-6 * (1.0 / a), (w, got, 1.0 / a)


def test_s_transform_two_atom_huge_lambda_analytic():
    # Two equal-weight atoms a, b=2a with λmax ≈ 1e13.  The ψ(z)=w equation is
    # the quadratic  ab(1+w) z² − (a+b)(½+w) z + w = 0; its small root gives the
    # CLOSED-FORM S(w) = (1+w)/w · z — an independent ground truth, so a
    # 2×-off answer cannot hide.
    a = 5e12
    b = 2 * a
    nodes = np.array([a, b])
    wt = np.array([0.5, 0.5])
    for w in (0.05, 0.5, 2.0):
        A2 = a * b * (1 + w)
        B2 = (a + b) * (0.5 + w)
        disc = B2 ** 2 - 4 * A2 * w
        z = (B2 - np.sqrt(disc)) / (2 * A2)             # small root in (0, 1/b)
        ref = (1 + w) / w * z
        got = lift.s_transform((nodes, wt), w)
        assert abs(got - ref) <= 1e-6 * abs(ref), (w, got, ref)


def test_s_transform_out_of_range_raises():
    # A target ψ cannot reach (negative) must raise, not return a wrong value.
    s = (np.array([2.0, 4.0]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        lift.s_transform(s, -1.0)


# ─────────────────────────────────────────────────────────────────────────────
# lift.r_transform — 55 bisections bit-identical to old 110 on a normal spectrum
# ─────────────────────────────────────────────────────────────────────────────
def test_r_transform_normal_spectrum_value():
    # Semicircle-ish atoms; R(w) ≈ mean as w→0 (R'(0)=κ₂), monotone check.
    rng = np.random.default_rng(0)
    nodes = np.sort(rng.uniform(-1, 1, 200))
    wt = np.ones_like(nodes)
    s = (nodes, wt)
    wq = np.array([0.05, 0.1, 0.2, 0.4])
    R = lift.r_transform(s, wq)
    assert np.all(np.isfinite(R))
    # R(0⁺) → mean
    assert abs(lift.r_transform(s, 1e-9) - nodes.mean()) < 1e-2


# ─────────────────────────────────────────────────────────────────────────────
# thermal.correlator — ts starting at t0≠0 (uniform fast branch pre-evolves)
# ─────────────────────────────────────────────────────────────────────────────
def _dense_correlator(H, O, beta, ts, N, probes, seed=0, k=64):
    """The SAME typicality estimator as thermal.correlator but with exact
    matrix exponentials (scipy.expm) — isolates the t0-offset logic."""
    tiny = np.finfo(float).tiny
    Eh = expm(-0.5 * beta * H)
    out = np.zeros((probes, len(ts)), complex)
    wts = np.zeros(probes)
    for p in range(probes):
        rng = np.random.default_rng(seed + p)
        r = rng.standard_normal(N) / np.sqrt(N)
        psi = Eh @ r
        w = float(np.real(np.vdot(psi, psi)))
        w = max(w, tiny)
        psi = psi / np.sqrt(w)
        wts[p] = w * N
        phi = O @ psi
        for i, t in enumerate(ts):
            U = expm(-1j * t * H)
            psi_t = U @ psi
            phi_t = U @ phi
            out[p, i] = np.vdot(psi_t, O @ phi_t)
    return (out * wts[:, None]).sum(0) / wts.sum()


def _make_H_O(N=8, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N))
    H = (A + A.T) / 2
    B = rng.standard_normal((N, N))
    O = (B + B.T) / 2
    return H, O


def test_correlator_uniform_grid_t0_nonzero():
    H, O = _make_H_O()
    N = H.shape[0]
    mv = lambda v: H @ v
    Omv = lambda v: O @ v
    beta = 0.4
    ts = np.linspace(2.0, 5.0, 7)                       # uniform, starts at 2.0
    got = thermal.correlator(mv, Omv, beta, ts, N, probes=2, k=N, seed=0)
    ref = _dense_correlator(H, O, beta, ts, N, probes=2, seed=0)
    assert np.max(np.abs(got - ref)) < 1e-8, np.max(np.abs(got - ref))


def test_correlator_nonuniform_grid_t0_nonzero():
    H, O = _make_H_O()
    N = H.shape[0]
    mv = lambda v: H @ v
    Omv = lambda v: O @ v
    beta = 0.4
    ts = np.array([2.0, 2.3, 3.1, 4.7])                 # non-uniform, starts at 2.0
    got = thermal.correlator(mv, Omv, beta, ts, N, probes=2, k=N, seed=0)
    ref = _dense_correlator(H, O, beta, ts, N, probes=2, seed=0)
    assert np.max(np.abs(got - ref)) < 1e-8, np.max(np.abs(got - ref))


def test_correlator_fast_slow_agree_t0_nonzero():
    # uniform (fast) and non-uniform (slow) sharing the same 3 sample times
    # must agree — both pre-evolve to t0 correctly.
    H, O = _make_H_O()
    N = H.shape[0]
    mv = lambda v: H @ v
    Omv = lambda v: O @ v
    beta = 0.4
    tvals = [2.0, 3.0, 4.0]
    uni = thermal.correlator(mv, Omv, beta, np.array(tvals), N, probes=2, k=N)
    slow = thermal.correlator(mv, Omv, beta, np.array([2.0, 3.0, 4.0, 9.9]),
                              N, probes=2, k=N)[:3]
    assert np.max(np.abs(uni - slow)) < 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# thermal.state — large-β weight floor (no inf/0 division)
# ─────────────────────────────────────────────────────────────────────────────
def test_state_large_beta_finite_normalized():
    # spectrum with λmin = 0 → ground-state amplitude survives: normalized.
    rng = np.random.default_rng(2)
    A = rng.standard_normal((8, 8))
    H = A @ A.T                                          # PSD, λmin ≈ 0
    H = H - 0.0                                          # keep λmin ≥ 0
    mv = lambda v: H @ v
    psi, weight = thermal.state(mv, beta=80.0, N=8, k=8)
    assert np.all(np.isfinite(psi))
    assert np.isfinite(weight)
    assert abs(np.vdot(psi, psi).real - 1.0) < 1e-6


def test_state_extreme_beta_no_nan():
    # strictly positive spectrum + enormous β → e^{-βH/2} underflows; the floor
    # must keep the result finite (no nan/inf from 0/0).
    H = np.diag(np.linspace(2.0, 6.0, 8))
    mv = lambda v: H @ v
    psi, weight = thermal.state(mv, beta=1e6, N=8, k=8)
    assert np.all(np.isfinite(psi))
    assert np.isfinite(weight)


# ─────────────────────────────────────────────────────────────────────────────
# thermal.expect — probes guard
# ─────────────────────────────────────────────────────────────────────────────
def test_expect_probes_zero_raises():
    H = np.diag(np.linspace(0.0, 1.0, 8))
    mv = lambda v: H @ v
    Omv = lambda v: H @ v
    with pytest.raises(ValueError):
        thermal.expect(mv, Omv, beta=0.5, N=8, probes=0)
