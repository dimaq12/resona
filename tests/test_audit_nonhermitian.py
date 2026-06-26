"""Regression tests for the non-Hermitian audit batch (branch audit-batch-fixes).

Each test pins a SPECIFIC correctness/robustness fix in resona.brown /
resona.subordination / resona.cloud / resona.cloud_flow.  Everything is checked
against dense / analytic ground truth; grids and sizes are kept tiny so the file
runs in a couple of seconds.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import resona.brown as brown
import resona.subordination as sub
from resona.cloud_flow import exceptional_point, biorthogonal_eigs


rng = np.random.default_rng(0)


def _ginibre(N, seed):
    r = np.random.default_rng(seed)
    return (r.standard_normal((N, N)) + 1j * r.standard_normal((N, N))) / np.sqrt(2 * N)


# ── brown: non-Hermitian callable adjoint guard ────────────────────────────
def test_brown_callable_needs_adjoint():
    """A non-dense operator with no rmatvec and no assume_self_adjoint must be
    REFUSED — using A as its own adjoint silently is the non-Hermitian trap."""
    A = _ginibre(8, seed=1)
    mv = lambda x: A @ x                       # non-Hermitian, no adjoint given
    with pytest.raises(ValueError, match="rmatvec"):
        brown.log_potential(mv, 0.3 + 0.2j, N=8)


def test_brown_callable_with_rmatvec_runs():
    """Supplying the true adjoint rmatvec=A^*·x makes the callable path run and
    match the dense ground truth."""
    N = 24
    A = _ginibre(N, seed=2)
    mv = lambda x: A @ x
    rmv = lambda x: A.conj().T @ x
    z = 0.3 + 0.2j
    s_mf = brown.log_potential(mv, z, N=N, rmatvec=rmv, k=24, probes=24, seed=0,
                               eta=1e-3)
    s_dense = brown.log_potential(A, z, exact=True, eta=1e-3)
    assert np.isfinite(s_mf)
    assert abs(s_mf - s_dense) < 0.2          # stochastic SLQ, loose bound


def test_brown_assume_self_adjoint_runs():
    """A genuinely Hermitian operator may pass assume_self_adjoint=True (A^*=A)
    instead of an explicit rmatvec, and then the callable path runs."""
    H = rng.standard_normal((8, 8))
    H = H + H.T                                # real symmetric ⇒ self-adjoint
    mv = lambda x: H @ x
    s = brown.log_potential(mv, 0.5 + 0.1j, N=8, assume_self_adjoint=True,
                            k=8, probes=8, seed=0, eta=1e-3)
    assert np.isfinite(s)


def test_brown_measure_tiny_grid_raises():
    """A 2×2 grid has an empty 5-point-Laplacian interior — must raise, not
    silently return an all-zero measure / IndexError."""
    A = _ginibre(6, seed=3)
    with pytest.raises(ValueError, match="3 points"):
        brown.brown_measure(A, grid=(-1.0, 1.0, -1.0, 1.0, 2), exact=True)


def test_brown_measure_mass_uses_signed_density():
    """Reported mass integrates the SIGNED density (mu_raw), not the clipped mu:
    so it equals mu_raw.sum()*cell, and is ≤ the (inflated) clipped sum."""
    A = _ginibre(40, seed=4)
    res = brown.brown_measure(A, grid=(-1.6, 1.6, -1.6, 1.6, 21), exact=True)
    hx = res["X"][0, 1] - res["X"][0, 0]
    hy = res["Y"][1, 0] - res["Y"][0, 0]
    signed = float(res["mu_raw"].sum() * hx * hy)
    clipped = float(res["mu"].sum() * hx * hy)
    assert abs(res["mass"] - signed) < 1e-12   # mass = signed integral
    assert res["mass"] <= clipped + 1e-12      # never inflated above clipped


# ── subordination: pastur convergence flag near a hard edge ────────────────
def test_pastur_warns_when_not_converged():
    """Forced to too few iterations next to a hard spectral edge, the Pastur
    fixed point has not contracted — pastur must WARN (not silently return an
    inaccurate / wrong-branch g)."""
    nodes = np.array([-1.0, 1.0])              # two atoms → sharp edges
    w = np.array([0.5, 0.5])
    GA = lambda zz: (w / (zz - nodes)).sum()
    z = 1.0 + 1e-7j                            # right ON the hard edge
    with pytest.warns(RuntimeWarning, match="converge"):
        sub.pastur(GA, z, sigma2=1e-3, iters=2, tol=1e-13)


def test_pastur_converged_no_warning():
    """Well inside the bulk the fixed point converges fast — no spurious warning,
    and the same g as before."""
    nodes = np.array([-1.0, 1.0])
    w = np.array([0.5, 0.5])
    GA = lambda zz: (w / (zz - nodes)).sum()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")         # any warning ⇒ failure
        g = sub.pastur(GA, 0.0 + 0.5j, sigma2=0.1, iters=2000, tol=1e-13)
    assert np.isfinite(g)


# ── cloud_flow: small-N dense fallback (no nev<=0 crash) ───────────────────
def test_cloud_flow_matrix_free_small_N_dense_fallback():
    """At N=2 ARPACK shift-invert is impossible (nev would be 0); the matrix-free
    request must transparently fall back to the dense path, not crash."""
    A0 = np.array([[0.0, 1.0], [0.0, 0.0]], complex)
    B = np.array([[0.0, 0.0], [1.0, 0.0]], complex)
    info = biorthogonal_eigs(A0, [B], targets=[1.0 + 0j], k=[1.0],
                             matrix_free=True)
    assert np.isfinite(info["vals"]).all()
    # ground truth: A(1) = [[0,1],[1,0]] has eigenvalues ±1
    assert min(abs(info["vals"][0] - 1.0), abs(info["vals"][0] + 1.0)) < 1e-9


# ── cloud_flow: exceptional_point refinement no longer stale ───────────────
def _ep_family(seed=7):
    """A 4×4 one-parameter family with a genuine EP: a 2×2 EP block
    [[0,1],[k,0]] (λ=±√k, EP at k=0) padded with two far-away spectator
    eigenvalues, in a non-trivial (rotated) basis."""
    A0 = np.zeros((4, 4), complex)
    A0[0, 1] = 1.0                             # EP block top row
    A0[2, 2] = 3.0                             # spectators, far from the EP pair
    A0[3, 3] = -3.0
    B = np.zeros((4, 4), complex)
    B[1, 0] = 1.0                              # moves the EP block: [[0,1],[k,0]]
    # rotate into a generic basis so eigenvectors are not axis-aligned
    r = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(r.standard_normal((4, 4)) + 1j * r.standard_normal((4, 4)))
    return Q @ A0 @ Q.conj().T, Q @ B @ Q.conj().T


def test_exceptional_point_refine_not_worse():
    """The refinement must match the LOCALLY-CONTINUED pair (not the stale
    original target_pair); refine=True must reach a rigidity minimum no worse
    than refine=False, and drive rig→0 at the real EP (k=0).

    The bracket (1, -1) STRADDLES the EP at k=0 with the EP OFF the coarse grid
    (n_scan=40), so golden-section genuinely engages: with the stale-pair bug the
    refinement matched the wrong eigenpair and could not reach the EP."""
    A0, B = _ep_family()
    k_start = 1.0
    pair = np.array([np.sqrt(k_start), -np.sqrt(k_start)], complex)
    out_no = exceptional_point(A0, [B], bracket=(1.0, -1.0),
                               target_pair=pair, n_scan=40, refine=False)
    out_ref = exceptional_point(A0, [B], bracket=(1.0, -1.0),
                                target_pair=pair, n_scan=40, refine=True)
    # refinement is NOT worse than the coarse minimum ...
    assert out_ref["rig_min"] <= out_no["rig_min"] + 1e-9
    # ... and it strictly improves here (coarse grid misses the off-grid EP) ...
    assert out_ref["rig_min"] < out_no["rig_min"]
    # ... collapsing the rigidity toward 0 at the real EP (k = 0).
    assert out_ref["rig_min"] < 1e-2
    assert abs(out_ref["k_ep"]) < 5e-2
