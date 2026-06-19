"""
Tests for resona.cloud_flow — the MOVE verb for the non-Hermitian corner.

Everything is verified against DENSE ground truth (scipy/numpy.linalg.eig with
eigenvalue continuation).  Nothing is faked: where the biorthogonal formula
diverges (near an exceptional point) that divergence is itself asserted.
"""
import numpy as np
import pytest

from resona.cloud_flow import (
    biorthogonal_eigs,
    cloud_wkernel,
    cloud_track,
    phase_rigidity,
    exceptional_point,
    _build,
)


def _nonnormal_family(N=12, M=3, seed=0):
    rng = np.random.default_rng(seed)
    A0 = np.triu(rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    A0 += 0.3 * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    Bs = [rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
          for _ in range(M)]
    return A0, Bs


# ──────────────────────────────────────────────────────────────────────────
# left/right eigenpairs are genuine — two-sided residuals tiny
# ──────────────────────────────────────────────────────────────────────────
def test_biorthogonal_residuals_tiny():
    A0, Bs = _nonnormal_family()
    w0 = np.linalg.eigvals(A0)
    targets = w0[np.argsort(-w0.real)[:4]]
    info = biorthogonal_eigs(A0, Bs, targets)
    # the targeted eigenvalues are actual eigenvalues
    assert np.max(np.abs(np.sort_complex(info["vals"]) -
                         np.sort_complex(targets))) < 1e-10
    # both left and right residuals at machine precision
    assert info["res_r"].max() < 1e-10
    assert info["res_l"].max() < 1e-10


def test_left_eigvec_is_actually_left():
    """u^* A = λ u^* (left eigenvector property), verified directly."""
    A0, Bs = _nonnormal_family(seed=3)
    A = _build(A0, Bs, np.zeros(len(Bs)))
    w0 = np.linalg.eigvals(A0)
    targets = w0[:3]
    info = biorthogonal_eigs(A0, Bs, targets)
    for i in range(3):
        u = info["L"][:, i]
        lam = info["vals"][i]
        # u^H A  vs  λ u^H
        lhs = u.conj() @ A
        rhs = lam * u.conj()
        assert np.linalg.norm(lhs - rhs) < 1e-9


# ──────────────────────────────────────────────────────────────────────────
# 1. biorthogonal ∂λ/∂k  vs finite difference (the core claim)
# ──────────────────────────────────────────────────────────────────────────
def test_wkernel_vs_finite_difference_nonnormal():
    A0, Bs = _nonnormal_family()
    M = len(Bs)
    w0 = np.linalg.eigvals(A0)
    targets = w0[np.argsort(-w0.real)[:5]]
    W, info = cloud_wkernel(A0, Bs, targets, return_pairs=True)

    eps = 1e-6
    Wfd = np.empty_like(W)
    for j in range(M):
        kp = np.zeros(M); kp[j] = eps
        km = np.zeros(M); km[j] = -eps
        vp = biorthogonal_eigs(A0, Bs, info["vals"], k=kp)["vals"]
        vm = biorthogonal_eigs(A0, Bs, info["vals"], k=km)["vals"]
        Wfd[:, j] = (vp - vm) / (2 * eps)

    err = np.max(np.abs(W - Wfd))
    assert err < 1e-6, f"biorthogonal vs FD disagree: {err:.3e}"


def test_wkernel_reduces_to_hermitian():
    """For a Hermitian family, cloud_wkernel == resona.wkernel.wkernel (real)."""
    from resona.wkernel import wkernel
    rng = np.random.default_rng(1)
    N = 10
    A0 = rng.standard_normal((N, N)); A0 = A0 + A0.T
    Bs = []
    for _ in range(2):
        B = rng.standard_normal((N, N)); Bs.append(B + B.T)
    w, V = np.linalg.eigh(A0)
    targets = w[[1, 4, 7]].astype(complex)
    W = cloud_wkernel(A0, Bs, targets)
    # Hermitian reference using the matching eigenvectors
    idx = [1, 4, 7]
    Wh = wkernel(V[:, idx], Bs)
    assert np.max(np.abs(W.imag)) < 1e-9          # sensitivity is real
    assert np.max(np.abs(W.real - Wh)) < 1e-8


# ──────────────────────────────────────────────────────────────────────────
# 2. complex eigenvalue continuation vs dense eig at each point
# ──────────────────────────────────────────────────────────────────────────
def test_cloud_track_vs_dense():
    A0, Bs = _nonnormal_family(seed=2)
    M = len(Bs)
    T = 11
    path = np.zeros((T, M))
    path[:, 0] = np.linspace(0, 1.0, T)
    path[:, 1] = np.linspace(0, -0.4, T)
    w0 = np.linalg.eigvals(A0)
    mode0 = w0[np.argmax(w0.real)]

    lams = cloud_track(A0, Bs, path, modes=mode0)

    truth = np.empty(T, complex)
    cur = mode0
    for t in range(T):
        w = np.linalg.eigvals(_build(A0, Bs, path[t]))
        cur = w[np.argmin(np.abs(w - cur))]
        truth[t] = cur
    err = np.max(np.abs(lams[:, 0] - truth))
    assert err < 1e-9, f"track vs dense eig: {err:.3e}"


def test_cloud_track_multi_mode():
    A0, Bs = _nonnormal_family(seed=5)
    M = len(Bs)
    T = 7
    path = np.zeros((T, M)); path[:, 0] = np.linspace(0, 0.6, T)
    w0 = np.linalg.eigvals(A0)
    modes = w0[np.argsort(-w0.real)[:3]]
    lams, rig = cloud_track(A0, Bs, path, modes=modes, return_rigidity=True)
    assert lams.shape == (T, 3)
    assert rig.shape == (T, 3)
    # each tracked mode matches an independent dense continuation
    for q in range(3):
        cur = modes[q]; truth = np.empty(T, complex)
        for t in range(T):
            w = np.linalg.eigvals(_build(A0, Bs, path[t]))
            cur = w[np.argmin(np.abs(w - cur))]; truth[t] = cur
        assert np.max(np.abs(lams[:, q] - truth)) < 1e-9


# ──────────────────────────────────────────────────────────────────────────
# 3. exceptional point — [[0,1],[k,0]] → λ = ±√k, EP at k=0
# ──────────────────────────────────────────────────────────────────────────
def test_exceptional_point_located():
    A0 = np.array([[0., 1.], [0., 0.]], complex)
    B = np.array([[0., 0.], [1., 0.]], complex)
    pair = np.array([1.0, -1.0], complex)        # λ=±√k at k=1
    out = exceptional_point(A0, [B], bracket=(1.0, -1e-12),
                            target_pair=pair, n_scan=401)
    # EP is at k=0; located minimum-rigidity locus must be near 0
    assert abs(out["k_ep"]) < 1e-3
    # rigidity and gap both collapse toward 0 at the EP
    assert out["rig_min"] < 1e-2
    assert out["gap_min"] < 1e-2


def test_exceptional_point_sensitivity_blowup():
    """∂λ/∂k ~ 1/(2√k) diverges as k→0 — the EP feature, vs analytic."""
    A0 = np.array([[0., 1.], [0., 0.]], complex)
    B = np.array([[0., 0.], [1., 0.]], complex)
    prev = 0.0
    for ks in [1.0, 1e-2, 1e-4, 1e-6]:
        W = cloud_wkernel(A0, [B], np.array([np.sqrt(ks)]), k=np.array([ks]))
        mag = abs(W[0, 0])
        # matches the analytic Puiseux derivative ∂(√k)/∂k = 1/(2√k)
        assert abs(mag - 1 / (2 * np.sqrt(ks))) / (1 / (2 * np.sqrt(ks))) < 1e-4
        assert mag > prev                       # strictly blowing up
        prev = mag


def test_phase_rigidity_normal_vs_defective():
    """Rigidity = 1 for a normal (Hermitian) eigenvalue; small near an EP."""
    # Hermitian → rigidity exactly 1
    rng = np.random.default_rng(7)
    N = 8
    A0 = rng.standard_normal((N, N)); A0 = A0 + A0.T
    Bs = [np.eye(N)]
    w = np.linalg.eigvalsh(A0)
    info = phase_rigidity(A0, Bs, w[[0, 3, 6]].astype(complex))
    assert np.allclose(info["rigidity"], 1.0, atol=1e-8)

    # near-EP 2x2 → rigidity small
    A0e = np.array([[0., 1.], [0., 0.]], complex)
    Be = np.array([[0., 0.], [1., 0.]], complex)
    ke = 1e-4
    infoe = phase_rigidity(A0e, [Be], np.array([np.sqrt(ke)]), k=np.array([ke]))
    assert infoe["rigidity"][0] < 0.05


# ──────────────────────────────────────────────────────────────────────────
# matrix-free shift-invert path agrees with the dense path
# ──────────────────────────────────────────────────────────────────────────
def test_matrix_free_targeting_agrees():
    rng = np.random.default_rng(11)
    N = 60
    A0 = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(N)
    A0 = np.triu(A0) + 0.1 * np.tril(A0, -1)        # non-normal
    Bs = [(rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(N)
          for _ in range(2)]
    w0 = np.linalg.eigvals(A0)
    # an interior eigenvalue (largest imaginary part)
    tgt = w0[np.argmax(w0.imag)]
    Wd = cloud_wkernel(A0, Bs, np.array([tgt]))
    Wm = cloud_wkernel(A0, Bs, np.array([tgt]), matrix_free=True, sigma=tgt)
    assert np.max(np.abs(Wd - Wm)) < 1e-6
