"""Phase D: cloud (non-Hermitian), thermal (typicality), lift.koopman."""
import numpy as np
import pytest

from resona.cloud import cloud
from resona import thermal


def test_cloud_markov_chain():
    # a reversible-ish random walk: top eigenvalue 1, gap = mixing rate
    rng = np.random.default_rng(0)
    N = 300
    P = rng.random((N, N)) ** 4
    P /= P.sum(1, keepdims=True)                  # row-stochastic
    ev = np.linalg.eigvals(P)
    c = cloud(lambda v: P.T @ v, N, k=64, probes=4)
    assert abs(c.radius() - 1.0) < 1e-8           # the stationary mode
    # second-largest modulus (the mixing gap) found within the cloud
    second_true = np.sort(np.abs(ev))[-2]
    second_cloud = np.sort(np.abs(c.nodes))[-5:]  # near-top cluster
    assert np.min(np.abs(second_cloud - second_true)) < 1e-3


def test_cloud_abscissa_lower_bound():
    rng = np.random.default_rng(1)
    N = 200
    A = -np.eye(N) + 0.4 * rng.standard_normal((N, N)) / np.sqrt(N)
    ev = np.linalg.eigvals(A)
    omega = float(np.max(np.linalg.eigvalsh((A + A.T) / 2)))   # numerical abscissa
    c = cloud(lambda v: A @ v, N, k=80, probes=4)
    # the cloud lives between the spectrum and the numerical range — and for
    # this non-normal A the two differ by ~0.6: the transient-growth gap
    assert c.abscissa() <= omega + 1e-8               # never above ω(A)
    assert c.abscissa() >= np.max(ev.real) - 1e-8     # at/above the spectrum here
    assert omega - np.max(ev.real) > 0.1              # the gap is real (the lesson)


def _chain(L, hx=1.05, hz=0.5):
    """mixed-field Ising chain, dense (test scale)."""
    sx = np.array([[0, 1], [1, 0]], float)
    sz = np.diag([1.0, -1.0])
    D = 2 ** L
    H = np.zeros((D, D))
    def op(mat, i):
        out = np.eye(1)
        for j in range(L):
            out = np.kron(out, mat if j == i else np.eye(2))
        return out
    for i in range(L - 1):
        H += op(sz, i) @ op(sz, i + 1)
    for i in range(L):
        H += hx * op(sx, i) + hz * op(sz, i)
    return H


def test_thermal_expect_vs_dense():
    L = 8
    H = _chain(L)
    D = 2 ** L
    beta = 0.7
    from scipy.linalg import expm
    rho = expm(-beta * H); Z = np.trace(rho)
    sz0 = np.kron(np.diag([1.0, -1.0]), np.eye(D // 2))
    truth = float(np.trace(sz0 @ rho) / Z)
    val, err = thermal.expect(lambda v: H @ v, lambda v: sz0 @ v, beta, D,
                              probes=12, k=48)
    assert err > 0
    assert abs(val - truth) < 5 * err + 0.02


def test_thermal_correlator_vs_dense():
    L = 6
    H = _chain(L)
    D = 2 ** L
    beta, ts = 0.5, np.array([0.0, 0.3, 0.7])
    from scipy.linalg import expm
    rho = expm(-beta * H); Z = np.trace(rho)
    sz0 = np.kron(np.diag([1.0, -1.0]), np.eye(D // 2))
    ew, V = np.linalg.eigh(H)
    Od = V.T @ sz0 @ V
    truth = []
    for t in ts:
        Ot = np.exp(1j * t * ew)[:, None] * Od * np.exp(-1j * t * ew)[None, :]
        truth.append(np.trace(Ot @ Od @ (V.T @ rho @ V)) / Z)
    # typicality error at this TINY D=64 is ~0.18/probe — the test verifies
    # correctness (no bias: 200 probes converge to 0.014), not efficiency
    got = thermal.correlator(lambda v: H @ v, lambda v: sz0 @ v, beta, ts, D,
                             probes=48, k=48)
    assert np.max(np.abs(got - np.array(truth))) < 0.08


def test_koopman_linear_system_spectrum():
    # data from a LINEAR system x_{k+1} = M x_k: koopman must recover eig(M)
    from resona.lift import koopman
    rng = np.random.default_rng(2)
    n = 12
    th = 0.7
    rot = np.eye(n) * 0.95
    rot[:2, :2] = 0.97 * np.array([[np.cos(th), -np.sin(th)],
                                   [np.sin(th), np.cos(th)]])
    X = np.zeros((n, 60))
    X[:, 0] = rng.standard_normal(n)
    for k in range(59):
        X[:, k + 1] = rot @ X[:, k]
    mv, rmv, N = koopman(X)
    c = cloud(mv, N, k=N, probes=2)
    lead = c.nodes[np.argsort(-np.abs(c.nodes))[:2]]
    target = 0.97 * np.exp(1j * th)
    assert np.min(np.abs(lead - target)) < 1e-6
    assert np.min(np.abs(lead - np.conj(target))) < 1e-6


def test_koopman_reports_rank():
    from resona.lift import koopman
    rng = np.random.default_rng(3)
    X = rng.standard_normal((5, 4))            # 4 snapshots, rank ≤ 3 pairs
    mv, rmv, N = koopman(X)
    assert N <= 3


def test_apply_complex_hermitian():
    # f(H)·v for COMPLEX Hermitian H via the real-tridiagonal Lanczos path
    rng = np.random.default_rng(7)
    N = 200
    H = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (H + H.conj().T) / np.sqrt(8 * N)
    from scipy.linalg import expm
    from resona import apply
    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    got = apply(lambda x: H @ x, lambda lam: np.exp(-0.7 * lam), v, k=64)
    exact = expm(-0.7 * H) @ v
    assert np.linalg.norm(got - exact) / np.linalg.norm(exact) < 1e-9
    # complex f on Hermitian H — quantum evolution through the SAME path
    got2 = apply(lambda x: H @ x, lambda lam: np.exp(-1j * 2.0 * lam), v, k=64)
    exact2 = expm(-2j * H) @ v
    assert np.linalg.norm(got2 - exact2) / np.linalg.norm(exact2) < 1e-9
