"""EPIC3 Phase 3 sockets: grad_trace (differentiable reads), the
LinearOperator/sparse/dense interop shim, generator_read_converged."""
import numpy as np
import pytest
import resona
from scipy import sparse as sp
from scipy.sparse.linalg import LinearOperator, aslinearoperator


def _spd(N=300, seed=11):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N))
    return A @ A.T / N + np.eye(N)


# ── grad_trace ────────────────────────────────────────────────────────────────
def test_grad_trace_logdet_single():
    # d/dθ log|A + θB| at θ=0  =  Tr(A⁻¹ B)
    N = 300
    A = _spd(N)
    rng = np.random.default_rng(1)
    B = rng.standard_normal((N, N)); B = (B + B.T) / 2
    truth = float(np.trace(np.linalg.solve(A, B)))
    g, se = resona.grad_trace(lambda v: A @ v, lambda v: B @ v,
                              lambda x: 1.0 / x, N, k=60, probes=32,
                              with_err=True)
    assert se > 0
    assert abs(g - truth) < 5 * se


def test_grad_trace_multi_param_and_sharpness():
    # f = x² (sharpness): d/dθ_j Tr (A+θ_j B_j)² = 2 Tr(A B_j), exact identity
    N = 200
    A = _spd(N, seed=2)
    rng = np.random.default_rng(3)
    Bs = []
    for _ in range(3):
        B = rng.standard_normal((N, N)); Bs.append((B + B.T) / 2)
    truth = np.array([2.0 * float(np.sum(A * B)) for B in Bs])  # Tr(AB), sym
    g = resona.grad_trace(lambda v: A @ v, [(lambda v, B=B: B @ v) for B in Bs],
                          lambda x: 2.0 * x, N, k=60, probes=64)
    _, se = resona.grad_trace(lambda v: A @ v,
                              [(lambda v, B=B: B @ v) for B in Bs],
                              lambda x: 2.0 * x, N, k=60, probes=64,
                              with_err=True)
    assert np.all(np.abs(g - truth) < 5 * se + 1e-9)


def test_grad_trace_accepts_operator_objects():
    N = 150
    A = _spd(N, seed=4)
    B = _spd(N, seed=5)
    g1 = resona.grad_trace(A, B, lambda x: 1.0 / x, k=50, probes=8)   # dense, no N
    g2 = resona.grad_trace(lambda v: A @ v, lambda v: B @ v,
                           lambda x: 1.0 / x, N, k=50, probes=8)
    assert g1 == g2                                # same draws, same path


# ── the interop shim ──────────────────────────────────────────────────────────
def test_of_accepts_linearoperator_sparse_dense():
    N = 250
    A = _spd(N, seed=6)
    As = sp.csr_matrix(A)
    Lo = aslinearoperator(As)
    s_callable = resona.of(lambda v: A @ v, N, k=40, probes=4)
    s_dense = resona.of(A, k=40, probes=4)                # N omitted
    s_sparse = resona.of(As, k=40, probes=4)
    s_linop = resona.of(Lo, k=40, probes=4)
    t0 = s_callable.trace("log")
    for s in (s_dense, s_sparse, s_linop):
        assert np.isclose(s.trace("log"), t0, rtol=1e-10)


def test_of_shape_contradiction_raises():
    A = _spd(100, seed=7)
    with pytest.raises(ValueError):
        resona.of(A, 99)


def test_bare_callable_still_needs_N():
    with pytest.raises(ValueError):
        resona.of(lambda v: v)


def test_apply_quadform_cloud_accept_operators():
    N = 200
    A = _spd(N, seed=8)
    Lo = aslinearoperator(sp.csr_matrix(A))
    b = np.ones(N)
    x1 = resona.apply(lambda v: A @ v, lambda l: 1.0 / l, b, k=60)
    x2 = resona.apply(Lo, lambda l: 1.0 / l, b, k=60)
    # sparse matvec sums in a different order than dense gemv — last-bit only
    assert np.allclose(x1, x2, rtol=1e-12, atol=1e-12)
    q1 = resona.quadform(lambda v: A @ v, "inv", b)
    q2 = resona.quadform(Lo, "inv", b)
    assert np.isclose(q1, q2, rtol=1e-12)
    rng = np.random.default_rng(0)
    M = rng.standard_normal((80, 80)) / 10
    c1 = resona.cloud(lambda v: M @ v, 80, k=30)
    c2 = resona.cloud(M, k=30)                            # N read off shape
    assert np.array_equal(np.sort_complex(c1.nodes), np.sort_complex(c2.nodes))


# ── generator_read_converged ──────────────────────────────────────────────────
def _be_solution(A, u0, t, n):
    """n backward-Euler steps to time t."""
    N = len(u0)
    M = np.linalg.inv(np.eye(N) + (t / n) * A)
    u = u0.copy()
    for _ in range(n):
        u = M @ u
    return u


def test_generator_read_converged():
    rng = np.random.default_rng(9)
    N = 40
    A = rng.standard_normal((N, N)); A = A @ A.T / N + 0.5 * np.eye(N)
    u0 = rng.standard_normal(N)
    from scipy.linalg import expm
    t, n = 0.5, 64
    truth = A @ A @ (expm(-t * A) @ u0)       # G = A²e^{−tA}u₀ (the BE defect law)
    P_n = _be_solution(A, u0, t, n)
    P_2n = _be_solution(A, u0, t, 2 * n)
    P_4n = _be_solution(A, u0, t, 4 * n)
    from resona.defect import generator_read, generator_read_converged
    G_plain = generator_read(P_n, P_2n, t, n)
    G_conv, rel_dev = generator_read_converged(P_n, P_2n, P_4n, t, n)
    assert 0 < rel_dev < 0.2                          # converged regime
    err_plain = np.linalg.norm(G_plain - truth) / np.linalg.norm(truth)
    err_conv = np.linalg.norm(G_conv - truth) / np.linalg.norm(truth)
    assert err_conv < err_plain                       # Richardson gains an order
    assert err_conv < 0.05


def test_generator_read_converged_refuses_cn():
    with pytest.raises(ValueError):
        from resona.defect import generator_read_converged
        z = np.zeros(4)
        generator_read_converged(z, z, z, 1.0, 8, solver="cn")
