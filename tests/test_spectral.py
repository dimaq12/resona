"""Tests for resona.Spectral — verified against dense ground truth (fixed seeds)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy import linalg
from resona import Spectral, apply

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
    # Φ₁ is defined for PSD operators (covariance / kernel / Hessian).
    N = 400
    U = rng.standard_normal((N, 6)); low = U @ U.T                  # PSD, rank 6
    B = rng.standard_normal((N, N)); full = (B @ B.T) / N           # PSD, full rank
    s_low = Spectral.of(lambda v: low @ v, N, k=60, probes=16)
    s_full = Spectral.of(lambda v: full @ v, N, k=60, probes=16)
    assert s_low.effective_rank() < 0.1 * N                         # ≈ 6
    assert s_full.effective_rank() > 5 * s_low.effective_rank()     # ≈ N/2


def test_apply_matrix_function():
    # exp(A)·v via resona.apply vs dense expm — the PDE-evolution primitive.
    N = 150
    A = _sym(N) / np.sqrt(N)                              # tame spectrum
    v = rng.standard_normal(N)
    got = apply(lambda x: A @ x, lambda lam: np.exp(lam), v, k=80)
    true = linalg.expm(A) @ v
    assert np.max(np.abs(got - true)) < 1e-7 * np.linalg.norm(true)


def test_apply_general_nonsymmetric_and_complex():
    # The general (Arnoldi) path: non-symmetric exp(tA)·v, and complex exp(-itH)·ψ.
    N = 120
    A = rng.standard_normal((N, N)) / np.sqrt(N)          # non-symmetric
    v = rng.standard_normal(N)
    got = apply(lambda x: A @ x, lambda l: np.exp(0.5 * l), v, k=60, hermitian=False)
    true = linalg.expm(0.5 * A) @ v
    assert np.max(np.abs(got.real - true)) < 1e-9 * np.linalg.norm(true)

    H = _sym(N) / np.sqrt(N)                               # Hermitian, complex f
    psi = rng.standard_normal(N) + 0j
    got = apply(lambda x: H @ x, lambda l: np.exp(-1j * 0.5 * l), psi, k=80, hermitian=False)
    true = linalg.expm(-1j * 0.5 * H) @ psi
    assert np.max(np.abs(got - true)) < 1e-9 * np.linalg.norm(true)
    assert abs(np.linalg.norm(got) - np.linalg.norm(psi)) < 1e-9 * np.linalg.norm(psi)  # unitary


def test_local_density_ldos():
    # LDOS at site i of a diagonal operator is a Lorentzian at d_i, mass 1.
    N = 200; d = rng.standard_normal(N); A = np.diag(d)
    e = np.zeros(N); e[3] = 1.0
    xs = np.linspace(d.min() - 1, d.max() + 1, 2000)
    from resona import local_density, local_spectrum
    r = local_density(lambda v: A @ v, e, xs, k=120, eta=5e-3)
    assert abs(np.trapezoid(r, xs) - 1.0) < 0.02            # probability measure
    assert abs(xs[np.argmax(r)] - d[3]) < 0.05             # peaks at the site's level
    nodes, w = local_spectrum(lambda v: A @ v, e, k=120)
    assert abs(w.sum() - 1.0) < 1e-9                        # weights sum to ‖e‖²=1


def test_from_measure_inverts_of():
    # from_measure ∘ (eigenvalues, e0-weights) recovers a well-conditioned Jacobi matrix.
    N = 25; d = rng.standard_normal(N); e = rng.uniform(0.5, 1.0, N - 1)
    A = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    lam, V = np.linalg.eigh(A); w = V[0, :] ** 2
    from resona import from_measure
    al, be = from_measure(lam, w)
    assert np.max(np.abs(al - d)) < 1e-7            # diagonal
    assert np.max(np.abs(be - e)) < 1e-7            # |off-diagonal| (positive gauge)


def test_from_eigenbasis_exact():
    # full eigenbasis reconstructs the tridiagonal band of V diag(lam) V^T exactly,
    # for ANY operator (sharp or smooth) — machine precision.
    N = 40; d = rng.standard_normal(N); e = rng.uniform(0.3, 1.0, N - 1)
    A = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    lam, V = np.linalg.eigh(A)
    from resona import from_eigenbasis
    diag_h, off_h = from_eigenbasis(lam, V)
    assert np.max(np.abs(diag_h - d)) < 1e-12
    assert np.max(np.abs(off_h - e)) < 1e-12


def test_block_probe_matches_sequential():
    # the BLAS-3 block path must reproduce the single-vector path: same probe
    # vectors, same Lanczos math — nodes/weights equal to rounding
    N = 1200                                                         # above the block threshold
    rng = np.random.default_rng(7)
    A = rng.standard_normal((N, N)); A = (A + A.T) / np.sqrt(2 * N)
    s_blk = Spectral.of(lambda x: A @ x, N, k=32, probes=6)          # block-capable
    s_seq = Spectral.of(lambda x: A @ np.asarray(x).reshape(N), N,   # force 1-D only
                        k=32, probes=6)
    assert np.max(np.abs(np.sort(s_blk.nodes) - np.sort(s_seq.nodes))) < 1e-10
    assert np.max(np.abs(np.sort(s_blk.weights) - np.sort(s_seq.weights))) < 1e-12


def test_hub_methods_delegate():
    # the Spectral hub: every lifted/derived quantity reads off the same object
    N = 400
    rng = np.random.default_rng(11)
    A = rng.standard_normal((N, N)); A = A @ A.T / N
    s = Spectral.of(lambda x: A @ x, N)
    from resona import lift, beta
    assert s.cauchy(9.0) == lift.cauchy(s, 9.0)
    assert s.r(0.2) == lift.r_transform(s, 0.2)
    assert s.s(0.5) == lift.s_transform(s, 0.5)
    assert np.allclose(s.levels(N), beta.beta_from(s, N))
    lo, hi = s.extreme()
    assert s.condition() == hi / lo
    k = s.cumulants(3)
    assert abs(k[0] - s.moment(1) / N) < 1e-12


def test_transform_full_name_aliases():
    assert Spectral.r_transform is Spectral.r
    assert Spectral.s_transform is Spectral.s


def test_boxplus_as_spectral_gauss():
    # semicircle ⊞ semicircle = semicircle(2σ²): the quadrature Spectral must
    # reproduce the free-convolved moments EXACTLY (that is its contract)
    N = 2000
    rng = np.random.default_rng(3)
    A = rng.standard_normal((N, N)); A = (A + A.T) / np.sqrt(2 * N)
    B = rng.standard_normal((N, N)); B = (B + B.T) / np.sqrt(2 * N)
    sA = Spectral.of(lambda x: A @ x, N)
    sB = Spectral.of(lambda x: B @ x, N)
    m = sA.boxplus(sB, order=6)
    q = sA.boxplus(sB, order=6, as_spectral=True)
    assert isinstance(q, Spectral)
    # an n=3-point Gauss quadrature reproduces m_1..m_{2n-1}=m_5 exactly
    for p in range(1, 6):
        mq = float(np.sum(q.weights * q.nodes ** p))
        assert abs(mq - m[p - 1]) < 1e-8 * max(1.0, abs(m[p - 1]))
    # m2 of the free sum ≈ m2(A) + m2(B) ≈ 2 (semicircles of variance 1)
    assert abs(m[1] - 2.0) < 0.1


def test_condition_polish_more_honest():
    # ill-conditioned PSD: plain condition() under-reads; polish must not shrink it
    N = 1500
    rng = np.random.default_rng(5)
    U = rng.standard_normal((N, N)); U = np.linalg.qr(U)[0]
    lam = np.concatenate([[1e-6], np.linspace(0.5, 2.0, N - 1)])
    A = (U * lam) @ U.T
    s = Spectral.of(lambda x: A @ x, N, k=64, probes=8)
    k_plain = s.condition()
    k_pol = s.condition(polish=True)
    assert k_pol >= k_plain * 0.999
    assert k_pol > 1e5            # polish actually finds the hidden small edge
