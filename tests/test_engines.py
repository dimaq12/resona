"""Phase B engines: of(deflate=K) and of(engine='kpm')."""
import numpy as np
import pytest

from resona.spectral import Spectral


def _spiked(N, K=8, seed=0, shift=1.0):
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((N, K)))[0]
    sp = np.linspace(40, 8, K)
    B = rng.standard_normal((N, N))
    return (U * sp) @ U.T + B @ B.T / N + shift * np.eye(N), sp


def test_deflate_variance_drop_on_top_weighted():
    N = 800
    A, _ = _spiked(N)
    mv = lambda v: A @ v
    truth = float(np.sum(np.linalg.eigvalsh(A) ** 2))
    plain, defl = [], []
    for seed in range(6):
        plain.append(Spectral.of(mv, N, k=32, probes=6, seed=seed).moment(2))
        defl.append(Spectral.of(mv, N, k=32, probes=6, seed=seed,
                                deflate=8).moment(2))
    plain, defl = np.array(plain), np.array(defl)
    assert defl.std() < plain.std() / 5            # ≥5× variance drop
    assert abs(defl.mean() - truth) / truth < 0.02


def test_deflate_atoms_are_top_eigenvalues():
    N = 600
    A, _ = _spiked(N)
    ev = np.sort(np.linalg.eigvalsh(A))
    s = Spectral.of(lambda v: A @ v, N, k=32, probes=4, deflate=8)
    atoms = np.sort(s.nodes[-8:])
    assert np.max(np.abs(atoms - ev[-8:])) < 1e-6  # Ritz-exact top-K
    assert np.allclose(s.weights[-8:], 1.0 / N)


def test_deflate_with_err_shrinks():
    N = 800
    A, _ = _spiked(N)
    mv = lambda v: A @ v
    _, se_plain = Spectral.of(mv, N, k=32, probes=8).moment(2, with_err=True)
    v, se_defl = Spectral.of(mv, N, k=32, probes=8, deflate=8).moment(2, with_err=True)
    assert se_defl < se_plain / 3
    truth = float(np.sum(np.linalg.eigvalsh(A) ** 2))
    assert abs(v - truth) < 6 * max(se_defl, 1e-9) + 0.01 * truth


def test_deflate_certified_bracket():
    N = 600
    A, _ = _spiked(N, shift=1.0)
    mv = lambda v: A @ v
    s_lo = Spectral.of(mv, N, k=12, probes=6, deflate=8)
    s_hi = Spectral.of(mv, N, k=64, probes=6, deflate=8)
    lo, hi = s_lo.trace_certified("log", support=(0.5, None))
    converged = s_hi.trace(np.log)
    assert lo <= converged <= hi


def test_kpm_same_object_contract():
    # moments of the kpm Spectral must match the lanczos engine's within the
    # engines' joint tolerance, across operator types
    rng = np.random.default_rng(3)
    N = 1200
    ops = []
    B = rng.standard_normal((N, N)); ops.append(B @ B.T / N + 0.2 * np.eye(N))
    d = rng.uniform(0.5, 3.0, N); e = 0.3 * rng.standard_normal(N - 1)
    ops.append(np.diag(d) + np.diag(e, 1) + np.diag(e, -1))
    U = rng.standard_normal((N, 6)); ops.append(U @ U.T / 6 + 0.5 * np.eye(N))
    for A in ops:
        ev = np.linalg.eigvalsh(A)
        s_k = Spectral.of(lambda v: A @ v, N, k=64, probes=8, engine="kpm")
        for p in (1, 2):
            tr = float(np.sum(ev ** p))
            assert abs(s_k.moment(p) - tr) / abs(tr) < 0.08, (p, s_k.moment(p), tr)


def test_kpm_density_matches_truth():
    rng = np.random.default_rng(4)
    N = 1500
    B = rng.standard_normal((N, N)); A = B @ B.T / N
    ev = np.linalg.eigvalsh(A)
    s = Spectral.of(lambda v: A @ v, N, k=96, probes=8, engine="kpm")
    xs = np.linspace(0.05, 3.9, 200)
    rho = s.density(xs, eta=0.05)
    hist = np.histogram(ev, bins=200, range=(0.05, 3.9))[0].astype(float)
    assert np.corrcoef(rho, hist)[0, 1] > 0.97


def test_engine_guards():
    N = 200
    rng = np.random.default_rng(5)
    H = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (H + H.conj().T) / 2
    with pytest.raises(ValueError):
        Spectral.of(lambda v: H @ v, N, engine="kpm")       # complex → refuse
    A = np.eye(N)
    with pytest.raises(ValueError):
        Spectral.of(lambda v: A @ v, N, engine="kpm", deflate=4)
    with pytest.raises(ValueError):
        Spectral.of(lambda v: A @ v, N, engine="fft")
    s = Spectral.of(lambda v: A @ v, N, k=8, probes=2, engine="kpm")
    with pytest.raises(ValueError):
        s.trace_certified("log", support=(0.5, None))  # no theorems on KPM
