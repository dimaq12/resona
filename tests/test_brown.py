"""Fast tests for resona.brown — the Brown measure (non-Hermitian eigenvalue law
in the complex plane) via Hermitization.  Grids are kept SMALL so the whole suite
runs in seconds: the Brown measure costs one spectral solve PER grid point, so a
fine grid is for a demo, not a unit test."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import resona.brown as brown

rng = np.random.default_rng(0)


def _disk_stats(res):
    mu = np.maximum(res["mu"], 0.0)
    X, Y = res["X"], res["Y"]
    R = np.hypot(X, Y)
    da = (X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0])
    return (mu[R < 1.0].sum() * da,            # mass inside unit disk
            mu[R > 1.10].sum() * da,           # mass well outside
            mu[R < 0.85].mean())               # interior density


def test_circular_law():
    """Ginibre ⇒ Brown measure = uniform on the unit disk (density 1/π)."""
    N = 300
    A = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2 * N)
    res = brown.brown_measure(A, grid=(-1.5, 1.5, -1.5, 1.5, 21), exact=True)
    mass_in, mass_out, dens_in = _disk_stats(res)
    assert mass_in > 0.9 and mass_out < 0.1            # mass on the disk
    assert abs(dens_in - 1 / np.pi) < 0.07             # circular-law density 1/π


def test_log_potential_matrixfree_vs_dense():
    """matrix-free SLQ Tr log((A−z)*(A−z))/2N matches the dense log-potential."""
    N = 80
    A = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2 * N)
    zs = np.array([0.0 + 0j, 0.5 + 0.3j, 1.2 - 0.2j])
    S_dense = brown.log_potential(A, zs, exact=True)
    S_mf = brown.log_potential(A, zs, k=60, probes=24, seed=1, eta=1e-3)
    assert np.max(np.abs(S_mf - S_dense)) < 0.05       # SLQ stochastic, but close


def test_normal_operator_brown_is_eigenvalues():
    """For a NORMAL operator the Brown measure sits on the eigenvalues."""
    N = 120
    d = rng.standard_normal(N) + 1j * rng.standard_normal(N)   # diagonal = normal
    A = np.diag(d)
    res = brown.brown_measure(A, grid=(-3, 3, -3, 3, 21), exact=True)
    mu = np.maximum(res["mu"], 0.0)
    X, Y = res["X"], res["Y"]
    da = (X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0])
    total = mu.sum() * da
    assert abs(total - 1.0) < 0.2                      # a probability measure
    # the mass concentrates where the eigenvalues are (in-support density > out)
    Z = X + 1j * Y
    near = np.min(np.abs(Z.ravel()[:, None] - d[None, :]), axis=1).reshape(X.shape) < 0.4
    assert mu[near].mean() > mu[~near].mean()


def test_free_sum_brown_matches_eigenvalues():
    """COMPOSE: the Brown measure of a *-free sum A⊞B matches the eig(A+B) cloud."""
    N = 200
    A = np.diag(rng.standard_normal(N)) + np.diag(np.ones(N - 1), 1)        # non-normal
    A2 = np.diag(rng.standard_normal(N)) + np.diag(np.ones(N - 1), 1)
    Qm, _ = np.linalg.qr(rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    B = Qm @ A2 @ Qm.conj().T                                              # *-free of A
    res = brown.brown_boxplus(A, B, grid=(-3.5, 3.5, -3.5, 3.5, 15), exact=True, nbins=70)
    mu = np.maximum(res["mu"], 0.0)
    X, Y = res["X"], res["Y"]
    da = (X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0])
    assert abs(mu.sum() * da - 1.0) < 0.2                                  # probability measure
    ev = np.linalg.eigvals(A + B)
    Z = (X + 1j * Y).ravel()
    supp = Z[mu.ravel() > mu.max() * 0.05]
    inside = np.mean([np.min(np.abs(e - supp)) < 0.5 for e in ev])
    assert inside > 0.85                                                   # eig cloud in support
