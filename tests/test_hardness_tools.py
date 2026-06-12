"""Tools ported from the spectral_hardness program: pseudospectrum (the ε^{1/q}
bloom law), conserved-charge search, level-spacing ratio."""
import numpy as np
from itertools import product

from resona.defect import sigma_min, pseudospectrum_radius, pseudospectrum
from resona.lift import conserved_charge
from resona.cost import level_spacing_ratio


def _jordan(q):
    J = np.zeros((q, q))
    for i in range(q - 1):
        J[i, i + 1] = 1.0
    return J


def test_pseudospectrum_bloom_law_dense():
    # an order-q Jordan defect blooms {0} into a disk of radius eps^{1/q}
    eps = 1e-6
    for q in (2, 3, 4, 5):
        rad = pseudospectrum_radius(_jordan(q), eps)
        pred = eps ** (1.0 / q)
        assert abs(rad - pred) / pred < 0.1, (q, rad, pred)


def test_pseudospectrum_normal_no_bloom():
    A = np.diag([0.0, 1.0, -2.0, 3.5])
    eps = 1e-6
    rad = pseudospectrum_radius(A, eps, z0=0.0, r_max=0.4)
    assert rad < 3 * eps                       # trivial ε-fattening only


def test_pseudospectrum_matrix_free():
    q, eps = 3, 1e-6
    J = _jordan(q)
    pad = np.zeros((40, 40)); pad[:q, :q] = J  # embed the defect in a bigger op
    pad += np.diag(np.linspace(2.0, 3.0, 40))  # well-separated normal bulk
    pad[:q, :q] = J                            # keep the defect block exact
    mv = lambda x: pad @ x
    rmv = lambda x: pad.T @ x
    rad = pseudospectrum_radius(mv, eps, z0=0.0, N=40, rmatvec=rmv, r_max=0.5)
    pred = eps ** (1.0 / q)
    assert abs(rad - pred) / pred < 0.15, (rad, pred)


def test_pseudospectrum_grid_mask():
    J = _jordan(3)
    zs = np.array([0.0, 0.5e-2 + 0j, 0.5])     # inside, inside, outside the bloom
    mask = pseudospectrum(J, zs, eps=1e-6)
    assert mask.tolist() == [True, True, False]


def test_sigma_min_matches_svd():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((30, 30))
    z = 0.3 + 0.2j
    exact = sigma_min(A, z)
    est = sigma_min(lambda x: A @ x, z, N=30, rmatvec=lambda x: A.T @ x, k=60)
    assert abs(est - exact) / exact < 1e-6


# ── conserved charges ──────────────────────────────────────────────────────────
_P = {0: np.eye(2, dtype=complex),
      1: np.array([[0, 1], [1, 0]], complex),
      2: np.array([[0, -1j], [1j, 0]], complex),
      3: np.array([[1, 0], [0, -1]], complex)}


def _string(types):
    m = np.array([[1.0 + 0j]])
    for t in types:
        m = np.kron(m, _P[t])
    return m


def _klocal_basis(L, width=2):
    ops = []
    for types in product(range(4), repeat=L):
        sup = [i for i, t in enumerate(types) if t]
        if not sup or max(sup) - min(sup) >= width:
            continue
        ops.append(_string(types))
    return ops


def test_conserved_charge_integrable_vs_chaotic():
    L = 5
    two = lambda a, t: _string(tuple(t if k in (a, a + 1) else 0 for k in range(L)))
    one = lambda a, t: _string(tuple(t if k == a else 0 for k in range(L)))
    H_xx = sum(two(i, 1) + two(i, 2) for i in range(L - 1))            # integrable
    H_ch = (sum(two(i, 3) for i in range(L - 1))                        # chaotic
            + 1.05 * sum(one(i, 1) for i in range(L))
            + 0.5 * sum(one(i, 3) for i in range(L)))
    basis = _klocal_basis(L, width=2)
    _, n_xx = conserved_charge(H_xx, basis)
    _, n_ch = conserved_charge(H_ch, basis)
    found_xx = int(np.sum(n_xx < 1e-6))
    found_ch = int(np.sum(n_ch < 1e-6))
    assert found_xx >= 2                       # finds H itself AND total Z (blind)
    assert found_ch >= 1                       # energy is always conserved
    assert found_xx > found_ch                 # the integrable side has MORE charges


def test_conserved_charge_finds_total_z():
    L = 4
    two = lambda a, t: _string(tuple(t if k in (a, a + 1) else 0 for k in range(L)))
    H_xx = sum(two(i, 1) + two(i, 2) for i in range(L - 1))
    basis = _klocal_basis(L, width=2)
    charges, norms = conserved_charge(H_xx, basis)
    Z_tot = sum(_string(tuple(3 if k == a else 0 for k in range(L))) for a in range(L))
    Z_tot /= np.linalg.norm(Z_tot)
    conserved = [Q for Q, n in zip(charges, norms) if n < 1e-6]
    overlap = max(abs(np.vdot(Q.ravel(), Z_tot.ravel())) for Q in conserved)
    # total Z lives in the conserved span (up to mixing among the zero modes)
    proj = sum(abs(np.vdot(Q.ravel(), Z_tot.ravel())) ** 2 for Q in conserved)
    assert proj > 0.99, (overlap, proj)


# ── level spacing ratio ────────────────────────────────────────────────────────
def test_level_spacing_poisson_vs_goe():
    rng = np.random.default_rng(0)
    poisson = np.cumsum(rng.exponential(size=8000))        # uncorrelated levels
    r_p = level_spacing_ratio(poisson)
    A = rng.standard_normal((2000, 2000)); A = (A + A.T) / 2
    r_g = level_spacing_ratio(np.linalg.eigvalsh(A))
    assert abs(r_p - 0.386) < 0.02
    assert abs(r_g - 0.531) < 0.02
    assert r_g > r_p + 0.1                                  # the detector separates
