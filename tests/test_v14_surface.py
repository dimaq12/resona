"""1.4 surface: epistemic parity (bars everywhere), the trace split, the
renames, and the R/S duals.  Every new read is checked against ground truth
AND against the bit-frozen default path (additive only — the ratchet's rule)."""
import numpy as np
import pytest
import resona
from resona import Spectral


def _spd(N=400, seed=3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N))
    return A @ A.T / N + np.eye(N)


# ── the trace split ───────────────────────────────────────────────────────────
def test_trace_certified_bracket():
    A = _spd()
    s = Spectral.of(lambda v: A @ v, A.shape[0], k=48, probes=8)
    lo, hi = s.trace_certified("log", support=(1e-6, None))
    assert lo <= hi
    truth = np.linalg.slogdet(A)[1]
    # the bracket certifies k-truncation of THESE probes; it should sit near
    # the plain trace read, and be ordered
    assert abs(0.5 * (lo + hi) - s.trace("log")) < 1e-6 * max(1.0, abs(truth))


# ── error-bar parity: density ─────────────────────────────────────────────────
def test_density_with_err():
    A = _spd()
    s = Spectral.of(lambda v: A @ v, A.shape[0], k=48, probes=16)
    xs = np.linspace(0.5, 4.0, 60)
    rho_default = s.density(xs)
    rho, err = s.density(xs, with_err=True)
    assert np.array_equal(rho, rho_default)          # default path bit-frozen
    assert err.shape == rho.shape and np.all(err >= 0) and err.max() > 0
    ev = np.linalg.eigvalsh(A)
    eta = 0.1
    truth = (eta / np.pi) / ((xs[:, None] - ev[None, :]) ** 2 + eta ** 2)
    truth = truth.sum(1) / len(ev)
    # the bar brackets the truth at (nearly) every point
    assert np.mean(np.abs(rho - truth) < 5 * err + 1e-12) > 0.9


def test_density_with_err_deflate_atoms_exact():
    A = _spd()
    s = Spectral.of(lambda v: A @ v, A.shape[0], k=48, probes=8, deflate=10)
    xs = np.linspace(0.5, 4.0, 20)
    rho, err = s.density(xs, with_err=True)
    assert np.all(np.isfinite(err))


# ── error-bar parity: extreme ─────────────────────────────────────────────────
def test_extreme_with_err():
    A = _spd()
    s = Spectral.of(lambda v: A @ v, A.shape[0], k=48, probes=16)
    (lo, hi), (lo_e, hi_e) = s.extreme(with_err=True)
    assert (lo, hi) == s.extreme()                   # default unchanged
    assert lo_e >= 0 and hi_e >= 0
    ev = np.linalg.eigvalsh(A)
    assert abs(hi - ev[-1]) < 10 * hi_e + 1e-6       # top is Lanczos-sharp


# ── error-bar parity: cumulants ───────────────────────────────────────────────
def test_cumulants_with_err():
    A = _spd()
    s = Spectral.of(lambda v: A @ v, A.shape[0], k=48, probes=16)
    k_default = s.cumulants(4)
    kappa, err = s.cumulants(4, with_err=True)
    assert np.allclose(kappa, k_default)
    assert len(err) == len(kappa) and np.all(err >= 0) and err.max() > 0
    ev = np.linalg.eigvalsh(A)
    m1 = ev.mean()
    assert abs(kappa[0] - m1) < 5 * err[0] + 1e-12   # κ₁ = mean


# ── error-bar parity: kappa_w full distribution ───────────────────────────────
def test_kappa_w_full():
    rng = np.random.default_rng(0)
    A0 = rng.standard_normal((30, 30)); A0 = (A0 + A0.T) / 2
    Bs = []
    for _ in range(3):
        B = rng.standard_normal((30, 30)); Bs.append((B + B.T) / 2)
    k0 = np.zeros(3)
    kmax = resona.wkernel.kappa_w(A0, Bs, k0)
    kfull, vals = resona.wkernel.kappa_w(A0, Bs, k0, full=True)
    assert kfull == kmax == vals.max()
    assert len(vals) == 8


# ── renames / aliases ─────────────────────────────────────────────────────────
def test_defect_barycentres():
    power = np.array([0.0, 4.0, 0.0, 0.0, 1.0, 1.0])
    bands = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    k1, s1 = resona.defect.defect_barycentres(power, bands)
    assert k1[0] == 1.0                              # all energy at index 1
    assert s1[0] == 2.0
    # 2.0: the legacy names are GONE
    assert not hasattr(resona.defect, "spectroscopy")
    assert not hasattr(resona.cost, "phi1")


def test_synthesize_is_from_measure():
    nodes = np.array([1.0, 2.0, 3.0])
    w = np.array([0.2, 0.3, 0.5])
    a1, b1 = resona.synthesize(nodes, w)
    a2, b2 = resona.from_measure(nodes, w)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)


# ── the R/S duals ─────────────────────────────────────────────────────────────
def test_r_inverse_roundtrip():
    A = _spd(seed=5)
    s = Spectral.of(lambda v: A @ v, A.shape[0], k=48, probes=8)
    w0 = 0.31
    val = resona.lift.r_transform(s, w0)
    w_back = resona.lift.r_inverse(s, val)
    assert abs(w_back - w0) < 1e-10
    # vectorized
    ws = np.array([0.1, 0.2, 0.4])
    back = resona.lift.r_inverse(s, resona.lift.r_transform(s, ws))
    assert np.allclose(back, ws, atol=1e-10)


def test_s_inverse_roundtrip():
    A = _spd(seed=6)
    s = Spectral.of(lambda v: A @ v, A.shape[0], k=48, probes=8)
    w0 = 0.25
    val = resona.lift.s_transform(s, w0)
    w_back = resona.lift.s_inverse(s, val)
    assert abs(w_back - w0) < 1e-10


def test_r_inverse_out_of_range_raises():
    A = _spd(seed=7)
    s = Spectral.of(lambda v: A @ v, A.shape[0], k=48, probes=8)
    with pytest.raises(ValueError):
        resona.lift.r_inverse(s, 1e9)


def test_version_matches_pyproject():
    # the REAL invariant (the one the sed/ENOSPC incident violated): the
    # package version and pyproject must agree — never a hardcoded literal
    import pathlib, re
    toml = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    v = re.search(r'version = "([^"]+)"', toml.read_text()).group(1)
    assert resona.__version__ == v
