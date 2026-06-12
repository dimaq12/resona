"""Certified Gauss–Radau brackets: trace(certified=True) and quadform."""
import numpy as np
import pytest

from resona.spectral import Spectral, quadform


def _psd(N, shift=0.5, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N))
    A = A @ A.T / N + shift * np.eye(N)
    return A


def test_quadform_certified_log_brackets_truth():
    # vᵀlog(A)v: NO stochastics → the bracket is a TRUE certificate
    N = 500
    A = _psd(N)
    from scipy.linalg import logm
    LA = np.real(logm(A))
    rng = np.random.default_rng(1)
    for k in (8, 16, 32):
        v = rng.standard_normal(N)
        exact = float(v @ LA @ v)
        lo, hi = quadform(lambda x: A @ x, "log", v, k=k, certified=True, support=(0.45, None))
        # containment up to the dense REFERENCE's own rounding (logm ~1e-12 rel):
        # at k=32 the bracket is TIGHTER than the reference can resolve
        tol = 1e-11 * abs(exact)
        assert lo - tol <= exact <= hi + tol, (k, lo, exact, hi)
        if k == 8:
            assert lo <= exact <= hi               # wide bracket: strict containment
        if k >= 16:
            assert hi - lo < 1e-6 * abs(exact)     # and it is TIGHT


def test_quadform_certified_inv():
    # vᵀA⁻¹v — GP posterior variance shape; f=1/x flips the bracket sides
    N = 400
    A = _psd(N)
    Ainv = np.linalg.inv(A)
    rng = np.random.default_rng(2)
    v = rng.standard_normal(N)
    exact = float(v @ Ainv @ v)
    lo, hi = quadform(lambda x: A @ x, "inv", v, k=24, certified=True, support=(0.45, None))
    assert lo <= exact <= hi
    assert hi - lo < 1e-8 * abs(exact)


def test_quadform_value_matches_apply_family():
    # certified=False path: plain Gauss value equals the local-spectrum read
    N = 300
    A = _psd(N)
    rng = np.random.default_rng(3)
    v = rng.standard_normal(N)
    val = quadform(lambda x: A @ x, np.exp, v, k=32)
    from scipy.linalg import expm
    exact = float(v @ expm(A) @ v)
    assert abs(val - exact) / abs(exact) < 1e-8


def test_trace_certified_brackets_quadrature():
    # the bracket certifies the k-truncation of the SLQ estimate:
    # the same-probes high-k estimate must lie inside the low-k bracket
    N = 600
    A = _psd(N)
    mv = lambda x: A @ x
    s_lo = Spectral.of(mv, N, k=12, probes=8)
    s_hi = Spectral.of(mv, N, k=64, probes=8)      # same seed → same probes
    lo, hi = s_lo.trace("log", certified=True, support=(0.45, None))
    converged = s_hi.trace(np.log)                 # quadrature error ~1e-13
    assert lo <= converged <= hi
    # the bracket does NOT certify Monte-Carlo scatter — documented contract;
    # dense truth may legitimately sit outside when probes are few


def test_trace_certified_needs_known_family():
    N = 100
    A = _psd(N)
    s = Spectral.of(lambda x: A @ x, N, k=8, probes=2)
    with pytest.raises(ValueError):
        s.trace(np.log, certified=True)            # callable: signs unknown
    with pytest.raises(ValueError):
        s.trace("log", certified=True, support=(2.0, None))  # a inside spectrum? no:
        # support left endpoint must be positive and below spectrum — here it
        # is ABOVE λmin≈0.5, the certificate must refuse


def test_certified_string_f_matches_callable():
    N = 200
    A = _psd(N)
    s = Spectral.of(lambda x: A @ x, N, k=16, probes=4)
    assert abs(s.trace("log") - s.trace(np.log)) < 1e-12


def test_zoom_resolves_interior():
    # diagonal operator: zoom must resolve INTERIOR eigenvalues that plain
    # k=24 SLQ blurs, and estimate the window count
    N = 2000
    rng = np.random.default_rng(5)
    d = np.sort(rng.uniform(0, 10, N))
    mv = lambda v: d * v
    s = Spectral.of(mv, N, k=24, probes=4)
    a, b = 4.0, 4.5
    z = s.zoom(a, b, k=32, probes=8, degree=300)
    inside = (z.nodes >= a) & (z.nodes <= b)
    assert z.weights[inside].sum() / z.weights.sum() > 0.85   # mass concentrated
    # count estimate: N * window mass ≈ true count (within ~12%)
    count_true = int(np.sum((d >= a) & (d <= b)))
    count_est = float(N * z.weights[inside].sum())
    assert abs(count_est - count_true) / count_true < 0.12
    # node accuracy: with 107 eigenvalues in the window and k=32, nodes are
    # QUADRATURE points of the local measure — within ~2 mean spacings is the
    # honest expectation (individual-eigenvalue convergence needs k > count,
    # tested in the sparse-window case below)
    spacing = 10.0 / N
    for nd in z.nodes[inside & (z.weights > 1e-4)]:
        assert np.min(np.abs(d - nd)) < 2 * spacing


def test_zoom_sparse_window_machine_nodes():
    # few eigenvalues in the window (< k): zoom nodes converge to THEM
    N = 1500
    rng = np.random.default_rng(6)
    d = np.concatenate([rng.uniform(0, 4, N - 3), [5.0, 5.2, 5.45],])
    mv = lambda v: d * v
    s = Spectral.of(mv, N, k=24, probes=4)
    z = s.zoom(4.8, 5.6, k=16, probes=4, degree=400)
    big = z.nodes[(z.nodes > 4.8) & (z.weights > 1e-5)]
    for target in (5.0, 5.2, 5.45):
        assert np.min(np.abs(big - target)) < 1e-9   # isolated: machine-grade


def test_polish_guards():
    import pytest as _pt
    N = 200
    A = _psd(N)
    s = Spectral.of(lambda v: A @ v, N, k=16, probes=4)
    with _pt.raises(ValueError):
        s.trace("log", certified=True, with_err=True, support=(0.4, None))
    with _pt.raises(ValueError):
        s.zoom(2.0, 1.0)                       # a >= b refused
    val, err = s.effective_rank(with_err=True)
    assert err > 0 and abs(val - s.effective_rank()) < 1e-12
    from resona.free import rie_clean
    with _pt.raises(ValueError):
        rie_clean(np.ones(10), q=3.0)          # q outside MP domain
    from resona.wkernel import track
    with _pt.raises(ValueError):
        track(np.array([[0., 9.], [0., 1.]]), [np.eye(2)],
              np.linspace([0.], [1.], 3))      # non-symmetric A0 refused
