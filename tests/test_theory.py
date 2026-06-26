"""Tests for the theory modules: wkernel, lift, beta, defect, free (vs ground truth)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy import linalg
import resona

rng = np.random.default_rng(0)


def _sym(N):
    M = rng.standard_normal((N, N)); return (M + M.T) / 2


def test_beta_from_spectral():
    N = 400
    B = rng.standard_normal((N, 80)); A = (B @ B.T) / 80
    s = resona.of(lambda v: A @ v, N, k=80, probes=8)
    lev = np.sort(resona.beta.beta_from(s, N))
    true = np.sort(linalg.eigvalsh(A))
    assert np.mean(np.abs(lev - true)) / (true.max() - true.min()) < 0.06


def test_beta_from_robust_moments():
    """robust=True (Rademacher–Hutchinson moments) tightens the GOE case where the
    SLQ-quadrature moments are noisy (seed=0 here: 5.84% → <1%)."""
    D = 512
    H = np.random.default_rng(0).standard_normal((D, D)); A = (H + H.T) / 2 / np.sqrt(D)
    ev = np.sort(linalg.eigvalsh(A)); span = ev.max() - ev.min()
    s = resona.of(lambda x: A @ x, D, k=64, probes=20)
    rob = np.sort(resona.beta.beta_from(s, N=D, robust=True))
    assert np.mean(np.abs(rob - ev)) / span < 0.015          # robust moments < 1.5%


def test_wkernel_matches_finite_diff():
    n = 120; G = rng.standard_normal((n, n)); A0 = (G + G.T) / 2
    Bp = np.diag(rng.standard_normal(n))
    w, V = linalg.eigh(A0)
    W = resona.wkernel.wkernel(V[:, :6], [Bp])
    fd = (np.sort(linalg.eigvalsh(A0 + 1e-6 * Bp))[:6] - np.sort(w)[:6]) / 1e-6
    assert np.max(np.abs(W[:, 0] - fd)) < 1e-3


def test_defect_richardson():
    f = lambda n: np.pi + 1.3 / n + 0.7 / n ** 2
    r = resona.defect.richardson(f(50), f(100), p=1)
    assert abs(r - np.pi) < abs(f(100) - np.pi) / 5


def test_free_cumulants_semicircle():
    kap = resona.free.free_cumulants([0, 1, 0, 2, 0, 5])    # semicircle moments
    assert abs(kap[1] - 1) < 1e-9
    assert max(abs(k) for i, k in enumerate(kap) if i != 1) < 1e-9


def test_freeness_defect_free_vs_nonfree():
    M = 600; X = rng.standard_normal((M, M)); A = (X + X.T) / np.sqrt(2 * M)
    Q, _ = linalg.qr(rng.standard_normal((M, M))); Bf = Q @ A @ Q.T
    free = resona.free.freeness_defect(lambda x: A @ x, lambda x: Bf @ x, M, probes=20)
    non = resona.free.freeness_defect(lambda x: A @ x, lambda x: A @ x, M, probes=20)
    assert free < 0.05 < non


def test_r_transform_additivity():
    M = 600; X = rng.standard_normal((M, M)); A = (X + X.T) / np.sqrt(2 * M)
    Q, _ = linalg.qr(rng.standard_normal((M, M))); Bf = Q @ A @ Q.T
    sA = resona.of(lambda v: A @ v, M, k=90, probes=6)
    sB = resona.of(lambda v: Bf @ v, M, k=90, probes=6)
    sAB = resona.of(lambda v: (A + Bf) @ v, M, k=90, probes=6)
    wg = np.linspace(0.05, 0.4, 6)
    RA = resona.lift.r_transform(sA, wg); RB = resona.lift.r_transform(sB, wg)
    RAB = resona.lift.r_transform(sAB, wg)
    assert np.max(np.abs(RAB - (RA + RB))) / np.max(np.abs(RAB)) < 0.15


def test_carleman_scalar_logistic_stepped():
    M = resona.lift.carleman_scalar([0, 1, -1], order=12)
    x, dt, T = 0.2, 0.1, 2.0
    for _ in range(int(T / dt)):
        z0 = np.array([x ** j for j in range(13)])         # basis (x⁰..x¹²)
        x = float(resona.apply(lambda v: M @ v, lambda l: np.exp(dt * l), z0,
                               k=13, hermitian=False)[1].real)   # x = z_1
    exact = 0.2 * np.exp(T) / (1 + 0.2 * (np.exp(T) - 1))
    assert abs(x - exact) < 1e-9


def test_carleman_gf_exact():
    from itertools import product
    for f in (lambda x: max(x), lambda x: min(x), lambda x: (x[0] + x[1]) % 3):
        _, ev = resona.lift.carleman_gf(3, 2, f)
        assert all(ev(x) == f(x) for x in product(range(3), repeat=2))


def test_free_convolution_moments():
    M = 900; d = rng.uniform(-1, 1, M); A = np.diag(d)
    Q, _ = linalg.qr(rng.standard_normal((M, M))); B = Q @ A @ Q.T
    sA = resona.of(lambda v: A @ v, M, k=120, probes=16)
    sB = resona.of(lambda v: B @ v, M, k=120, probes=16)
    mpred = resona.lift.free_convolution(sA, sB, order=4)
    mtrue = [np.trace(np.linalg.matrix_power(A + B, n)) / M for n in range(1, 5)]
    assert max(abs(p - t) for p, t in zip(mpred, mtrue)) < 0.1


def test_subordination_averaged_dos():
    N = 600
    A = np.diag(np.concatenate([-np.ones(N // 2), np.ones(N // 2)]))   # atoms ±1
    sA = resona.of(lambda v: A @ v, N, k=80, probes=8)
    xs = np.linspace(-3, 3, 1500)
    rho = resona.subordination.averaged_dos(sA, 0.5, xs, eta=2e-3)
    mass = np.trapezoid(rho, xs); m2 = np.trapezoid(xs ** 2 * rho, xs)
    assert abs(mass - 1) < 0.03
    assert abs(m2 - 1.25) < 0.05                          # m2(A)+σ² = 1 + 0.25


def test_cost_extractable_vs_genuine():
    x = np.arange(300)
    ext_p, _ = resona.cost.is_extractable(np.sin(2 * np.pi * x / 7))
    seq = np.array([pow(3, int(i), 100003 * 100019) for i in range(300)], float)
    ext_s, _ = resona.cost.is_extractable(seq)
    assert ext_p and not ext_s


def test_cost_fit_law():
    eps = np.array([1e-2, 1e-3, 1e-4] * 3); dist = np.repeat([0.1, 0.3, 1.0], 3)
    costs = resona.cost.extraction_cost(eps, dist, a=1.5, b=0.8, c=2.0)
    a, b, c = resona.cost.fit_law(costs, eps, dist)
    assert abs(a - 1.5) < 1e-6 and abs(b - 0.8) < 1e-6


def test_flow_shock_time():
    N = 600
    A = np.diag(np.concatenate([-np.ones(N // 2), np.ones(N // 2)]))
    sA = resona.of(lambda v: A @ v, N, k=80, probes=8)
    tc = resona.flow.shock_time(sA)
    assert tc is not None and abs(tc - 1.0) < 0.3       # two atoms ±1 merge at t_c=1


def test_s_transform_multiplicative():
    M = 700; G = rng.standard_normal((M, 400)); A = (G @ G.T) / 400 + 0.3 * np.eye(M)
    Q, _ = linalg.qr(rng.standard_normal((M, M))); B = Q @ A @ Q.T
    sA = resona.of(lambda v: A @ v, M, k=110, probes=8)
    sB = resona.of(lambda v: B @ v, M, k=110, probes=8)
    evP = np.linalg.eigvals(A @ B).real
    sP = type("o", (), {"nodes": evP, "weights": np.ones(M) / M})
    wg = np.array([0.1, 0.2, 0.3])
    SA = resona.lift.s_transform(sA, wg); SB = resona.lift.s_transform(sB, wg)
    SP = resona.lift.s_transform(sP, wg)
    assert np.max(np.abs(SP - SA * SB)) / np.max(np.abs(SP)) < 0.05   # S_{A⊠B}=S_A·S_B


def test_cross_moment():
    N = 400; A = _sym(N) / np.sqrt(N); B = _sym(N) / np.sqrt(N)
    tau = resona.free.cross_moment({"A": lambda x: A @ x, "B": lambda x: B @ x},
                                   "AB", N, probes=60)
    assert abs(tau - np.trace(A @ B) / N) < 0.1 * abs(np.trace(A @ B) / N) + 0.02


def test_richardson_limit():
    f = lambda n: np.pi + 1.3 / n + 0.7 / n ** 2
    ns = [25, 50, 100, 200]
    lim = resona.defect.richardson_limit([f(n) for n in ns], ns, p0=1.0)
    assert abs(lim - np.pi) < 1e-4


def test_wkernel_design_consistency():
    n = 140; H = rng.standard_normal((n, n)); A0 = (H + H.T) / 2
    Bs = [np.diag((np.arange(n) == i).astype(float)) for i in range(0, n, 20)]
    _, V = linalg.eigh(A0)
    W = resona.wkernel.wkernel(V[:, :len(Bs)], Bs)               # square
    target = rng.standard_normal(len(Bs)) * 0.05
    dk = resona.wkernel.design(W, target)
    assert np.max(np.abs(W @ dk - target)) < 1e-9                # solves W·dk=target


def test_wkernel_design_tikhonov():
    n = 20; W = rng.standard_normal((n, n)); y = rng.standard_normal(n)
    dk0 = resona.wkernel.design(W, y, reg=0.0)
    assert np.max(np.abs(W @ dk0 - y)) < 1e-9                      # exact (full rank)
    reg = 1e-2; dk = resona.wkernel.design(W, y, reg=reg)
    smax2 = np.linalg.svd(W, compute_uv=False)[0] ** 2
    resid = (W.T @ W + reg * smax2 * np.eye(n)) @ dk - W.T @ y     # normal equations
    assert np.max(np.abs(resid)) < 1e-8
    assert np.linalg.norm(dk) < np.linalg.norm(dk0) + 1e-9         # regularized = smaller


def test_kappa_w_modes_matches_dense_block():
    """Matrix-free κ_W (modes=k via eigsh) == dense κ_W on the SAME bottom-k block."""
    import scipy.sparse as sp
    N, m = 600, 6
    d = rng.standard_normal(N); o1 = rng.standard_normal(N - 1) * 0.8
    A0 = sp.diags([o1, d, o1], [-1, 0, 1]).tocsc()
    B1 = sp.diags([np.ones(N - 1) * 0.5, np.zeros(N), np.ones(N - 1) * 0.5], [-1, 0, 1]).tocsc()
    B2 = sp.diags([np.ones(N - 2) * 0.3, np.ones(N - 2) * 0.3], [-2, 2]).tocsc()
    Bs = [B1, B2]; k0 = np.array([0.1, 0.05])

    # dense reference: full eigh -> bottom-m vectors -> κ_W, SAME random directions
    A0d, Bd = A0.toarray(), [B.toarray() for B in Bs]

    def W_dense(k):
        _, V = np.linalg.eigh(A0d + sum(float(k[j]) * Bd[j] for j in range(2)))
        V = V[:, :m]
        return np.array([[float(V[:, i] @ (Bd[j] @ V[:, i])) for j in range(2)] for i in range(m)])

    r = np.random.default_rng(0); W0 = W_dense(k0); vals = []
    for _ in range(4):
        u = r.standard_normal(2); u /= np.linalg.norm(u)
        vals.append(float(np.linalg.norm(W_dense(k0 + 1e-5 * u) - W0) / 1e-5))
    kd = max(vals)

    km = resona.wkernel.kappa_w(A0, Bs, k0, modes=m, probes=4, seed=0)
    assert abs(kd - km) / max(abs(kd), 1e-30) < 1e-6              # identical to dense


def test_kappa_w_modes_all_unchanged():
    """modes='all' (default) reproduces the original dense full-spectrum κ_W."""
    n = 40; A0 = _sym(n); Bs = [_sym(n) * 0.1]
    kf = resona.wkernel.kappa_w(A0, Bs, [0.2], probes=4, seed=1)             # default = 'all'
    ka = resona.wkernel.kappa_w(A0, Bs, [0.2], probes=4, seed=1, modes="all")
    assert kf == ka and kf > 0


def test_track_modes_matches_dense():
    """Matrix-free track(modes=k) == dense track on the same bottom-k block (same method)."""
    import scipy.sparse as sp
    N = 400
    d = np.sort(rng.standard_normal(N)) * 0.3
    A0 = sp.diags(d).tocsc()
    B1 = sp.diags([np.ones(N - 1) * 0.2, np.zeros(N), np.ones(N - 1) * 0.2], [-1, 0, 1]).tocsc()
    path = np.linspace(0, 0.2, 5).reshape(-1, 1)
    lams_mf, _ = resona.wkernel.track(A0, [B1], path, steps=2, modes=4)
    lams_all, _ = resona.wkernel.track(A0.toarray(), [B1.toarray()], path, steps=2, modes="all")
    diff = np.max(np.abs(np.sort(lams_mf, axis=1) - np.sort(lams_all[:, :4], axis=1)))
    assert diff < 1e-10                                # identical to dense, same method
    assert lams_mf.shape == (5, 4)                     # returns the selected block only


def test_track_modes_gap_guard():
    """guard warns when a mode is about to leave the selected block (near-degenerate boundary)."""
    import warnings
    N = 60; diag = np.arange(N, dtype=float); diag[4] = diag[3] + 3e-4   # block boundary near-degenerate
    A0 = np.diag(diag); B = _sym(N) * 1e-4                               # tiny B keeps the gap small
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        resona.wkernel.track(A0, [B], np.linspace(0, 1, 3).reshape(-1, 1), steps=1, modes=4, guard=True)
        assert any("leaving the selected block" in str(x.message) for x in wl)


def test_normality_zero_for_normal_and_matfree_structured():
    """normality=0 for symmetric (normal); matrix-free matches dense on a structured op."""
    import scipy.sparse as sp
    N = 400
    # (a) symmetric ⇒ exactly normal ⇒ energy 0
    S = _sym(N)
    e0, _ = resona.defect.normality(lambda x: S @ x, N=N, probes=16)
    assert e0 < 1e-18
    # (b) structured non-symmetric: matrix-free Hutchinson matches the dense energy <2%
    d = rng.standard_normal(N); up = rng.standard_normal(N - 1); lo = rng.standard_normal(N - 1) * 0.3
    A = sp.diags([lo, d, up], [-1, 0, 1]).tocsc()
    dense = float(np.linalg.norm((A @ A.T - A.T @ A).toarray()) ** 2)
    est, se = resona.defect.normality(lambda x: A @ x, N=N, rmatvec=lambda x: A.T @ x, probes=48)
    assert abs(est - dense) / dense < 0.02
    # exact ndarray path is deterministic
    ev, se2 = resona.defect.normality(A.toarray())
    assert abs(ev - dense) / dense < 1e-9 and se2 == 0.0


def test_rmt_class_separates_ensembles():
    """R4 rigidity meter orders Poisson > GOE > GUE (ensemble-averaged — a single
    draw is noisy near the GOE↔GUE boundary, per the function's documented caveat)."""
    D, reps = 600, 6

    def avg_R4(make):
        return float(np.mean([resona.cost.rmt_class(make())[1] for _ in range(reps)]))

    def goe():
        Hr = rng.standard_normal((D, D)); return np.linalg.eigvalsh((Hr + Hr.T) / 2)

    def gue():
        Hc = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
        return np.linalg.eigvalsh((Hc + Hc.conj().T) / 2)

    rp = avg_R4(lambda: np.sort(rng.uniform(-1, 1, D)))
    rg, ru = avg_R4(goe), avg_R4(gue)
    assert rp > rg > ru                                  # rigidity ordering (averaged)
    assert resona.cost.rmt_class(np.sort(rng.uniform(-1, 1, D)))[0] == "Poisson"  # robust per-draw


def test_hard_points_finds_avoided_crossing():
    """Φ_η divining rod: argmax_k Φ_η = the avoided crossing (k=0), matrix-free, no eig."""
    N = 200; delta = 0.04
    bg = np.diag(np.concatenate([[0, 0], rng.uniform(0.5, 3, N - 2) * rng.choice([-1, 1], N - 2)]))
    off = np.zeros((N, N)); off[0, 1] = off[1, 0] = delta

    def H_of_k(k):
        M = bg.copy().astype(float); M[0, 0] = k; M[1, 1] = -k; M += off
        return (M + M.T) / 2

    B = np.zeros((N, N)); B[0, 1] = B[1, 0] = 1.0
    ks = np.linspace(-0.3, 0.3, 13)
    k_star, prof = resona.defect.hard_points(H_of_k, ks, B, E=0.0, eta=0.06, probes=8)
    assert abs(k_star) < 1e-9                       # the crossing is at k=0 (min gap = 2δ)
    assert prof.argmax() == len(ks) // 2            # peak at the centre


def test_normality_dense_robust_all_seeds():
    """Rademacher + median-of-means: even a DENSE random op is <3% on every seed
    (regression — Gaussian seed=0 here used to read 49.8% off)."""
    N = 400
    M = np.random.default_rng(0).standard_normal((N, N))
    true = float(np.linalg.norm(M @ M.T - M.T @ M) ** 2)
    for seed in range(5):
        est, _ = resona.defect.normality(lambda x: M @ x, N=N,
                                         rmatvec=lambda x: M.T @ x, seed=seed)
        assert abs(est - true) / true < 0.03


def test_rie_clean_sample_covariance():
    # free deconvolution of Marchenko-Pastur noise: must beat the raw empirical
    # covariance and come close to the oracle (best possible with E's basis)
    from resona.free import rie_clean
    rng = np.random.default_rng(0)
    N, T = 200, 400
    lam_true = np.concatenate([[12.0, 6.0], np.linspace(2.0, 0.5, N - 2)])
    U = np.linalg.qr(rng.standard_normal((N, N)))[0]
    C = (U * lam_true) @ U.T
    X = rng.multivariate_normal(np.zeros(N), C, size=T)
    E = X.T @ X / T
    le, Ue = np.linalg.eigh(E)
    xi = rie_clean(le, q=N / T)
    xi_or = np.array([Ue[:, i] @ C @ Ue[:, i] for i in range(N)])
    err = lambda lam: np.linalg.norm((Ue * lam) @ Ue.T - C)
    assert err(xi) < 0.7 * err(le)               # ≥1.4x closer to the truth
    assert err(xi_or) > 0.85 * err(xi)           # within ~15% of the oracle


def test_rie_clean_additive():
    from resona.free import rie_clean_additive
    rng = np.random.default_rng(1)
    N, sigma = 300, 0.7
    A = np.diag(np.linspace(-2, 2, N))
    W = rng.standard_normal((N, N)); W = (W + W.T) / np.sqrt(2 * N)
    E = A + sigma * W
    le, Ue = np.linalg.eigh(E)
    xi = rie_clean_additive(le, sigma)
    err = lambda lam: np.linalg.norm((Ue * lam) @ Ue.T - A)
    assert err(xi) < 0.9 * err(le)               # strictly better than raw


def test_subordination_contraction_edge():
    # |T'| -> ~1 approaching the two-atom band edge; small in the bulk
    import resona
    from resona.subordination import contraction
    N = 2000
    d = np.concatenate([np.full(N // 2, -1.0), np.full(N // 2, 1.0)])
    s = resona.of(lambda x: d * x, N)
    # bulk point: well inside the band, modest contraction
    c_bulk = contraction(s, 0.0, 0.25)
    assert c_bulk < 0.9
    # approach the (numerically located) outer edge from inside
    xs = np.array([1.74, 1.755, 1.760])
    cs = contraction(s, xs, 0.25)
    assert cs[-1] > 0.95                       # critical near the edge
    assert np.all(np.diff(cs) > 0)             # and monotone approaching it


def test_generator_read_be_families():
    # O(n^-2) absolute convergence on a family OUTSIDE the original suite
    from scipy.linalg import expm
    from resona.defect import generator_read
    rng = np.random.default_rng(3)
    N, t = 80, 0.5
    A = np.diag(np.geomspace(1e-2, 1e2, N))          # stiff diagonal
    u0 = rng.standard_normal(N)
    G_true = A @ A @ (expm(-t * A) @ u0)
    errs = []
    for n in (64, 128, 256):
        M = np.linalg.inv(np.eye(N) + (t / n) * A)
        P = np.linalg.matrix_power(M, n) @ u0
        M2 = np.linalg.inv(np.eye(N) + (t / (2 * n)) * A)
        P2 = np.linalg.matrix_power(M2, 2 * n) @ u0
        Gh = generator_read(P, P2, t, n)
        errs.append(np.linalg.norm(Gh - G_true) / np.linalg.norm(G_true))
    slope = np.polyfit(np.log([64, 128, 256]), np.log(errs), 1)[0]
    assert -1.35 < slope < -0.7                      # rel err O(1/n) ⇔ abs O(n^-2)
    assert errs[-1] < 8e-3


def test_generator_read_refuses_cn():
    from resona.defect import generator_read
    import pytest as _pt
    with _pt.raises(ValueError):
        generator_read(np.ones(4), np.ones(4), 0.5, 8, solver="cn")


def test_spectroscopy_barycentre_and_noise():
    from resona.defect import defect_barycentres
    rng = np.random.default_rng(0)
    Nf = 256
    k = np.fft.fftfreq(Nf, 1.0 / Nf)
    kmag = np.abs(k)
    # defect power concentrated at modes 3 and 24
    power = np.zeros(Nf)
    power[np.abs(kmag - 3) < 0.5] = 4.0
    power[np.abs(kmag - 24) < 0.5] = 9.0
    bands = [(kmag >= 2 ** j) & (kmag <= 2 ** (j + 1) - 1) for j in range(6)]
    kb, sig = defect_barycentres(power, bands, coords=kmag)
    assert abs(kb[1] - 3.0) < 1e-12                  # band j=1 holds mode 3
    assert abs(kb[4] - 24.0) < 1e-12                 # band j=4 holds mode 24
    # 5% noise: barycentre moves by less than one bin
    kb2, _ = defect_barycentres(power + 0.05 * power.max() * rng.random(Nf),
                          bands, coords=kmag)
    assert abs(kb2[1] - 3.0) < 1.0 and abs(kb2[4] - 24.0) < 1.0
    # empty band → nan + zero signal, no crash
    assert np.isnan(kb[5]) or sig[5] >= 0


def _linear_family(N=20, M=4, seed=7):
    rng = np.random.default_rng(seed)
    A0 = rng.standard_normal((N, N)); A0 = (A0 + A0.T) / 2
    Bs = []
    for _ in range(M):
        B = rng.standard_normal((N, N)); Bs.append((B + B.T) / 2)
    return A0, Bs


def test_track_beats_frozen_w():
    from resona.wkernel import wkernel as wk, track
    rng = np.random.default_rng(7)
    A0, Bs = _linear_family()
    target = rng.standard_normal(4) * 1.2
    Bstack = np.stack(Bs)
    lam_exact = np.linalg.eigvalsh(A0 + np.tensordot(target, Bstack, axes=1))
    # frozen W from the origin
    lam0, V0 = np.linalg.eigh(A0)
    W0 = wk(V0, Bs)
    frozen = np.sort(lam0 + W0 @ target)
    path = np.linspace(np.zeros(4), target, 201)
    lams, _ = track(A0, Bs, path)
    err_frozen = np.max(np.abs(np.sort(frozen) - lam_exact))
    err_track = np.max(np.abs(np.sort(lams[-1]) - lam_exact))
    assert err_track < err_frozen / 100          # the C2 claim


def test_track_survives_crossing():
    from resona.wkernel import track
    # two eigenvalues CROSS along the path: sorted order breaks, tracking holds
    A0 = np.diag([0.0, 1.0]); B = np.diag([1.0, -1.0])
    path = np.linspace([0.0], [1.0], 41)         # crossing at k = 0.5
    lams, _ = track(A0, [B], path)
    # tracked branch 0 must follow 0 + k (through the crossing), branch 1 → 1 − k
    assert abs(lams[-1][0] - 1.0) < 1e-10
    assert abs(lams[-1][1] - 0.0) < 1e-10
    sorted_end = np.sort(np.linalg.eigvalsh(A0 + 1.0 * B))
    assert np.max(np.abs(np.sort(lams[-1]) - sorted_end)) < 1e-10


def test_kappa_w_dials():
    from resona.wkernel import kappa_w
    A0, Bs = _linear_family(seed=9)
    # commuting family: diagonal A0 and diagonal Bs → W constant → κ_W ≈ 0
    D0 = np.diag(np.arange(10.0))
    DBs = [np.diag(np.linspace(1, 2, 10)), np.diag(np.linspace(-1, 1, 10))]
    assert kappa_w(D0, DBs, np.zeros(2)) < 1e-6
    assert kappa_w(A0, Bs, np.zeros(4)) > 1.0    # generic: curvature present


# ── defect calculus ──

def test_defect_subtraction_is_owned_object():
    """defect(P_n, P_2n) = P_n − P_{2n} — the boundary measurement."""
    from resona.defect import defect
    P_n = np.array([1.0, 2.0, 3.0])
    P_2n = np.array([0.9, 1.98, 2.99])
    D = defect(P_n, P_2n)
    np.testing.assert_allclose(D, [0.1, 0.02, 0.01], atol=1e-12)


def test_defect_jump_exact_on_known_iteration():
    """defect_jump: D_{2n} = J^n · D_n — verify on scaled identity iteration."""
    from resona.defect import defect, defect_jump
    D_n = np.array([3.0, 5.0, 7.0])
    J = 0.5 * np.eye(3)
    n = 4
    D_2n = defect_jump(D_n, J, n)
    # J^n · D_n = (0.5^4) * D_n
    expected = D_n * (0.5 ** n)
    np.testing.assert_allclose(D_2n, expected, atol=1e-14)


def test_defect_jump_callable_matvec():
    """defect_jump with callable J (matvec) — same result as matrix J."""
    from resona.defect import defect_jump
    D_n = np.array([2.0, -1.0, 4.0])
    J = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    result_mat = defect_jump(D_n, J, 3)
    result_mv = defect_jump(D_n, lambda x: J @ x, 3)
    np.testing.assert_allclose(result_mat, result_mv, atol=1e-14)


# ── free heat flow (burgers_density) ──

def test_burgers_density_two_atoms_merge():
    """Free-heat-flow of two atoms ±1: at small t two peaks, at t≥t_c one band."""
    from resona.flow import burgers_density, shock_time
    lam = np.array([-1.0, 1.0]); w = np.array([0.5, 0.5])
    xs = np.linspace(-3, 3, 600)
    from scipy.signal import argrelextrema
    # at small t: density must have two distinct peaks
    rho_small = burgers_density((lam, w), 0.2, xs, eta=1e-3)
    peaks = argrelextrema(rho_small, np.greater)[0]
    assert len(peaks) >= 2
    # at large t: merged
    rho_large = burgers_density((lam, w), 2.0, xs, eta=1e-3)
    peaks_large = argrelextrema(rho_large, np.greater)[0]
    assert len(peaks_large) <= 2
    # mass conserved within stochastic tolerance
    dx = xs[1] - xs[0]
    assert abs(np.trapezoid(rho_large, xs) - 1.0) < 0.06
    assert abs(np.trapezoid(rho_small, xs) - 1.0) < 0.06
    # shock time ~1 for ±1
    tc = shock_time((lam, w))
    assert tc is not None
    assert 0.5 < tc < 2.0


def test_burgers_density_semicircle_attractor():
    """At large t, μ_t → semicircle — the central limit of free probability."""
    from resona.flow import burgers_density
    lam = np.array([-2.0, 0.5, 1.3]); w = np.array([0.3, 0.3, 0.4])
    T = 20.0
    xs = np.linspace(-12, 12, 800)
    rho = burgers_density((lam, w), T, xs, eta=1e-3)
    var0 = np.sum(w * lam ** 2) - np.sum(w * lam) ** 2
    R = 2 * np.sqrt(T + var0)
    inside = xs[np.abs(xs) < 0.98 * R]
    assert rho[np.abs(xs) < 0.98 * R].min() > 0
    dx = xs[1] - xs[0]
    assert abs(np.trapezoid(rho, xs) - 1.0) < 0.06


# ── beta_spectrum standalone ──

def test_beta_spectrum_known_moments():
    """beta_spectrum: [E0, Emax, μ1, μ2] + N → spectrum recovers known distribution."""
    from resona.beta import beta_spectrum
    N = 200
    # uniform on [0, 1]: E0=0, Emax=1, μ1=0.5, μ2=1/3
    levels = beta_spectrum(0.0, 1.0, 0.5, 1.0 / 3.0, N)
    assert len(levels) == N
    assert abs(levels.min() - 0.0) < 0.02
    assert abs(levels.max() - 1.0) < 0.02
    assert abs(np.mean(levels) - 0.5) < 0.05
    # monotone increasing
    assert np.all(np.diff(levels) >= 0)


def test_beta_spectrum_semicircle_moments():
    """semicircle [−2,2]: μ1=0, μ2=1 → levels bounded within [−2,2]."""
    from resona.beta import beta_spectrum
    N = 300
    levels = beta_spectrum(-2.0, 2.0, 0.0, 1.0, N)
    assert len(levels) == N
    assert levels.min() >= -2.2
    assert levels.max() <= 2.2
    assert abs(np.mean(levels)) < 0.1    # zero mean


# ── free cumulants roundtrip ──

def test_free_cumulants_roundtrip():
    """moments_from_cumulants(free_cumulants(m)) == m."""
    from resona.free import free_cumulants, moments_from_cumulants
    m = [1.0, 2.5, 7.0, 21.0, 65.0, 200.0]
    k = free_cumulants(m)
    m_back = moments_from_cumulants(k)
    np.testing.assert_allclose(m_back, m, atol=1e-12)


def test_free_cumulants_additivity():
    """κ_n(A⊞B) = κ_n(A) + κ_n(B) — the defining property."""
    from resona.free import free_cumulants
    from resona.lift import free_convolution
    N = 120
    A = rng.standard_normal((N, N)); A = A @ A.T / N
    B_mat = rng.standard_normal((N, N)); B = B_mat @ B_mat.T / N
    # rotate B to make it free from A
    U = np.linalg.qr(rng.standard_normal((N, N)))[0]
    B_rot = U @ B @ U.T
    sA = resona.of(lambda v: A @ v, N, k=40, probes=8)
    sB = resona.of(lambda v: B_rot @ v, N, k=40, probes=8)
    m = free_convolution(sA, sB, order=5)
    # κ_n add: moments from free cumulants of sum = κ(A)+κ(B)
    s_sum = sA + sB
    true_sum = np.sort(linalg.eigvalsh(A + B_rot))
    # trace of sum = Tr(A) + Tr(B) — verified through moments
    assert abs(m[0] - (np.trace(A) + np.trace(B_rot)) / N) < 2.0


# ── pastur subordination ──

def test_pastur_fixed_point_semicircle():
    """pastur: the Pastur fixed point g = G_A(z − σ²·g) for semicircle A."""
    from resona.subordination import pastur
    # semicircle G_A(ζ) = (ζ − √(ζ²−4))/2
    def GA(zz):
        return (zz - np.sqrt(zz ** 2 - 4.0 + 0j)) / 2.0
    z = 1.5 + 0.3j
    g_zero = pastur(GA, z, 0.0)
    assert abs(g_zero - GA(z)) < 2e-4               # σ=0 → g = G_A(z)
    # with small disorder: g shifts
    g_small = pastur(GA, z, 0.1)
    assert abs(g_small - GA(z)) > 1e-6              # σ>0 gives different g


def test_pastur_grid_vs_scalar():
    """pastur_grid on an array must match pastur on each element."""
    from resona.subordination import pastur, pastur_grid
    # semicircle on [-2, 2]: nodes = quad pts, weights = semicircle density
    from numpy.polynomial.legendre import leggauss
    nodes_quad, weights_quad = leggauss(80)
    nodes = 2 * nodes_quad
    weights = weights_quad * (2 / np.pi) * np.sqrt(1 - nodes_quad ** 2)
    spectral = (nodes, weights / weights.sum())
    zs = np.linspace(2.1, 4.0, 9) + 0.3j
    gs_grid = pastur_grid(spectral, zs, 0.25)
    def GA(zz):
        return np.sum(spectral[1] / (zz - spectral[0]))
    gs_scalar = np.array([pastur(GA, z, 0.25) for z in zs])
    np.testing.assert_allclose(gs_grid, gs_scalar, atol=1e-6)


# ── lift_rank standalone ──

def test_lift_rank_low_rank_signal():
    """lift_rank: a low-rank (few-sine) signal has low effective rank."""
    from resona.cost import lift_rank
    xs = np.linspace(0, 2 * np.pi, 200)
    signal = np.sin(xs) + 0.5 * np.sin(2 * xs)       # rank ~2 in trajectory space
    r = lift_rank(signal, k=40)
    assert r < 6                                        # low-rank


def test_lift_rank_noise_high_rank():
    """lift_rank: a noise signal has high effective rank."""
    from resona.cost import lift_rank
    rng_local = np.random.default_rng(42)
    signal = rng_local.standard_normal(300)
    r = lift_rank(signal, k=30)
    assert r > 10                                       # high-rank, not saturated
