"""
resona.defect — the defect calculus: error-as-information.

The root idea of the whole program.  Run a solver / discretization at resolution
n and get P_n; the DEFECT

        D_n = P_n − P_{2n}

is not waste — it is the dominant error mode, and it CARRIES the operator's
spectrum.  Two consequences are exposed here:

• RICHARDSON annihilation: if the error scales as n^{-p}, the combination
  (2^p P_{2n} − P_n)/(2^p − 1) cancels the leading defect → a higher-order answer
  for free.  Iterated, it is a convergence-acceleration tower.

• The DEFECT-JUMP law: for a linear stationary iteration x ← J x + c (gradient
  descent, power iteration, Gauss–Seidel, PageRank), the defect of a doubling is
  EXACTLY D_{2n} = J^n · D_n — so one cheap defect predicts the converged answer
  with no extra iterations.

• The PSEUDOSPECTRUM — the geometric half of the defect calculus.  For a NORMAL
  operator the spectrum is the whole story; for a defective one (a Jordan block)
  the spectrum LIES: an order-q defect blooms the point eigenvalue into a disk
  of radius ε^{1/q} — the same catastrophe exponent as the hardness law
  (`theory/hardness_exponents.py`).  GMRES/Arnoldi convergence follows the
  pseudospectrum, not the spectrum.
"""
import numpy as np


def defect(P_n, P_2n):
    """The defect D_n = P_n − P_{2n} — the module's OWNED OBJECT.

    Yes, it is a subtraction.  It exists as a named function because the
    whole module is a calculus on this quantity (richardson, defect_jump,
    generator_read, spectroscopy all consume it): naming the boundary is
    the point, the arithmetic is incidental."""
    return np.asarray(P_n) - np.asarray(P_2n)


def richardson(P_n, P_2n, p=1):
    """One Richardson step for error ~ n^{-p}: (2^p P_{2n} − P_n)/(2^p − 1)."""
    f = 2.0 ** p
    return (f * np.asarray(P_2n) - np.asarray(P_n)) / (f - 1.0)


def richardson_limit(values, ns, p0=1.0):
    """Extrapolate a sequence values[k] sampled at resolutions ns[k] (assumed a
    geometric doubling) to n→∞ via a Neville/Richardson tableau on error ~ n^{-p}.

    Returns the top-of-tableau estimate of the limit.
    """
    v = [np.asarray(x, float) for x in values]
    ns = list(ns); L = len(v)
    T = [v[:]]                                              # T[0] = raw column
    for col in range(1, L):
        prev = T[col - 1]; newcol = [None] * L
        for k in range(col, L):
            r = (ns[k] / ns[k - col]) ** (p0 * col)         # ratio of resolutions^p
            newcol[k] = (r * prev[k] - prev[k - 1]) / (r - 1.0)
        T.append(newcol)
    return T[L - 1][L - 1]


def sigma_min(A, z, N=None, rmatvec=None, k=96, seed=0):
    """Smallest singular value of (A − zI).

    A : square ndarray (exact, via SVD) OR a matvec callable (then pass N and,
    for a NON-symmetric operator, `rmatvec` for Aᵀx — σ_min needs both actions).
    Matrix-free path: Lanczos extreme of the PSD operator (A−zI)ᵀ(A−zI); for
    complex z the system is realified to 2N (σ values are unchanged).
    """
    if not callable(A):
        M = np.asarray(A, complex) - z * np.eye(len(A))
        return float(np.linalg.svd(M, compute_uv=False)[-1])
    if N is None:
        raise ValueError("sigma_min(matvec, ...) needs N")
    rmv = rmatvec or A                                    # symmetric default
    zr, zi = float(np.real(z)), float(np.imag(z))

    def B(x):                                             # (A − zI) on realified C^N
        u, v = x[:N], x[N:]
        Au, Av = A(u), A(v)
        return np.concatenate([Au - zr * u + zi * v, Av - zr * v - zi * u])

    def Bt(x):                                            # (A − zI)ᵀ (realified adjoint)
        u, v = x[:N], x[N:]
        Au, Av = rmv(u), rmv(v)
        return np.concatenate([Au - zr * u - zi * v, Av - zr * v + zi * u])

    from .spectral import _lanczos
    rng = np.random.default_rng(seed)
    al, be = _lanczos(lambda x: Bt(B(x)), rng.standard_normal(2 * N), k)
    kk = len(al)
    T = np.diag(al) + np.diag(be[:kk - 1], 1) + np.diag(be[:kk - 1], -1)
    lam_min = float(np.linalg.eigvalsh(T)[0])
    return float(np.sqrt(max(lam_min, 0.0)))


def normality(A, N=None, rmatvec=None, probes=48, groups=6, seed=0):
    """Departure-from-normality ENERGY  ‖[A, A*]‖²_F = ‖A A* − A* A‖²_F.

    Zero ⇔ A is NORMAL (commutes with its adjoint — Hermitian / skew / unitary).
    The cheap GLOBAL non-normality scalar — the companion to the per-point
    `sigma_min` / `pseudospectrum` (which are LOCAL).  When this is large the
    spectrum lies (defective / pseudospectrum-driven); when ≈0 it is the whole story.

    MATRIX-FREE via Hutchinson:  ‖[A,A*]‖²_F = 𝔼_z‖A(A*z) − A*(Az)‖²  — each probe
    is one matvec A and one adjoint A* (`rmatvec`), O(probes · matvec), no matrix.

    Variance control (what makes it robust):
      • RADEMACHER probes (z_i = ±1) — the diagonal of G = [A,A*]ᵀ[A,A*] is then
        captured with ZERO variance (only off-diagonal contributes), and G is
        diagonally heavy here → far tighter than Gaussian probes (measured: a
        Gaussian seed that read 49.8% off drops to 0.1% with this estimator);
      • MEDIAN-OF-MEANS over `groups` blocks — robust to a bad probe block.

    A : square ndarray (exact, two matmuls) OR a matvec callable (then pass N and
        `rmatvec`=A*·x; a symmetric/Hermitian A with the default rmatvec=A gives 0).
    Returns (value, stderr) — the energy and the spread of the group means / √groups.
    A STOCHASTIC estimate; never labelled exact.  The =0-iff-normal property is exact.
    """
    if not callable(A):
        M = np.asarray(A, complex)
        C = M @ M.conj().T - M.conj().T @ M
        return float(np.linalg.norm(C) ** 2), 0.0
    if N is None:
        raise ValueError("normality(matvec, ...) needs N (and rmatvec=A*·x)")
    rmv = rmatvec or A                                    # symmetric default → 0
    g = max(1, int(groups)); probes = max(probes, g) - (max(probes, g) % g)
    rng = np.random.default_rng(seed)
    vals = np.empty(probes)
    for p in range(probes):
        z = rng.integers(0, 2, N).astype(float) * 2.0 - 1.0     # Rademacher ±1
        c = np.asarray(A(rmv(z))) - np.asarray(rmv(A(z)))       # [A, A*] z
        vals[p] = float(np.vdot(c, c).real)
    means = vals.reshape(g, -1).mean(axis=1)                    # block means
    est = float(np.median(means))                              # median-of-means
    se = float(means.std(ddof=1) / np.sqrt(g)) if g > 1 else float("nan")
    return est, se


def pseudospectrum_radius(A, eps, z0=0.0, N=None, rmatvec=None, direction=1.0,
                          r_max=None, k=96, iters=60):
    """The local ε-pseudospectrum radius at z0: the largest r along `direction`
    with  σ_min(A − (z0 + r·direction)I) < eps.

    THE LAW (verified exactly on Jordan blocks): an order-q defect blooms the
    point spectrum into a disk of radius ε^{1/q}.  So:
        radius ≈ eps        → benign (normal-like): the eigenvalue is trustworthy;
        radius ≈ eps^{1/q}  → an order-q defect: the eigenvalue is uncertain to
                              that radius, and iterative solvers converge slowly.
    Read q ≈ ln eps / ln radius.

    Bisection in log-r on the monotone growth of σ_min away from the bloom
    (`iters` halvings).  HONEST LIMIT: with a matvec, σ_min is itself a Lanczos
    estimate; near a deep defect it is ill-conditioned — that IS the phenomenon.
    Report the resolution, never a headline.
    """
    direction = complex(direction)
    direction /= abs(direction)
    sm = lambda r: sigma_min(A, z0 + r * direction, N=N, rmatvec=rmatvec, k=k)
    if r_max is None:
        r_max = 4.0
        while sm(r_max) < eps and r_max < 1e6:
            r_max *= 4.0
    if sm(r_max) < eps:
        return float(r_max)                               # bloom exceeds the search box
    lo, hi = 0.0, float(r_max)
    for _ in range(iters):
        mid = 0.5 * (lo + hi) if lo == 0.0 else float(np.sqrt(lo * hi))
        if sm(mid) < eps:
            lo = mid
        else:
            hi = mid
    return float(lo)


def pseudospectrum(A, zs, eps, N=None, rmatvec=None, k=96):
    """Boolean mask of the ε-pseudospectrum  Λ_ε = { z : σ_min(A − zI) < eps }
    over an array of complex grid points zs."""
    zs = np.asarray(zs)
    return np.array([sigma_min(A, z, N=N, rmatvec=rmatvec, k=k) < eps
                     for z in zs.ravel()]).reshape(zs.shape)


def defect_jump(D_n, J, n):
    """The exact defect-jump  D_{2n} = J^n · D_n  for a linear iteration matrix J
    (or matvec).  Predicts the next-doubling defect without running the iteration.
    """
    mv = J if callable(J) else (lambda x, J=J: J @ x)
    x = np.asarray(D_n, float)
    for _ in range(n):
        x = mv(x)
    return x


def generator_read(P_n, P_2n, t, n, solver="be"):
    """Read the GENERATOR term a one-step solver already computed and threw
    away: for backward Euler,  D_n = P_n − P_2n = (t²/4n)·A²e^{−tA}u₀ + O(n⁻²),
    so  (4n/t²)·D_n  estimates the Koopman-generator observable A²e^{−tA}u₀ —
    from two runs of an UNMODIFIED solver, treated as a black box.

    Family: defect.py owns D_n = P_n − P_2n; this is D_n's leading
    coefficient under the solver's defect expansion.

    Verified (FA/revise_stress): absolute error O(n⁻²) exactly, on families
    far beyond the original suite (graph Laplacians, complex Hermitian,
    stiff κ=1e6, defective Jordan bands; the original: slopes −2.3…−2.0).

    HONEST LIMITS: the constant is SOLVER-SPECIFIC — this formula is for
    backward Euler ("be"); Crank–Nicolson deviates O(1) (measured rel. dev.
    ≈ 1.0) and is refused rather than approximated.  float32 snapshots sit
    at the noise floor (slope ≈ 0): use float64 runs.  The order check
    (does YOUR data follow n⁻²?) is one Richardson line away — see the
    stand examples/defect_spectroscopy.py.
    """
    if solver != "be":
        raise ValueError(
            f"solver={solver!r}: the defect constant is solver-specific; this "
            "read is verified for backward Euler only (Crank–Nicolson deviates "
            "O(1), measured rel. dev. ~1.0 — FA/revise_stress)")
    D = np.asarray(P_n) - np.asarray(P_2n)
    return (4.0 * n / t ** 2) * D


def generator_read_converged(P_n, P_2n, P_4n, t, n, solver="be"):
    """`generator_read` with its own CONVERGENCE CHECK — the Richardson line
    the plain read tells you to write, written.

    Three resolutions give two independent generator reads:
        G_n  from (P_n,  P_2n)  at n,
        G_2n from (P_2n, P_4n)  at 2n,
    and their disagreement IS the truncation estimate.  Returns
    (G, rel_dev):
        G       — the finer read G_2n, Richardson-extrapolated one order
                  using the defect law's own O(1/n) leading correction:
                  G = 2·G_2n − G_n;
        rel_dev — ‖G_2n − G_n‖ / ‖G_2n‖, the convergence certificate: if
                  this is not small, the law's regime (smooth u₀, resolved
                  dynamics) does not hold yet and NO budget of algebra fixes
                  it — refine the solver instead.

    Same verified domain as `generator_read` (backward Euler; CN refused).
    """
    G_n = generator_read(P_n, P_2n, t, n, solver=solver)
    G_2n = generator_read(P_2n, P_4n, t, 2 * n, solver=solver)
    num = float(np.linalg.norm(np.asarray(G_2n) - np.asarray(G_n)))
    den = float(np.linalg.norm(np.asarray(G_2n)))
    rel_dev = num / den if den > 0 else np.inf
    return 2.0 * np.asarray(G_2n) - np.asarray(G_n), rel_dev


def defect_barycentres(power, bands, coords=None):
    """COMPRESS each band of a defect power distribution to ONE coordinate —
    its energy barycentre (the BDS read).  One (location, amplitude) pair per
    band — that compression is exactly what survives noise where ratio
    estimators die.

    (Named `spectroscopy` before 2.0 — that name promised a spectrum it
    does not deliver; see MIGRATION.md.)

    `power`  : |D̂|² — the defect's energy in the caller's diagonalizing
               basis (resona does NOT choose your basis: you transform).
    `bands`  : list of index masks/arrays (the shells).
    `coords` : the mode coordinate per index (default: the index itself).
    Returns (kbar, signal): per band, the energy barycentre ⟨k⟩ and the
    band's total amplitude √Σpower.  The caller's one-liner
    ``lam = symbol[round(kbar)]`` recovers eigenvalues where a symbol
    lookup exists.

    Family: the defect's power spectrum is a measure; this is its per-band
    barycentre — a read on D_n.

    Verified (FA/revise_stress + method8_35): 35/35 PDE suite; the
    barycentre stays stable under 5% snapshot noise where the norm-RATIO
    estimator collapses at 1e-5.  HONEST LIMITS: integer rounding of ⟨k⟩
    costs ±1 bin when a band's energy splits between modes (λ error
    ~2Δk/k); ~5× the matvec cost of the ratio method; verified on
    band-decomposable (e.g. Fourier-diagonal) discretizations.
    """
    power = np.asarray(power, float)
    coords = np.arange(len(power), dtype=float) if coords is None \
        else np.asarray(coords, float)
    kbar, signal = [], []
    for b in bands:
        w = power[b]
        tot = float(w.sum())
        if tot <= 0.0:
            kbar.append(np.nan); signal.append(0.0); continue
        kbar.append(float((coords[b] * w).sum() / tot))
        signal.append(float(np.sqrt(tot)))
    return np.array(kbar), np.array(signal)
