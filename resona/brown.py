"""
resona.brown — the COMPOSE / plane-READ verb for the non-Hermitian corner: the
BROWN MEASURE (Haagerup–Larsen / Girko).

`cloud` (resona.cloud) gives the Arnoldi Ritz cloud of a non-normal operator —
honestly labelled as an UNRELIABLE read (the Ritz values sit in the numerical
range, not on the spectrum; the gap IS the pseudospectrum).  This module gives
the RELIABLE 2-D law that the cloud only hints at: the BROWN MEASURE μ_A, the
right notion of "eigenvalue distribution in the complex plane" for a
non-self-adjoint operator.

THE MATH
--------
For a non-normal A the eigenvalues fill a 2-D region of the plane.  The Brown
measure is built by HERMITIZATION:

    H(z) = [[0,        A − zI],
            [(A−zI)*,  0      ]]            (Hermitian, size 2N)

whose eigenvalues are ±σ_i(A − zI), the singular values of A − zI.  From the
symmetrized singular-value distribution one reads the LOG-POTENTIAL

    S(z) = (1/N) log|det(A − zI)| = (1/N) Σ_i log σ_i(A − zI),

and the Brown measure is its (distributional) Laplacian over the plane z=x+iy:

    μ_A = (1/2π) Δ_z S(z).

MATRIX-FREE S(z)
----------------
    S(z) = (1/2N) · Tr log( (A − zI)*(A − zI) ),

a matrix-free log-det of the Hermitian PSD operator (A−z)*(A−z) — exactly
resona's SLQ Tr(log(·)) territory (`resona.of(matvec, N).trace(np.log)`).

We REALIFY the complex operator (A−z) to a real operator on R^{2N}: a complex
vector w = u + i v is carried as [u; v] ∈ R^{2N}, and the *complex* action
A·(u+iv) = A u + i A v is read through real probes (Re / Im of A applied to the
real sub-blocks).  The realified (A−z)^T(A−z) is real-symmetric PSD and its 2N
eigenvalues are the N values σ_i² EACH DOUBLED, so

    Tr log( realified (A−z)^T(A−z) )  =  2 Σ_i log σ_i²  =  4 Σ_i log σ_i,

and therefore  S(z) = Tr_log / (4N).  μ_A then follows from a 5-point numerical
Laplacian of S on a complex grid.

(IMPORTANT — this module does NOT reuse `resona.defect.sigma_min`'s realified
operator: that helper feeds the *complex* matvec output back into a real block
decomposition and silently casts away the imaginary part, so it returns
σ_min = 0 for a genuinely complex A.  The realification below is the corrected
one and is verified eigenvalue-for-eigenvalue against dense SVD in the tests.)

A REGULARISER η.  Right on supp μ_A the smallest singular value σ_min(A−z) → 0,
so log σ is singular and the SLQ estimate is ill-conditioned.  We optionally
compute  S_η(z) = (1/2N) Tr log( (A−z)*(A−z) + η² ); η = 0 is the bare
log-potential.  The η used is reported by the caller.

HONEST (the hard rule).  The Brown measure equals the empirical eigenvalue
distribution of A only under conditions; for a non-normal A the eigenvalues are
pseudospectrally UNSTABLE (a tiny perturbation moves them a lot), so eig(A) is
itself an unreliable "ground truth".  μ_A is the STABLE object.  The SLQ
Tr(log) is a STOCHASTIC estimate — never exact — and log is ill-conditioned
exactly at the σ_min → 0 ridge (on supp μ_A), where more probes / Lanczos steps
(and an η) are needed.  This module reports numbers, not headlines, and uses the
CIRCULAR LAW (where the law is clean and analytically known) as the primary
check.
"""
import numpy as np

__all__ = ["log_potential", "brown_measure", "brown_boxplus"]


# ─────────────────────────────────────────────────────────────────────────────
# Realification of (A − zI) and its Gram operator (A−zI)^*(A−zI) on R^{2N}.
#
# Carry w = u + i v ∈ C^N as [u; v] ∈ R^{2N}.  For a complex linear map A and a
# REAL vector u, A u is a complex vector whose real / imaginary parts give the
# real and imaginary ACTION of A on that vector — that is all we need, and it is
# obtained with the black-box matvec alone (no access to A's entries).  With
# z = zr + i zi,
#
#     (A − z)(u + i v) = [ Re(Au) − Im(Av) − zr u + zi v ]
#                      + i[ Im(Au) + Re(Av) − zi u − zr v ].
#
# The adjoint (A − z)^* uses rmatvec = A^*·x and flips the z-imaginary signs:
#
#     (A − z)^*(u + i v) = [ Re(A^*u) − Im(A^*v) − zr u − zi v ]
#                        + i[ Im(A^*u) + Re(A^*v) + zi u − zr v ].
# ─────────────────────────────────────────────────────────────────────────────
def _realified_AHA(matvec, rmatvec, N, z):
    """The realified Hermitian PSD operator (A − zI)^*(A − zI) on R^{2N}.

    Returns a callable x ↦ (A−z)^*((A−z)·x) acting on REAL x ∈ R^{2N}.  The
    operator is real-symmetric PSD; its 2N eigenvalues are the N singular values
    of (A − zI) SQUARED, each appearing twice (verified vs dense SVD in tests).
    """
    zr, zi = float(np.real(z)), float(np.imag(z))

    def B(x):                                    # (A − zI) on realified C^N
        u, v = x[:N], x[N:]
        Au, Av = matvec(u), matvec(v)            # complex C^N each
        re = np.real(Au) - np.imag(Av) - zr * u + zi * v
        im = np.imag(Au) + np.real(Av) - zi * u - zr * v
        return np.concatenate([re, im])

    def Bt(x):                                   # (A − zI)^* on realified C^N
        u, v = x[:N], x[N:]
        Au, Av = rmatvec(u), rmatvec(v)
        re = np.real(Au) - np.imag(Av) - zr * u - zi * v
        im = np.imag(Au) + np.real(Av) + zi * u - zr * v
        return np.concatenate([re, im])

    return lambda x: Bt(B(x))


def _coerce(A, N, rmatvec):
    """Return (matvec, rmatvec, N) for a dense array or matvec+rmatvec pair.

    The matvecs are wrapped so they ALWAYS receive a complex input and return a
    complex ndarray (the realification reads Re/Im of the output).
    """
    if callable(A):
        if N is None:
            raise ValueError("brown: a matvec needs N (and rmatvec = A^*·x for a "
                             "non-symmetric / non-Hermitian operator)")
        rmv = rmatvec if rmatvec is not None else A     # symmetric default
        mv_c = lambda x, f=A: np.asarray(f(np.asarray(x, complex)), complex)
        rmv_c = lambda x, f=rmv: np.asarray(f(np.asarray(x, complex)), complex)
        return mv_c, rmv_c, int(N)
    M = np.asarray(A, complex)
    n = M.shape[0]
    Mc = M.conj().T
    return (lambda x: M @ np.asarray(x, complex)), \
           (lambda x: Mc @ np.asarray(x, complex)), n


def log_potential(A, z, N=None, rmatvec=None, k=64, probes=16, seed=0,
                  eta=0.0, exact=False):
    """The log-potential  S(z) = (1/N) Σ_i log σ_i(A − zI)  at one or many z.

    A        : dense square ndarray, OR a matvec callable (then pass N, and
               rmatvec = A^*·x for a non-symmetric/non-Hermitian operator).
    z        : scalar or array of complex grid points.
    eta      : regulariser; returns (1/2N) Tr log((A−z)^*(A−z) + η²).  η = 0 is
               the bare log-potential.  Raise η near the σ_min→0 ridge.
    exact    : if True and A is dense, use the exact SVD per z (ground truth);
               otherwise the matrix-free SLQ Tr(log) estimate.

    Returns S(z) (float for scalar z, ndarray matching z otherwise).

    HONEST: the SLQ path is STOCHASTIC (a Hutchinson/Lanczos-quadrature estimate
    of Tr log), and log is ill-conditioned where σ_min(A−z) → 0 — i.e. ON the
    support of the Brown measure.  Raise `probes`/`k`/`eta` there; the exact
    path is the per-z dense ground truth.
    """
    zs = np.atleast_1d(np.asarray(z, complex))
    scalar = np.ndim(z) == 0
    eta2 = float(eta) ** 2

    if exact:
        if callable(A):
            raise ValueError("log_potential(exact=True) needs a dense array A")
        M = np.asarray(A, complex)
        n = M.shape[0]
        I = np.eye(n)
        out = np.empty(len(zs))
        for i, zz in enumerate(zs):
            sig = np.linalg.svd(M - zz * I, compute_uv=False)
            out[i] = float(np.mean(0.5 * np.log(sig ** 2 + eta2)))
        return float(out[0]) if scalar else out

    import resona
    matvec, rmv, n = _coerce(A, N, rmatvec)
    f = np.log if eta2 == 0.0 else (lambda x: np.log(x + eta2))
    out = np.empty(len(zs))
    for i, zz in enumerate(zs):
        AHA = _realified_AHA(matvec, rmv, n, zz)
        s = resona.of(AHA, 2 * n, k=k, probes=probes, seed=seed)
        trlog = s.trace(f)                       # Tr log = 4 Σ_i log σ_i (η=0)
        out[i] = trlog / (4.0 * n)
    return float(out[0]) if scalar else out


def _grid(grid):
    """Normalize the `grid` argument → (X, Y, Z, hx, hy) on a regular mesh.

    `grid` may be (xmin, xmax, ymin, ymax, n)  — a square n×n mesh, or
            (xs, ys)                            — two 1-D coordinate arrays.
    """
    if len(grid) == 5:
        xmin, xmax, ymin, ymax, n = grid
        xs = np.linspace(xmin, xmax, int(n))
        ys = np.linspace(ymin, ymax, int(n))
    elif len(grid) == 2:
        xs, ys = np.asarray(grid[0], float), np.asarray(grid[1], float)
    else:
        raise ValueError("grid = (xmin,xmax,ymin,ymax,n) or (xs, ys)")
    X, Y = np.meshgrid(xs, ys)                   # X[i,j]=xs[j], Y[i,j]=ys[i]
    Z = X + 1j * Y
    hx = float(xs[1] - xs[0])
    hy = float(ys[1] - ys[0])
    return X, Y, Z, hx, hy


def brown_measure(A, N=None, grid=(-1.6, 1.6, -1.6, 1.6, 41), rmatvec=None,
                  k=64, probes=16, seed=0, eta=0.0, exact=False):
    """The Brown measure μ_A on a complex grid — the matrix-free eigenvalue
    DENSITY in the plane.

    Computes the log-potential S(z) = (1/N) Σ log σ_i(A − zI) on the grid
    (matrix-free SLQ Tr(log), or exact SVD with `exact=True` on a dense A), then

        μ_A = (1/2π) Δ S            (5-point numerical Laplacian over the plane).

    A      : dense square ndarray, OR a matvec callable (pass N and rmatvec=A^*·x).
    grid   : (xmin,xmax,ymin,ymax,n) for a square mesh, or (xs, ys) coordinates.
    eta    : log-potential regulariser (see `log_potential`).
    exact  : per-z exact SVD ground truth (dense A only).

    Returns a dict:
        'X','Y'  — the mesh (meshgrid form);
        'Z'      — X + iY;
        'S'      — the log-potential on the grid;
        'mu'     — μ_A density (clipped ≥ 0; the raw Laplacian can dip slightly
                   negative from SLQ noise / discretization — that dip is the
                   honest error bar, not a measure);
        'mu_raw' — the unclipped Laplacian (1/2π)ΔS, for diagnostics;
        'mass'   — ∫ mu dx dy over the grid (≈ 1 if the support is inside the
                   box; < 1 means the box clipped the support).

    HONEST: μ_A is the STABLE eigenvalue law; for a non-normal A the dense
    eig(A) is pseudospectrally unstable and is NOT a trustworthy ground truth.
    The SLQ S(z) is stochastic and log-singular on supp μ_A — read `mu` as a
    density estimate with the noise its own caveat.  Use the circular law as the
    clean check.
    """
    X, Y, Z, hx, hy = _grid(grid)
    flatZ = Z.ravel()
    S = log_potential(A, flatZ, N=N, rmatvec=rmatvec, k=k, probes=probes,
                      seed=seed, eta=eta, exact=exact).reshape(Z.shape)

    # 5-point Laplacian  ΔS ≈ (S_xx + S_yy); interior only, edges → 0.
    lap = np.zeros_like(S)
    lap[1:-1, 1:-1] = (
        (S[1:-1, 2:] - 2 * S[1:-1, 1:-1] + S[1:-1, :-2]) / hx ** 2 +   # ∂²/∂x²
        (S[2:, 1:-1] - 2 * S[1:-1, 1:-1] + S[:-2, 1:-1]) / hy ** 2)    # ∂²/∂y²
    mu_raw = lap / (2.0 * np.pi)
    mu = np.maximum(mu_raw, 0.0)
    mass = float(mu.sum() * hx * hy)
    return {"X": X, "Y": Y, "Z": Z, "S": S, "mu": mu, "mu_raw": mu_raw,
            "mass": mass}


# ─────────────────────────────────────────────────────────────────────────────
# brown_boxplus — the Brown measure of a *-FREE sum A + B in the plane.
#
# For *-free A, B the symmetrized singular distribution of (A+B)−z is obtained by
# HERMITIZING both operands at the SAME z and freely ADDING the two Hermitian
# 2N×2N hermitizations H_A(z) ⊞ H_B(z) (Hermitian free additive convolution).
# From the free-sum density of states ρ we read S_{A+B}(z) = ∫ log|t| ρ(t) dt,
# and μ_{A+B} = (1/2π) Δ S as before.  ρ is computed by the Belinschi–Bercovici
# subordination fixed point on the two SLQ singular measures (no matrix formed).
# ─────────────────────────────────────────────────────────────────────────────
def _signed_sv_measure(matvec, rmatvec, N, z, k, probes, seed):
    """SLQ symmetrized signed-singular measure (t, w) of (A − zI).

    The hermitization eigenvalues are ±σ_i.  We read the SLQ measure of the
    realified (A−z)^*(A−z) (nodes ≈ σ², each doubled) and map σ² → ±σ, splitting
    each weight in half — giving the symmetric measure ν_{A−z} on the real line.
    """
    import resona
    AHA = _realified_AHA(matvec, rmatvec, N, z)
    s = resona.of(AHA, 2 * N, k=k, probes=probes, seed=seed)
    sig = np.sqrt(np.clip(np.asarray(s.nodes, float), 0.0, None))
    w = np.asarray(s.weights, float)
    t = np.concatenate([sig, -sig])
    ww = np.concatenate([w, w]) * 0.5
    return t, ww / ww.sum()


def _free_add_dos(tA, wA, tB, wB, xs, eta, iters=600, tol=1e-12):
    """Density of states of the free additive convolution μ_A ⊞ μ_B on `xs`.

    Belinschi–Bercovici subordination fixed point: there are ω_A(z), ω_B(z) in
    the upper half-plane with
        G_{A⊞B}(z) = G_A(ω_A(z)) = G_B(ω_B(z)),
        ω_A(z) + ω_B(z) = z + 1/G_{A⊞B}(z),
    equivalently, with h_X = 1/G_X − id,
        ω_A = z + h_B(ω_B),   ω_B = z + h_A(ω_A).
    Solved by a damped iteration on (ω_A, ω_B), VECTORIZED over the whole
    frequency grid (each Cauchy transform is an outer sum over atoms).
    ρ(x) = −Im G_A(ω_A(x + iη)) / π.
    """
    tA = np.asarray(tA, float); wA = np.asarray(wA, float)
    tB = np.asarray(tB, float); wB = np.asarray(wB, float)

    def GA(zc):
        return (wA[None, :] / (zc[:, None] - tA[None, :])).sum(1)

    def GB(zc):
        return (wB[None, :] / (zc[:, None] - tB[None, :])).sum(1)

    z = np.asarray(xs, float) + 1j * eta
    wAo = z.copy()
    wBo = z.copy()
    for _ in range(iters):
        nA = z + (1.0 / GB(wBo) - wBo)                # z + h_B(ω_B)
        nA = nA.real + 1j * np.maximum(nA.imag, eta)  # keep in upper half-plane
        nB = z + (1.0 / GA(nA) - nA)                  # z + h_A(ω_A)
        nB = nB.real + 1j * np.maximum(nB.imag, eta)
        if np.max(np.abs(nA - wAo) + np.abs(nB - wBo)) < tol:
            wAo, wBo = nA, nB
            break
        wAo = 0.5 * wAo + 0.5 * nA
        wBo = 0.5 * wBo + 0.5 * nB
    G = GA(wAo)
    return np.maximum(-G.imag / np.pi, 0.0)


def _bin_atoms(t, w, nbins=80):
    """Collapse an atomic measure (t, w) to ≤nbins weighted bins — the per-z free
    convolution cost is O(nfreq · #atoms), so binning 2N atoms → ~80 is a ~Nx win
    with no measurable loss (the DOS is η-broadened anyway)."""
    t = np.asarray(t, float); w = np.asarray(w, float)
    if len(t) <= nbins:
        return t, w
    lo, hi = float(t.min()), float(t.max())
    if hi <= lo:
        return np.array([lo]), np.array([w.sum()])
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, nbins - 1)
    wb = np.zeros(nbins); tb = np.zeros(nbins)
    np.add.at(wb, idx, w)
    np.add.at(tb, idx, t * w)
    m = wb > 0
    return tb[m] / wb[m], wb[m]


def _sv_measure_dense(Ad, z, N):
    """Symmetrized singular measure ν_{A−z} (atoms ±σ_i) by a direct dense SVD —
    fast for small N, the clean ground-truth path (no SLQ stochasticity)."""
    sig = np.linalg.svd(Ad - z * np.eye(N), compute_uv=False)
    t = np.concatenate([sig, -sig])
    w = np.full(2 * N, 0.5 / N)
    return t, w


def brown_boxplus(A, B, N=None, grid=(-2.2, 2.2, -2.2, 2.2, 21),
                  rmatvecA=None, rmatvecB=None, k=64, probes=16, seed=0,
                  eta_free=1e-2, sigma_floor=1e-3, span_pad=0.5, nfreq=121,
                  exact=False, nbins=80):
    """(STRETCH) The Brown measure of a *-FREE sum A + B, from the per-z free
    convolution of the two HERMITIZED symmetrized singular distributions.

    At each grid point z we hermitize A and B separately, read each symmetrized
    singular measure ν_{A−z}, ν_{B−z} (eigenvalues ±σ_i), free-ADD the two
    Hermitian measures with the Belinschi–Bercovici subordination fixed point
    (`_free_add_dos`), and read the log-potential

        S_{A+B}(z) = ∫ log|x| · ρ_{A+B−z}(x) dx,

    then μ_{A+B} = (1/2π) Δ S as in `brown_measure`.

    Returns the same dict as `brown_measure`.

    HONEST STATUS — see the module report.  The σ→0 ridge log-integral is
    regularised at `sigma_floor` and the DOS is η-broadened at `eta_free`; both
    bias S a little.  This is a sanity-level free-sum read, not a sharp law.
    """
    matA, rmvA, n = _coerce(A, N, rmatvecA)
    matB, rmvB, nB = _coerce(B, N, rmatvecB)
    if n != nB:
        raise ValueError(f"A and B must act on the same dimension ({n} vs {nB})")
    Ad = np.asarray(A) if exact else None        # dense fast path for the symm. measure
    Bd = np.asarray(B) if exact else None

    X, Y, Z, hx, hy = _grid(grid)
    flatZ = Z.ravel()
    S = np.empty(len(flatZ))
    for i, zz in enumerate(flatZ):
        if exact:                                # direct SVD — no SLQ, clean & fast
            tA, wA = _sv_measure_dense(Ad, zz, n)
            tB, wB = _sv_measure_dense(Bd, zz, n)
        else:
            tA, wA = _signed_sv_measure(matA, rmvA, n, zz, k, probes, seed)
            tB, wB = _signed_sv_measure(matB, rmvB, n, zz, k, probes, seed)
        tA, wA = _bin_atoms(tA, wA, nbins)       # 2N atoms → ~nbins: ~Nx cheaper free conv
        tB, wB = _bin_atoms(tB, wB, nbins)
        span = float(np.max(np.abs(tA)) + np.max(np.abs(tB))) + span_pad
        xs = np.linspace(-span, span, int(nfreq))
        dos = _free_add_dos(tA, wA, tB, wB, xs, eta_free, iters=120)
        Z0 = np.trapz(dos, xs)
        if Z0 > 0:
            dos = dos / Z0                       # renormalize the η-broadened DOS
        xr = np.where(np.abs(xs) < sigma_floor, sigma_floor, np.abs(xs))
        S[i] = float(np.trapz(np.log(xr) * dos, xs))

    S = S.reshape(Z.shape)
    lap = np.zeros_like(S)
    lap[1:-1, 1:-1] = (
        (S[1:-1, 2:] - 2 * S[1:-1, 1:-1] + S[1:-1, :-2]) / hx ** 2 +
        (S[2:, 1:-1] - 2 * S[1:-1, 1:-1] + S[:-2, 1:-1]) / hy ** 2)
    mu_raw = lap / (2.0 * np.pi)
    mu = np.maximum(mu_raw, 0.0)
    mass = float(mu.sum() * hx * hy)
    return {"X": X, "Y": Y, "Z": Z, "S": S, "mu": mu, "mu_raw": mu_raw,
            "mass": mass}


# ─────────────────────────────────────────────────────────────────────────────
# Demo / self-check — run:  PYTHONPATH=. python3 -m resona.brown
# ─────────────────────────────────────────────────────────────────────────────
def _demo():
    import resona  # noqa: F401

    def ginibre(N, seed=0):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((N, N))
                + 1j * rng.standard_normal((N, N))) / np.sqrt(2 * N)

    print("=" * 72)
    print("resona.brown — Brown measure / circular-law demo")
    print("=" * 72)

    # --- matrix-free S(z) vs dense ground truth ---
    N = 80
    A = ginibre(N, seed=1)
    zs = np.array([0.0, 0.3 + 0.2j, -0.5 + 0.4j, 0.9 + 0.0j, 1.3 + 0.0j])
    S_dense = log_potential(A, zs, exact=True)
    S_mf = log_potential(A, zs, k=80, probes=24, seed=3, eta=1e-3)
    err = np.abs(S_mf - S_dense)
    print("\n[matrix-free S(z) vs dense]  (k=80, probes=24, eta=1e-3)")
    for zz, sd, sm, e in zip(zs, S_dense, S_mf, err):
        print(f"   z={zz:+.2f}  dense={sd:+.4f}  mf={sm:+.4f}  |err|={e:.3e}")
    print(f"   max abs err = {err.max():.3e}")

    # --- circular law (exact path = clean ground truth) ---
    print("\n[CIRCULAR LAW]  Ginibre N=200, exact per-z SVD log-potential")
    A2 = ginibre(200, seed=5)
    res = brown_measure(A2, grid=(-1.6, 1.6, -1.6, 1.6, 65), exact=True)
    X, Y, mu, mass = res["X"], res["Y"], res["mu"], res["mass"]
    R = np.hypot(X, Y)
    cell = (X[0, 1] - X[0, 0]) ** 2
    mu_in = mu[R < 0.8].mean()
    mu_out = mu[R > 1.25].mean()
    mass_in = float(mu[R < 1.0].sum() * cell)
    mass_out = float(mu[R > 1.0].sum() * cell)
    ev = np.linalg.eigvals(A2)
    print(f"   interior density mu_in  = {mu_in:.4f}  (target 1/pi = {1/np.pi:.4f})")
    print(f"   exterior density mu_out = {mu_out:.4f}  (target 0)")
    print(f"   total mass = {mass:.4f}   mass inside disk = {mass_in:.4f}   "
          f"outside = {mass_out:.4f}")
    print(f"   spectral radius max|eig(A)| = {np.max(np.abs(ev)):.4f}  (target 1)")

    # --- matrix-free circular law ---
    print("\n[CIRCULAR LAW — matrix-free SLQ]  Ginibre N=120")
    A3 = ginibre(120, seed=7)
    resm = brown_measure(A3, grid=(-1.6, 1.6, -1.6, 1.6, 41),
                         k=70, probes=20, seed=4, eta=1e-2)
    Rm = np.hypot(resm["X"], resm["Y"])
    print(f"   mf interior density = {resm['mu'][Rm < 0.7].mean():.4f}  "
          f"(target {1/np.pi:.4f})")
    print(f"   mf exterior density = {resm['mu'][Rm > 1.3].mean():.4f}")
    print(f"   mf total mass = {resm['mass']:.4f}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    _demo()
