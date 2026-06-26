"""
resona.cloud_flow — the MOVE verb for the NON-HERMITIAN corner.

The biorthogonal spectral Jacobian and complex eigenvalue flow.  Where
`resona.wkernel` owns the Hermitian sensitivity  ∂λ_i/∂k_j = v_iᵀ B_j v_i
(one eigenvector, real spectrum), this module owns its non-self-adjoint
generalization, where the spectrum is COMPLEX and a single eigenvector is
not enough.  For a parametric family

        A(k) = A0 + Σ_j k_j B_j        (A0, B_j NOT assumed symmetric)

standard non-Hermitian (non-normal) perturbation theory gives the EXACT
first-order eigenvalue sensitivity in terms of the LEFT and RIGHT
eigenvectors (the biorthogonal pair):

        ∂λ_i/∂k_j  =  (u_i^* B_j v_i) / (u_i^* v_i),

    A v_i = λ_i v_i        (right eigenvector, column of V)
    u_i^* A = λ_i u_i^*    (left  eigenvector, i.e. A^H u_i = conj(λ_i) u_i)

For a Hermitian A this collapses to u_i = v_i and  u_i^* v_i = 1, recovering
`resona.wkernel.wkernel` exactly.

THE NEW PHYSICS — exceptional points.  The denominator  u_i^* v_i  is the
(unnormalized) PHASE RIGIDITY.  At an EXCEPTIONAL POINT two eigenvalues and
their eigenvectors coalesce; there left ⊥ right, so  u_i^* v_i → 0  and the
sensitivity  ∂λ/∂k → ∞  (it diverges like ε^{-1/2} as a Puiseux series in the
distance ε to the EP).  That blow-up is not a bug: it is the defining feature
of a defective spectrum, and `phase_rigidity` / `exceptional_point` locate it
matrix-free.

HONEST LIMITS (read `resona.cloud` first).  Arnoldi Ritz vectors for a
non-normal operator are NOT reliable far from convergence — the cloud sits in
the numerical range, not on the spectrum.  For TARGETED interior eigenvalues
we therefore use shift-invert Arnoldi (`scipy.sparse.linalg.eigs(sigma=...)`)
and ALWAYS report the two-sided residuals ‖A v − λ v‖ and ‖u^H A − λ u^H‖.
Near an EP the biorthogonal formula legitimately diverges — that is the read,
not an error.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "biorthogonal_eigs",
    "cloud_wkernel",
    "cloud_track",
    "phase_rigidity",
    "exceptional_point",
]


# ──────────────────────────────────────────────────────────────────────────
# left+right eigenpairs near a target, with two-sided residuals
# ──────────────────────────────────────────────────────────────────────────
def _build(A0, Bs, k):
    """Assemble A(k) = A0 + Σ k_j B_j as a dense complex array (for ground
    truth / small problems) — Bs are dense or sparse matrices."""
    import scipy.sparse as sp
    A = np.array(A0, dtype=complex) if not sp.issparse(A0) else A0.toarray().astype(complex)
    for kj, B in zip(k, Bs):
        Bd = B.toarray() if sp.issparse(B) else np.asarray(B)
        A = A + kj * Bd
    return A


def _match_targets(vals, targets):
    """Greedy nearest-value matching of computed eigenvalues to `targets`
    (each target consumed once).  Returns the index array into `vals`."""
    vals = np.asarray(vals, complex)
    idx = []
    used = np.zeros(len(vals), bool)
    for t in np.atleast_1d(targets):
        d = np.abs(vals - t)
        d[used] = np.inf
        j = int(np.argmin(d))
        used[j] = True
        idx.append(j)
    return np.array(idx, int)


def biorthogonal_eigs(A0, Bs, targets, k=None, matrix_free: bool = False,
                      sigma=None, tol: float = 0.0) -> dict:
    """LEFT and RIGHT eigenvectors of A(k) for the eigenvalues nearest each
    target value, with two-sided residuals.

    A0, Bs   : the linear family  A(k) = A0 + Σ k_j B_j.
    targets  : complex values; the returned eigenpairs are the spectral
               eigenvalues NEAREST these (one each).
    k        : parameter point (default zeros → A(0) = A0).
    matrix_free : if True use shift-invert Arnoldi `eigs(sigma=...)` (targeted,
               for large/sparse A); else a dense `scipy.linalg.eig` (ground
               truth, exact biorthogonal pair).
    sigma    : shift for the matrix-free path (default: mean of targets).

    Returns dict with
        vals  (m,)        the targeted eigenvalues λ_i
        R     (N, m)      right eigenvectors v_i (columns)
        L     (N, m)      left  eigenvectors u_i (columns), so u_i^* A = λ_i u_i^*
        denom (m,)        u_i^* v_i  (the phase-rigidity numerator)
        res_r (m,)        ‖A v_i − λ_i v_i‖   (right residual)
        res_l (m,)        ‖u_i^H A − λ_i u_i^H‖ (left residual)
    """
    import scipy.sparse as sp
    targets = np.atleast_1d(np.asarray(targets, complex))
    m = len(targets)
    if k is None:
        k = np.zeros(len(Bs))
    k = np.asarray(k, float)

    # ARPACK shift-invert needs 1 ≤ nev ≤ N−1 (here nev ≥ 1 only if N ≥ 3); at
    # N < 3 it cannot run, so fall back to the exact dense path (no information
    # lost — dense eig IS the ground truth at this size).
    n0 = A0.shape[0]
    if matrix_free and n0 < 3:
        matrix_free = False

    if not matrix_free:
        A = _build(A0, Bs, k)
        # right pairs
        w, V = np.linalg.eig(A)
        # left pairs from A^H:  A^H u = conj(λ) u  ⇒ u^* A = λ u^*
        wL, U = np.linalg.eig(A.conj().T)
        idxR = _match_targets(w, targets)
        vals = w[idxR]
        R = V[:, idxR]
        # match left eigvecs to the SAME eigenvalues (conjugate side)
        idxL = _match_targets(wL.conj(), vals)
        L = U[:, idxL]
        A_for_res = A
    else:
        from scipy.sparse.linalg import eigs, LinearOperator
        if sigma is None:
            sigma = complex(np.mean(targets))
        A = _build(A0, Bs, k)            # used only to form sparse operators
        As = sp.csc_matrix(A)
        kk = k if k is not None else None
        # number of eigenpairs to request around the shift
        nev = max(m + 2, 6)
        nev = min(nev, As.shape[0] - 2)
        valsR, R_all = eigs(As, k=nev, sigma=sigma, tol=tol)
        valsL, L_all = eigs(As.conj().T.tocsc(), k=nev, sigma=np.conj(sigma),
                            tol=tol)
        idxR = _match_targets(valsR, targets)
        vals = valsR[idxR]
        R = R_all[:, idxR]
        idxL = _match_targets(valsL.conj(), vals)
        L = L_all[:, idxL]
        A_for_res = As

    # two-sided residuals
    res_r = np.empty(m)
    res_l = np.empty(m)
    denom = np.empty(m, complex)
    for i in range(m):
        v = R[:, i]
        u = L[:, i]
        v = v / np.linalg.norm(v)
        u = u / np.linalg.norm(u)
        R[:, i] = v
        L[:, i] = u
        Av = A_for_res @ v
        uA = A_for_res.conj().T @ u            # (u^H A)^H = A^H u
        res_r[i] = np.linalg.norm(Av - vals[i] * v)
        res_l[i] = np.linalg.norm(uA - np.conj(vals[i]) * u)
        denom[i] = np.vdot(u, v)               # u^* v = u^H v

    return dict(vals=vals, R=R, L=L, denom=denom, res_r=res_r, res_l=res_l)


# ──────────────────────────────────────────────────────────────────────────
# 1. the biorthogonal spectral Jacobian
# ──────────────────────────────────────────────────────────────────────────
def cloud_wkernel(A0, Bs, targets, k=None, matrix_free: bool = False,
                  sigma=None, return_pairs: bool = False):
    """W[i,j] = (u_i^* B_j v_i) / (u_i^* v_i) = ∂λ_i/∂k_j — the NON-HERMITIAN
    spectral Jacobian for the targeted COMPLEX eigenvalues of
    A(k) = A0 + Σ_j k_j B_j.

    The biorthogonal (left+right eigenvector) generalization of
    `resona.wkernel.wkernel`; reduces to it exactly when A is Hermitian.

    A0, Bs   : the linear family.
    targets  : complex values; W is computed for the eigenvalues nearest them.
    k        : parameter point (default A(0)).
    matrix_free : shift-invert Arnoldi targeting (large/sparse) vs dense eig.
    return_pairs : also return the `biorthogonal_eigs` dict (eigvecs, residuals,
               phase rigidity) — recommended so the caller can check residuals.

    Returns W of shape (m, M)  (complex), or (W, info) if return_pairs.
    """
    import scipy.sparse as sp
    info = biorthogonal_eigs(A0, Bs, targets, k=k, matrix_free=matrix_free,
                             sigma=sigma)
    R, L, denom, vals = info["R"], info["L"], info["denom"], info["vals"]
    m = R.shape[1]
    M = len(Bs)
    W = np.empty((m, M), complex)
    for j, B in enumerate(Bs):
        Bd = B if not sp.issparse(B) else B
        for i in range(m):
            v = R[:, i]
            u = L[:, i]
            Bv = Bd @ v
            W[i, j] = np.vdot(u, Bv) / denom[i]      # (u^* B v)/(u^* v)
    return (W, info) if return_pairs else W


# ──────────────────────────────────────────────────────────────────────────
# 2. complex eigenvalue continuation
# ──────────────────────────────────────────────────────────────────────────
def cloud_track(A0, Bs, path, modes, matrix_free: bool = False, sigma=None,
                return_rigidity: bool = False):
    """Follow chosen COMPLEX eigenvalues λ(k) along a parameter path — the
    non-Hermitian analogue of `resona.wkernel.track`.

    At each path point we re-solve the eigenproblem near the CURRENT λ values
    (continuation in the complex plane): the eigenvalue is tracked by nearest
    -value matching from the previous point, so crossings / complex orbits do
    not scramble the labelling.

    A0, Bs   : the linear family  A(k) = A0 + Σ k_j B_j.
    path     : (T, M) array of parameter points  k(0) … k(T-1).
    modes    : complex target value(s) at path[0]; these eigenvalues are
               followed.  Scalar or length-m sequence.
    matrix_free : shift-invert targeting vs dense eig at each point.
    sigma    : base shift for the matrix-free path (default: tracks the values).
    return_rigidity : also return |u^* v| along the path (EP proximity).

    Returns lams (T, m)  — the tracked complex eigenvalues; or (lams, rig)
    with rig (T, m) = |u_i^* v_i| at each point if return_rigidity.

    HONEST: continuation assumes the step is small enough that nearest-value
    matching is unambiguous — near an EP, where two eigenvalues coalesce, the
    labelling and the sensitivity both legitimately break down (use the
    phase-rigidity read to detect it).
    """
    path = np.asarray(path, float)
    T = path.shape[0]
    cur = np.atleast_1d(np.asarray(modes, complex))
    m = len(cur)
    lams = np.empty((T, m), complex)
    rig = np.empty((T, m))
    for t in range(T):
        sig = sigma if (sigma is not None) else (complex(np.mean(cur)) if matrix_free else None)
        info = biorthogonal_eigs(A0, Bs, cur, k=path[t],
                                 matrix_free=matrix_free, sigma=sig)
        cur = info["vals"]
        lams[t] = cur
        rig[t] = np.abs(info["denom"])
    return (lams, rig) if return_rigidity else lams


# ──────────────────────────────────────────────────────────────────────────
# 3. phase rigidity & exceptional-point detector
# ──────────────────────────────────────────────────────────────────────────
def phase_rigidity(A0, Bs, targets, k=None, matrix_free: bool = False,
                   sigma=None) -> dict:
    """The PHASE RIGIDITY r_i = |u_i^* v_i| / (‖u_i‖ ‖v_i‖) of the targeted
    eigenvalues — the EP proximity gauge.

    For unit-normalized biorthogonal pairs r_i ∈ (0, 1]:  r_i = 1 for a normal
    eigenvalue, r_i → 0 as the eigenvalue approaches an EXCEPTIONAL POINT (left
    and right eigenvectors become orthogonal, the eigenvalue becomes defective,
    and ∂λ/∂k ~ r_i^{-1} diverges).

    Returns dict with  rigidity (m,), vals (m,), denom, res_r, res_l.
    """
    info = biorthogonal_eigs(A0, Bs, targets, k=k, matrix_free=matrix_free,
                             sigma=sigma)
    # R, L already unit-normalized in biorthogonal_eigs → |denom| IS the rigidity
    info["rigidity"] = np.abs(info["denom"])
    return info


def exceptional_point(A0, Bs, bracket, target_pair, n_scan: int = 201,
                      refine: bool = True) -> dict:
    """Locate an EXCEPTIONAL POINT along a ONE-parameter scan, matrix-free in
    the read: scan k over `bracket`, track the two coalescing eigenvalues, and
    find where the phase rigidity |u^* v| is minimal (eigenvectors coalesce).

    A0, Bs       : the family; the scan moves the FIRST parameter (Bs[0]).
    bracket      : (k_lo, k_hi) scan range for k_0.
    target_pair  : the two complex eigenvalues (at k_lo) that coalesce at the EP.
    n_scan       : number of scan points.
    refine       : golden-section refine the rigidity minimum.

    Returns dict:
        k_ep      the parameter value of minimum rigidity (the EP locus)
        rig_min   the minimum |u^* v| reached (→ 0 at a true EP)
        gap_min   the minimum |λ_1 − λ_2| reached (→ 0 at a true EP)
        k_scan, rig_scan, gap_scan   the full scan curves
    """
    klo, khi = bracket
    M = len(Bs)

    def _rig_and_gap(k0, pair):
        # match against `pair` — the LOCALLY-continued eigenvalues, not the
        # original target_pair (which has drifted far from the EP).
        k = np.zeros(M)
        k[0] = k0
        info = biorthogonal_eigs(A0, Bs, pair, k=k)
        return float(np.min(np.abs(info["denom"]))), \
            float(np.abs(info["vals"][0] - info["vals"][1])), info["vals"]

    ks = np.linspace(klo, khi, n_scan)
    rigs = np.empty(n_scan)
    gaps = np.empty(n_scan)
    vals_scan = [None] * n_scan             # the CONTINUED pair at each scan point
    last = np.atleast_1d(np.asarray(target_pair, complex))
    for i, k0 in enumerate(ks):
        k = np.zeros(M); k[0] = k0
        info = biorthogonal_eigs(A0, Bs, last, k=k)
        rigs[i] = float(np.min(np.abs(info["denom"])))
        gaps[i] = float(np.abs(info["vals"][0] - info["vals"][1]))
        last = info["vals"]                 # continuation along the scan
        vals_scan[i] = last

    i0 = int(np.argmin(rigs))
    k_ep = ks[i0]
    rig_min = rigs[i0]
    gap_min = gaps.min()

    if refine and 0 < i0 < n_scan - 1:
        # golden-section on rigidity over the neighbouring bracket, matching
        # against the locally-continued pair at the coarse minimum (NOT the
        # stale target_pair: that is the bug the dead `cur = last` hinted at).
        a, b = ks[i0 - 1], ks[i0 + 1]
        gr = (np.sqrt(5) - 1) / 2
        pair0 = vals_scan[i0]
        def f(k0):
            return _rig_and_gap(k0, pair0)[0]
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(60):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = f(d)
            if abs(b - a) < 1e-12:
                break
        k_ep = 0.5 * (a + b)
        rig_min, gap_min, _ = _rig_and_gap(k_ep, pair0)

    return dict(k_ep=float(k_ep), rig_min=float(rig_min), gap_min=float(gap_min),
                k_scan=ks, rig_scan=rigs, gap_scan=gaps)


# ──────────────────────────────────────────────────────────────────────────
# __main__ demo — numbers vs dense ground truth
# ──────────────────────────────────────────────────────────────────────────
def _demo():
    rng = np.random.default_rng(0)
    print("=" * 70)
    print("MOVE — non-Hermitian spectral Jacobian & flow (cloud_flow)")
    print("=" * 70)

    # ---- 1. biorthogonal ∂λ/∂k vs finite difference on a NON-normal A0 ----
    N = 12
    # strongly non-normal: upper triangular + small random perturbation
    A0 = np.triu(rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    A0 += 0.3 * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    M = 3
    Bs = [rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
          for _ in range(M)]
    # pick a few well-separated eigenvalues as targets
    w0 = np.linalg.eigvals(A0)
    targets = w0[np.argsort(-w0.real)[:4]]

    W, info = cloud_wkernel(A0, Bs, targets, return_pairs=True)
    # central finite difference with eigenvalue continuation
    eps = 1e-6
    Wfd = np.empty_like(W)
    for j in range(M):
        kp = np.zeros(M); kp[j] = eps
        km = np.zeros(M); km[j] = -eps
        vp = biorthogonal_eigs(A0, Bs, info["vals"], k=kp)["vals"]
        vm = biorthogonal_eigs(A0, Bs, info["vals"], k=km)["vals"]
        Wfd[:, j] = (vp - vm) / (2 * eps)
    err = np.max(np.abs(W - Wfd))
    print(f"\n[1] biorthogonal ∂λ/∂k  vs finite difference (non-normal A0)")
    print(f"    targets (Re):     {np.round(targets.real, 3)}")
    print(f"    max |W - W_fd|  = {err:.3e}")
    print(f"    max right resid = {info['res_r'].max():.2e}, "
          f"max left resid = {info['res_l'].max():.2e}")
    print(f"    phase rigidity  = {np.round(np.abs(info['denom']), 3)}")

    # ---- 2. track a complex eigenvalue along a path vs dense eig ----
    T = 9
    path = np.zeros((T, M))
    path[:, 0] = np.linspace(0, 1.0, T)
    path[:, 1] = np.linspace(0, 0.5, T)
    mode0 = targets[0]
    lams = cloud_track(A0, Bs, path, modes=mode0)
    # dense ground truth via independent continuation
    truth = np.empty(T, complex)
    cur = mode0
    for t in range(T):
        A = _build(A0, Bs, path[t])
        w = np.linalg.eigvals(A)
        cur = w[np.argmin(np.abs(w - cur))]
        truth[t] = cur
    terr = np.max(np.abs(lams[:, 0] - truth))
    print(f"\n[2] complex eigenvalue continuation along path (T={T})")
    print(f"    λ start = {mode0:.4f}   λ end = {lams[-1,0]:.4f}")
    print(f"    max |track - dense eig| = {terr:.3e}")

    # ---- 3. EP demo: [[0,1],[k,0]] → λ = ±√k, EP at k=0 ----
    A0_ep = np.array([[0., 1.], [0., 0.]], complex)
    B_ep = np.array([[0., 0.], [1., 0.]], complex)
    # at k = k0 eigenvalues are ±√k0; pick a starting pair away from EP
    k_start = 1.0
    pair = np.array([np.sqrt(k_start), -np.sqrt(k_start)], complex)
    out = exceptional_point(A0_ep, [B_ep], bracket=(1.0, -1e-9),
                            target_pair=pair, n_scan=401)
    # sensitivity blow-up: W at small k
    kspots = [1.0, 1e-2, 1e-4, 1e-6]
    print(f"\n[3] exceptional point of [[0,1],[k,0]] (λ=±√k, EP at k=0)")
    print(f"    located k_ep   = {out['k_ep']:.3e}  "
          f"(true EP at k=0)")
    print(f"    min |u*v|      = {out['rig_min']:.3e}  (→0 at EP)")
    print(f"    min |λ1-λ2|    = {out['gap_min']:.3e}  (→0 at EP)")
    print(f"    |∂λ/∂k| blow-up as k→0:")
    for ks in kspots:
        W = cloud_wkernel(A0_ep, [B_ep], np.array([np.sqrt(ks)]),
                          k=np.array([ks]))
        # analytic ∂(√k)/∂k = 1/(2√k)
        print(f"      k={ks:8.0e} : |W|={abs(W[0,0]):11.3e}  "
              f"theory 1/(2√k)={1/(2*np.sqrt(ks)):11.3e}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    _demo()
