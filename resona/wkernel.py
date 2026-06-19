"""
resona.wkernel — the spectral Jacobian  W[i,j] = ∂λ_i/∂k_j  (Hellmann–Feynman).

The conjugate of the matrix-free response.  For a parametric operator
A(k) = A0 + Σ_j k_j B_j, perturbation theory gives the EXACT first-order
eigenvalue sensitivity (Hellmann–Feynman):

        ∂λ_i/∂k_j  =  v_iᵀ B_j v_i ,        v_i = eigenvector of λ_i.

W needs the eigenvectors (the "expensive" side of the program — O(N³) in general,
but only a few modes via Lanczos for the bottom/top of the spectrum), and in
return gives the exact spectral DESIGN map: predict how the spectrum shifts under
any parameter change, and INVERT it to choose k that hits a target spectrum.
This is the W-kernel that the FA program is built on, made a first-class primitive.
"""
import numpy as np


def wkernel(eigvecs, perturbations):
    """W[i,j] = v_iᵀ B_j v_i = ∂λ_i/∂k_j.

    eigvecs       : (N, m) array, columns are the eigenvectors of interest.
    perturbations : list of the B_j (each a dense/sparse matrix OR a matvec callable).
    returns       : (m, M) spectral Jacobian.
    """
    V = np.asarray(eigvecs, float)
    m = V.shape[1]
    W = np.empty((m, len(perturbations)))
    for j, B in enumerate(perturbations):
        mv = B if callable(B) else (lambda x, B=B: B @ x)
        for i in range(m):
            v = V[:, i]
            W[i, j] = float(v @ mv(v))
    return W


def design(W, target_shift, reg=0.0, rcond=None):
    """Parameter step dk so that W·dk ≈ target_shift (λ_target − λ_0).

    One Hellmann–Feynman design step — no finite differences, no per-parameter
    eigensolve.  Iterate (recompute W at the new k) for nonlinear targets.

    reg > 0 adds TIKHONOV regularization (relative to the largest singular value²):
    dk = Σ s_i/(s_i²+reg·s_max²) (u_iᵀy) v_i.  This is the bias–variance dial of an
    ILL-POSED inverse (e.g. inverse spectral / conductivity): reg=0 is the exact
    least-squares step (machine-precise when well-posed, but blows up on
    rank-deficient / under-determined W); reg>0 is bounded and ROBUST, recovering a
    SMOOTHED solution — the trade-off a regularized full-response inverse makes.
    """
    W = np.asarray(W, float); y = np.asarray(target_shift, float)
    if reg <= 0:
        return np.linalg.lstsq(W, y, rcond=rcond)[0]
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    return Vt.T @ ((s / (s ** 2 + reg * s[0] ** 2)) * (U.T @ y))


def track(A0, perturbations, path, steps=1, modes="all", guard=True):
    """Integrate the spectral flow  dλ = W(k)·dk  along a parameter path —
    eigenvalues followed by EIGENVECTOR CONTINUATION, so crossings do not
    scramble them (sorted eigenvalues break by O(1) at a crossing; the
    tracked path stays exact — measured 8.9e-15 vs 3.3 in the source
    program's criterion C9).

    Family: wkernel owns ∂λ_i/∂k_j (Hellmann–Feynman); `track` is its line
    integral.  (`design` is the INVERSE step: target Δλ → Δk.)

    A0            : base matrix (the family is LINEAR: A(k) = A0 + Σ k_j B_j).
    perturbations : list of B_j (matrices or matvec callables).
    path          : (T, M) array of parameter points, k(0) … k(T−1).
    steps         : midpoint sub-steps per path segment.

    ``modes`` — the MATRIX-FREE dial:
        'all' (default) → ALL N eigenvalues, dense eigh per midpoint (O(N³));
        k>0  → track only the BOTTOM-k modes via `eigsh` (matrix-free, O(N·k) for
               sparse A0/B); k<0 → the TOP-|k|.  Returns lams of shape (T, |k|).
    ``guard`` (modes≠'all') — warn if a tracked mode is about to LEAVE the block
        (the gap to mode |k|+1 closes): selected-block continuation is exact only
        while the mode group stays spectrally isolated (measured 1e-15 when gapped,
        ~1e-3 when a mode crosses the boundary).  Raise |modes| or use 'all' then.

    Returns (lams, V): lams[(T, m)] — eigenvalues along the path in CONSISTENT
    order (m = N for 'all', else |modes|); V — eigenvectors at the final point.

    HONEST LIMITS: 'all' is dense (one eigh per midpoint — ~100× more accurate
    than frozen-W per eigh, 44–302× fewer eigh than finite-difference); frozen-W
    error grows as C_H·‖dk‖² between refreshes (Theorem E) — `kappa_w` is that dial.
    """
    import scipy.sparse as sp
    path = np.asarray(path, float)
    Bs = list(perturbations)

    if modes == "all":
        A0 = np.asarray(A0, float)
        if not np.allclose(A0, A0.T, atol=1e-10):
            raise ValueError("track's verified domain is SYMMETRIC linear families "
                             "(eigh-based continuation); A0 is not symmetric — for "
                             "non-Hermitian spectra see resona.cloud")
        if any(callable(B) for B in Bs):
            raise ValueError("track needs explicit matrices for the linear family "
                             "A(k) = A0 + Σ k_j B_j (callables hide the domain)")
        Bstack = np.stack([np.asarray(B, float) for B in Bs])
        build = lambda k: A0 + np.tensordot(k, Bstack, axes=1)
        lam, V = np.linalg.eigh(build(path[0]))
        lams = [lam.copy()]
        for seg in range(1, len(path)):
            for s in range(steps):
                a = path[seg - 1] + (path[seg] - path[seg - 1]) * s / steps
                b = path[seg - 1] + (path[seg] - path[seg - 1]) * (s + 1) / steps
                _, Vm = np.linalg.eigh(build(0.5 * (a + b)))
                perm = np.argmax(np.abs(V.T @ Vm), axis=1)
                Wm = np.einsum('ni,mni->im', Vm[:, perm], Bstack @ Vm[:, perm])
                lam = lam + Wm @ (b - a)
                V = Vm[:, perm]
            lams.append(lam.copy())
        return np.array(lams), V

    # ── MATRIX-FREE selected-block path ──────────────────────────────────────
    import warnings
    from scipy.sparse.linalg import eigsh
    m = abs(int(modes)); which = "SA" if modes > 0 else "LA"
    sparse_in = sp.issparse(A0)
    Mp = path.shape[1]
    kg = m + 1 if guard else m

    def build(k):
        if sparse_in:
            return (A0 + sum(float(k[j]) * Bs[j] for j in range(Mp))).tocsc()
        return np.asarray(A0, float) + sum(float(k[j]) * np.asarray(Bs[j], float)
                                           for j in range(Mp))

    def Bmv(j, v):
        B = Bs[j]; return B(v) if callable(B) else (B @ v)

    vals0, V0 = eigsh(build(path[0]), k=kg, which=which)
    o = np.argsort(vals0); vals0, V0 = vals0[o], V0[:, o]
    V = V0[:, :m]; lam = vals0[:m].copy(); lams = [lam.copy()]
    warned = False
    for seg in range(1, len(path)):
        for s in range(steps):
            a = path[seg - 1] + (path[seg] - path[seg - 1]) * s / steps
            b = path[seg - 1] + (path[seg] - path[seg - 1]) * (s + 1) / steps
            vm, Vm = eigsh(build(0.5 * (a + b)), k=kg, which=which)
            o = np.argsort(vm); vm, Vm = vm[o], Vm[:, o]
            if guard and not warned and (vm[m] - vm[m - 1]) < 1e-3 * max(vm[-1] - vm[0], 1e-30):
                warnings.warn(f"track(modes={modes}): a mode is leaving the selected "
                              "block (boundary gap closing) — selected-block tracking "
                              "may lose accuracy; raise |modes| or use modes='all'.")
                warned = True
            Vm = Vm[:, :m]
            perm = np.argmax(np.abs(V.T @ Vm), axis=1)
            Vsel = Vm[:, perm]
            Wm = np.array([[float(Vsel[:, i] @ Bmv(j, Vsel[:, i])) for j in range(Mp)]
                           for i in range(m)])
            lam = lam + Wm @ (b - a)
            V = Vsel
        lams.append(lam.copy())
    return np.array(lams), V


def _selected_W(A0, perturbations, k, modes):
    """W-block of the `modes` selected eigenvectors — MATRIX-FREE when modes≠'all'.

    modes : 'all'  → dense eigh, W over ALL eigenvectors (the original, O(N³));
            k>0    → the BOTTOM k eigenpairs via Lanczos shift (`eigsh`, which='SA');
            k<0    → the TOP |k| eigenpairs (which='LA').
    For the selected block this needs only matvecs (sparse A0/B → genuinely
    matrix-free, O(N·k) ≪ O(N³)).  Returns W = (m, M).
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    Bs = list(perturbations)
    M = len(k)
    if modes == "all":
        A0d = np.asarray(A0, float)
        Bstack = np.stack([np.asarray(B, float) for B in Bs])
        _, V = np.linalg.eigh(A0d + np.tensordot(k, Bstack, axes=1))
        return np.einsum('ni,mni->im', V, Bstack @ V)
    m = abs(int(modes)); which = "SA" if modes > 0 else "LA"
    if sp.issparse(A0):
        A = (A0 + sum(float(k[j]) * Bs[j] for j in range(M))).tocsc()
    else:
        A = np.asarray(A0, float) + sum(float(k[j]) * np.asarray(Bs[j], float)
                                        for j in range(M))
    _, V = eigsh(A, k=m, which=which)
    def Bmv(j, v):
        B = Bs[j]; return B(v) if callable(B) else (B @ v)
    return np.array([[float(V[:, i] @ Bmv(j, V[:, i])) for j in range(M)]
                     for i in range(m)])


def kappa_w(A0, perturbations, k0, eps=1e-5, probes=8, seed=0, full=False, modes="all"):
    """κ_W — the local curvature of the spectral-flow kernel:
    max over random unit directions of ‖W(k₀+εu) − W(k₀)‖_F / ε.

    WHAT IT PREDICTS (and the first thing to know): the ACCURACY of a
    frozen-W prediction over a step (Spearman ρ = 0.929 against measured
    frozen-W error, criterion C8).  WHAT IT DOES NOT PREDICT: global
    computational cost — on blind random families ρ(κ_W, cost) ≈ 0.05
    across fresh seeds; cost follows the PATH LENGTH, not the local
    curvature.  Use κ_W to size a trust region for `track`/`design` steps,
    never as a difficulty oracle.

    ``full=True`` → (max, values): the whole per-direction distribution.

    ``modes`` — the MATRIX-FREE dial (the conjugate W-side, completed):
        'all' (default) → curvature of the FULL spectral Jacobian (dense eigh, O(N³));
        k>0  → curvature of the BOTTOM-k modes' sub-Jacobian, MATRIX-FREE via `eigsh`
               (only matvecs — for SPARSE A0/B this is O(N·k), measured 9×→1913×
               vs dense at N=1k→8k with rel.err ~1e-9 on the same block);
        k<0  → the TOP-|k| modes.
    HONEST: modes=k reads κ_W of the SELECTED block — the right scope when you
    design/track only those modes — NOT the full-spectrum κ_W (a different number).
    The matrix-free win is real only for sparse/structured A0,B (cheap matvec);
    on a dense A0 `eigsh` still avoids the full spectrum but pays a denser solve.

    Family: wkernel owns W; this is W's local Lipschitz read.
    """
    k0 = np.asarray(k0, float)
    rng = np.random.default_rng(seed)
    W0 = _selected_W(A0, perturbations, k0, modes)
    vals = []
    for _ in range(probes):
        u = rng.standard_normal(len(k0)); u /= np.linalg.norm(u)
        Wu = _selected_W(A0, perturbations, k0 + eps * u, modes)
        vals.append(float(np.linalg.norm(Wu - W0) / eps))
    return (max(vals), np.array(vals)) if full else max(vals)
