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


def track(A0, perturbations, path, steps=1):
    """Integrate the spectral flow  dλ = W(k)·dk  along a parameter path —
    eigenvalues followed by EIGENVECTOR CONTINUATION, so crossings do not
    scramble them (sorted eigenvalues break by O(1) at a crossing; the
    tracked path stays exact — measured 8.9e-15 vs 3.3 in the source
    program's criterion C9).

    Family: wkernel owns ∂λ_i/∂k_j (Hellmann–Feynman); `track` is its line
    integral.  (`design` is the INVERSE step: target Δλ → Δk.)

    A0            : base matrix (the family is LINEAR: A(k) = A0 + Σ k_j B_j —
                    the theorem's verified domain; nonlinear-in-k families
                    are out of scope by design).
    perturbations : list of B_j (matrices or matvec callables).
    path          : (T, M) array of parameter points, k(0) … k(T−1).
    steps         : midpoint sub-steps per path segment.

    Returns (lams, V): lams[(T, N)] — every eigenvalue along the path in a
    CONSISTENT order; V — the eigenvectors at the final point.

    HONEST LIMITS: dense (one eigh per midpoint — this is the point: ~100×
    more accurate than frozen-W per eigh spent, and 44–302× fewer eigh than
    finite-difference continuation, measured); frozen-W error grows as
    C_H·‖dk‖² between refreshes (Theorem E) — `kappa_w` is that dial.
    """
    A0 = np.asarray(A0, float)
    if not np.allclose(A0, A0.T, atol=1e-10):
        raise ValueError("track's verified domain is SYMMETRIC linear families "
                         "(eigh-based continuation); A0 is not symmetric — for "
                         "non-Hermitian spectra see resona.cloud")
    path = np.asarray(path, float)
    Bs = [B if not callable(B) else None for B in perturbations]
    if any(b is None for b in Bs):
        raise ValueError("track needs explicit matrices for the linear family "
                         "A(k) = A0 + Σ k_j B_j (callables hide the domain)")
    Bstack = np.stack([np.asarray(B, float) for B in Bs])

    def build(k):
        return A0 + np.tensordot(k, Bstack, axes=1)

    lam, V = np.linalg.eigh(build(path[0]))
    lams = [lam.copy()]
    for seg in range(1, len(path)):
        for s in range(steps):
            a = path[seg - 1] + (path[seg] - path[seg - 1]) * s / steps
            b = path[seg - 1] + (path[seg] - path[seg - 1]) * (s + 1) / steps
            km = 0.5 * (a + b)
            lam_m, Vm = np.linalg.eigh(build(km))
            # eigenvector continuation: match midpoint modes to OUR order
            perm = np.argmax(np.abs(V.T @ Vm), axis=1)
            Wm = np.einsum('ni,mni->im', Vm[:, perm], Bstack @ Vm[:, perm])
            lam = lam + Wm @ (b - a)
            V = Vm[:, perm]
        lams.append(lam.copy())
    return np.array(lams), V


def kappa_w(A0, perturbations, k0, eps=1e-5, probes=8, seed=0):
    """κ_W — the local curvature of the spectral-flow kernel:
    max over random unit directions of ‖W(k₀+εu) − W(k₀)‖_F / ε.

    WHAT IT PREDICTS (and the first thing to know): the ACCURACY of a
    frozen-W prediction over a step (Spearman ρ = 0.929 against measured
    frozen-W error, criterion C8).  WHAT IT DOES NOT PREDICT: global
    computational cost — on blind random families ρ(κ_W, cost) ≈ 0.05
    across fresh seeds; cost follows the PATH LENGTH, not the local
    curvature.  Use κ_W to size a trust region for `track`/`design` steps,
    never as a difficulty oracle.

    Family: wkernel owns W; this is W's local Lipschitz read.
    """
    A0 = np.asarray(A0, float)
    Bstack = np.stack([np.asarray(B, float) for B in perturbations])
    k0 = np.asarray(k0, float)
    rng = np.random.default_rng(seed)

    def W_at(k):
        _, V = np.linalg.eigh(A0 + np.tensordot(k, Bstack, axes=1))
        return np.einsum('ni,mni->im', V, Bstack @ V)

    W0 = W_at(k0)
    out = 0.0
    for _ in range(probes):
        u = rng.standard_normal(len(k0)); u /= np.linalg.norm(u)
        out = max(out, float(np.linalg.norm(W_at(k0 + eps * u) - W0) / eps))
    return out
