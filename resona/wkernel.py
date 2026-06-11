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
