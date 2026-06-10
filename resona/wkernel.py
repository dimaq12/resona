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


def design(W, target_shift, rcond=None):
    """Least-squares parameter step dk so that W·dk ≈ target_shift (λ_target − λ_0).

    One Hellmann–Feynman design step — no finite differences, no per-parameter
    eigensolve.  Iterate (recompute W at the new k) for nonlinear targets.
    """
    dk, *_ = np.linalg.lstsq(W, np.asarray(target_shift, float), rcond=rcond)
    return dk
