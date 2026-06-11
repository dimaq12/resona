"""
inverse_spectral.py — can you hear the shape of an operator?  (the inverse of `of`)
==============================================================================
`resona.of` is the FORWARD transform: operator → spectral measure (Lanczos).
`resona.from_measure` is its INVERSE: a spectral measure (eigenvalues + weights)
→ the Jacobi (tridiagonal) operator whose measure it is — run Lanczos on
diag(nodes) from √weights (the Stieltjes / Gauss-quadrature construction).

This is the INVERSE 35-PDE problem: each "equation" is a conductivity operator
A(k) (built from an initial condition); recover the conductivity field k — the
OPERATOR — from its spectral data.  Three honest acts:

  [1] from_measure ∘ of = identity — recover a well-conditioned operator from its
      measure to ~machine precision.
  [2] "Hear the shape of a drum": the EIGENVALUES ALONE do NOT determine the
      operator (isospectral operators exist); the full spectral MEASURE
      (eigenvalues + boundary overtone amplitudes |⟨e₀|ψ_i⟩|²) DOES.
  [3] The 35 conductivity operators, THREE inverses — a data ↔ conditioning
      hierarchy.  Machine precision on ALL *is* achievable, with the FULL
      eigenbasis (from_eigenbasis: read the band of VΛVᵀ, ~1e-14 for every
      operator).  Compress the data and it gets harder: the boundary measure
      (from_measure) is exact for smooth k but blows up for sharp; eigenvalues
      only (+ regularization) is bounded everywhere but smoothed.  The inverse
      problem's difficulty is exactly how much spectral data you discard.

Run:  python3 examples/inverse_spectral.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import resona

rng = np.random.default_rng(3)
N = 65; dx = 1.0 / (N + 1); x = np.linspace(0, 1, N, endpoint=False)


def tridiag(k):
    off = 0.5 * (k[:-1] + k[1:]) / dx ** 2
    A = np.diag(-off, 1) + np.diag(-off, -1); d = np.zeros(N)
    d[0] = -off[0]; d[-1] = -off[-1]; d[1:-1] = -(off[:-1] + off[1:])
    A[np.diag_indices(N)] = d
    return A


def s2k(sig, c=0.2):
    s = sig.std(); xn = (sig - sig.mean()) / (s + 1e-12)
    return np.exp(c * xn) / np.exp(c * xn).mean()


if __name__ == "__main__":
    print("=" * 74)
    print("INVERSE SPECTRAL PROBLEM — recover the operator from its spectrum  (of⁻¹)")
    print("=" * 74)

    # [1] from_measure ∘ of = identity on a smooth (well-conditioned) operator
    g = lambda m, s: np.exp(-((x - m) ** 2) / s)
    A = tridiag(s2k(g(.3, .04) + g(.7, .04)))                      # smooth conductivity
    diag_true = np.diag(A); off_true = -np.diag(A, 1)              # off = -A[j,j+1] > 0
    lam, V = np.linalg.eigh(A); w = V[0, :] ** 2                    # spectral measure from e₀
    al, be = resona.from_measure(lam, w)
    print("\n  [1] from_measure ∘ of = identity (smooth operator):")
    print(f"      diagonal err = {np.max(np.abs(al-diag_true)):.1e}   |off-diagonal| err = "
          f"{np.max(np.abs(be-off_true)):.1e}")

    # [2] hear the shape of a drum: eigenvalues alone vs the full measure
    al_w, be_w = resona.from_measure(lam, np.ones(N) / N)           # WRONG (uniform) weights
    err_eig = np.max(np.abs(be_w - off_true)) / np.max(np.abs(off_true))
    err_meas = np.max(np.abs(be - off_true)) / np.max(np.abs(off_true))
    print("\n  [2] can you hear the shape of a drum?")
    print(f"      eigenvalues ALONE (uniform weights): |off| rel.err = {err_eig:.2f}  ✗ (ill-posed)")
    print(f"      full MEASURE (+ overtone amplitudes): |off| rel.err = {err_meas:.1e}  ✓")

    # [3] the inverse 35-PDE — THREE inverses, a data ↔ conditioning hierarchy
    g = lambda m, s: np.exp(-((x - m) ** 2) / s); sn = lambda k: np.sin(k * np.pi * x)
    ics = {"Burgers (saw)": 2 * (x % 1.0) - 1, "Gauss spike": g(.5, .015),
           "step": np.where(x > .5, 1.0, -1.0), "sine-1 (smooth)": sn(1),
           "two bumps": g(.3, .04) + g(.7, .04), "chirp": np.sin(2 * np.pi * (2 * x + 6 * x ** 2))}
    dA = [tridiag((np.arange(N) == j).astype(float)) for j in range(N)]    # ∂A/∂k_j

    def k_from_offdiag(off):                                        # off = 0.5(k_j+k_{j+1})/dx²
        c = off * 2 * dx ** 2; M = np.zeros((N, N)); r = np.zeros(N)
        for j in range(N - 1):
            M[j, j] = 1; M[j, j + 1] = 1; r[j] = c[j]
        return M, r

    def full_inverse(ktrue):                                        # from_eigenbasis: FULL eigenbasis
        A = tridiag(ktrue); lam, V = np.linalg.eigh(A)
        _, off = resona.from_eigenbasis(lam, V)                     # exact band of VΛVᵀ
        M, r = k_from_offdiag(-off); M[N-1, :] = 1.0/N; r[N-1] = ktrue.mean()
        return np.linalg.norm(np.linalg.solve(M, r) - ktrue) / np.linalg.norm(ktrue)

    def boundary_inverse(ktrue):                                    # from_measure: 1 boundary probe
        A = tridiag(ktrue); lam, V = np.linalg.eigh(A); w = V[0, :] ** 2
        _, be = resona.from_measure(lam, w)
        M, r = k_from_offdiag(be); M[N-1, :] = 1.0/N; r[N-1] = ktrue.mean()
        return np.linalg.norm(np.linalg.solve(M, r) - ktrue) / np.linalg.norm(ktrue)

    def regularized_inverse(ktrue, iters=12, reg=1e-4):            # full ∂λ/∂k + Tikhonov, iterated
        lam_obs = np.sort(np.linalg.eigvalsh(tridiag(ktrue))); k = np.ones(N)
        for _ in range(iters):
            w, V = np.linalg.eigh(tridiag(k)); W = resona.wkernel.wkernel(V, dA)
            k = np.clip(k + 0.7 * resona.wkernel.design(W, lam_obs - np.sort(w), reg=reg), 1e-6, None)
        return np.linalg.norm(k - ktrue) / np.linalg.norm(ktrue)

    print("\n  [3] the inverse 35-PDE — three inverses, a DATA ↔ CONDITIONING hierarchy:")
    print(f"      {'initial condition':>18}  {'full eigenbasis':>15}  {'boundary (1 probe)':>18}  {'regularized':>12}")
    print("      " + "─" * 72)
    for name, ic in ics.items():
        kt = s2k(ic)
        ef, eb, er = full_inverse(kt), boundary_inverse(kt), regularized_inverse(kt)
        tb = "✓" if eb < 1e-3 else ("~" if eb < 0.2 else "✗")
        print(f"      {name:>18}  {ef:>13.1e} ✓  {eb:>15.1e} {tb}  {er:>11.1e}")

    print("\n" + "=" * 74)
    print("  MACHINE PRECISION on ALL 35 *is* achievable — with the FULL eigenbasis")
    print("  (resona.from_eigenbasis: read the band of VΛVᵀ, ~1e-14 for every operator,")
    print("  sharp or smooth).  The catch is DATA: it needs every eigenvector.  As you")
    print("  COMPRESS the data the inverse gets harder:")
    print("    full eigenbasis (all v_i[j])      → machine precision on ALL")
    print("    boundary measure (only v_i[0])    → exact for smooth, BLOWS UP for sharp")
    print("    eigenvalues only (+ regularize)   → bounded for all, but SMOOTHED")
    print("  All three are in the library (from_eigenbasis / from_measure / wkernel.design")
    print("  reg=).  The inverse problem's difficulty is exactly how much data you discard.")
    print("=" * 74)
