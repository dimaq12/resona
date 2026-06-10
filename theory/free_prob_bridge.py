"""
free_probability/free_prob_bridge.py
=============================================================================
THE ANALYTICAL HEART — both branches collapse into FREE PROBABILITY.

Response branch  (moments Tr(A^p))      = the *-distribution of A.
Resolvent branch (G(z)=Tr((z-A)^{-1}))  = the Cauchy transform of that distribution.
Bridge:  G(z) = Σ_p m_p / z^{p+1}   (resolvent = generating function of moments).

The canonical coordinates of the response algebra are NOT the raw moments but the
FREE CUMULANTS κ_n (Voiculescu / Speicher).  In them, operator addition of FREE
elements is LITERALLY addition:

        κ_n(A ⊞ B) = κ_n(A) + κ_n(B)            (free additivity)

and the generating function of the κ_n is the R-TRANSFORM, R(z)=G^{-1}(z)-1/z,
which lives in the resolvent picture.  So: response gives moments, resolvent
inverts G to R, R linearizes composition.  Closure = free additivity.  The
non-closable residue = the FREENESS DEFECT (eigenbasis correlation), measured by
the cross-moments.

This script: (1) moment->free-cumulant inversion via M(z)=C(zM(z));
             (2) verify κ_n(A+B)=κ_n(A)+κ_n(B) for a FREE pair (Haar-rotated);
             (3) show it FAILS for a NON-free pair (shared basis) — the defect.
"""
import numpy as np

rng = np.random.default_rng(11)


def moments(M, K):
    """Normalized moments m_p = Tr(M^p)/N, p=1..K."""
    N = M.shape[0]
    out = []
    P = np.eye(N)
    for _ in range(K):
        P = P @ M
        out.append(float(np.trace(P)) / N)
    return out


def free_cumulants(m):
    """Free cumulants κ_1..κ_K from moments m_1..m_K via M(z)=C(zM(z)):
        κ_n = m_n - Σ_{k=1}^{n-1} κ_k · [z^{n-k}] M(z)^k ,   M(z)=1+Σ m_j z^j."""
    K = len(m)
    Mc = np.array([1.0] + list(m))          # coeffs of M(z), z^0..z^K
    kappa = np.zeros(K + 1)                  # 1-indexed
    for n in range(1, K + 1):
        s = 0.0
        Mk = np.array([1.0])                 # M(z)^0
        for k in range(1, n):
            Mk = np.convolve(Mk, Mc)         # M(z)^k
            idx = n - k
            s += kappa[k] * (Mk[idx] if idx < len(Mk) else 0.0)
        kappa[n] = m[n - 1] - s
    return kappa[1:]


def haar_orthogonal(N):
    Q, R = np.linalg.qr(rng.standard_normal((N, N)))
    return Q * np.sign(np.diag(R))


if __name__ == "__main__":
    K = 5
    print("=" * 75)
    print("SANITY — free cumulants of the semicircle (Wigner): expect (0,1,0,0,0)")
    print("=" * 75)
    N = 3000
    G = rng.standard_normal((N, N)); W = (G + G.T) / np.sqrt(2 * N)   # semicircle, var 1
    ks = free_cumulants(moments(W, K))
    print("  κ =", np.array2string(ks, precision=3, suppress_small=True),
          "  (κ_2≈1, rest≈0 — free cumulants of a semicircle)\n")

    # A with NON-trivial higher cumulants: 3-point spectrum
    N = 2500
    spec = rng.choice([-1.5, 0.3, 1.4], size=N)
    A = np.diag(spec)
    mA = moments(A, K); kA = free_cumulants(mA)

    print("=" * 75)
    print("FREE pair:  B = Q A Qᵀ  (Haar Q)  →  A, B asymptotically FREE")
    print("=" * 75)
    Q = haar_orthogonal(N)
    Bf = Q @ A @ Q.T
    kBf = free_cumulants(moments(Bf, K))
    kSf = free_cumulants(moments(A + Bf, K))
    print(f"  {'n':>2} {'κ_n(A)':>10} {'κ_n(B)':>10} {'κ_n(A)+κ_n(B)':>14} {'κ_n(A+B)':>12} {'err':>9}")
    for n in range(K):
        add = kA[n] + kBf[n]
        print(f"  {n+1:>2} {kA[n]:>10.4f} {kBf[n]:>10.4f} {add:>14.4f} {kSf[n]:>12.4f} "
              f"{abs(add-kSf[n]):>9.1e}")
    err_free = max(abs(kA[n] + kBf[n] - kSf[n]) for n in range(1, K))
    print(f"\n  FREE additivity κ_n(A+B)=κ_n(A)+κ_n(B) holds to {err_free:.1e} "
          f"(O(1/N) freeness corrections).")
    print(f"  => composition is EXACTLY addition in free-cumulant coordinates.\n")

    print("=" * 75)
    print("NON-FREE pair:  B = A  (same eigenbasis)  →  NOT free.  A+B = 2A")
    print("=" * 75)
    kSn = free_cumulants(moments(A + A, K))
    print(f"  {'n':>2} {'κ_n(A)+κ_n(A)=2κ_n':>18} {'κ_n(2A)=2^n κ_n':>16} {'κ_n(A+B) actual':>16}")
    for n in range(K):
        print(f"  {n+1:>2} {2*kA[n]:>18.4f} {(2**(n+1))*kA[n]:>16.4f} {kSn[n]:>16.4f}")
    defect = max(abs(2 * kA[n] - kSn[n]) for n in range(1, K))
    print(f"\n  free additivity FAILS: κ_n(2A)=2^n κ_n(A) ≠ 2 κ_n(A).  defect = {defect:.2f}")
    print(f"  the FREENESS DEFECT = the non-closable residue = eigenbasis correlation,")
    print(f"  measured by the cross-moments the spectrum discards.\n")

    print("=" * 75)
    print("VERDICT")
    print("=" * 75)
    print(f"  response algebra = free probability.  canonical coords = free cumulants.")
    print(f"  composition = free additivity (exact for free pairs: err {err_free:.1e}).")
    print(f"  R-transform (resolvent picture) = generating function of the cumulants.")
    print(f"  non-closable residue = freeness defect ({defect:.1f}) = the orientation")
    print(f"  the spectrum throws away.  Two branches, one theory: Voiculescu.")
    print("=" * 75)
