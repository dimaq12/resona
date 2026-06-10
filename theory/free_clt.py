"""
response_algebra/free_clt.py
=============================================================================
Two analytical pillars of the response algebra, verified numerically.

ACT 1 — FREE CENTRAL LIMIT THEOREM (why "free/semicircle" is the attractor).
  Sum of K free copies, normalized: (X_1+…+X_K)/√K → SEMICIRCLE.
  In free cumulants: κ_n of the normalized sum = K^{1-n/2} κ_n(X).
    n=2 preserved; n>2 → 0.  So everything flows to the semicircle (only κ_2).
  This is WHY random matrices are semicircular and why generic/disordered
  systems gravitate to "free" — semicircle is the free Gaussian, the universal
  attractor.  We verify κ_4 ∝ 1/K.

ACT 2 — MULTIPLICATIVE CLOSURE (composition under PRODUCTS, not just sums).
  For FREE A,B the product moments Tr((AB)^k) depend ONLY on the marginal
  spectra of A and B (free multiplicative convolution; linearized by the
  S-transform).  Verified: invariant under relative rotation for free pairs,
  broken for non-free.  Ties to products of random weight matrices (the
  universal carrier / deep nets).

Run:  python3 response_algebra/free_clt.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import linalg
from free_prob_bridge import moments, free_cumulants, haar_orthogonal   # reuse our primitives

rng = np.random.default_rng(3)


def act1_free_clt(N=2500, K_list=(1, 2, 4, 8, 16)):
    print("=" * 72)
    print("ACT 1 — FREE CLT: sum of K free copies → semicircle (κ_4 ∝ 1/K)")
    print("=" * 72)
    # X: zero-mean unit-variance, NON-semicircle (Bernoulli ±1 spectrum → κ_4≠0)
    spec = rng.choice([-1.0, 1.0], size=N)
    X = np.diag(spec)
    k0 = free_cumulants(moments(X, 5))
    print(f"  base X (Bernoulli spectrum): κ = "
          f"{np.array2string(k0, precision=2, suppress_small=True)}  (κ_4≠0 ⇒ not free)\n")
    print(f"  {'K':>3} {'κ_2(sum/√K)':>13} {'κ_4(sum/√K)':>13} {'κ_4·K':>9}")
    for K in K_list:
        S = np.zeros((N, N))
        for _ in range(K):
            Q = haar_orthogonal(N)
            S += Q @ X @ Q.T                      # free copies
        S /= np.sqrt(K)                           # CLT normalization
        k = free_cumulants(moments(S, 5))
        print(f"  {K:>3} {k[1]:>13.3f} {k[3]:>13.4f} {k[3]*K:>9.3f}")
    print(f"\n  κ_2 stays ≈1, κ_4 → 0 like 1/K (κ_4·K ≈ const): the sum flows to the")
    print(f"  SEMICIRCLE.  Free/semicircle is the universal attractor (the free Gaussian).\n")


def act2_multiplicative_closure(N=1500):
    print("=" * 72)
    print("ACT 2 — MULTIPLICATIVE closure: product moments close for FREE pairs")
    print("=" * 72)
    # positive operators (so AB has real spectrum via A^{1/2}BA^{1/2})
    A = np.diag(rng.uniform(0.5, 3.0, N))
    B0 = np.diag(rng.uniform(0.5, 3.0, N))
    Q = haar_orthogonal(N)
    B_free = Q @ B0 @ Q.T                          # free copy (rotated): A ⊠ B_free
    B_nonfree = B0                                 # same eigenbasis: NOT free

    def prod_moments(A, B, K=5):
        Ah = linalg.sqrtm(A).real
        M = Ah @ B @ Ah                            # same spectrum as AB, symmetric
        return moments(M, K)

    mf = prod_moments(A, B_free)
    mn = prod_moments(A, B_nonfree)
    # marginal-only prediction sanity: m_1(AB)=m_1(A)m_1(B) for free
    pred_m1 = (np.trace(A) / N) * (np.trace(B0) / N)
    print(f"  {'k':>2} {'Tr((A·B_free)^k)/N':>20} {'Tr((A·B_same)^k)/N':>20} {'differ?':>9}")
    for k in range(5):
        print(f"  {k+1:>2} {mf[k]:>20.4f} {mn[k]:>20.4f} "
              f"{'yes' if abs(mf[k]-mn[k])>1e-2 else '—':>9}")
    print(f"\n  m_1 free = {mf[0]:.4f}  vs  marginal prediction m_1(A)·m_1(B) = {pred_m1:.4f}")
    print(f"  FREE product moments depend only on the marginals (closed); the NON-free")
    print(f"  product differs — the freeness defect again.  The S-transform linearizes")
    print(f"  this (S_{{A⊠B}}=S_A·S_B), as R does for sums.  (= products of random weights")
    print(f"  in deep nets / the universal carrier.)\n")


if __name__ == "__main__":
    act1_free_clt()
    act2_multiplicative_closure()
    print("=" * 72)
    print("  + closes by R-transform (sums) ;  × closes by S-transform (products).")
    print("  Free CLT: the semicircle is the attractor.  Response algebra = free probability,")
    print("  closed under BOTH operations, with the freeness defect as the only residue.")
    print("=" * 72)
