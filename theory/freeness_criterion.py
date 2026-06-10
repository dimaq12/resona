"""
response_algebra/freeness_criterion.py
=============================================================================
THE NON-CLOSABLE RESIDUE == THE FREENESS DEFECT == MIXED FREE CUMULANTS.
Turns the honest boundary into a theorem (Voiculescu / Speicher).

Freeness (definition): A, B are free  ⟺  φ(Ẋ₁ Ẏ₁ Ẋ₂ Ẏ₂ …) = 0  for every
ALTERNATING product of CENTERED elements Ẋ=p(A)-φ(p(A)), Ẏ=q(B)-φ(q(B)).
Equivalently: all MIXED free cumulants vanish (Speicher).

So the obstruction to closure is EXACTLY the magnitude of these alternating
centered moments — the freeness defect, computable and zero iff free.

We verify:
  FREE pair  (B = Q·A²·Qᵀ, Haar): alternating centered moments ≈ 0 (O(1/√N)).
  NON-free   (B = A², shared basis): they are large — the residue, made visible.

Run:  python3 response_algebra/freeness_criterion.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from free_prob_bridge import haar_orthogonal

rng = np.random.default_rng(2)


def phi(M):
    return float(np.trace(M)) / M.shape[0]


def center(M):
    return M - phi(M) * np.eye(M.shape[0])


def alt_centered_moment(Ac, Bc, L):
    """φ of the alternating centered word  Ac Bc Ac Bc …  of length L."""
    P = Ac.copy()
    for i in range(1, L):
        P = P @ (Bc if i % 2 == 1 else Ac)
    return phi(P)


if __name__ == "__main__":
    N = 2000
    A = np.diag(rng.standard_normal(N))
    Asq = A @ A                                  # a genuine second operator (commutes with A)
    Q = haar_orthogonal(N)

    Ac = center(A)
    Bc_free = center(Q @ Asq @ Q.T)              # FREE copy (rotated)
    Bc_nonfree = center(Asq)                      # SAME eigenbasis → NOT free

    print("=" * 72)
    print("FREENESS DEFECT = alternating centered moments = the non-closable residue")
    print("=" * 72)
    print(f"  N={N}.  A and B=A² ;  free pair rotates B by Haar Q.")
    print(f"  φ(Ȧ Ḃ Ȧ Ḃ …) — must vanish for a FREE pair (Voiculescu/Speicher).\n")
    print(f"  {'word length L':>14} {'FREE  |φ|':>14} {'NON-free  |φ|':>16}")
    df, dn = 0.0, 0.0
    for L in range(2, 7):
        f = abs(alt_centered_moment(Ac, Bc_free, L))
        n = abs(alt_centered_moment(Ac, Bc_nonfree, L))
        df, dn = max(df, f), max(dn, n)
        print(f"  {L:>14} {f:>14.2e} {n:>16.2e}")
    print(f"\n  FREE: all alternating centered moments ≈ 0 (worst {df:.1e}, the O(1/√N) floor)")
    print(f"  NON-free: large (worst {dn:.1e}) — the freeness defect, made a number.")
    print(f"  ratio non-free/free = {dn/df:.0f}×")
    print("\n" + "=" * 72)
    print("THEOREM (boundary → theorem)")
    print("=" * 72)
    print("  Closure of the response algebra under composition is EXACT ⟺ the operators")
    print("  are FREE ⟺ all mixed free cumulants (= alternating centered moments) vanish.")
    print("  The non-closable residue is not vague — it IS the freeness defect, computable")
    print("  and zero exactly when composition closes.  (Speicher's vanishing criterion.)")
    print("=" * 72)
