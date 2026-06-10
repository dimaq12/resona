"""
frontier/manifold_extraction.py
=============================================================================
IS THE ANSWER EXTRACTABLE EVEN WHERE IT SEEMS NOT?  — honest check.

Claim: a shock is linear in the lifted coordinate, so the apparent wall is a
COORDINATE ARTIFACT — lift to the manifold and the answer extracts smoothly.

True for REMOVABLE singularities (shock, EP, isolated pole): a FINITE lift makes
them linear → answer extractable through the apparent wall.
NOT true for NON-removable ones (structureless / Shor): NO finite lift linearizes
them — the effective rank keeps growing at every lift order.  The Extraction Law
is exactly this dichotomy.

ACT 1 — REMOVABLE: extract the eigenvalue THROUGH an exceptional point.
ACT 2 — the dial: lift to higher order; removable rank SATURATES (finite chart,
        extractable), structureless rank GROWS (no chart, genuine wall).

Run:  python3 frontier/manifold_extraction.py
"""
import numpy as np
from scipy.linalg import hankel, svdvals

rng = np.random.default_rng(0)


def act1_extract_through_EP():
    print("=" * 72)
    print("ACT 1 — REMOVABLE: extract the eigenvalue THROUGH an exceptional point")
    print("=" * 72)
    print("  A(z)=[[0,1],[z,0]], eig=±√z.  at z=0 (EP) the bare eig has ∞ derivative —")
    print("  'seems' un-extractable.  the lift u=λ²=z is LINEAR → extract exactly.\n")
    print(f"  {'z':>7} {'eig=√z (bare, singular)':>24} {'u=λ²=z (lifted, linear)':>24}")
    for z in [-0.04, -0.02, 0.0, 0.02, 0.04]:
        lam = np.sqrt(complex(z))
        print(f"  {z:>7.3f} {str(np.round(lam,4)):>24} {np.round((lam*lam).real,4):>24}")
    # extraction: target a spectral value, recover z THROUGH the EP, one lifted step
    lam_t = 0.3
    z_rec = (lam_t ** 2)                     # one step in the lift, exact at the EP
    print(f"\n  extract z from target λ={lam_t}:  lifted one-step z = {z_rec:.4f} "
          f"(exact, through the wall)")
    print("  the apparent singularity is a coordinate artifact — the lift removes it.\n")


def eff_rank(seq, k):
    seq = np.asarray(seq, float); seq = (seq - seq.mean()) / (seq.std() + 1e-12)
    H = hankel(seq[:k], seq[k - 1:2 * k - 1])
    s = svdvals(H)
    num_rank = int(np.sum(s > 1e-9 * s[0]))           # numerical rank
    phi1 = float((s ** 2).sum() ** 2 / (s ** 4).sum())  # effective rank Φ₁
    return num_rank, phi1


def act2_removable_vs_genuine():
    print("=" * 72)
    print("ACT 2 — the dial: does a finite lift linearize it?  (rank saturates or not)")
    print("=" * 72)
    L = 400
    x = np.arange(L)
    removable = np.cos(0.3 * x) + np.cos(0.7 * x) + np.cos(1.1 * x)   # 3 modes → rank 6
    N = 100003 * 100019
    structureless = np.array([pow(3, int(i), N) for i in x], float)    # a^x mod N (Shor)

    print(f"  Hankel rank / Φ₁ vs lift window k:\n")
    print(f"  {'k':>5} {'REMOVABLE rank':>16} {'Φ₁':>7}   {'STRUCTURELESS rank':>20} {'Φ₁':>7}")
    for k in [10, 20, 40, 80]:
        r1, p1 = eff_rank(removable, k)
        r2, p2 = eff_rank(structureless, k)
        print(f"  {k:>5} {r1:>16} {p1:>7.1f}   {r2:>20} {p2:>7.1f}")
    print(f"\n  REMOVABLE: rank SATURATES at 6 (finite chart exists) → answer extractable")
    print(f"  even where it 'looked' nonlinear.  STRUCTURELESS: rank GROWS with k (no")
    print(f"  finite chart) → genuine wall, extraction stays extensive (Shor).\n")


if __name__ == "__main__":
    act1_extract_through_EP()
    act2_removable_vs_genuine()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("  YES — the answer is extractable even where it SEEMS not, WHENEVER the")
    print("  singularity is REMOVABLE: a finite lift to the manifold makes it linear")
    print("  (shock→R, EP→λ^q, pole→deflate).  Most apparent walls are this kind.")
    print("  BUT not always: where the lift never saturates (structureless / Shor),")
    print("  no chart linearizes it — the wall is genuine.  The Extraction Law is the")
    print("  test: rank saturates ⇒ extractable; rank grows ⇒ real wall.")
    print("=" * 72)
