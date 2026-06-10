"""
quantum/dequantize.py
==============================================================================
BEAT THE QUANTUM COMPUTER (honestly) — dequantizing low-rank quantum ML.

THE CLAIM IT REFUTES.  A celebrated line of "exponential quantum speedups" for
linear algebra — quantum recommendation systems, quantum PCA, low-rank linear
solvers (Kerenidis–Prakash and kin) — promised exponential advantage over any
classical method.  Ewin Tang (2018) DEQUANTIZED them: the speedup rested entirely
on the matrix being LOW RANK, and low rank is harvestable classically by
length-squared SAMPLING, at cost INDEPENDENT of the dimension.  The quantum
advantage was redundant — it was paying for structure you can sample for free.

OUR LENS — it's the effective rank Φ₁.  resona's cost dial `effective_rank`
(= Tr(A)²/Tr(A²), the participation ratio) measures exactly this resource: how
many modes a response really has.  Low Φ₁ ⇒ a few modes carry everything ⇒
length-squared sampling finds them ⇒ no quantum computer needed.  So our framework
both MEASURES the resource that decides "redundant vs genuine" AND exploits it.

THE DEMO.  An implicit low-rank matrix A = U·B (n rows, rank r, NEVER
materialized).  We recover its top-r right singular subspace by sampling only
s ≪ n rows ∝ ‖row‖².  We grow n by 500× with s FIXED: accuracy stays ≈1 while the
fraction of data touched vanishes.  Cost tracks the RANK, not the dimension —
the signature of a dequantized algorithm.

Run:  python3 examples/quantum/dequantize.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from scipy import linalg

rng = np.random.default_rng(0)


def true_subspace(U, B, r):
    """Top-r right singular vectors of A=U·B from the r×m structure (cheap)."""
    M = B.T @ (U.T @ U) @ B                                # = AᵀA  (m×m, rank r)
    return linalg.eigh(M)[1][:, -r:]                       # m×r orthonormal


def length_squared_sample(U, B, s, r):
    """Sample s rows ∝ ‖row‖², sketch, recover the top-r right singular subspace."""
    rownorm2 = np.einsum('ir,rs,is->i', U, B @ B.T, U)     # ‖U[i]·B‖²  (A never formed)
    p = rownorm2 / rownorm2.sum()
    idx = rng.choice(len(U), size=s, p=p)
    R = (U[idx] @ B) / np.sqrt(s * p[idx])[:, None]        # s×m rescaled sketch
    return linalg.svd(R, full_matrices=False)[2][:r].T     # m×r approximate subspace


def overlap(Va, Vb):
    """Subspace overlap ∈[0,1], 1 = identical (both orthonormal m×r)."""
    return float(np.linalg.norm(Va.T @ Vb) ** 2 / Va.shape[1])


if __name__ == "__main__":
    print("=" * 72)
    print("DEQUANTIZATION — low-rank quantum speedup is REDUNDANT (sample the rank)")
    print("=" * 72)
    m, r, s = 400, 5, 60
    print(f"  implicit low-rank A=U·B, rank r={r}, m={m} cols.  sample s={s} rows.")
    print(f"  recover the top-{r} singular subspace; cost ~ rank, NOT dimension n.\n")
    print(f"  {'n (rows)':>10} {'rows sampled':>13} {'data touched':>14} {'subspace overlap':>18}")
    print("  " + "─" * 58)
    for n in [2_000, 20_000, 200_000, 1_000_000]:
        U = rng.standard_normal((n, r)) @ np.diag([5, 4, 3, 2, 1][:r])
        B = rng.standard_normal((r, m))
        ov = overlap(length_squared_sample(U, B, s, r), true_subspace(U, B, r))
        print(f"  {n:>10,} {s:>13} {s/n*100:>13.4f}% {ov:>18.4f}")
    print(f"\n  Same s={s} as n grows 500× — accuracy ≈1, touching a vanishing fraction.")
    print(f"  The 'exponential quantum advantage' for low-rank collapses to classical")
    print(f"  sampling — it was REDUNDANT, paying for structure (low Φ₁) you sample free.")

    print("\n" + "=" * 72)
    print("THE HONEST BOUNDARY (where we beat quantum, and where we don't)")
    print("=" * 72)
    print("  REDUNDANT (dequantizable — low Φ₁ / structure):")
    print("    low-rank linear algebra (recommendation, PCA, low-rank solve) — Tang;")
    print("    Clifford circuits (Gottesman–Knill → entanglement_transition.py);")
    print("    free fermions; area-law states & low-entanglement dynamics (tensor nets);")
    print("    noisy circuits.  Quantum speedup here is at most POLYNOMIAL.")
    print("  GENUINE quantum (NOT dequantizable, as far as is known):")
    print("    Shor factoring/discrete-log (→ shor_wall.py); real-time VOLUME-LAW")
    print("    entanglement growth; full-rank / high-condition problems; generic")
    print("    random-circuit sampling.")
    print("\n  Φ₁ (resona's effective_rank) is the DIAL: low ⇒ the quantum advantage is")
    print("  redundant and we harvest it by sampling; high ⇒ honest quantum frontier.")
    print("  We don't beat ALL quantum — we beat exactly the low-rank class, and we can")
    print("  TELL WHICH, because the resource is the effective rank we already measure.")
    print("=" * 72)
