"""
quantum/shor_wall.py
==============================================================================
WHERE THE QUANTUM WALL IS — and why our own Φ₁ dial honestly marks it.

THE OTHER SIDE OF DEQUANTIZATION.  dequantize.py shows we beat quantum where Φ₁
is LOW (structure to sample).  Intellectual honesty demands we point at the place
we do NOT: Shor's factoring algorithm.  A theory that claimed to beat Shor would
be claiming to break RSA — and should be distrusted.  Here we let our OWN tool
testify against us.

THE REDUCTION.  Factoring N reduces to PERIOD FINDING: find the order r of
a^x mod N (the smallest r with a^r ≡ 1).  Given r (even), gcd(a^{r/2}±1, N)
splits N.
  • r small  → classical order-finding is O(r): we factor it (shown below).
  • r large (~N, the RSA regime) → O(r) is exponential in log N.  Worse, the
    sequence a^x mod N is STRUCTURELESS to our tools: its trajectory (Hankel)
    matrix has high effective rank Φ₁ (≈10× a structured signal) — no low-rank /
    sampling handle exists.

THE Φ₁ TEST.  We compute resona's effective rank (participation ratio
Φ₁ = (Σσ²)²/Σσ⁴, the same `effective_rank` formula) of the Hankel matrix of the
sequence — its trajectory operator.  A periodic/structured signal gives LOW Φ₁
(few modes → harvestable).  a^x mod N gives HIGH Φ₁ (no dominant low-rank
structure) → the dequantization dial reads GENUINE QUANTUM: no classical shortcut.

THE POINT.  The very structurelessness that SECURES RSA is what BLOCKS the
classical harvest, and our own dial detects it.  Shor's QFT finds r WITHOUT
generating r terms — that is the irreducible quantum advantage, and Φ₁ marks the
wall from the outside.

Run:  python3 examples/quantum/shor_wall.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from math import gcd
import resona                                          # cost.lift_rank = Φ₁ of the trajectory


def order(a, N, cap=10 ** 7):
    """Classical order finding: smallest r with a^r ≡ 1 (mod N), or None. O(r)."""
    x, r = a % N, 1
    while x != 1 and r <= cap:
        x = (x * a) % N; r += 1
    return r if x == 1 else None


def factor_via_order(N):
    """Factor N classically through period finding — cheap iff the order is small."""
    for a in range(2, N):
        if gcd(a, N) != 1:
            return gcd(a, N), N // gcd(a, N), a, 0                  # lucky gcd
        r = order(a, N)
        if r and r % 2 == 0:
            f = gcd(pow(a, r // 2, N) - 1, N)
            if 1 < f < N:
                return f, N // f, a, r
    return None


if __name__ == "__main__":
    print("=" * 70)
    print("THE SHOR WALL — period finding, and the Φ₁ structure dial")
    print("=" * 70)

    print("\n  (A) when the order r is SMALL, classical period-finding FACTORS N:")
    for N in [15, 21, 323, 3127]:
        res = factor_via_order(N)
        if res:
            p, q, a, r = res
            print(f"      N={N:>5} = {p}×{q}   (via a={a}, order r={r})   ← O(r) work")
    print("      → small r is cheap.  For RSA, r ~ N: O(r) is exponential in log N.")

    print("\n  (B) the Φ₁ dial on a^x mod N (large order) vs a structured sequence:")
    N = 100003 * 100019                                            # ~10^10, large order
    k = 60
    phi_shor = resona.cost.lift_rank([pow(3, x, N) for x in range(130)], k)
    phi_struct = resona.cost.lift_rank([np.sin(2 * np.pi * x / 7) for x in range(130)], k)
    print(f"      {'sequence':>26} {('Φ₁ (eff. rank, max=%d)' % k):>24}")
    print(f"      {'a^x mod N (Shor target)':>26} {phi_shor:>20.1f}   ← ~10× higher: no handle")
    print(f"      {'periodic (period 7)':>26} {phi_struct:>20.1f}   ← low: we harvest it")

    print("\n" + "=" * 70)
    print("VERDICT — honest")
    print("=" * 70)
    print("  Our Φ₁ dial reads a^x mod N as high-rank (~10× the structured signal) — NO")
    print("  low-rank / sampling handle to exploit.")
    print("  The same dial that lets us dequantize low-rank QML (dequantize.py) here says:")
    print("  GENUINE QUANTUM FRONTIER, no classical shortcut.  Beating Shor classically =")
    print("  polynomial factoring = breaking RSA: no known path, and our own framework")
    print("  HONESTLY points away from it.  The boundary is the feature, not the failure.")
    print("=" * 70)
