"""
quantum/integrability_detector.py
==============================================================================
IS THIS HAMILTONIAN INTEGRABLE?  Two dials, no prior knowledge.

"Integrable" = enough conserved charges = a LIFT exists (the system is secretly
linear in some chart) = cheap to compute.  "Chaotic" = no charges, no lift —
the genuine frontier.  resona reads the answer two independent ways:

  DIAL 1 — resona.cost.level_spacing_ratio(λ): the spectral statistics.
     ⟨r⟩ ≈ 0.386 (Poisson)  → levels don't repel → integrable, a lift exists
     ⟨r⟩ ≈ 0.531 (GOE)      → level repulsion    → chaotic, no lift
     Unfolding-free, 3 lines, O(N log N).

  DIAL 2 — resona.lift.conserved_charge(H, basis): the CONSTRUCTIVE check.
     A blind variational search over k-local operators for the Q that best
     commutes with H (the commutator-Gram eigenproblem).  It does not test
     integrability — it FINDS the charges, with no prior knowledge of what
     they are; for a chaotic H it honestly returns none beyond H itself.

THE TRAP (shown live below): dial 1 measured WITHOUT resolving symmetry
sectors fakes Poisson — levels from different sectors don't repel, so a
chaotic Hamiltonian reads "integrable".  Project into ONE sector first
(here: magnetization + reflection).  This was a real debugging catch in the
research program; the library docstring carries the warning.

Run:  python3 examples/quantum/integrability_detector.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from itertools import combinations, product
from resona.cost import level_spacing_ratio, rmt_class
from resona.lift import conserved_charge


# ── spin-chain builder, magnetization sector + reflection block ──────────────
def bit(s, i):
    return (s >> i) & 1


def build_xxz_sector(L, J2=0.0, delta=0.5, project_reflection=True):
    """Heisenberg XXZ (+ next-nearest J2) in the half-filling Sz sector;
    optionally projected onto the reflection-symmetric block."""
    ups = (L + 1) // 2
    states = [sum(1 << i for i in c) for c in combinations(range(L), ups)]
    idx = {s: i for i, s in enumerate(states)}
    D = len(states)
    H = np.zeros((D, D))
    bonds = ([(i, i + 1, 1.0) for i in range(L - 1)]
             + ([(i, i + 2, J2) for i in range(L - 2)] if J2 else []))
    for a, s in enumerate(states):
        for i, j, J in bonds:
            zi, zj = 2 * bit(s, i) - 1, 2 * bit(s, j) - 1
            H[a, a] += J * delta * 0.25 * zi * zj
            if bit(s, i) != bit(s, j):
                t = s ^ (1 << i) ^ (1 << j)
                H[idx[t], a] += J * 0.5
    if not project_reflection:
        return H
    refl = lambda s: sum(1 << (L - 1 - i) for i in range(L) if bit(s, i))
    cols, used = [], set()
    for s in states:
        r = refl(s)
        if s in used or (r in idx and r < s):
            continue
        v = np.zeros(D); v[idx[s]] += 1
        if r != s:
            v[idx[r]] += 1
        cols.append(v / np.linalg.norm(v)); used.add(s)
    P = np.array(cols).T
    return P.T @ H @ P


# ── Pauli-string basis for the blind charge search ───────────────────────────
_P = {0: np.eye(2, dtype=complex),
      1: np.array([[0, 1], [1, 0]], complex),
      2: np.array([[0, -1j], [1j, 0]], complex),
      3: np.array([[1, 0], [0, -1]], complex)}


def string(L, types):
    m = np.array([[1.0 + 0j]])
    for t in types:
        m = np.kron(m, _P[t])
    return m


def klocal_basis(L, width=2):
    ops = []
    for types in product(range(4), repeat=L):
        sup = [i for i, t in enumerate(types) if t]
        if not sup or max(sup) - min(sup) >= width:
            continue
        ops.append(string(L, types))
    return ops


if __name__ == "__main__":
    print("=" * 74)
    print("IS IT INTEGRABLE?  ⟨r⟩ statistics + blind conserved-charge search")
    print("=" * 74)

    # ── DIAL 1: level-spacing ratio, sector-resolved ─────────────────────────
    L = 13
    print(f"\n  DIAL 1 — level_spacing_ratio, XXZ chain L={L}, one (Sz, reflection) sector")
    print(f"  (middle 70% of the spectrum; references: Poisson 0.386 / GOE 0.531)\n")
    for name, J2 in [("XXZ  (integrable, Bethe ansatz)", 0.0),
                     ("XXZ + NNN coupling  (chaotic)  ", 1.0)]:
        ev = np.linalg.eigvalsh(build_xxz_sector(L, J2=J2))
        n = len(ev)
        bulk = ev[int(0.15 * n):int(0.85 * n)]
        r = level_spacing_ratio(bulk)
        cls, R4 = rmt_class(bulk)                     # DIAL 1b: the universality class
        verdict = "lift EXISTS" if r < 0.45 else "NO lift (chaos)"
        print(f"    {name}  ⟨r⟩ = {r:.3f}  rmt_class = {cls:<8}(R4={R4:+.2f})  → {verdict}")

    # the trap: same chaotic H, sectors NOT resolved → fake Poisson
    ev_mixed = np.linalg.eigvalsh(build_xxz_sector(L, J2=1.0, project_reflection=False))
    n = len(ev_mixed)
    r_fake = level_spacing_ratio(ev_mixed[int(0.15 * n):int(0.85 * n)])
    print(f"\n    THE TRAP: the same chaotic H with sectors UNRESOLVED reads "
          f"⟨r⟩ = {r_fake:.3f}")
    print(f"    — levels from different sectors don't repel, faking Poisson.  Always")
    print(f"    project into one symmetry sector first (see the library docstring).")

    # ── DIAL 2: blind charge search ──────────────────────────────────────────
    L2 = 6
    print(f"\n  DIAL 2 — conserved_charge: blind search over all ≤2-local Pauli strings")
    print(f"  (L={L2}; ‖[H,Q]‖/‖Q‖ < 1e-6 ⇒ a genuine conserved charge, found blind)\n")
    two = lambda a, t: string(L2, tuple(t if k in (a, a + 1) else 0 for k in range(L2)))
    one = lambda a, t: string(L2, tuple(t if k == a else 0 for k in range(L2)))
    H_xx = sum(two(i, 1) + two(i, 2) for i in range(L2 - 1))
    H_ch = (sum(two(i, 3) for i in range(L2 - 1))
            + 1.05 * sum(one(i, 1) for i in range(L2))
            + 0.5 * sum(one(i, 3) for i in range(L2)))
    basis = klocal_basis(L2, width=2)
    for name, H in [("XX chain (integrable)     ", H_xx),
                    ("mixed-field Ising (chaotic)", H_ch)]:
        _, norms = conserved_charge(H, basis)
        found = int(np.sum(norms < 1e-6))
        nontriv = norms[norms > 1e-6][0]
        print(f"    {name}  charges found: {found}   "
              f"best non-charge: ‖[H,Q]‖/‖Q‖ = {nontriv:.3f}")
    print(f"\n    The integrable side yields MORE exact charges (H, total Z, free-fermion")
    print(f"    bilinears); the chaotic side keeps only energy — honest 'no hidden handle'.")

    print("\n" + "=" * 74)
    print("  Two independent dials, one verdict: spectral statistics (⟨r⟩) detect the")
    print("  lift, the commutator-Gram search CONSTRUCTS it.  Same library, same")
    print("  primitive: hardness is measurable before you pay for the computation.")
    print("=" * 74)
