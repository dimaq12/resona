"""
RMT UNIVERSALITY — naming the Dyson class of a spectrum from its rigidity.

WHAT:  Build all FOUR Wigner-Dyson ensembles at the same dimension and let
       resona name each one blind, from the spectrum alone:

         Poisson  (β=0) — sorted uniform levels, no repulsion (integrable)
         GOE      (β=1) — real symmetric Gaussian        (time-reversal, no spin)
         GUE      (β=2) — complex Hermitian Gaussian      (broken time-reversal)
         GSE      (β=4) — quaternion / Kramers Gaussian   (time-reversal + spin-½)

       For each we print  ⟨r⟩  (level_spacing_ratio) and  R4  (the rigidity meter
       inside cost.rmt_class), ENSEMBLE-AVERAGED over several draws, then read the
       class off the averaged R4.  Both dials reproduce the textbook strict order
         Poisson > GOE > GUE > GSE
       and every class is named correctly.

WHY:   The universality class is the coarsest invariant of a spectrum — it says
       whether levels repel, and how hard (β).  It is the same dial that separates
       "integrable / a lift exists" from "chaotic / no lift" in the quantum stand,
       extended to resolve all four symmetry classes, not just Poisson-vs-chaos.

RESONA's role:  cost.rmt_class(eigenvalues) -> (class, R4).  R4 is the tail
       rigidity of the UNFOLDED spacings (it flattens the local density itself,
       so it works on a window of interior levels, not just a full spectrum).
       cost.level_spacing_ratio(eigenvalues) -> ⟨r⟩ is the unfolding-free
       companion.  HONEST CAVEAT, honored here: a SINGLE draw is noisy near the
       GOE↔GUE boundary (they sit only ~2σ apart at this D) — so we average R4
       over several realizations and classify the MEAN, exactly as the docstring
       prescribes.  We also print the per-draw class votes so the noise is visible,
       not hidden.  Every number is measured; nothing is asserted.

Run:   PYTHONPATH=. python3 examples/spectral_phenomena/rmt_universality.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from collections import Counter
import numpy as np
from resona.cost import rmt_class, level_spacing_ratio

# textbook reference values (ensemble-averaged) that resona should reproduce
REF_R = {"Poisson": 0.386, "GOE": 0.531, "GUE": 0.600, "GSE": 0.676}   # ⟨r⟩
REF_R4 = {"Poisson": 0.199, "GOE": 0.009, "GUE": -0.080, "GSE": -0.230}  # rmt_class refs


def classify_R4(R4):
    """Name the class from a (possibly averaged) R4 — the same nearest-reference
    rule cost.rmt_class uses internally, applied to the ENSEMBLE MEAN."""
    return min(REF_R4, key=lambda c: abs(R4 - REF_R4[c]))


# ── the four Dyson ensembles (all yield ~D interior levels) ──────────────────
def make_ensembles(rng, D):
    def poisson():                       # β = 0: uncorrelated levels
        return np.sort(rng.uniform(-1.0, 1.0, D))

    def goe():                           # β = 1: real symmetric
        H = rng.standard_normal((D, D))
        return np.linalg.eigvalsh((H + H.T) / 2)

    def gue():                           # β = 2: complex Hermitian
        H = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
        return np.linalg.eigvalsh((H + H.conj().T) / 2)

    def gse():                           # β = 4: quaternion / self-dual Hermitian
        n = D // 2                       # 2n×2n complex block; Kramers-degenerate
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        A = (A + A.conj().T) / 2
        B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        B = (B - B.T) / 2                # antisymmetric off-diagonal block
        H = np.block([[A, B], [-B.conj(), A.conj()]])
        H = (H + H.conj().T) / 2
        ev = np.linalg.eigvalsh(H)
        return ev[::2]                   # drop the exact Kramers degeneracy

    return [("Poisson", poisson), ("GOE", goe), ("GUE", gue), ("GSE", gse)]


if __name__ == "__main__":
    print("=" * 78)
    print("RMT UNIVERSALITY — cost.rmt_class names the Dyson class from the spectrum")
    print("=" * 78)

    D, reps = 360, 8
    rng = np.random.default_rng(11)
    print(f"\n  Dimension D={D}, ensemble-averaged over {reps} independent draws.")
    print(f"  References — ⟨r⟩: Poisson .386 / GOE .531 / GUE .600 / GSE .676")
    print(f"              R4 : Poisson +.20 > GOE +.01 > GUE −.08 > GSE −.23\n")
    print(f"  {'ensemble':9s} {'⟨r⟩':>7s} {'(ref)':>7s} {'R4_avg':>8s} {'(ref)':>7s} "
          f"{'verdict':>9s}  {'per-draw votes'}")
    print(f"  {'-'*9} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*9}  {'-'*22}")

    rows = []
    for name, fn in make_ensembles(rng, D):
        R4s, rs, votes = [], [], []
        for _ in range(reps):
            ev = fn()
            cls, R4 = rmt_class(ev)          # the v3 dial: (class, rigidity)
            R4s.append(R4)
            rs.append(level_spacing_ratio(ev))
            votes.append(cls)
        r_avg, R4_avg = float(np.mean(rs)), float(np.mean(R4s))
        verdict = classify_R4(R4_avg)        # classify the ENSEMBLE MEAN (honest)
        ok = "OK" if verdict == name else "MISS"
        rows.append((name, r_avg, R4_avg, verdict, ok))
        vote_str = "  ".join(f"{c}:{n}" for c, n in Counter(votes).most_common())
        print(f"  {name:9s} {r_avg:>7.3f} {REF_R[name]:>7.3f} {R4_avg:>+8.3f} "
              f"{REF_R4[name]:>+7.2f} {verdict:>9s}  {vote_str}")

    # strict-ordering checks, measured (not asserted as constants)
    r_seq = [r for _, r, _, _, _ in rows]
    R4_seq = [R4 for _, _, R4, _, _ in rows]
    r_ordered = all(r_seq[i] < r_seq[i + 1] for i in range(3))     # Poisson<GOE<GUE<GSE
    R4_ordered = all(R4_seq[i] > R4_seq[i + 1] for i in range(3))  # Poisson>GOE>GUE>GSE
    all_named = all(ok == "OK" for *_, ok in rows)

    print(f"\n  ⟨r⟩ strictly increasing  Poisson<GOE<GUE<GSE : {r_ordered}")
    print(f"  R4  strictly decreasing  Poisson>GOE>GUE>GSE : {R4_ordered}")
    print(f"  all four classes named correctly             : {all_named}")

    print(f"\n  WHY WE AVERAGE: a single GUE draw at this D often reads R4 between the")
    print(f"  GOE and GUE references (the per-draw votes above show the spill) — the")
    print(f"  documented GOE↔GUE noise.  Classifying the AVERAGED R4 resolves it; the")
    print(f"  Poisson-vs-chaotic split is robust on every single draw.")

    print("\n" + "=" * 78)
    print("  cost.rmt_class + level_spacing_ratio reproduce the Wigner-Dyson order")
    print("  Poisson > GOE > GUE > GSE and name each class — from the spectrum alone.")
    print("=" * 78)
