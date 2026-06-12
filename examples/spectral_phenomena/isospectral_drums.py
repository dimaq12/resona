"""
YOU CANNOT HEAR THE SHAPE OF A DRUM — BUT YOU CAN HEAR IT FROM A POINT.

Kac's 1966 question ("can one hear the shape of a drum?") has the famous
answer NO: non-isomorphic objects can be exactly isospectral — the spectrum
discards information.  This stand makes both halves of that statement
executable on the smallest classical pair:

    G1 = the star K(1,4)        G2 = C4 + an isolated vertex

Their adjacency spectra are IDENTICAL: {−2, 0, 0, 0, +2}.  Every GLOBAL
spectral read agrees — density, moments, trace of any f.  What tells them
apart is the LOCAL response: μ_v = Σ |⟨v|ψ_i⟩|² δ(λ_i) seen from a vertex —
exactly resona's `local_spectrum`, the vector-resolved measure that the
trace averages away.

This is the library's quotient story made audible (THEORY: "operators modulo
basis"): the spectrum is the invariant of the quotient; the BASIS layer —
who sits where — is what a measure discards (Horn), and the local probe is
the cheapest read that reaches back into it.

Run:  python3 examples/spectral_phenomena/isospectral_drums.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import resona

# the classical cospectral pair (Schwenk): star K(1,4) vs C4 ∪ {v}
A1 = np.zeros((5, 5)); A1[0, 1:] = 1; A1[1:, 0] = 1          # star, hub = 0
A2 = np.zeros((5, 5))
for i in range(4):
    A2[i, (i + 1) % 4] = A2[(i + 1) % 4, i] = 1              # C4; vertex 4 isolated

if __name__ == "__main__":
    print("=" * 72)
    print("ISOSPECTRAL DRUMS — global reads agree, the LOCAL response does not")
    print("=" * 72)
    e1 = np.sort(np.linalg.eigvalsh(A1)); e2 = np.sort(np.linalg.eigvalsh(A2))
    print(f"\n  spectra (dense check):  star {np.round(e1, 10)}")
    print(f"                          C4+v {np.round(e2, 10)}")
    print(f"  max |difference| = {np.max(np.abs(e1 - e2)):.1e}  — EXACTLY isospectral")

    # every global resona read agrees
    xs = np.linspace(-2.5, 2.5, 7)
    f = lambda x: np.exp(0.3 * x)
    tr = [float(np.sum(f(e))) for e in (e1, e2)]
    print(f"\n  global reads:  Tr e^{{0.3A}}: {tr[0]:.6f} vs {tr[1]:.6f}  (equal)")
    print(f"  → no trace functional, density, moment or extreme can separate them.")

    # the LOCAL response from each vertex — the basis layer speaks
    print(f"\n  local_spectrum (the vector-resolved measure), per vertex:")
    print(f"    {'vertex':>7} {'star: weight on λ=±2':>22} {'C4+v: weight on λ=±2':>22}")
    rows = []
    for v in range(5):
        ws = []
        for A in (A1, A2):
            e = np.zeros(5); e[v] = 1.0
            th, w = resona.local_spectrum(lambda x: A @ x, e, k=5)
            ws.append(float(w[np.abs(np.abs(th) - 2.0) < 1e-9].sum()))
        rows.append(ws)
        print(f"    {v:>7} {ws[0]:>22.4f} {ws[1]:>22.4f}")
    star_prof = sorted(r[0] for r in rows)
    c4_prof = sorted(r[1] for r in rows)
    print(f"\n  sorted local profiles: star {np.round(star_prof, 3)}")
    print(f"                         C4+v {np.round(c4_prof, 3)}")
    assert star_prof != c4_prof
    print(f"  → DIFFERENT: the star's hub holds weight 1.0 on the band edges, the")
    print(f"    leaf holds 0.25; the cycle spreads 0.5 evenly; the isolated vertex")
    print(f"    holds 0.  One Lanczos from ONE chosen vector hears the shape that")
    print(f"    the whole spectrum cannot.")

    print("\n" + "=" * 72)
    print("  Kac, executable: the spectrum is the quotient invariant (basis")
    print("  forgotten); local_spectrum reaches back into the basis layer.  What")
    print("  the drum hides from the concert hall, it tells the stethoscope.")
    print("=" * 72)
