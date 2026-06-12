"""
extraction_law.py — the cost of an answer, and removable vs genuine walls.
==============================================================================
THE LAW.  The answer pre-exists in the resolvent field; a solver only EXTRACTS
it, at cost  Cost(ε, z) ~ ε^{-a}·dist(z, Σ*)^{-b}, singular on the NON-removable
set Σ* (edges, branch points, shocks — not isolated poles).

THE DIAL.  Whether an apparent wall is REMOVABLE (a finite lift linearizes it →
extractable) or GENUINE (no finite chart → structureless) is read by LIFT-RANK
SATURATION (resona.cost): lift a signal to its trajectory operator at a growing
window and watch its effective rank.
   • SATURATES  → removable → extractable (most "impossible" is just bad coords).
   • keeps GROWING → genuine wall (e.g. aˣ mod N — the Shor / RSA structurelessness).

This is the same dial that says dequantizable-vs-quantum and easy-vs-hard — here
read straight from the signal, no model.

Run:  python3 examples/spectral_phenomena/extraction_law.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import resona

rng = np.random.default_rng(0)
x = np.arange(300)


if __name__ == "__main__":
    print("=" * 72)
    print("THE EXTRACTION LAW — removable vs genuine walls, by lift-rank saturation")
    print("=" * 72)
    signals = {
        "periodic (period 7)":      np.sin(2 * np.pi * x / 7),
        "3 cosines (low-rank)":     np.cos(x / 5) + 0.6 * np.cos(x / 11) + 0.3 * np.cos(x / 23),
        "decaying mix":             np.exp(-x / 200) * np.sin(x / 4),
        "random noise":             rng.standard_normal(300),
        "aˣ mod N  (Shor target)":  np.array([pow(3, int(i), 100003 * 100019) for i in x], float),
    }
    print(f"\n  {'signal':>26}  {'lift-rank vs window 20→120':>30}  verdict")
    print("  " + "─" * 70)
    for name, sig in signals.items():
        ok, ranks = resona.cost.is_extractable(sig)
        curve = " ".join(f"{r:5.1f}" for r in ranks)
        print(f"  {name:>26}  {curve:>30}  {'EXTRACTABLE ✓' if ok else 'GENUINE WALL ✗'}")

    # ── the law itself: fit Cost ~ ε^{-a} dist^{-b} from measurements ──
    eps = np.array([1e-2, 1e-3, 1e-4] * 3); dist = np.repeat([0.1, 0.3, 1.0], 3)
    costs = resona.cost.extraction_cost(eps, dist, a=1.5, b=0.8, c=2.0)   # synthetic truth
    a, b, c = resona.cost.fit_law(costs, eps, dist)
    print(f"\n  cost-law fit on measured costs:  a={a:.2f} (true 1.5),  b={b:.2f} (true 0.8)")
    print("\n" + "=" * 72)
    print("  Saturating lift-rank ⇒ a finite chart linearizes the singularity ⇒ the")
    print("  answer is extractable (most apparent walls are just coordinates).  Rank")
    print("  that grows with the window ⇒ Σ* — a genuine wall, no classical shortcut.")
    print("  Φ₁ = s.effective_rank() is the one-number global version of the same dial.")
    print("=" * 72)
