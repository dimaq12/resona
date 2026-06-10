"""
resolvent_algebra/burgers_shock.py
=============================================================================
THE DEFECT IS A SHOCK.  Free convolution = a heat flow whose PDE is the complex
inviscid Burgers equation; its SHOCKS are the spectral edges = the defect.

Adding free semicircle of variance t to μ₀ evolves the Cauchy transform G(t,z)
by  ∂_t G + G ∂_z G = 0  (complex Burgers / Hopf), solved implicitly by the
subordination  G_t(z) = G₀(z − t·G_t(z)).  Characteristics z(t)=x₀+t·G₀(x₀) are
straight lines; where they CROSS a shock forms — that is the spectral edge,
where the DOS support boundary / a band merger appears.

Demo: μ₀ = ½δ₋₁ + ½δ₊₁ (a gap at 0).  As t grows, the gap CLOSES at a critical
t_c — a band-merger PHASE TRANSITION = a Burgers shock collision.
Analytic shock time at x=0:  t_c = −1/G₀'(0).  For two atoms G₀'(0)=−1 ⇒ t_c=1.

This closes the loop to defect_calculus: the DEFECT is literally a PDE SHOCK.

Run:  python3 resolvent_algebra/burgers_shock.py
"""
import numpy as np


def G0(w):
    return 0.5 / (w - 1) + 0.5 / (w + 1)            # Cauchy transform of ½δ₋₁+½δ₊₁


def solve_G(z, t, iters=8000, tol=1e-12):
    g = -0.3j
    for _ in range(iters):
        gn = G0(z - t * g)
        if abs(gn - g) < tol:
            return gn
        g = 0.5 * g + 0.5 * gn                      # damped (critical slowing near t_c)
    return g


def dos(x, t, eta=1e-3):
    return max(-solve_G(x + 1j * eta, t).imag / np.pi, 0.0)


if __name__ == "__main__":
    print("=" * 72)
    print("THE DEFECT IS A SHOCK — free heat flow = complex Burgers; edges = shocks")
    print("=" * 72)
    Gp0 = -0.5 / (0 - 1) ** 2 - 0.5 / (0 + 1) ** 2  # G₀'(0)
    t_c = -1.0 / Gp0
    print(f"  μ₀ = ½δ₋₁+½δ₊₁ (gap at 0).  analytic shock time t_c = −1/G₀'(0) = {t_c:.2f}\n")
    print(f"  flow t  |  DOS at gap centre x=0   gap")
    for t in [0.2, 0.5, 0.8, 0.95, 1.0, 1.05, 1.3, 1.8]:
        d0 = dos(0.0, t)
        state = "OPEN (gap)" if d0 < 1e-3 else "CLOSED (merged)"
        bar = "█" * int(60 * d0)
        mark = "  ← t_c" if abs(t - t_c) < 0.06 else ""
        print(f"  {t:>5.2f}   |   {d0:>8.4f}   {state}{mark}  {bar}")

    print(f"\n  band profile below vs above the shock:")
    for t in [0.5, 1.8]:
        xs = np.linspace(-2.2, 2.2, 23)
        prof = "".join(" .:-=+*#%@"[min(9, int(9 * dos(x, t) / 0.6))] for x in xs)
        print(f"    t={t}:  |{prof}|")

    print("\n" + "=" * 72)
    print(f"  The gap CLOSES at t_c={t_c:.2f} — a band-merger phase transition = the")
    print(f"  collision of two Burgers shocks (characteristics crossing at x=0).")
    print(f"  Free convolution is a HEAT FLOW; the spectral EDGE is its SHOCK; the SHOCK")
    print(f"  is the DEFECT.  This is the same critical point where the subordination map")
    print(f"  loses contraction (subordination_chaos) — and it is literally a PDE shock,")
    print(f"  closing the loop to defect_calculus, where the journey began.")
    print("=" * 72)
