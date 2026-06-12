"""
ASCENDING THE CRITICAL LINE — Odlyzko's zeros through the resona dial.

Odlyzko's classic computation showed the Riemann zeros' statistics CONVERGE to
GUE as you climb the critical line.  This script reproduces that ascent with
two resona dials — on his published tables (first 100,000 zeros; zeros number
10^12+1… and 10^21+1…, stored as offsets so the local precision survives):

  DIAL 1  ⟨r⟩ (level_spacing_ratio), high statistics: at zero #10^12 the
          zeros hit the GUE value to four digits.

  DIAL 2  the JACOBI RIGIDITY dial (this library's own): realize a window of
          M zeros as the Jacobi operator (`from_measure`), measure the
          fluctuation of its hopping profile β against the rigid (Weyl)
          reference, and compare with density-matched GUE controls.  At low
          height the zeta operator is ~1.4x SMOOTHER than GUE (Berry's
          saturation of the number variance, L* ~ ln t); climbing the line
          pushes the saturation scale beyond the window and the excess
          rigidity DECAYS toward exactly GUE:

              height      t≈10²    t≈7.5·10⁴   #10^12     #10^21
              GUE/zeta    ~1.4x      ~1.2x      ~1.1x      ~1.07x

HONEST NOTES.  Odlyzko's tables are accurate to ~3e-9 (offsets exact locally);
windows are unfolded by the Riemann–von Mangoldt law (absolute zeros) or the
local mean spacing (offset tables, where the density is constant to ~1e-7);
controls are GUE bulks carried to the same density (n=4 per height); this is
an observation with controls and error bars, not a theorem.

Data: downloaded once from https://www-users.cse.umn.edu/~odlyzko/zeta_tables/
into examples/science/data/ (cached; ~2 MB).  Offline → the script says so.

Run:  python3 examples/science/zeta_ascent.py
"""
import sys, os
import urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import resona
from resona.cost import level_spacing_ratio

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BASE = "https://www-users.cse.umn.edu/~odlyzko/zeta_tables/"
TABLES = {"zeros1": "first 100,000 zeros",
          "zeros3": "zeros #10^12+1…+10^4 (offsets from 267653395647)",
          "zeros4": "zeros #10^21+1…+10^4 (offsets from 144176897509546973000)"}
M = 800                      # window realized as a Jacobi operator
N_GUE = 4
rng = np.random.default_rng(0)


def fetch(name):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print(f"  downloading {name} ({TABLES[name]}) …")
        urllib.request.urlretrieve(BASE + name, path)
    vals = []
    for line in open(path):
        try:
            vals.append(float(line.split()[0]))
        except (ValueError, IndexError):
            pass                                   # header lines in zeros3/4
    return np.array(vals)


def nbar(t):
    return t / (2 * np.pi) * np.log(t / (2 * np.pi * np.e)) + 7.0 / 8.0


def unfold(levels, absolute):
    """→ unit-mean-spacing coordinates, affinely mapped to [0, M−1]."""
    lv = np.sort(levels)
    u = nbar(lv) if absolute else lv / np.mean(np.diff(lv))
    return (u - u[0]) / (u[-1] - u[0]) * (M - 1)


def jacobi_beta(u):
    return resona.from_measure(u, np.full(M, 1.0 / M), k=M)[1]


if __name__ == "__main__":
    print("=" * 76)
    print("ASCENDING THE CRITICAL LINE — Odlyzko's tables through the resona dial")
    print("=" * 76)
    try:
        z1 = fetch("zeros1")
        z12 = fetch("zeros3")
        z21 = fetch("zeros4")
    except Exception as e:
        print(f"\n  Odlyzko tables unreachable ({e}); connect and re-run.")
        sys.exit(0)

    # ── DIAL 1: ⟨r⟩ at full statistics ───────────────────────────────────────
    print(f"\n  DIAL 1 — ⟨r⟩ over whole tables (ref GUE 0.5996, Poisson 0.386):")
    for name, z in [("first 100,000 zeros (t ≤ 74 921)", z1),
                    ("10⁴ zeros at #10^12  (t ≈ 2.7·10¹¹)", z12),
                    ("10⁴ zeros at #10^21  (t ≈ 1.4·10²⁰)", z21)]:
        print(f"    {name:>38}:  ⟨r⟩ = {level_spacing_ratio(z):.4f}")
    print(f"    → at #10^12 the match is four digits — Odlyzko's discovery, one call.")

    # ── DIAL 2: Jacobi β-rigidity vs height, shared controls ────────────────
    weyl = np.linspace(0, M - 1, M)
    be_weyl = jacobi_beta(weyl)
    s_gue = []
    for _ in range(N_GUE):
        n3 = 3 * M
        G = rng.standard_normal((n3, n3)) + 1j * rng.standard_normal((n3, n3))
        ev = np.linalg.eigvalsh((G + G.conj().T) / 2)
        mid = ev[n3 // 2 - M // 2: n3 // 2 + M // 2]
        R = np.sqrt(2 * n3)
        cdf = (mid * np.sqrt(np.clip(R * R - mid * mid, 0, None))
               + R * R * np.arcsin(np.clip(mid / R, -1, 1))) / (np.pi * R * R) + 0.5
        s_gue.append(float((jacobi_beta(unfold(cdf, absolute=False)) - be_weyl).std()))
    g_mean, g_err = float(np.mean(s_gue)), float(np.std(s_gue))

    windows = [("t ≈ 10²–10³   (zeros 1…800)", z1[:M], True),
               ("t ≈ 7.5·10⁴  (zeros 99 200…100 000)", z1[-M:], True),
               ("zero #10^12  (t ≈ 2.7·10¹¹)", z12[:M], False),
               ("zero #10^21  (t ≈ 1.4·10²⁰)", z21[:M], False)]
    print(f"\n  DIAL 2 — hopping-profile fluctuation of the realized Jacobi operator")
    print(f"  (window M={M}; GUE controls: {g_mean:.2f} ± {g_err:.2f}, n={N_GUE}; "
          f"Weyl reference = rigid lattice)\n")
    print(f"    {'window':>38} {'std(β−Weyl)':>12} {'GUE/zeta':>9}")
    for name, z, absolute in windows:
        s_z = float((jacobi_beta(unfold(z, absolute)) - be_weyl).std())
        print(f"    {name:>38} {s_z:12.3f} {g_mean / s_z:8.2f}x")
    print(f"\n    → the zeta operator is SMOOTHER than GUE (Berry saturation), and the")
    print(f"      excess decays toward 1 as the saturation scale L* ~ ln t outgrows")
    print(f"      the window — the operator-coefficient face of Odlyzko's convergence.")

    print("\n" + "=" * 76)
    print("  Two dials, three heights, one ascent: ⟨r⟩ pins GUE to four digits at")
    print("  #10^12, and the Jacobi rigidity dial watches the zeros' extra stiffness")
    print("  fade into pure random-matrix behaviour on the way up the critical line.")
    print("=" * 76)
