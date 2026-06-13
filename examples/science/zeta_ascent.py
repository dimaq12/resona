"""
ASCENDING THE CRITICAL LINE — Odlyzko's zeros through the resona dial.

Odlyzko's classic computation showed the Riemann zeros' statistics CONVERGE to
GUE as you climb the critical line.  This script reproduces that ascent with
two resona dials — on his published tables (first 100,000 zeros; zeros number
10^12+1… and 10^21+1…, stored as offsets so the local precision survives):

  DIAL 1  ⟨r⟩ (level_spacing_ratio), high statistics: climbing the line the
          zeros' ⟨r⟩ falls from 0.611 (first 100k, low t) onto the GUE bulk
          value 0.5996.  HONESTLY: each high table is only 10^4 zeros, so ⟨r⟩
          there is pinned to ~±0.002 (1σ) — about TWO digits, not four.  At
          #10^12 ⟨r⟩ = 0.600 sits right on GUE, well inside that one error bar;
          that agreement-within-noise IS the convergence, but it is a 2-digit
          read, and the printout carries its bootstrap error bar to say so.

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
# Cap BLAS threads BEFORE numpy import: oversubscription on many-core / shared
# boxes can turn the n=2400 eigvalsh controls from seconds into minutes (the
# audit's 300-550s timeout).  4 threads is plenty for these dense solves.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
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


def r_values(levels, eps=1e-9):
    """The per-pair min/max spacing ratios whose mean is level_spacing_ratio."""
    s = np.diff(np.sort(np.asarray(levels, float)))
    s = s[s > eps]
    return np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])


def r_mean_err(levels):
    """⟨r⟩ and its 1σ sampling error (analytic SE = std/√n; the ratios are
    weakly correlated so this is a slight under-estimate — good enough to show
    that 10^4 zeros pin ⟨r⟩ to ~±0.002, i.e. ~two digits)."""
    r = r_values(levels)
    return float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r)))


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
    print(f"\n  DIAL 1 — ⟨r⟩ over whole tables (ref GUE bulk 0.5996, Poisson 0.386):", flush=True)
    for name, z in [("first 100,000 zeros (t ≤ 74 921)", z1),
                    ("10⁴ zeros at #10^12  (t ≈ 2.7·10¹¹)", z12),
                    ("10⁴ zeros at #10^21  (t ≈ 1.4·10²⁰)", z21)]:
        m, se = r_mean_err(z)
        print(f"    {name:>38}:  ⟨r⟩ = {m:.4f} ± {se:.4f}", flush=True)
    print(f"    → ⟨r⟩ falls onto GUE as t climbs; at #10^12 it reads 0.600 — bang on")
    print(f"      GUE 0.5996, but with only 10⁴ zeros that is a ~±0.002 (1σ) read:")
    print(f"      agreement to TWO digits, well inside one error bar.  (Odlyzko used")
    print(f"      ~10⁸ zeros to push this to many digits; one table here ≠ four digits.)")

    # ── DIAL 2: Jacobi β-rigidity vs height, shared controls ────────────────
    print(f"\n  DIAL 2 — building {N_GUE} GUE controls (n={3 * M} dense eigvalsh) …",
          flush=True)
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
        print(f"    {name:>38} {s_z:12.3f} {g_mean / s_z:8.2f}x", flush=True)
    print(f"\n    → the zeta operator is SMOOTHER than GUE (Berry saturation), and the")
    print(f"      excess decays toward 1 as the saturation scale L* ~ ln t outgrows")
    print(f"      the window — the operator-coefficient face of Odlyzko's convergence.")

    print("\n" + "=" * 76)
    print("  Two dials, three heights, one ascent: ⟨r⟩ settles onto GUE 0.5996 (a")
    print("  ~2-digit, ±0.002 read at 10⁴ zeros — not four), and the Jacobi rigidity")
    print("  dial watches the zeros' extra stiffness fade toward pure random-matrix")
    print("  behaviour on the way up the critical line.")
    print("=" * 76)
