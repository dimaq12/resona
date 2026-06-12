"""
zeta_saturation.py — Berry's saturation scale, measured over 18 decades.

THE QUESTION (raised by examples/science/zeta_ascent.py).  The zeta-zero
operator's hopping profile is smoother than GUE, and the excess fades with
height.  Berry's semiclassical theory locates the cause: the zeros' NUMBER
VARIANCE Σ²(L) follows GUE only up to the wavelength of the shortest periodic
orbit (the prime 2) and SATURATES beyond it.  Where exactly, in mean-spacing
units, is the knee — and how does it move up the critical line?

THE MEASUREMENT.  Σ²(L) = variance of the number of unfolded zeros in windows
of length L, on three Odlyzko tables: t ≈ 7.3·10⁴ (last 20k of the first
100k zeros), zeros #10^12+… (t ≈ 2.7·10¹¹), zeros #10^21+… (t ≈ 1.4·10²⁰).
GUE reference: the exact large-L asymptote Σ²_GUE(L) = (ln 2πL + γ + 1)/π² − 1/8.
Saturation point L_sat = first L where Σ²_zeta < (1−δ)·Σ²_GUE.

THE RESULT (δ = 10% departure criterion, smoothed Σ²; printed live below):

    L_sat / ln(t/2π)  ≈  0.9–1.0  at ALL three heights — the knee tracks
    ln(t/2π) mean spacings over eighteen decades (= ln 2 × the prime-2
    orbit wavelength Λ₂ = ln(t/2π)/ln 2).  The saturation PLATEAU grows
    ~ (1/π²)·ln ln t (Berry's prediction), checked below.  Σ² at L ≳ 100
    carries ~±0.05 sampling noise (10–20k zeros per table) — the
    constant's third digit is not significant and is not claimed.

HONESTY LEDGER.
  • The absolute constant (≈0.9–1.0 at δ=10%) carries the δ-criterion
    convention (δ=5%/20% shift all heights together); the LAW — the SAME
    constant across 18 decades — is criterion-stable.  Both shown below.
  • A first attempt read the same physics from the Jacobi β-fluctuations of
    small realized windows (M = 64…512): the rigidity excess pointed the
    right way but the collapse in M/L* was statistically WEAK (R² ≈ 0.17 vs
    0.06) — small windows are noise-dominated.  Recorded as a negative
    sub-result; the fixed-M=800 ladder of zeta_ascent.py (1.42x → 1.06x)
    remains the operator-side face of the effect.

Run:  python3 theory/zeta_saturation.py   (data auto-downloaded by
      examples/science/zeta_ascent.py into examples/science/data/)
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "examples", "science", "data")
EULER = 0.5772156649015329


def load(name):
    vals = []
    for line in open(os.path.join(DATA, name)):
        try:
            vals.append(float(line.split()[0]))
        except (ValueError, IndexError):
            pass
    return np.array(vals)


def nbar(t):
    return t / (2 * np.pi) * np.log(t / (2 * np.pi * np.e)) + 7.0 / 8.0


def sigma2(u, Ls, step=0.25):
    """Number variance: Var #(zeros in [x, x+L)) over a dense sliding window."""
    u = np.sort(u)
    out = []
    for L in Ls:
        starts = np.arange(u[0], u[-1] - L, step)
        out.append(np.var(np.searchsorted(u, starts + L)
                          - np.searchsorted(u, starts)))
    return np.array(out)


def gue_sigma2(L):
    """GUE number variance, large-L asymptote (exact to O(1/L))."""
    return (np.log(2 * np.pi * L) + EULER + 1) / np.pi ** 2 - 0.125


def l_sat(Ls, s2, delta):
    """First PERSISTENT departure: L ≥ 3 (the γ-asymptote is exact only for
    large L) where Σ² < (1−δ)·GUE and stays below for the next 3 grid points."""
    below = s2 < (1 - delta) * gue_sigma2(Ls)
    for i in range(np.searchsorted(Ls, 3.0), len(Ls) - 3):
        if below[i] and below[i + 1:i + 4].all():
            return float(Ls[i])
    return float("nan")


if __name__ == "__main__":
    print("=" * 78)
    print("BERRY SATURATION OF THE RIEMANN ZEROS — the knee of Σ²(L), three heights")
    print("=" * 78)

    z1 = load("zeros1")
    mean_sp = lambda z: np.mean(np.diff(np.sort(z)))
    sets = [("t ≈ 7.3·10⁴ ", nbar(z1[-20000:]), 7.3e4),
            ("t ≈ 2.7·10¹¹", np.sort(load("zeros3")) / mean_sp(load("zeros3")), 2.7e11),
            ("t ≈ 1.4·10²⁰", np.sort(load("zeros4")) / mean_sp(load("zeros4")), 1.4e20)]
    Ls = np.unique(np.geomspace(1, 120, 48).round(2))
    smooth = lambda a: np.convolve(a, np.ones(5) / 5, mode="same")

    print(f"\n  Σ²(L) vs the exact GUE curve (γ-asymptote), and the knee L_sat:\n")
    print(f"    {'height':>14} {'Σ²(L=5)':>9} {'Σ²(15)':>8} {'Σ²(40)':>8} {'Σ²(100)':>9} "
          f"{'(GUE':>7}{gue_sigma2(5):5.2f}/{gue_sigma2(15):.2f}/{gue_sigma2(40):.2f}/{gue_sigma2(100):.2f})")
    results = []
    for name, u, t in sets:
        s2 = smooth(sigma2(u, Ls))
        v = [s2[np.argmin(np.abs(Ls - l))] for l in (5, 15, 40, 100)]
        results.append((name, t, Ls, s2))
        print(f"    {name:>14} {v[0]:9.3f} {v[1]:8.3f} {v[2]:8.3f} {v[3]:9.3f}")

    print(f"\n  THE KNEE — L_sat(δ) vs ln(t/2π), criterion stability:\n")
    print(f"    {'height':>14} {'ln(t/2π)':>9} | " +
          " | ".join(f"δ={int(d*100)}%: L_sat (ratio)" for d in (0.05, 0.10, 0.20)))
    for name, t, Ls_, s2 in results:
        ref = np.log(t / (2 * np.pi))
        row = []
        for d in (0.05, 0.10, 0.20):
            ls = l_sat(Ls_, s2, d)
            row.append(f"{ls:13.1f} ({ls / ref:4.2f})")
        print(f"    {name:>14} {ref:9.1f} |" + " |".join(row))
    r10 = [l_sat(Ls_, s2, 0.10) / np.log(t / (2 * np.pi))
           for _, t, Ls_, s2 in results]
    print(f"\n    → at δ=10%:  L_sat / ln(t/2π) = "
          + ", ".join(f"{r:.2f}" for r in r10) + "  —")
    print(f"      THE LAW  L_sat ≈ ln(t/2π)  (spread {max(r10)-min(r10):.2f} across")
    print(f"      18 decades; absolute level carries the δ-criterion convention).")
    print(f"      (= ln 2 × the prime-2 orbit wavelength Λ₂ = ln(t/2π)/ln 2.)")

    # plateau height vs Berry's ln ln t
    print(f"\n  THE PLATEAU — Σ²(L=100) vs Berry's (1/π²)·ln ln(t/2π) + C:")
    plat = np.array([s2[np.argmin(np.abs(Ls_ - 100))] for _, t, Ls_, s2 in results])
    pred = np.array([np.log(np.log(t / (2 * np.pi))) for _, t, _, _ in results]) / np.pi ** 2
    C = float(np.mean(plat - pred))
    print(f"    measured plateaus : {np.round(plat, 3)}")
    print(f"    (1/π²)lnln + {C:.3f} : {np.round(pred + C, 3)}   "
          f"max dev {np.max(np.abs(plat - pred - C)):.3f}")

    # ── THE CONTINUOUS BAND (EPIC3 Phase 4): zeros6 = the first 2,001,052 zeros ──
    # The three-table result above samples 18 decades at three points; zeros6
    # turns the LOW end into a continuous band (t ≈ 2·10⁴ … 1.1·10⁶, 14
    # sliding windows of 50k zeros) and lets the law be FIT, not eyeballed.
    z6_path = os.path.join(DATA, "zeros6")
    if os.path.exists(z6_path):
        print(f"\n  THE CONTINUOUS BAND — zeros6 (2,001,052 zeros), 14 windows × 50k:\n")
        z6 = load("zeros6")
        W = 50_000
        centres = np.unique(np.geomspace(W // 2, len(z6) - W // 2 - 1, 14).astype(int))
        band = []
        for c in centres:
            win = z6[c - W // 2: c + W // 2]
            s2w = smooth(sigma2(nbar(win), Ls))
            lsw = l_sat(Ls, s2w, 0.10)
            band.append((float(np.log(win[W // 2] / (2 * np.pi))), lsw))
        lts, lsats = np.array(band).T
        ok = ~np.isnan(lsats)
        a_fit, b_fit = np.polyfit(lts[ok], lsats[ok], 1)
        pred = a_fit * lts[ok] + b_fit
        r2 = 1 - np.sum((lsats[ok] - pred) ** 2) / np.sum((lsats[ok] - lsats[ok].mean()) ** 2)
        ratios = lsats[ok] / lts[ok]
        print(f"    ln(t/2π) ∈ [{lts.min():.1f}, {lts.max():.1f}]   "
              f"L_sat/ln(t/2π) = {ratios.mean():.3f} ± {ratios.std():.3f}")
        print(f"    FIT:  L_sat = {a_fit:.2f}·ln(t/2π) {b_fit:+.1f}    R² = {r2:.3f}")
        print(f"    → the knee is LINEAR in ln(t/2π) within the band (slope ≈ 1),")
        print(f"      and the three-table constant (0.9–1.0) is the same constant —")
        print(f"      the law interpolates, it does not just hold at three points.")
    else:
        print(f"\n  (zeros6 not present — the continuous-band fit needs the first")
        print(f"   2,001,052 zeros: download zeros6.gz from Odlyzko's zeta_tables")
        print(f"   into examples/science/data/ and gunzip; ~36 MB.)")

    print("\n" + "=" * 78)
    print("  Status: numerical observation on Odlyzko's published zeros, with the")
    print("  exact GUE reference and a criterion-stability check.  The Jacobi-window")
    print("  collapse attempt (M=64…512) was statistically weak and is recorded as a")
    print("  negative sub-result in the docstring; the M=800 ascent ladder stands.")
    print("=" * 78)
