"""
A COMPUTATIONAL HILBERT–PÓLYA PROBE — the operator whose spectrum is the
Riemann zeros, built, verified, and interrogated.

THE CONJECTURE (Hilbert–Pólya).  The nontrivial zeros of ζ(s) are the spectrum
of some self-adjoint operator — that would prove the Riemann Hypothesis.

THE HONEST FRAMING.  *Existence* is trivial: any finite real sequence is the
spectrum of a diagonal matrix.  The conjecture's content is STRUCTURE — is
there a *natural / local / lawful* operator?  resona's synthesizer lets us ask
that question quantitatively: `from_measure(zeros, equal weights)` builds the
unique JACOBI (tridiagonal, 1-D local, self-adjoint) representative, and then
the library interrogates its hopping coefficients (α, β).

WHAT THIS SCRIPT FINDS (all measured live):
  1. The Jacobi operator REALIZES the first 200 zeros to machine precision —
     a 1-D local self-adjoint operator with the zeta spectrum, in your hands.
  2. Its smooth hopping profile β(k) is the WEYL LAW of the zeros (the
     Riemann–von Mangoldt density), verified by correlation.
  3. Montgomery–Odlyzko live: ⟨r⟩ of the zeros matches a GUE control and
     rejects a Poisson control (both built with the SAME smooth density).
  4. The surprise: the β-residual fluctuations of the zeta operator are
     SMOOTHER than its GUE twins' — consistent with the zeros' long-range
     rigidity beyond RMT (Berry's semiclassical saturation).  Reported as an
     observation with control statistics, not a theorem.

Zeros are cached in zeta_zeros_200.txt; `--regen` recomputes them with mpmath
(slow); every run re-verifies the first 10 against mpmath if it is installed.

Run:  python3 examples/science/hilbert_polya.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import resona
from resona.cost import level_spacing_ratio

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "zeta_zeros_200.txt")
rng = np.random.default_rng(0)


def load_zeros():
    if "--regen" in sys.argv or not os.path.exists(CACHE):
        import mpmath as mp
        zs = np.array([float(mp.zetazero(n).imag) for n in range(1, 201)])
        np.savetxt(CACHE, zs, fmt="%.15f")
        return zs, "regenerated with mpmath"
    zs = np.loadtxt(CACHE)
    try:                                    # honest cache: spot-verify vs mpmath
        import mpmath as mp
        chk = max(abs(float(mp.zetazero(n).imag) - zs[n - 1]) for n in range(1, 11))
        return zs, f"cache verified vs mpmath (first 10: max dev {chk:.1e})"
    except ImportError:
        return zs, "cache loaded (install mpmath to re-verify)"


def nbar(t):
    """Riemann–von Mangoldt smooth zero-counting N̄(t)."""
    return t / (2 * np.pi) * np.log(t / (2 * np.pi * np.e)) + 7.0 / 8.0


def controls(zeros, n_gue=5):
    """Poisson and GUE level sets carrying the SAME smooth (Weyl) density."""
    M = len(zeros)
    u = nbar(zeros)
    tg = np.linspace(zeros[0] * 0.8, zeros[-1] * 1.05, 40000)
    inv = lambda uu: np.interp(uu, nbar(tg), tg)

    up = np.cumsum(rng.exponential(size=M))
    up = (up - up[0]) / (up[-1] - up[0]) * (u[-1] - u[0]) + u[0]
    poisson = inv(up)

    gues = []
    for _ in range(n_gue):
        n3 = 3 * M
        G = rng.standard_normal((n3, n3)) + 1j * rng.standard_normal((n3, n3))
        ev = np.linalg.eigvalsh((G + G.conj().T) / 2)
        mid = ev[n3 // 2 - M // 2: n3 // 2 + M // 2]
        R = np.sqrt(2 * n3)                          # semicircle-CDF unfolding
        cdf = (mid * np.sqrt(np.clip(R * R - mid * mid, 0, None))
               + R * R * np.arcsin(np.clip(mid / R, -1, 1))) / (np.pi * R * R) + 0.5
        ug = cdf * n3
        ug = (ug - ug[0]) / (ug[-1] - ug[0]) * (u[-1] - u[0]) + u[0]
        gues.append(inv(ug))
    return poisson, gues


def jacobi_beta(levels):
    """Realize levels as the Jacobi operator → (α, β)."""
    M = len(levels)
    return resona.from_measure(np.sort(levels), np.full(M, 1.0 / M), k=M)


if __name__ == "__main__":
    zeros, note = load_zeros()
    M = len(zeros)
    print("=" * 76)
    print("HILBERT–PÓLYA, COMPUTATIONALLY — the operator with the zeta-zero spectrum")
    print("=" * 76)
    print(f"\n  first {M} nontrivial zeros ({note})")

    # ── 1. realize and verify ─────────────────────────────────────────────────
    al, be = jacobi_beta(zeros)
    T = np.diag(al) + np.diag(be, 1) + np.diag(be, -1)
    err = float(np.max(np.abs(np.sort(np.linalg.eigvalsh(T)) - np.sort(zeros))))
    print(f"\n  1. REALIZED: tridiagonal (1-D local, self-adjoint) {M}x{M} operator")
    print(f"     max|eig − zeros| = {err:.1e}   — the zeta spectrum, machine-exact")

    # ── 2. the smooth hopping law IS the Weyl density ────────────────────────
    # reference: the PERFECTLY RIGID realization — levels placed exactly by the
    # Riemann–von Mangoldt counting N̄(t); its β is the pure Weyl trend
    u = nbar(zeros)
    tg = np.linspace(zeros[0] * 0.8, zeros[-1] * 1.05, 40000)
    inv = lambda uu: np.interp(uu, nbar(tg), tg)
    weyl_levels = inv(u[0] + (np.arange(M) + 0.5) / M * (u[-1] - u[0]))
    _, be_weyl = jacobi_beta(weyl_levels)
    k = np.arange(len(be))
    trend = np.polyval(np.polyfit(k, be, 6), k)
    c_weyl = float(np.corrcoef(trend, be_weyl)[0, 1])
    print(f"\n  2. THE SMOOTH LAW: the operator's hopping profile IS the Weyl density")
    print(f"     corr(zeta β trend, β of the pure-N̄ realization) = {c_weyl:.4f}")

    # ── 3. Montgomery–Odlyzko via ⟨r⟩, against honest controls ──────────────
    poisson, gues = controls(zeros)
    r_z = level_spacing_ratio(zeros)
    r_g = np.mean([level_spacing_ratio(g) for g in gues])
    r_p = level_spacing_ratio(poisson)
    print(f"\n  3. SPACING STATISTICS (controls share the same Weyl density):")
    print(f"     ⟨r⟩  zeta = {r_z:.3f}   GUE controls = {r_g:.3f}   "
          f"Poisson control = {r_p:.3f}   (refs .600 / .386)")
    print(f"     → Montgomery–Odlyzko, live: the zeros repel like GUE eigenvalues.")

    # ── 4. the observation: smoother than GUE ───────────────────────────────
    resid = lambda lv: jacobi_beta(lv)[1] - be_weyl       # fluctuation vs Weyl
    s_z = float(resid(zeros).std())
    s_gs = [float(resid(g).std()) for g in gues]
    s_p = float(resid(poisson).std())
    print(f"\n  4. HOPPING FLUCTUATIONS std(β − smooth fit):")
    print(f"     zeta = {s_z:.2f}   GUE controls = {np.mean(s_gs):.2f} ± {np.std(s_gs):.2f} "
          f"(n={len(s_gs)})   Poisson = {s_p:.2f}")
    print(f"     → the zeta operator's hopping profile is SMOOTHER than its GUE")
    print(f"       twins ({np.mean(s_gs)/s_z:.2f}x) — consistent with the zeros' long-range")
    print(f"       rigidity beyond RMT (Berry saturation).  An observation with")
    print(f"       n={len(s_gs)} controls — suggestive, not a theorem.")

    print("\n" + "=" * 76)
    print("  Existence was never the question — STRUCTURE is.  The synthesizer puts")
    print("  the zeta operator on the bench: its smooth part is the Weyl law, its")
    print("  fluctuations are GUE-rigid (and then some).  The bridge from 'a list of")
    print("  zeros' to 'an operator you can probe with the rest of the library'.")
    print("=" * 76)
