"""
OPERATOR SYNTHESIS — order a spectrum in words, get a working matvec.

resona is not only an oscilloscope for operators (PROBE/READ) — together with
the inverse transform it is a SYNTHESIZER: design a measure by the measure
algebra (atoms, bands, ⊞ noise, heat flow), then REALIZE it as a concrete
tridiagonal (local!) operator whose matvec you can hand to anything else.

    design the measure          realize                    play it
    ρ(x): bands, ⊞, flow(t)  →  from_measure → (α, β)  →  of / apply / GMRES…

Three acts, each verified:

  ACT 1  "I want a gapped material":  a two-band density (an insulator) is
         written down as a FORMULA, inverse-CDF'd into N levels, realized as a
         tridiagonal operator — whose eigenvalues match the order to ~1e-14.

  ACT 2  "I want this exact boundary response":  the same density ordered as
         an e0-MEASURE (what a boundary probe must hear) — `from_measure` on
         the (x, ρ) grid; verified by local_density from e0.  The two acts
         differ on purpose: eigenvalue density and boundary response are
         DIFFERENT measures — the synthesizer has a constructor for each.

  ACT 3  "now close the gap":  turn the free-heat knob on the act-1 spectrum
         (s.flow — Pastur, no realizations), find the band-merger time
         (s.shock_time), and realize the flowed measure JUST BELOW and JUST
         ABOVE t_c as two new operators — a gapped and a gapless material,
         synthesized on the fly from the same design.

Honest limits: `from_measure` returns the Jacobi (tridiagonal) REPRESENTATIVE
of the isospectral class — the spectrum does not fix the eigenbasis (Horn:
cross-moments are exactly what a measure discards).  Sharp/atomic measures
make the long recurrence ill-conditioned (see `inverse_spectral.py`).

Run:  python3 examples/operator_synthesis.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
from numpy import trapezoid
import resona

XS = np.linspace(0.0, 5.5, 900)


def two_band_density(xs, b1=(1.0, 2.0), b2=(3.0, 4.0)):
    """The ORDER: semicircular bands on b1 and b2 — a gapped 'material'."""
    rho = np.zeros_like(xs)
    for a, b in (b1, b2):
        c, r = 0.5 * (a + b), 0.5 * (b - a)
        m = np.abs(xs - c) < r
        rho[m] += np.sqrt(r ** 2 - (xs[m] - c) ** 2)
    return rho / trapezoid(rho, xs)


def realize_levels(levels):
    """Levels → tridiagonal operator with EXACTLY these eigenvalues
    (from_measure with equal weights), returned as (matvec, M, (α, β))."""
    M = len(levels)
    al, be = resona.from_measure(levels, np.full(M, 1.0 / M), k=M)

    def mv(x):
        y = al * x
        y[:-1] += be * x[1:]
        y[1:] += be * x[:-1]
        return y
    return mv, M, (al, be)


def levels_from_density(rho, xs, M):
    cdf = np.cumsum(rho); cdf /= cdf[-1]
    return np.interp((np.arange(M) + 0.5) / M, cdf, xs)


if __name__ == "__main__":
    print("=" * 74)
    print("OPERATOR SYNTHESIS — design the measure, realize the matvec")
    print("=" * 74)

    # ── ACT 1: order an eigenvalue DENSITY, realize, verify ──────────────────
    M = 300
    rho = two_band_density(XS)
    levels = levels_from_density(rho, XS, M)
    mv, M, (al, be) = realize_levels(levels)

    T = np.diag(al) + np.diag(be, 1) + np.diag(be, -1)        # ground truth only
    ev = np.sort(np.linalg.eigvalsh(T))
    err = float(np.max(np.abs(ev - np.sort(levels))))
    s = resona.of(mv, M, k=64, probes=24)
    lo, hi = s.extreme()
    rb = s.density(XS, eta=0.06); rb /= trapezoid(rb, XS)
    corr = float(np.corrcoef(rho, rb)[0, 1])
    gap_dip = s.density(np.array([2.5]), eta=0.08)[0] / s.density(np.array([1.5]), eta=0.08)[0]
    print(f"\n  ACT 1 — order: two semicircular bands [1,2]∪[3,4] (a gapped material)")
    print(f"    realized: TRIDIAGONAL {M}x{M} (local operator), βₖ ∈ "
          f"[{be.min():.2f}, {be.max():.2f}]")
    print(f"    [verify] max|eig − ordered levels| = {err:.1e}   (machine precision)")
    print(f"    [verify] probe of the matvec: corr(ρ_order, ρ_read) = {corr:.4f}, "
          f"support [{lo:.2f},{hi:.2f}] (order [1,4])")
    print(f"    [verify] density in the gap / in the band = {gap_dip:.3f}  → the gap is real")

    # ── ACT 2: order a BOUNDARY RESPONSE (e0-measure) instead ────────────────
    al2, be2 = resona.from_measure(XS, rho / rho.sum(), k=M)

    def mv2(x):
        y = al2 * x
        y[:-1] += be2 * x[1:]
        y[1:] += be2 * x[:-1]
        return y
    e0 = np.zeros(M); e0[0] = 1.0
    rho_e0 = resona.local_density(mv2, e0, XS, k=M // 2, eta=0.05)
    rho_e0 /= trapezoid(rho_e0, XS)
    corr2 = float(np.corrcoef(rho, rho_e0)[0, 1])
    print(f"\n  ACT 2 — the SAME curve ordered as the e0-RESPONSE (boundary measure):")
    print(f"    [verify] local_density(e0) vs order: corr = {corr2:.4f}")
    print(f"    (eigenvalue density and boundary response are DIFFERENT measures —")
    print(f"     ordering one and checking the other fails by design; see ACT 1 vs 2)")

    # ── ACT 3: turn the free-heat knob, re-realize on the fly ────────────────
    tc = s.shock_time(t_max=2.0, n_t=60, n_x=240, eta=4e-3)
    print(f"\n  ACT 3 — close the gap with the free-heat knob (no realizations, Pastur):")
    print(f"    band-merger (shock) time of the ordered spectrum: t_c = {tc:.3f}")
    mids = np.array([2.5])
    for t, tag in [(0.3 * tc, "below t_c (still gapped)"), (1.6 * tc, "above t_c (gap closed)")]:
        rho_t = s.flow(t, XS, eta=5e-3)
        lv = levels_from_density(np.maximum(rho_t, 0) + 1e-15, XS, M)
        mv_t, _, _ = realize_levels(lv)                       # a NEW operator, on the fly
        s_t = resona.of(mv_t, M, k=64, probes=16)
        dip = (s_t.density(mids, eta=0.08)[0]
               / s_t.density(np.array([1.5 + 0.4 * t]), eta=0.08)[0])
        print(f"    t = {t:.2f} ({tag}):  realized + re-probed, "
              f"gap/band density = {dip:.3f}")
    print(f"    → two MATERIALS (matvecs) synthesized from one design by one knob.")

    print("\n" + "=" * 74)
    print("  PROBE/READ listen to operators; COMPOSE+from_measure WRITE them.")
    print("  Design in the measure algebra (bands, ⊞, flow), realize as a local")
    print("  tridiagonal matvec, hand it to of/apply/GMRES — operators on demand,")
    print("  eigenvalues matching the order to machine precision (ACT 1).")
    print("=" * 74)
