"""sff_typicality.py — the SFF retry by FILTERED TYPICALITY: measured verdict.

HISTORY.  EPIC1 tried the spectral form factor K(t) = |Tr e^{−iHt}|² from the
SLQ quadrature measure and PARKED it honestly (corr 0.45 vs dense: the
measure does not carry eigenvalue PAIR correlations).  The parking note named
the honest path: filtered-typicality estimators.  EPIC3 ran that path
(pre-registered: may fail again).  This file is the record.

THE ESTIMATOR (and why it is the right one).  Per Gaussian probe z_r:
    e_r(t) = ⟨z_r| e^{−iHt} g(H) |z_r⟩ ,   E[e_r] = Tr(e^{−iHt} g(H)),
with g a bulk filter (Gaussian in H here).  The naive |mean e_r|² carries a
D²/p bias; the CROSS-PAIR estimator
    K_est(t) = mean over r≠s of  Re( e_r(t) · conj(e_s(t)) )
is UNBIASED for |Tr U g|² — independence kills the diagonal bias exactly.
Evolution is incremental Lanczos stepping (`resona.apply`, ~t·‖H‖ matvecs
total): fully matrix-free.

THE VERDICT (GOE, D = 512, p = 32 probes, dense-eigh ground truth; the
79-minute run, numbers recorded here — the script below reruns a smaller
instance):

    slope   (t ≲ 2)        : tracks to ~1%                     — WORKS
    plateau (t > 1.5·D)    : ratio est/true = 1.049            — WORKS
    dip + ramp (t ~ 4…600) : SWAMPED — estimates fluctuate
                             NEGATIVE (−15 where truth is 12);
                             ramp median est/true 0.73,
                             log-log corr 0.66 overall          — FAILS

WHY, quantitatively (the new content of this second parking).  Var[e_r] ≈
Tr(g²) = O(D) at every t, so the cross-pair average over p probes carries
noise ~ Tr(g²)/p ~ D/p — INDEPENDENT of the signal.  The connected dip/ramp
signal is K(t) = O(1…t/t_H·D).  Resolving it therefore needs
    p ≳ D / K(t)
probes — at the dip bottom (K ~ O(1)) that is p ~ D: no cheaper than exact
diagonalization.  The bias was the SLQ attempt's disease and typicality cures
it; the VARIANCE at the connected scale is the real wall, and it is a wall of
statistics, not of estimator design.

STATUS: PARKED, SECOND CONFIRMATION — now with the price tag (p ≳ D/K).
What survives matrix-free: the slope, the plateau, and everything
`thermal_response.py` already reads (S(ω), correlators — one-point in the
spectrum).  Two-point spectral statistics at fluctuation scale remain
exact-spectrum territory.

Run:  python3 theory/sff_typicality.py    (D=192 miniature, ~6 min; the same
      three-regime verdict at small scale — measured: slope 1.14, dip+ramp
      0.76 with 6/19 negative points, plateau 1.44; at p=24 pairs even the
      plateau is noisy, which is itself the point)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import resona

D, P, DT = 192, 24, 1.0

if __name__ == "__main__":
    print("=" * 74)
    print("SFF BY FILTERED TYPICALITY — the pre-registered retry, measured")
    print("=" * 74)
    rng = np.random.default_rng(0)
    G = rng.standard_normal((D, D))
    H = (G + G.T) / np.sqrt(2 * D)
    ew = np.linalg.eigvalsh(H)
    lam_max = float(np.abs(ew).max())
    g = lambda x: np.exp(-x ** 2 / (2 * (0.4 * lam_max) ** 2))

    ts = np.unique(np.geomspace(0.5, 3.0 * D, 60))
    truth = np.array([abs(np.sum(g(ew) * np.exp(-1j * ew * t))) ** 2 for t in ts])

    mv = lambda v: H @ v
    E = np.zeros((P, len(ts)), complex)
    t0 = time.perf_counter()
    for r in range(P):
        z = rng.standard_normal(D)
        y = resona.apply(mv, g, z, k=64).astype(complex)
        t_cur, it = 0.0, 0
        while it < len(ts):
            if ts[it] <= t_cur + 1e-9:
                E[r, it] = z @ y
                it += 1
                continue
            step = min(DT, ts[it] - t_cur)
            y = resona.apply(mv, lambda l: np.exp(-1j * step * l), y, k=24)
            t_cur += step
    print(f"\n  evolution: {time.perf_counter()-t0:.0f}s "
          f"({P} probes, ~{int(ts[-1]/DT)} steps, matrix-free)")

    iu = np.triu_indices(P, 1)
    K_est = np.array([float(np.mean(np.real(E[iu[0], j] * np.conj(E[iu[1], j]))))
                      for j in range(len(ts))])

    plateau = ts > 1.5 * D
    ramp = (ts > 15) & (ts < 0.8 * D)
    slope = ts < 2
    print(f"\n  {'regime':>10} {'truth scale':>12} {'est/true':>9}")
    print(f"  {'slope':>10} {np.mean(truth[slope]):12.1f} "
          f"{np.mean(K_est[slope])/np.mean(truth[slope]):9.3f}")
    print(f"  {'dip+ramp':>10} {np.mean(truth[ramp]):12.1f} "
          f"{np.mean(K_est[ramp])/np.mean(truth[ramp]):9.3f}   "
          f"(negative excursions: {int(np.sum(K_est[ramp] < 0))}/{int(ramp.sum())} points)")
    print(f"  {'plateau':>10} {np.mean(truth[plateau]):12.1f} "
          f"{np.mean(K_est[plateau])/np.mean(truth[plateau]):9.3f}")
    print(f"\n  noise floor ~ Tr(g²)/p = {np.sum(g(ew)**2)/P:.1f}  vs ramp signal "
          f"{np.mean(truth[ramp]):.1f} — the wall is p ≳ D/K(t), statistics not bias.")
    print("\n" + "=" * 74)
    print("  VERDICT (second parking confirmation): slope and plateau read")
    print("  matrix-free; the connected dip/ramp needs p ~ D/K(t) probes — at the")
    print("  dip that is p ~ D, no cheaper than eigh.  Bias cured, variance is")
    print("  the wall.  Two-point fluctuation statistics stay exact-spectrum.")
    print("=" * 74)
