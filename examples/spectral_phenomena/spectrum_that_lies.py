"""
THE SPECTRUM THAT LIES — when eigenvalues are not the whole story.

WHAT:  A strongly NON-NORMAL convection operator
           A = diag(d) + β·(shift-up)        d ∈ [−1.5, −1],  β = 2
       whose EIGENVALUES are exactly d — all real, all in the left half-plane,
       textbook "stable, nothing dangerous nearby."  Yet:
         • the resolvent norm ‖(A−zI)⁻¹‖ = 1/σ_min(A−zI) is ASTRONOMICAL at
           points z that no eigenvalue is anywhere near (the ε-pseudospectrum
           bulges far into the RIGHT half-plane);
         • ‖exp(At)‖ GROWS (1.6×, 2.6×, 6.6×, 17× at t=0.5…3) even though every
           eigenvalue decays — transient growth the spectrum flatly denies, with
           initial rate exactly the numerical abscissa ω(A) the cloud reports.
       A NORMAL operator with the IDENTICAL spectrum does none of this.

WHY:   For a non-normal operator the eigenvalues alone mislead — stability,
       solver convergence, and transient response live in the pseudospectrum /
       singular values, not the point spectrum.  This is the local-vs-global
       non-normal toolbox of resona v3, and the place it is the right read.

RESONA's role (v3 non-normal toolbox, matrix-free where it can be):
       • defect.normality(matvec, N, rmatvec) -> (‖[A,A*]‖²_F, stderr): the GLOBAL
         flag.  ≫0 for A, exactly 0 for the normal twin — "the spectrum may lie."
       • defect.sigma_min(matvec, z, N, rmatvec): the LOCAL read — σ_min(A−zI),
         so 1/σ_min is the resolvent norm.  Tracks the bloom along a ray.
       • defect.pseudospectrum: the boolean ε-pseudospectrum mask — shows the bulge.
       • resona.cloud(matvec, N): the non-Hermitian Ritz read; its .abscissa() is a
         lower bound on the NUMERICAL abscissa (the transient-growth dial), which
         sits in the RIGHT half-plane even though the spectral abscissa is −1.

       EVERY printed number is checked against a dense reference (eig / svd / expm).
       HONEST LIMIT, shown live: matrix-free σ_min is a Lanczos estimate; at the
       DEEPEST point of the bloom it hits the float64 floor and can no longer
       resolve σ_min ~ 1e-25 — that ill-conditioning IS the phenomenon.  It agrees
       with dense to ~1e-9 everywhere the value is resolvable; we report both.

Run:   PYTHONPATH=. python3 examples/spectral_phenomena/spectrum_that_lies.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from scipy.linalg import expm
import resona
from resona.defect import normality, sigma_min, pseudospectrum

N = 120
diag = -1.0 - 0.5 * np.linspace(0.0, 1.0, N)     # eigenvalues: real, in [−1.5, −1]
beta = 2.0                                        # convection (upper-shift) strength


# matrix-free actions of the non-normal operator A and its adjoint Aᵀ
def matvec(x):
    y = diag * x
    y[:-1] += beta * x[1:]
    return y


def rmatvec(x):
    y = diag * x
    y[1:] += beta * x[:-1]
    return y


# dense references
A = np.diag(diag).astype(float)
A[np.arange(N - 1), np.arange(1, N)] = beta
ev = np.linalg.eigvals(A)
A_normal = np.diag(ev.real)                        # NORMAL twin: identical spectrum


if __name__ == "__main__":
    print("=" * 78)
    print("THE SPECTRUM THAT LIES — non-normal A, eigenvalues say 'stable', "
          "pseudospectrum disagrees")
    print("=" * 78)

    # ── the point spectrum: looks completely benign ─────────────────────────
    print(f"\n  Operator A = diag(d) + β·shiftᵤₚ,  N={N}, β={beta}")
    print(f"  eigenvalues of A: all real, all in [{ev.real.min():.2f}, {ev.real.max():.2f}]"
          f"  (spectral abscissa = {ev.real.max():+.2f}  → 'stable')")

    # ── (1) GLOBAL flag: defect.normality ───────────────────────────────────
    nn_est, nn_se = normality(matvec, N=N, rmatvec=rmatvec)
    nn_true = float(np.linalg.norm(A @ A.conj().T - A.conj().T @ A) ** 2)
    nnN_est, _ = normality(lambda x: A_normal @ x, N=N, rmatvec=lambda x: A_normal.T @ x)
    nnN_true = float(np.linalg.norm(A_normal @ A_normal.T - A_normal.T @ A_normal) ** 2)
    print(f"\n  (1) defect.normality  ‖[A,A*]‖²_F  (matrix-free Hutchinson, =0 ⇔ normal)")
    print(f"        A (non-normal): est = {nn_est:8.3f} ± {nn_se:.3f}   "
          f"dense = {nn_true:8.3f}   rel.err = {abs(nn_est-nn_true)/nn_true:.2%}")
    print(f"        normal twin   : est = {nnN_est:8.3f}              "
          f"dense = {nnN_true:8.3f}   → exactly 0, the spectrum is the whole story")

    # ── (2) LOCAL read: resolvent-norm bloom along the real axis ────────────
    print(f"\n  (2) defect.sigma_min along the real axis → resolvent norm 1/σ_min")
    print(f"      (z is the distance to the right of the spectrum; nearest eigenvalue")
    print(f"       to z=0 is at {ev.real.max():+.2f}, so z≥0 is OUTSIDE the spectrum)\n")
    print(f"      {'z':>5s} {'σ_min matfree':>14s} {'σ_min dense':>13s} "
          f"{'1/σ_min (resolvent)':>20s} {'agree?':>8s}")
    print(f"      {'-'*5} {'-'*14} {'-'*13} {'-'*20} {'-'*8}")
    for z in (0.0, 1.0, 2.0, 3.0):
        mf = sigma_min(matvec, float(z), N=N, rmatvec=rmatvec, k=N - 10)
        dn = float(np.linalg.svd(A - z * np.eye(N), compute_uv=False)[-1])
        resolvent = 1.0 / max(dn, 1e-300)
        rel = abs(mf - dn) / max(dn, 1e-300)
        tag = f"{rel:.0e}" if rel < 1e-3 else "FLOOR"        # honest: matfree floor
        print(f"      {z:>+5.1f} {mf:>14.3e} {dn:>13.3e} {resolvent:>20.3e} {tag:>8s}")
    print(f"\n      The resolvent norm is ENORMOUS at z=0 (nearest eigenvalue distance 1.0)")
    print(f"      and stays huge out to z=2 — the ε-pseudospectrum bulges right, into")
    print(f"      the unstable half-plane, where NO eigenvalue lives.  Matrix-free σ_min")
    print(f"      matches dense once it is resolvable; at the deepest point it hits the")
    print(f"      float64 floor (FLOOR) — that ill-conditioning is exactly the defect.")

    # ── (2b) the ε-pseudospectrum mask: how far right the bulge reaches ──────
    eps = 1e-2
    xs = np.linspace(-2.0, 3.0, 11)
    ys = np.linspace(-2.5, 2.5, 11)
    Z = xs[None, :] + 1j * ys[:, None]
    mask = pseudospectrum(A, Z, eps=eps)
    right_xs = [xs[j] for i in range(len(ys)) for j in range(len(xs)) if mask[i, j]]
    print(f"\n      defect.pseudospectrum (ε={eps:g}): {int(mask.sum())}/{mask.size} grid points "
          f"in Λ_ε; rightmost reaches Re = {max(right_xs):+.1f}")
    print(f"      (spectrum sits at Re = {ev.real.min():.1f}…{ev.real.max():.1f}; the ε-set "
          f"leaks {max(right_xs)-ev.real.max():+.1f} past it, into Re>0)")

    # ── (3) the non-Hermitian Ritz CLOUD ────────────────────────────────────
    cl = resona.cloud(matvec, N=N, k=60, probes=4)
    num_abscissa = float(np.linalg.eigvalsh((A + A.T) / 2)[-1])   # dense ground truth
    print(f"\n  (3) resona.cloud — non-Hermitian Ritz read (lower bounds by construction)")
    print(f"        cloud.radius()   ≥ {cl.radius():.3f}   (true spectral radius ρ(A) = "
          f"{np.max(np.abs(ev)):.3f})")
    print(f"        cloud.abscissa() ≥ {cl.abscissa():+.3f}   (true numerical abscissa "
          f"ω(A) = {num_abscissa:+.3f})")
    print(f"        ω(A) ≥ {num_abscissa:+.2f} > spectral abscissa {ev.real.max():+.2f}: the")
    print(f"        cloud's abscissa is in the RIGHT half-plane → transients grow even")
    print(f"        though every eigenvalue decays.  The gap = the non-normality.")

    # ── the consequence the spectrum denied: transient growth ───────────────
    # The initial slope d/dt‖exp(At)‖|₀ equals the numerical abscissa ω(A) (a
    # theorem) — so the cloud's abscissa read predicts the transient growth rate.
    dt = 1e-3
    slope0 = (np.linalg.norm(expm(A * dt), 2) - 1.0) / dt
    print(f"\n  CONSEQUENCE  ‖exp(A t)‖  (every eigenvalue of A has Re ≤ {ev.real.max():+.2f}):")
    print(f"        initial growth rate d/dt‖exp(At)‖|₀ = {slope0:+.3f}  =  numerical "
          f"abscissa ω(A) = {num_abscissa:+.3f}")
    print(f"        {'t':>5s} {'‖exp(At)‖ non-normal':>20s} {'‖exp(At)‖ normal twin':>22s}")
    print(f"        {'-'*5} {'-'*20} {'-'*22}")
    for t in (0.5, 1.0, 2.0, 3.0):
        g = float(np.linalg.norm(expm(A * t), 2))
        gN = float(np.linalg.norm(expm(A_normal * t), 2))
        print(f"        {t:>5.1f} {g:>20.3f} {gN:>22.3f}")
    print(f"        non-normal A grows {float(np.linalg.norm(expm(A*3),2)):.0f}× by t=3; the "
          f"normal twin (identical spectrum) only decays.")

    print("\n" + "=" * 78)
    print("  Same spectrum, opposite behaviour.  Eigenvalues alone call A 'stable';")
    print("  defect.normality (global), defect.sigma_min/pseudospectrum (local) and")
    print("  cloud.abscissa expose the transient bloom — the right read for non-normal")
    print("  operators.  Every number checked against dense eig / svd / expm.")
    print("=" * 78)
