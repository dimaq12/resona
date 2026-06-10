"""
free_convolution_flow.py — composition without a joint matvec, and the shock.
==============================================================================
COMPOSE, the theory way.  `s + t` in resona re-probes the combined operator
(A+B)x = Ax+Bx.  But free probability says you can compose the SPECTRA THEMSELVES —
no joint matvec, just the two measures:

      κ_n(A ⊞ B) = κ_n(A) + κ_n(B)        (free cumulants add)

So the spectrum of the sum is read off from μ_A and μ_B alone (resona.lift.
free_convolution).  Adding free SEMICIRCULAR noise is disorder averaging, and its
density is the PASTUR / subordination fixed point g = G_A(z − σ²g), in closed form
with no realization loop (resona.subordination).  Push the noise variance as a
TIME t and you get the complex BURGERS flow ∂_tG + G∂_zG = 0: two spectral bands
collide into one at a critical time t_c — a literal PDE SHOCK, which is the
program's defect = spectral edge = edge of chaos (resona.flow).

Run:  python3 examples/spectral_phenomena/free_convolution_flow.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from scipy import linalg
import resona

rng = np.random.default_rng(0)


if __name__ == "__main__":
    print("=" * 72)
    print("FREE CONVOLUTION + THE BURGERS SHOCK — composition without a joint matvec")
    print("=" * 72)

    # ── Act 1: spectrum of A⊞B from the two measures alone ──
    M = 900; d = rng.uniform(-1, 1, M); A = np.diag(d)
    Q, _ = linalg.qr(rng.standard_normal((M, M))); B = Q @ A @ Q.T          # free copy
    sA = resona.of(lambda v: A @ v, M, k=120, probes=16)
    sB = resona.of(lambda v: B @ v, M, k=120, probes=16)
    mpred = resona.lift.free_convolution(sA, sB, order=4)                   # no joint matvec
    mtrue = [np.trace(np.linalg.matrix_power(A + B, n)) / M for n in range(1, 5)]
    print("\n  [1] κ_n(A⊞B)=κ_n(A)+κ_n(B): moments of A⊞B from μ_A,μ_B ALONE")
    print(f"      {'n':>3} {'predicted':>12} {'measured':>12}")
    for n in range(4):
        print(f"      {n+1:>3} {mpred[n]:>12.4f} {mtrue[n]:>12.4f}")
    print(f"      max|Δm| = {max(abs(p-t) for p,t in zip(mpred,mtrue)):.4f}")

    # ── Act 2: disorder-averaged DOS of A + σ·GOE in closed form (vs Monte-Carlo) ──
    N = 600
    A2 = np.diag(np.concatenate([-np.ones(N // 2), np.ones(N // 2)]))        # atoms ±1
    s2 = resona.of(lambda v: A2 @ v, N, k=80, probes=8)
    sigma = 0.5; xs = np.linspace(-3, 3, 1500)
    rho = resona.subordination.averaged_dos(s2, sigma, xs, eta=2e-3)
    m2_closed = float(np.trapezoid(xs ** 2 * rho, xs))
    mc = np.concatenate([linalg.eigvalsh(A2 + sigma * ((W := rng.standard_normal((N, N)))
                         + W.T) / np.sqrt(2 * N)) for _ in range(40)])
    print("\n  [2] ⟨DOS⟩ of A+σ·GOE via Pastur (closed form, no eig, no realizations)")
    print(f"      ∫ρ dx = {np.trapezoid(rho, xs):.3f}   m2: closed {m2_closed:.3f}"
          f"  Monte-Carlo {np.mean(mc**2):.3f}   (= m2(A)+σ² = {1+sigma**2:.3f})")

    # ── Act 3: the Burgers shock — two bands merge at t_c ──
    tc = resona.flow.shock_time(s2)
    print("\n  [3] free heat flow μ_t = μ_0 ⊞ semicircle(t): the bands ±1 COLLIDE")
    print(f"      shock (band-merger) time  t_c ≈ {tc:.2f}   (exact t_c = 1.0)")
    print("\n" + "=" * 72)
    print("  Composition closes in the free cumulants (no joint matvec); free addition")
    print("  with a semicircle is disorder averaging in closed form (Pastur); as a flow")
    print("  it is complex Burgers, and the band merger is a SHOCK = spectral edge =")
    print("  the defect.  Three faces of one object, now callable from the library.")
    print("=" * 72)
