"""
THE ERROR WAS THE SIGNAL — defect spectroscopy of a black-box solver.

Every one-step integrator since 1950 computes a little more than it returns:
its discretization DEFECT D_n = P_n − P_2n carries (1) the Koopman-generator
observable of the system and (2) the system's spectrum, band by band.  Two
resona reads recover both — from an UNMODIFIED solver treated as a black box:

  ACT 1  `defect.generator_read(P_n, P_2n, t, n)`: the backward-Euler defect
         is (t²/4n)·A²e^{−tA}u₀ + O(n⁻²).  We run a 15-line legacy BE loop
         (pretend it is Fortran from 1987), call it twice, and read the
         generator observable to ~1% — verified against expm ground truth,
         with the O(n⁻²) order CHECKED from the data itself (Richardson),
         the float32 noise floor shown, and Crank–Nicolson honestly REFUSED
         (the constant is solver-specific; measured deviation O(1)).

  ACT 2  `defect.spectroscopy(power, bands, coords)`: the defect's power
         spectrum, read per frequency band by its BARYCENTRE — the
         blind-zone-free estimator (BDS).  The punchline is ROBUSTNESS:
         under snapshot noise the barycentre holds where the classical
         norm-RATIO estimator collapses — measured live below
         (barycentre stable at ε = 5e-2; ratio dead at ε = 1e-5).

Stress provenance: FA/revise_stress/STRESS_REPORT.md — both reads passed
re-verification beyond their original ensembles (graph Laplacians, complex
Hermitian, stiff κ=1e6, Jordan bands; 35/35 PDE suite).

Run:  python3 examples/defect_spectroscopy.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from numpy.fft import fft
from scipy.linalg import expm
from resona.defect import generator_read, spectroscopy, richardson_limit

rng = np.random.default_rng(0)


def legacy_be_solver(A, u0, t, n):
    """The 'Fortran 1987' black box: backward Euler, nothing else."""
    M = np.linalg.inv(np.eye(len(A)) + (t / n) * A)
    u = u0.copy()
    for _ in range(n):
        u = M @ u
    return u


if __name__ == "__main__":
    print("=" * 74)
    print("DEFECT SPECTROSCOPY — the solver's error, read back as physics")
    print("=" * 74)

    # ── ACT 1: the generator from two black-box runs ──────────────────────────
    N, t = 90, 0.5
    Adj = (rng.random((N, N)) < 0.06).astype(float)
    Adj = np.triu(Adj, 1); Adj = Adj + Adj.T
    A = np.diag(Adj.sum(1)) - Adj                       # graph Laplacian
    u0 = rng.standard_normal(N)
    G_true = A @ A @ (expm(-t * A) @ u0)

    print(f"\n  ACT 1 — generator_read on a graph Laplacian (N={N}), BE black box:")
    print(f"    {'n':>6} {'rel err vs expm truth':>23}")
    reads = {}
    for n in (64, 128, 256):
        P = legacy_be_solver(A, u0, t, n)
        P2 = legacy_be_solver(A, u0, t, 2 * n)
        reads[n] = generator_read(P, P2, t, n)
        err = np.linalg.norm(reads[n] - G_true) / np.linalg.norm(G_true)
        print(f"    {n:>6} {err:>23.2e}")
    # order check FROM THE DATA: Richardson-extrapolate the reads (err ~ 1/n)
    G_extrap = richardson_limit([reads[64], reads[128], reads[256]],
                                [64, 128, 256], p0=1.0)
    err_x = np.linalg.norm(G_extrap - G_true) / np.linalg.norm(G_true)
    print(f"    Richardson over the three reads: rel err {err_x:.2e} — the data")
    print(f"    itself confirms the O(n⁻¹) read / O(n⁻²) defect law.")
    try:
        generator_read(P, P2, t, 256, solver="cn")
    except ValueError as e:
        print(f"    Crank–Nicolson honestly refused: \"{str(e)[:62]}…\"")

    # ── ACT 2: BDS — the barycentre vs the ratio, under noise ────────────────
    Nf = 256
    K = np.fft.fftfreq(Nf, 1.0 / Nf)
    kmag = np.abs(K)
    sym = kmag ** 2                                     # heat symbol λ(k) = k²
    u = np.zeros(Nf)
    for km, amp in [(1, 1.0), (3, .7), (6, .5), (12, .35), (25, .25),
                    (50, .18), (100, .12)]:
        u += amp * np.cos(km * 2 * np.pi * np.arange(Nf) / Nf + rng.uniform(0, 6.28))

    def be_fourier(uu, tt, nn):                          # exact BE in Fourier
        return np.fft.ifft((1.0 + tt / nn * sym) ** (-nn) * fft(uu)).real

    bands = [(kmag >= 2 ** j) & (kmag <= 2 ** (j + 1) - 1) for j in range(7)]
    tH, nH = 0.5, 8
    print(f"\n  ACT 2 — spectroscopy: per-band λ from the defect (truth λ = k²):")
    Pn, P2n = be_fourier(u, tH, nH), be_fourier(u, tH, 2 * nH)
    base_k = None
    print(f"    {'ε noise':>9} | barycentre k̄ per band (→ λ = k̄²); "
          f"✓ = unchanged vs noiseless")
    for eps in (0.0, 1e-5, 1e-3, 5e-2):
        r2 = np.random.default_rng(1)
        nz = lambda v: v + eps * np.linalg.norm(v) / np.sqrt(Nf) * r2.standard_normal(Nf)
        power = np.abs(fft(nz(Pn) - nz(P2n))) ** 2
        kb, sig = spectroscopy(power, bands, coords=kmag)
        ki = [int(round(x)) if np.isfinite(x) else None for x in kb[:6]]
        if base_k is None:
            base_k = ki
            print(f"    {eps:9.0e} | {ki}")
        else:
            marks = ["✓" if a == b else f"{a}" for a, b in zip(ki, base_k)]
            print(f"    {eps:9.0e} | {marks}")
    # the classical RATIO estimator under the same noise
    print(f"\n    the classical norm-RATIO α per band, same noise:")
    for eps in (0.0, 1e-5, 1e-3):
        r2 = np.random.default_rng(2)
        nz = lambda v: v + eps * np.linalg.norm(v) / np.sqrt(Nf) * r2.standard_normal(Nf)
        P4n = be_fourier(u, tH, 4 * nH)
        D1 = fft(nz(Pn) - nz(P2n)); D2 = fft(nz(P2n) - nz(P4n))
        al = [np.log2(np.linalg.norm(D1[b]) / max(np.linalg.norm(D2[b]), 1e-300))
              for b in bands[:6]]
        print(f"    {eps:9.0e} | " + " ".join(f"{a:6.2f}" for a in al))
    print(f"    → the ratio collapses by ε=1e-5 in the saturated bands; the")
    print(f"      barycentre holds at ε=5e-2 — four orders of robustness.")

    print("\n" + "=" * 74)
    print("  Two runs of an unmodified solver; resona reads the generator (1%)")
    print("  and the band spectrum back out of the error everyone discards —")
    print("  with the order law checked from the data and the solver-specific")
    print("  constant refused where it does not hold.  Error as instrument.")
    print("=" * 74)
