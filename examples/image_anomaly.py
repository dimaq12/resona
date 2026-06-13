"""
resona on ANY image — per-region spectral descriptor + anomaly + control.

Same pipeline as a JWST κ_W scan, but the descriptor is the effective rank Φ₁
of the region's PATCH COVARIANCE — which reads spatial GEOMETRY (structured ⇒
low Φ₁; noise ⇒ high Φ₁), not just brightness.  Matrix-free, with the shuffle
control built in (shuffle destroys geometry ⇒ Φ₁ jumps).

Sister example of signals.py: the SAME covariance-operator recipe, the SAME
hub readouts — s.effective_rank(), s.condition() (dynamic range), and the
top-mode energy share λ_max/Tr from s.extreme()/s.moment(1).

Run:  python3 examples/image_anomaly.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from resona import Spectral

rng = np.random.default_rng(0)


def region_patches(region, patch=8, stride=3):
    """The centered patch matrix P (n_patches × patch²) of the region."""
    H, W = region.shape
    rows = []
    for i in range(0, H - patch + 1, stride):
        for j in range(0, W - patch + 1, stride):
            rows.append(region[i:i+patch, j:j+patch].ravel())
    P = np.asarray(rows, float)                       # (n_patches, patch²)
    return P - P.mean(0)


def region_spectral(region, patch=8, stride=3):
    """Spectral object of the region's patch covariance — the SAME recipe as
    signals.py's trajectory covariance: build a matvec, hand it to the hub."""
    P = region_patches(region, patch, stride)
    d = P.shape[1]
    return Spectral.of(lambda v: P.T @ (P @ v), d, k=40, probes=10)   # covariance, matrix-free


def region_phi1(region, patch=8, stride=3):
    """Φ₁ of the region's patch covariance = number of structural components."""
    return region_spectral(region, patch, stride).effective_rank()


def structured_patch(n):
    """a smooth, spatially-correlated feature (galaxy/edge-like)."""
    y, x = np.mgrid[0:n, 0:n] / n
    return (np.sin(6 * x) * np.cos(5 * y) + 2 * np.exp(-((x-.5)**2+(y-.5)**2)/0.05))


if __name__ == "__main__":
    n = 64
    noise = rng.standard_normal((n, n))
    struct = structured_patch(n) + 0.3 * rng.standard_normal((n, n))

    print("=" * 64)
    print("resona on ANY image — geometry descriptor Φ₁ (low=structured, high=noise)")
    print("=" * 64)
    phi_noise = region_phi1(noise)
    phi_struct = region_phi1(struct)
    print(f"   noise region        Φ₁ = {phi_noise:>6.1f}   (high → no structure)")
    print(f"   structured region   Φ₁ = {phi_struct:>6.1f}   (low  → real geometry)")

    # shuffle control — destroys geometry, keeps the histogram
    flat = struct.ravel().copy(); rng.shuffle(flat)
    phi_struct_shuf = region_phi1(flat.reshape(n, n))
    print(f"\n   structured, SHUFFLED Φ₁ = {phi_struct_shuf:>6.1f}   "
          f"(jumps toward noise → Φ₁ measures GEOMETRY, not the histogram)")
    print(f"   geometry signal: structured {phi_struct:.1f} vs shuffled {phi_struct_shuf:.1f} "
          f"(Δ = {phi_struct_shuf-phi_struct:+.1f})")

    # hub readouts on the same operators — the signals.py covariance family:
    # κ = s.condition() (dynamic range) and the top-mode energy share λ_max/Tr.
    # NOTE: λ_max is captured EXACTLY by Lanczos (s.extreme()), but on a spiked
    # operator the stochastic SLQ trace s.moment(1) is biased LOW (its scatter is
    # ~25% here — the spike makes a handful of probe directions dominate), which
    # would INFLATE the ratio.  For this covariance Tr(PᵀP) = ‖P‖_F² is free and
    # EXACT (matrix-free, no eig), so we use it — the share then matches dense truth.
    print(f"\n   hub readouts (same patch-covariance operators, signals.py family):")
    print(f"   {'region':<12} {'Φ₁':>6}  {'κ = λmax/λmin':>14}  {'top-mode share':>15}")
    for tag, region in [("noise", noise), ("structured", struct),
                        ("shuffled", flat.reshape(n, n))]:
        s = region_spectral(region)
        P = region_patches(region)
        tr_exact = float(np.sum(P * P))            # Tr(PᵀP) = ‖P‖_F², exact & free
        share = s.extreme()[1] / tr_exact          # λ_max exact / Tr exact
        print(f"   {tag:<12} {s.effective_rank():>6.1f}  {s.condition():>14.1f}  "
              f"{share:>14.1%}")
    print(f"   (structured: one mode carries most of the energy → tiny Φ₁, huge κ;")
    print(f"    noise/shuffled: energy spreads over ~all 64 modes → flat spectrum)")

    print("\n   → one primitive, any image, matrix-free, control built in.")
    print("     (a full anomaly MAP = region_phi1 over a sliding grid.)")
    print("=" * 64)
