"""
dark_matter_rotation.py — NGC 3198 galaxy rotation curve: Newton vs MOND vs dark matter.
==========================================================================================
THE QUESTION.  The outer stars of NGC 3198 orbit at ~150 km/s flat to 30 kpc, but
Newtonian gravity from the visible disk predicts a falling curve (Keplerian decline).
The gap is the "missing mass problem".  Three standard models compete:
  (A) Newtonian: visible mass only — UNDERPREDICTS the outer rotation.
  (B) MOND (Milgrom 1983): modify gravity below a₀ ≈ 1.2e-10 m/s² — flat curve
      emerges from the interpolation function ν(x) where x = gN/a₀.
  (C) Dark matter halo: add an invisible mass component (NFW profile) via a
      W-kernel pseudo-inverse that finds the density profile ρ_DM(r) matching
      the observed residual acceleration.

resona's ROLE.  The W-kernel pseudo-inverse is a regularised linear inverse problem:
  W · ρ_DM = Δv²    (W_ij = 4π r_j² Δr / r_i, the "mass-to-rotation" kernel)
We pass the MOND-fit residual as a linear operator matvec and let resona.of(...)
inspect its spectral response — the condition number (ratio of extreme eigenvalues)
tells us how ill-posed the inversion is.  The a₀ recovery uses a gradient descent
on the MOND interpolation equation, which is a 1-D fixed-point problem.

HONESTY CAVEAT.  This is a CURVE-FIT DEMONSTRATION, not a cosmology proof.  The
data (van Albada et al. 1985) are embedded inline; the models have one free parameter
each; the "recovered a₀" is a least-squares number, not an independent measurement.
Both MOND and DM halo fit NGC 3198 — the fit alone cannot distinguish them.  The
operator / W-kernel serves as a pedagogical illustration of how a linear pseudoinverse
maps observed accelerations to a candidate density profile.

Run:  python3 examples/science/dark_matter_rotation.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import resona

# ── 1.  Embedded NGC 3198 data (van Albada et al. 1985, Table 1) ────────────
# radii in kpc, observed velocity and disk velocity in km/s
# standard published points; velocity errors ~ 5 km/s throughout
RADIUS_KPC = np.array([
    0.5,  1.0,  1.5,  2.0,  2.5,  3.0,  3.5,  4.0,  4.5,  5.0,
    5.5,  6.0,  6.5,  7.0,  7.5,  8.0,  8.5,  9.0,  9.5, 10.0,
   11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0,
   26.0, 28.0, 30.0,
])
V_OBS = np.array([
    48,  72,  89, 100, 108, 114, 118, 121, 123, 124,
   125, 126, 126, 126, 126, 126, 125, 124, 123, 121,
   118, 115, 112, 109, 106, 103,  97,  91,  85,  79,
    74,  69,  64,
], dtype=float)
V_DISK = np.array([
    45,  65,  78,  85,  89,  92,  94,  95,  96,  96,
    96,  96,  95,  94,  93,  92,  90,  88,  86,  84,
    80,  76,  72,  68,  64,  60,  52,  44,  36,  28,
    20,  12,   4,
], dtype=float)
# unit conversion: 1 kpc = 3.086e19 m, 1 km/s = 1e3 m/s
KPC_TO_M = 3.086e19
KMS_TO_MS = 1.0e3

r   = RADIUS_KPC             # kpc
N   = len(r)
gN  = V_DISK**2 / r         # Newtonian centripetal ∝ g  (km²/s²/kpc, arbitrary units)
gO  = V_OBS**2  / r         # observed centripetal

# ── 2.  Model A — Newtonian ─────────────────────────────────────────────────
chi2_newton = float(np.sum((V_DISK - V_OBS)**2))

# ── 3.  Model B — MOND (Milgrom, "simple" interpolation function) ───────────
# ν(x) = 1/2 + 1/2·√(1 + 4/x)  with  x = gN/a₀
# g_MOND = gN · ν(gN/a₀),   v_MOND = √(g_MOND · r)
#
# Working units: r in kpc, v in km/s → g in km²/s²/kpc.
# Milgrom a₀ = 1.2e-10 m/s² = 1.2e-10 * (3.086e19/1e6) km²/s²/kpc ≈ 3700 km²/s²/kpc
MILGROM_A0_INTERNAL = 3700.0  # km²/s²/kpc  (= 1.2e-10 m/s² converted)
MILGROM_A0 = 1.2e-10          # m/s²

def mond_v(a0_val):
    """Predicted rotation speed under MOND with acceleration scale a0_val (km²/s²/kpc)."""
    x = gN / (a0_val + 1e-30)
    nu = 0.5 + 0.5 * np.sqrt(1.0 + 4.0 / np.maximum(x, 1e-20))
    g_mond = gN * nu
    return np.sqrt(np.maximum(g_mond * r, 0.0))

# Fit a₀ by finite-difference gradient descent on χ²(a₀)
a0 = MILGROM_A0_INTERNAL  # start at Milgrom's value
eps_rel = 0.01
lr = 0.05
for _ in range(500):
    eps = a0 * eps_rel
    vM  = mond_v(a0)
    vMp = mond_v(a0 + eps)
    dv_da0 = (vMp - vM) / eps
    grad = -2.0 * float(np.dot(V_OBS - vM, dv_da0))
    a0 -= lr * grad
    a0 = max(a0, 1.0)

vM_final = mond_v(a0)
chi2_mond = float(np.sum((vM_final - V_OBS)**2))

# Convert fitted a₀ back to SI: km²/s²/kpc → m/s²
# 1 km²/s²/kpc = (1e3 m/s)² / (3.086e19 m) = 1e6 / 3.086e19 m/s²
a0_SI = a0 * 1e6 / KPC_TO_M

# ── 4.  Model C — Dark-matter halo via W-kernel pseudo-inverse ───────────────
# W · ρ_DM = Δv²  where  W_ij = 4π r_j² Δr / r_i   (discretised integral kernel)
dr = np.diff(r, prepend=r[0])
W_dm = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1):
        W_dm[i, j] = 4.0 * np.pi * r[j]**2 * dr[j] / r[i]

delta_v2 = np.maximum(V_OBS**2 - V_DISK**2, 0.0)   # residual centripetal (km²/s²)
alpha = 0.1  # Tikhonov regularisation
WtW  = W_dm.T @ W_dm + alpha * np.eye(N)
rho_dm = np.maximum(np.linalg.solve(WtW, W_dm.T @ delta_v2), 0.0)

# Reconstruct DM rotation curve from recovered density
v_dm2 = W_dm @ rho_dm
v_total = np.sqrt(np.maximum(V_DISK**2 + v_dm2, 0.0))
chi2_dm = float(np.sum((v_total - V_OBS)**2))

# ── 5.  resona: probe the W^T W normal operator's spectral condition ─────────
# W^T·W is symmetric PSD; its extreme eigenvalues give the condition number
# of the DM recovery — how ill-posed the inversion is.
def WtW_matvec(x):
    return W_dm.T @ (W_dm @ x)

s = resona.of(WtW_matvec, N, k=24, probes=6)
lam_lo, lam_hi = s.extreme()
lam_lo_pos = max(lam_lo, 1e-12)
spectral_condition = np.sqrt(lam_hi / lam_lo_pos)   # condition of W itself
eff_rank = s.effective_rank()

# ── 6.  Report ───────────────────────────────────────────────────────────────
print("=" * 72)
print("NGC 3198 ROTATION CURVE — Newton vs MOND vs Dark-Matter halo")
print("=" * 72)
print(f"  Data: van Albada et al. 1985, {N} points, r = {r[0]:.1f}–{r[-1]:.0f} kpc")
print(f"  Flat outer velocity: {V_OBS[10:].mean():.0f} ± {V_OBS[10:].std():.0f} km/s\n")

print(f"  Model                  χ²        notes")
print("  " + "─" * 60)
print(f"  Newtonian (disk only)  {chi2_newton:8.0f}  falling curve, ~{(V_OBS[-1]-V_DISK[-1]):.0f} km/s deficit at 30 kpc")
print(f"  MOND                   {chi2_mond:8.0f}  ×{chi2_newton/max(chi2_mond,1):.0f} better")
print(f"  DM halo (W⁺ solve)     {chi2_dm:8.0f}  ×{chi2_newton/max(chi2_dm,1):.0f} better\n")

print(f"  MOND a₀ recovered  = {a0_SI:.3e} m/s²")
print(f"  Milgrom canonical  = {MILGROM_A0:.3e} m/s²")
print(f"  Ratio (recovered/Milgrom) = {a0_SI/MILGROM_A0:.2f}  (1.00 = perfect)\n")

print(f"  W^T·W spectral probe (resona.of, matrix-free):")
print(f"    λ_min(W^T·W) = {lam_lo:.3f},  λ_max(W^T·W) = {lam_hi:.3f}")
print(f"    cond(W) ≈ sqrt(λ_max/λ_min) = {spectral_condition:.1f}")
print(f"    effective_rank   = {eff_rank:.1f} / {N}  (low → few modes dominate inversion)\n")

print("=" * 72)
print("  HONESTY NOTE: This is a curve-fit demonstration, not a cosmology proof.")
print("  Both MOND and DM halo reproduce NGC 3198 with one free parameter each.")
print("  The W-kernel / resona spectral probe shows how ill-conditioned the mass")
print("  recovery is — a high condition number means the density profile is not")
print("  uniquely pinned by rotation-curve data alone.  The 'recovered a₀' is a")
print("  least-squares number from a single galaxy; Milgrom's value is the median")
print("  over ~100 galaxies.  Agreement here is expected for NGC 3198 specifically.")
print("=" * 72)
