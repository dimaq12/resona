"""
sacred_constants.py — 100+ mathematical constants packed into ONE operator.
=============================================================================
WHAT.  We embed 115 mathematical constants (pi, e, phi, sqrt2, Euler-gamma,
Riemann-zeta zeros, fine-structure constant, Planck length, etc.) as the
diagonal entries of a symmetric matrix A = diag(sorted constants) and probe
its spectrum with resona.  The resulting spectral fingerprint (effective_rank,
extreme eigenvalues, density) is genuinely meaningful as a description of
the *distribution* of these numbers — even if the specific patterns are
largely accidents of our choice of constant set.

WHY / resona's ROLE.  resona.of(matvec, N) is a matrix-free spectral probe.
A diagonal matrix is the simplest self-adjoint operator: its eigenvalues ARE
its diagonal entries.  So `resona.of(A_diag_matvec, N)` literally reads the
empirical spectral distribution of the constant collection — the density of
states is just the histogram of the constants, and the effective_rank measures
how "spread" vs "concentrated" the collection is on the positive axis.

The W-kernel idea from the FA program: we sort the constants and look at
spectral gaps — places where the "spectrum" has a large jump, potentially
flagging which decades of scale are well-sampled and which are empty.

Three more resona reads carry the analysis further:
  - s.levels(N): the Beta max-entropy closure reconstructs ALL N log-eigenvalues
    from just four numbers (support + two moments) — we check it against the
    true sorted log-values.
  - resona.local_spectrum(matvec, v): the measure seen from an INDICATOR vector
    over a class of constants (pure math vs physics) — class-resolved spectral
    statistics, read matrix-free from the same operator.

HONESTY CAVEAT.  This is a WHIMSICAL TOY.  The "spectral structure" reflects
nothing deep about the constants themselves — just the distribution of their
numerical values after we arrange them.  The effective_rank, for instance,
is dominated by the few very large constants (Planck temperature ~1e32, Age
of Universe ~4e17, etc.).  The interesting lesson is that resona is agnostic
about WHAT the numbers mean; it just probes the operator they define.

Run:  python3 examples/wild/sacred_constants.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import resona

# ── 115 embedded constants ────────────────────────────────────────────────────
# (name, value)  — all positive (we take abs where needed)
phi = (1.0 + np.sqrt(5)) / 2.0

CONSTANTS = [
    # ── Pure mathematics ──────────────────────────────────────────────────
    ("pi",              np.pi),
    ("e",               np.e),
    ("phi (golden)",    phi),
    ("sqrt(2)",         np.sqrt(2)),
    ("sqrt(3)",         np.sqrt(3)),
    ("sqrt(5)",         np.sqrt(5)),
    ("sqrt(7)",         np.sqrt(7)),
    ("ln(2)",           np.log(2)),
    ("ln(10)",          np.log(10)),
    ("Euler-gamma",     0.5772156649015329),
    ("Catalan G",       0.9159655941772190),
    ("Khinchin K",      2.6854520010653064),
    ("Feigenbaum d",    4.6692016091029906),
    ("Feigenbaum a",    2.5029078750958928),
    ("Omega (W(1))",    0.5671432904097839),
    ("Landau-Ramanujan",0.7642356354152948),
    ("pi^2",            np.pi**2),
    ("pi^e",            np.pi**np.e),
    ("e^pi",            np.exp(np.pi)),
    ("e^pi - pi",       np.exp(np.pi) - np.pi),
    ("Ramanujan-hardy", np.exp(np.pi * np.sqrt(163))),  # ≈ 262537412640768744
    ("zeta(2)=pi^2/6",  np.pi**2 / 6.0),
    ("zeta(3) Apery",   1.2020569031595943),
    ("zeta(4)=pi^4/90", np.pi**4 / 90.0),
    ("1/ln(phi)",       1.0 / np.log(phi)),
    ("2^(1/12)",        2**(1.0/12)),           # equal-tempered semitone
    ("phi^2",           phi**2),
    ("phi^3",           phi**3),
    ("2/pi",            2.0 / np.pi),
    ("pi/4",            np.pi / 4.0),
    # ── Riemann zeta zeros Im(rho_n) ──────────────────────────────────────
    ("zeta-zero-1",     14.134725141734693),
    ("zeta-zero-2",     21.022039638771554),
    ("zeta-zero-3",     25.010857580145688),
    ("zeta-zero-4",     30.424876125859513),
    ("zeta-zero-5",     32.935061587739189),
    ("zeta-zero-6",     37.586178158825671),
    ("zeta-zero-7",     40.918719012147495),
    ("zeta-zero-8",     43.327073280914999),
    ("zeta-zero-9",     48.005150881167159),
    ("zeta-zero-10",    49.773832477672302),
    ("zeta-zero-11",    52.970321477714460),
    ("zeta-zero-12",    56.446247697063394),
    ("zeta-zero-13",    59.347044002602353),
    ("zeta-zero-14",    60.831778524609809),
    ("zeta-zero-15",    65.112544048081606),
    # ── Musical / harmonic ratios ─────────────────────────────────────────
    ("octave 2:1",      2.0),
    ("fifth 3:2",       1.5),
    ("fourth 4:3",      4.0 / 3.0),
    ("major-third 5:4", 1.25),
    ("minor-third 6:5", 1.2),
    ("tritone 45:32",   45.0 / 32.0),
    ("A440",            440.0),
    ("A432",            432.0),
    ("Schumann 7.83",   7.83),
    ("middle-C",        261.626),
    # ── Sacred / combinatorial ────────────────────────────────────────────
    ("12 (months)",     12.0),
    ("60 (sexagesimal)",60.0),
    ("360 (degrees)",   360.0),
    ("108",             108.0),
    ("216=6^3",         216.0),
    ("432=216*2",       432.0),
    ("666",             666.0),
    ("1000",            1000.0),
    ("1729 Hardy-Ram.", 1729.0),
    ("5040=7!",         5040.0),
    ("pi*10^4",         np.pi * 1e4),
    # ── Physics constants (SI) ────────────────────────────────────────────
    ("alpha^-1 (FSC)",  137.035999084),
    ("alpha (FSC)",     1.0 / 137.035999084),
    ("c (m/s)",         2.99792458e8),
    ("h (J·s)",         6.62607015e-34),
    ("hbar (J·s)",      1.054571817e-34),
    ("G (m^3/kg/s^2)",  6.67430e-11),
    ("k_B (J/K)",       1.380649e-23),
    ("N_A (/mol)",      6.02214076e23),
    ("e_charge (C)",    1.602176634e-19),
    ("m_e (kg)",        9.1093837015e-31),
    ("m_p (kg)",        1.67262192369e-27),
    ("epsilon_0",       8.8541878128e-12),
    ("mu_0",            1.25663706212e-6),
    ("R_gas (J/mol/K)", 8.314462618),
    ("sigma_SB",        5.670374419e-8),
    ("atm (Pa)",        101325.0),
    ("Planck_l (m)",    1.616255e-35),
    ("Planck_m (kg)",   2.176434e-8),
    ("Planck_t (s)",    5.391247e-44),
    ("Planck_T (K)",    1.416784e32),
    ("Hubble (1/s)",    2.192e-18),
    ("CMB_T (K)",       2.7255),
    ("age_univ (s)",    4.355e17),
    ("G_F (GeV^-2)",    1.1663788e-5),
    ("sin^2(theta_W)",  0.23122),
    ("m_W (GeV)",       80.377),
    ("m_Z (GeV)",       91.1876),
    ("m_H (GeV)",       125.25),
    ("m_top (GeV)",     172.76),
    # ── Chemistry / atomic numbers ────────────────────────────────────────
    ("Z_H=1",           1.0),
    ("Z_He=2",          2.0),
    ("Z_C=6",           6.0),
    ("Z_N=7",           7.0),
    ("Z_O=8",           8.0),
    ("Z_Fe=26",         26.0),
    ("Z_Au=79",         79.0),
    ("Z_U=92",          92.0),
    # ── Chess piece values ────────────────────────────────────────────────
    ("pawn",            1.0),
    ("knight/bishop",   3.0),
    ("rook",            5.0),
    ("queen",           9.0),
    ("board 64",        64.0),
    # ── Misc curiosities ──────────────────────────────────────────────────
    ("ln(pi)",          np.log(np.pi)),
    ("exp(1/e)",        np.exp(1.0 / np.e)),
    ("pi+e",            np.pi + np.e),
    ("pi*e",            np.pi * np.e),
    ("Brun's const",    1.9021605823),
    ("Copeland-Erdos",  0.2357111317),
    ("Champernowne",    0.1234567891),
]

# ── Build the operator: diag matrix of sorted constants ───────────────────────
names  = [c[0] for c in CONSTANTS]
values = np.array([abs(float(c[1])) for c in CONSTANTS])   # take abs; must be > 0

# Log-scale the values so the operator is not dominated by a few huge numbers.
# We probe BOTH: the raw-value diagonal and the log-value diagonal.
log_values = np.log(values + 1.0)   # log(1 + |c|) to handle < 1

order      = np.argsort(values)
vals_sorted = values[order]
logv_sorted = log_values[order]
N = len(vals_sorted)

def raw_matvec(x):
    return vals_sorted * x      # diagonal operator — eigenvalues = sorted values

def log_matvec(x):
    return logv_sorted * x      # log-scale version

# ── Probe with resona ─────────────────────────────────────────────────────────
# Raw operator: 73-decade range → use exact min/max from the sorted diagonal
# (Lanczos numerics are unreliable over such a range; extreme() still reports
# the Ritz bounds from random probes, which will miss tiny eigenvalues).
lam_lo_raw = float(vals_sorted[0])
lam_hi_raw = float(vals_sorted[-1])

# Log-scale operator: well-conditioned, resona probes reliably
s_log = resona.of(log_matvec, N, k=48, probes=12)
lam_lo_log, lam_hi_log = s_log.extreme()
eff_rank_log = s_log.effective_rank()

# For raw effective_rank we compute analytically (diagonal: Tr(A)^2/Tr(A^2))
m1_raw = float(vals_sorted.mean())       # Tr(A)/N
m2_raw = float(np.mean(vals_sorted**2))  # Tr(A^2)/N
eff_rank_raw = float(m1_raw**2 / m2_raw) if m2_raw > 0 else 1.0

# Spectral gaps: find the top-5 largest multiplicative jumps in the sorted values
ratios  = vals_sorted[1:] / np.maximum(vals_sorted[:-1], 1e-300)
top5_gap_idx = np.argsort(ratios)[-5:][::-1]

# Density of states in log-space (where the constants are spread more uniformly)
log_range = np.linspace(logv_sorted.min(), logv_sorted.max(), 80)
density   = s_log.density(log_range, eta=0.3)

# Find the 3 peaks in the density
from scipy.signal import argrelmax
peaks, = argrelmax(density, order=3)
top3_peaks = peaks[np.argsort(density[peaks])[-3:]][::-1]

# ── Beta max-entropy closure: ALL N levels from 4 numbers (s.levels) ──────────
# resona reads support + two moments matrix-free; the Beta closure unfolds the
# whole log-spectrum.  Compare against the true sorted log-values.
levels_beta  = np.sort(s_log.levels(N))
beta_med_err = float(np.median(np.abs(levels_beta - logv_sorted)))
beta_max_err = float(np.max(np.abs(levels_beta - logv_sorted)))
beta_span    = float(logv_sorted.max() - logv_sorted.min())

# ── Class-resolved measures via resona.local_spectrum ─────────────────────────
# Probe the SAME log-operator from indicator vectors over constant classes:
# the local spectral measure mu_v gives exactly the class's log-value statistics.
def class_stats(idx_lo, idx_hi):
    """Mean +/- spread of log(1+|c|) over CONSTANTS[idx_lo:idx_hi+1], read as
    the local spectral measure of the indicator vector (one Lanczos run)."""
    v = np.isin(order, np.arange(idx_lo, idx_hi + 1)).astype(float)
    nodes_c, w_c = resona.local_spectrum(log_matvec, v, k=64)
    w_c = w_c / w_c.sum()
    mean = float(np.sum(w_c * nodes_c))
    var  = max(float(np.sum(w_c * nodes_c**2)) - mean**2, 0.0)
    return mean, np.sqrt(var)

i_math_lo, i_math_hi = names.index("pi"), names.index("pi/4")
i_phys_lo, i_phys_hi = names.index("alpha^-1 (FSC)"), names.index("m_top (GeV)")
math_mean, math_spread = class_stats(i_math_lo, i_math_hi)
phys_mean, phys_spread = class_stats(i_phys_lo, i_phys_hi)

# ── Report ────────────────────────────────────────────────────────────────────
print("=" * 72)
print(f"SACRED CONSTANTS OPERATOR — {N} numbers, one spectrum")
print("=" * 72)
print(f"  {N} constants embedded as eigenvalues of a diagonal operator.")
print(f"  Probed matrix-free by resona.of(matvec, N).\n")

print(f"  RAW-VALUE OPERATOR  (eigenvalues = constant values, SI/natural units):")
print(f"    λ_min = {lam_lo_raw:.4e}  ({names[order[0]]})")
print(f"    λ_max = {lam_hi_raw:.4e}  ({names[order[-1]]})")
print(f"    span in decades : {np.log10(lam_hi_raw/max(lam_lo_raw,1e-300)):.1f}")
print(f"    effective_rank  : {eff_rank_raw:.4f} / {N}  (analytical, Tr^2/Tr^2)")
print(f"    (near-zero eff_rank → spectrum dominated by Planck_T ~1e32)\n")

print(f"  LOG-SCALE OPERATOR  (eigenvalues = log(1+|c|), resona.of, matrix-free):")
print(f"    λ_min = {lam_lo_log:.4f}  λ_max = {lam_hi_log:.4f}")
print(f"    effective_rank  : {eff_rank_log:.2f} / {N}")
print(f"    (eff_rank closer to N → more uniform spread in log-space)\n")

print(f"  TOP-5 SPECTRAL GAPS (largest multiplicative jumps between neighbours):")
for idx in top5_gap_idx:
    print(f"    {names[order[idx]]:24s} ({vals_sorted[idx]:.3e})"
          f"  →  {names[order[idx+1]]:24s} ({vals_sorted[idx+1]:.3e})"
          f"   ×{ratios[idx]:.1f}")

print(f"\n  LOG-DENSITY PEAKS (where constants cluster on the log scale):")
for p in top3_peaks:
    exp_val = np.exp(log_range[p]) - 1.0
    print(f"    log(1+c) ≈ {log_range[p]:.2f}  →  c ≈ {exp_val:.3e}")

print(f"\n  BETA CLOSURE (s.levels): all {N} log-eigenvalues from FOUR numbers")
print(f"  (support endpoints + two trace moments, all read matrix-free):")
print(f"    median |level error| = {beta_med_err:.2f}   max = {beta_max_err:.2f}"
      f"   (span = {beta_span:.1f})")
print(f"    (the heavy Planck-scale tail breaks the smooth-density assumption —")
print(f"     the misfit itself is the diagnostic: this measure is ATOMIC, not smooth)")

print(f"\n  CLASS-RESOLVED MEASURES (resona.local_spectrum from indicator vectors):")
print(f"    pure-math constants : log(1+c) = {math_mean:5.2f} ± {math_spread:.2f}"
      f"   (c ~ {np.exp(math_mean)-1:.1f})")
print(f"    physics constants   : log(1+c) = {phys_mean:5.2f} ± {phys_spread:.2f}"
      f"   (c ~ {np.exp(phys_mean)-1:.3e})")
print(f"    (physics constants live ~{abs(phys_mean-math_mean):.0f} log-units away"
      f" and are ~{phys_spread/max(math_spread,1e-12):.0f}x more spread — SI units, not mysticism)")

print()
print("=" * 72)
print("  HONESTY NOTE: This is a whimsical toy demonstration.")
print("  The 'spectral structure' reflects the numerical distribution of the")
print("  chosen constant set — not any deep mathematical relationship between")
print("  them.  The low effective_rank in raw units is dominated by Planck_T")
print(f"  (~1e32) and age_univ (~4e17) dwarfing everything else.")
print("  In log-space the spread is more uniform (eff_rank ≈ "
      f"{eff_rank_log:.0f}/{N}).")
print("  What resona genuinely provides: a matrix-free spectral probe that")
print("  treats ANY collection of numbers as an operator — no matrix formed,")
print("  no eig called.  The lesson is the API, not the numerology.")
print("=" * 72)
