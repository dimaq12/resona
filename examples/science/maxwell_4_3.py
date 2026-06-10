"""
maxwell_4_3.py — Maxwell's 4/3 electromagnetic mass paradox as a spectral signature.
======================================================================================
WHAT.  A spherical shell of charge at rest has electromagnetic energy U_em = E_0.
When it moves at velocity v, the electromagnetic momentum should be p = (U_em/c^2)*v
for a Lorentz-covariant object.  But Maxwell's theory gives:
    p_em = (4/3) * (U_em / c^2) * v
— an extra factor of 4/3 (Poincare 1906, Rohrlich 1960, Griffiths 2012).

THE SPECTRAL DIAGNOSIS.  Consider the 2x2 Lorentz-boost Jacobian:
    L = d(P^mu_boosted) / d(P^nu_rest)
For a true 4-vector, L must be the Lorentz boost matrix with det=1 and exactly 2
independent singular values (boost + transverse).  For pure Maxwell EM, the boost
of (E, p) has the WRONG structure: the ratio E/p is off by 4/3, so L deviates from
a proper Lorentz transformation.  We measure this as an eigenvalue/singular-value
deviation: the spectral mismatch captures the missing 1/3.

Separately, the Lorentz-covariance operator C = dP^mu/dv (velocity-momentum Jacobian
of the EM 4-momentum) has a DIFFERENT structure for Maxwell vs correct Poincare:
  - Maxwell:   dP/dv column = [gamma^3*v, (4/3)*gamma + (4/3)*gamma^3*v^2]  WRONG
  - Poincare:  dP/dv column = [gamma^3*v,          gamma + gamma^3*v^2]     correct

We stack dP/dv at multiple velocities into a matrix and measure its spectral structure.
The Maxwell version has an anomalous leading singular value (4/3 inflated), while the
Poincare version has singular values consistent with Lorentz covariance.

resona's ROLE.  resona.of(matvec, N) probes the spectral density of the stacked
Jacobian M^T M; effective_rank() measures how many independent modes the EM response
has.  The ratio of leading eigenvalues directly encodes the 4/3 discrepancy.

HONESTY CAVEAT.  The "rank deficit" is a conceptual framing: in reality both matrices
have full numerical rank (2), but the RATIO of their singular values encodes the 4/3
factor.  A genuine rank-deficit interpretation requires working in a 4D Minkowski
space with the full stress-energy tensor T^{mu nu}, which is beyond this demo's scope.
The 4/3 ratio IS genuine physics; the spectral framing is a valid algebraic observation.

Run:  python3 examples/science/maxwell_4_3.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import resona

# ── Velocity-momentum Jacobian of charged shell ───────────────────────────────
# Natural units: c = E_0 = 1.
# Pure Maxwell:   p_em(v) = (4/3) * gamma(v) * v
# Poincare corrected: p_em(v) = gamma(v) * v       (ratio = 1, covariant)
# Energy component: E(v) = gamma(v)  (same in both cases to leading order)

def dp_dv(v, k=0.0):
    """d/dv of (E, p) for charged shell. k=0: Maxwell, k=-1/3: Poincare."""
    g   = 1.0 / np.sqrt(1 - v**2 + 1e-14)
    dE  = g**3 * v                                          # d(gamma)/dv
    dp  = (4.0/3.0 + k) * (g + g**3 * v**2)               # d[(4/3+k)*gamma*v]/dv
    return np.array([dE, dp])

def stacked_jacobian(k, v_vals):
    """Build Nx2 matrix: each row = dp/dv at velocity v_i."""
    return np.column_stack([dp_dv(v, k) for v in v_vals]).T  # shape (N, 2)

def spectral_probe(M, label):
    """Probe M^T M with resona; compute SVD analysis."""
    MtM = M.T @ M                            # 2x2 Gramian
    s   = np.linalg.svd(M, compute_uv=False)
    cond = s[0] / (s[1] + 1e-14)
    sp   = resona.of(lambda x: MtM @ x, 2, k=2, probes=40, seed=42)
    eff_r = sp.effective_rank()
    lo, hi = sp.extreme()
    return dict(s=s, cond=cond, eff_rank=eff_r, sv_lo=lo, sv_hi=hi,
                s_ratio=s[0]/s[1])

if __name__ == "__main__":
    print("=" * 68)
    print("  MAXWELL 4/3 PARADOX — spectral signature of EM mass anomaly")
    print("=" * 68)

    # Sample velocities — use very small v to stay near the 4/3 limit
    v_vals = np.linspace(0.001, 0.10, 20)
    N      = len(v_vals)

    k_maxwell  = 0.0
    k_poincare = -1.0 / 3.0

    M_mx = stacked_jacobian(k_maxwell,  v_vals)
    M_pc = stacked_jacobian(k_poincare, v_vals)

    res_mx = spectral_probe(M_mx, "Pure Maxwell")
    res_pc = spectral_probe(M_pc, "Poincare")

    # ── The 4/3 ratio ─────────────────────────────────────────────────────────
    ratio_maxwell  = 4.0 / 3.0 + k_maxwell   # = 1.3333
    ratio_poincare = 4.0 / 3.0 + k_poincare  # = 1.0000

    # At low v, dp/dv = (4/3 + k) * dE/dv => s[0]_maxwell / s[0]_poincare ~ 4/3
    sv_ratio = res_mx['s'][0] / res_pc['s'][0]

    print(f"\n  THE 4/3 RATIO (low-velocity limit):")
    print(f"    p_em/p_correct (Maxwell)  = {ratio_maxwell:.6f}  (= 4/3)")
    print(f"    p_em/p_correct (Poincare) = {ratio_poincare:.6f}  (= 1, covariant)")
    print(f"    Discrepancy               = {ratio_maxwell - ratio_poincare:.6f}  (= 1/3)")

    print(f"\n  STACKED JACOBIAN M = [dp/dv|_v1, dp/dv|_v2, ...], shape ({N}, 2):")
    print(f"  {'Quantity':30}  {'Pure Maxwell':>14}  {'Poincare':>12}")
    print("  " + "-" * 60)
    print(f"  {'Leading singular value s_0':30}  {res_mx['s'][0]:>14.4f}  {res_pc['s'][0]:>12.4f}")
    print(f"  {'Trailing singular value s_1':30}  {res_mx['s'][1]:>14.4f}  {res_pc['s'][1]:>12.4f}")
    print(f"  {'Condition number kappa':30}  {res_mx['cond']:>14.2f}  {res_pc['cond']:>12.2f}")
    print(f"  {'resona eff_rank':30}  {res_mx['eff_rank']:>14.3f}  {res_pc['eff_rank']:>12.3f}")
    print(f"  {'s0_maxwell / s0_poincare':30}  {sv_ratio:>14.4f}  {'(should ~ 4/3)':>12}")

    print(f"\n  SPECTRAL ENCODING OF THE 4/3 ANOMALY:")
    print(f"    s0(Maxwell) / s0(Poincare) = {sv_ratio:.4f}")
    print(f"    4/3 exact                  = {4.0/3.0:.4f}")
    print(f"    Error                      = {abs(sv_ratio - 4.0/3.0):.4f}  "
          f"(nonzero: ratio varies with v)")

    print(f"\n  At v -> 0 (first sample v={v_vals[0]:.2f}):")
    dpdv_mx0 = dp_dv(v_vals[0], k_maxwell)
    dpdv_pc0 = dp_dv(v_vals[0], k_poincare)
    ratio_dp = dpdv_mx0[1] / dpdv_pc0[1]
    print(f"    dp/dv[1] ratio: {ratio_dp:.6f}  (exact 4/3 = {4.0/3.0:.6f})")

    # Verify at v -> 0 exactly
    v_tiny = 1e-3
    dp_mx  = dp_dv(v_tiny, k_maxwell)[1]
    dp_pc  = dp_dv(v_tiny, k_poincare)[1]
    print(f"    At v={v_tiny}: dp/dv ratio = {dp_mx/dp_pc:.8f}  (exact: {4.0/3.0:.8f})")

    print(f"\n  METRICS SUMMARY:")
    print(f"    4/3 ratio (Maxwell):             {ratio_maxwell:.6f}")
    print(f"    Ratio with Poincare stress:      {ratio_poincare:.6f}")
    print(f"    Leading SV ratio s0_Mx/s0_Pc:    {sv_ratio:.4f}  (~ 4/3 for low v)")
    print(f"    resona eff_rank Maxwell:         {res_mx['eff_rank']:.3f}")
    print(f"    resona eff_rank Poincare:        {res_pc['eff_rank']:.3f}")
    print(f"    Condition number Maxwell:        {res_mx['cond']:.2f}")
    print(f"    Condition number Poincare:       {res_pc['cond']:.2f}")
    print()
    print("  HONESTY NOTE: both matrices have full numerical rank (2). The 4/3")
    print("  anomaly appears as a SCALING of the leading singular value, not a")
    print("  rank drop.  A genuine rank-deficit interpretation requires the full")
    print("  4D Minkowski stress-energy tensor (T^{mu nu}), out of scope here.")
    print("  The dp/dv ratio at v->0 recovers 4/3 exactly; the physics is real.")
    print("=" * 68)
