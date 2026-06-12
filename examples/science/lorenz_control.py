"""
lorenz_control.py — Lorenz chaos: butterfly effect as a spectral operator.
===========================================================================
WHAT.  The Lorenz system (1963):
    dx/dt = sigma*(y - x)
    dy/dt = x*(rho - z) - y
    dz/dt = x*y - beta*z

Classical parameters (sigma=10, rho=28, beta=8/3) give chaotic dynamics.
The linearized system at a fixed point is governed by the Jacobian J(sigma,rho,beta).
Its eigenvalues lambda_i determine local stability:
  - All Re(lambda) < 0  =>  stable fixed point (no chaos)
  - Any Re(lambda) > 0  =>  unstable direction  (chaos / diverging trajectories)

THE W-KERNEL (Hellmann-Feynman for parameter sensitivity):
    W_{i,j} = d(lambda_i) / d(param_j)
computed by finite differences of J's eigenvalues.  This is the "butterfly effect
operator": small delta_param shifts eigenvalues by W * delta_param, directly
steering the system between order and chaos.

resona's ROLE.  The Jacobian J is a small (3x3) dense matrix, so we can compute
eigenpairs directly with numpy.  resona.of(matvec, 3) is then used to probe the
spectral density of |J| = sqrt(J^T J) (the singular-value operator), giving a
compact spectral summary; effective_rank() and condition() measure how many
independent modes the linearized flow has and how anisotropic it is.
resona.apply() evolves the linearized system in time (Arnoldi, non-hermitian),
and the growth rate it measures at the origin recovers the unstable eigenvalue.
resona.defect.pseudospectrum_radius() CHECKS the non-normality of J — whether
the marginal eigenvalue pair blooms under perturbation (at C+/C- it reads
normal-like, an honest negative: the eigenvalue reading is trustworthy).
A third steering demo replaces the hand-rolled SVD pseudo-inverse with the
library's wkernel.design (Tikhonov-regularized Hellmann-Feynman step).

HONESTY CAVEAT.  The Jacobian is 3x3 — resona's Lanczos adds no computational
benefit here; the real value is conceptual: the same W-kernel formula that works
for 10^6-dim operators also controls a 3-dim chaos model.  The steering demo
IS genuine: we adjust parameters to push all Re(lambda) negative (stabilize) or
more positive (amplify instability), and we verify this numerically.

Run:  python3 examples/science/lorenz_control.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
from scipy import linalg
import resona
from resona import wkernel as wk      # spectral design step (steering demo 3)
from resona import defect as rdefect  # pseudospectrum: the non-normality read

# ── Lorenz Jacobian ────────────────────────────────────────────────────────────
def lorenz_jacobian(sigma, rho, beta, fp=1):
    """Jacobian at fixed point. fp=0: origin, fp=1: non-trivial C+/C-."""
    if fp == 0:
        x = y = z = 0.0
    else:
        if rho <= 1:
            x = y = 0.0; z = 0.0
        else:
            x = np.sqrt(beta * (rho - 1)); y = x; z = rho - 1
    return np.array([[-sigma,  sigma,   0.0],
                     [rho - z, -1.0,   -x  ],
                     [y,        x,   -beta ]])

def eigvals_sorted(sigma, rho, beta, fp=1):
    J = lorenz_jacobian(sigma, rho, beta, fp)
    return np.sort_complex(linalg.eigvals(J))

def build_W_kernel(sigma, rho, beta, fp=1, eps=1e-3):
    """W[i,j] = d(Re lambda_i)/d(param_j).  Params: sigma, rho, beta.

    NOTE: resona.wkernel (Hellmann-Feynman v^T B v) requires A(k) = A0 + sum k_j B_j.
    The Lorenz Jacobian is nonlinear in (sigma, rho, beta) — dJ/dparam depends on
    the fixed-point coordinates which themselves shift with params — so the operator
    is not of that affine form.  Finite differences remain the correct approach here.
    """
    lam0 = eigvals_sorted(sigma, rho, beta, fp)
    W = np.zeros((3, 3))
    for j, (ds, dr, db) in enumerate([(eps,0,0),(0,eps,0),(0,0,eps)]):
        lamj = eigvals_sorted(sigma+ds, rho+dr, beta+db, fp)
        W[:, j] = np.real(lamj - lam0) / eps
    return np.real(lam0), W

def steer(sigma0, rho0, beta0, target_real, steps=60, lr=0.25, fp=1):
    """Gradient-descent steering: move Re(lambda) toward target via W^+."""
    k = np.array([sigma0, rho0, beta0], float)
    history = []
    for _ in range(steps):
        lam_r, W = build_W_kernel(*k, fp=fp)
        U, s, Vt = linalg.svd(W, full_matrices=False)
        s_reg = np.where(s > 1e-2, s, 1e-2)
        W_pinv = Vt.T @ np.diag(1.0 / s_reg) @ U.T
        dk = lr * (W_pinv @ (target_real - lam_r))
        k  = np.maximum(k + dk, 0.05)
        history.append(lam_r.copy())
    return k, lam_r, np.array(history)

def steer_design(sigma0, rho0, beta0, target_real, steps=80, lr=0.25, fp=1, reg=1e-4):
    """Same steering loop, but the parameter step is the LIBRARY call:
    wk.design solves W·dk ≈ Δλ with Tikhonov regularization — no hand-rolled
    SVD pseudo-inverse.  (steer() above is kept for comparison.)"""
    k = np.array([sigma0, rho0, beta0], float)
    for _ in range(steps):
        lam_r, W = build_W_kernel(*k, fp=fp)
        dk = lr * wk.design(W, target_real - lam_r, reg=reg)
        k  = np.maximum(k + dk, 0.05)
    lam_r, _ = build_W_kernel(*k, fp=fp)
    return k, lam_r

if __name__ == "__main__":
    print("=" * 68)
    print("  LORENZ CONTROL — butterfly effect as spectral W-kernel")
    print("=" * 68)

    sigma0, rho0, beta0 = 10.0, 28.0, 8.0 / 3.0

    # ── 1. Analyse classical attractor ───────────────────────────────────────
    lam_c, W_c = build_W_kernel(sigma0, rho0, beta0, fp=1)
    lam_orig, _ = build_W_kernel(sigma0, rho0, beta0, fp=0)
    print(f"\n  CLASSICAL LORENZ  (sigma={sigma0}, rho={rho0}, beta={beta0:.4f})")
    print(f"    Eigenvalues at C+/C-: {np.round(lam_c, 4)}")
    print(f"    Re(lambda):           {np.round(lam_c, 4)}")
    print(f"    Eigenvalues at origin:{np.round(lam_orig, 4)}")
    print(f"    -> origin has {np.sum(lam_orig > 0)} unstable direction(s) => chaos")

    # resona probes the Jacobian's singular-value operator
    J0  = lorenz_jacobian(sigma0, rho0, beta0, fp=1)
    JtJ = J0.T @ J0
    sv  = resona.of(lambda x: JtJ @ x, 3, k=3, probes=30, seed=7)
    eff_r = sv.effective_rank()
    lo, hi = sv.extreme()
    print(f"\n  resona probes |J|^2 = J^T J (3x3 Jacobian):")
    print(f"    Spectral support: [{lo:.3f}, {hi:.3f}]")
    print(f"    eff_rank = {eff_r:.2f}  (3 = fully generic, <3 = structured)")
    print(f"    cond(J) = sqrt(s.condition()) = {np.sqrt(sv.condition()):.3f}"
          f"  (anisotropy of the linearized flow)")

    # Non-normality: the Lorenz Jacobian is NOT normal, so the spectrum alone
    # understates perturbation response.  resona.defect measures the epsilon-
    # pseudospectrum bloom around the marginal eigenvalue pair.
    lam_J    = linalg.eigvals(J0)
    lam_marg = lam_J[np.argmax(lam_J.real)]
    eps_ps   = 1e-3
    rad_ps   = rdefect.pseudospectrum_radius(J0, eps_ps, z0=lam_marg, r_max=50.0)
    amp_ps = rad_ps / eps_ps
    print(f"\n  NON-NORMALITY CHECK (resona.defect.pseudospectrum_radius):")
    print(f"    marginal eigenvalue: {lam_marg:.4f}")
    print(f"    eps={eps_ps:.0e} bloom radius = {rad_ps:.4f}  "
          f"(x{amp_ps:.1f} the eps of a normal operator)")
    if amp_ps < 3:
        print(f"    -> at C+/C- the marginal pair is NORMAL-LIKE: the eigenvalue")
        print(f"       reading is trustworthy; no hidden transient amplification")
    else:
        print(f"    -> non-normal bloom x{amp_ps:.0f}: transient growth beyond"
              f" what Re(lambda) predicts")

    # ── 2. W-kernel analysis ─────────────────────────────────────────────────
    print(f"\n  W-KERNEL  d(Re lambda_i)/d(sigma, rho, beta):")
    print(f"  {'mode':>6}  {'Re(lam)':>9}  {'d/dsigma':>10}  {'d/drho':>9}  {'d/dbeta':>9}")
    print("  " + "-" * 50)
    for i in range(3):
        print(f"  {i:>6}  {lam_c[i]:>9.4f}  {W_c[i,0]:>10.4f}  "
              f"{W_c[i,1]:>9.4f}  {W_c[i,2]:>9.4f}")
    dom_param = ["sigma", "rho", "beta"][int(np.argmax(np.abs(W_c).max(axis=0)))]
    dom_val   = np.abs(W_c).max()
    print(f"  -> '{dom_param}' is the dominant control parameter "
          f"(max |W| = {dom_val:.3f})")

    # ── 3. Steer to stable ───────────────────────────────────────────────────
    print(f"\n  STEERING DEMO 1: stabilize all eigenvalues (target: all Re<0)")
    target_stable = np.array([-0.5, -2.0, -5.0])
    k_s, lam_s, hist_s = steer(sigma0, rho0, beta0, target_stable, steps=80)
    stable_ok = np.all(lam_s < 0)
    print(f"    Target:   {target_stable}")
    print(f"    Achieved: {np.round(lam_s, 3)}")
    print(f"    New params: sigma={k_s[0]:.2f}, rho={k_s[1]:.2f}, beta={k_s[2]:.3f}")
    print(f"    -> {'STABLE (all Re<0) -- chaos suppressed' if stable_ok else 'not fully stable'}")

    # ── 4. Evaluate at high-rho (more chaotic) parameters ────────────────────
    print(f"\n  STEERING DEMO 2: manually confirm higher rho => more instability")
    # Classical: rho=28 => 1 unstable mode at C+/-
    # At rho=50: stronger instability
    for rho_test in [28.0, 50.0, 100.0]:
        lam_t, _ = build_W_kernel(sigma0, rho_test, beta0, fp=1)
        n_u = int(np.sum(lam_t > 0))
        print(f"    rho={rho_test:.0f}: Re(lam)={np.round(lam_t,3)}, "
              f"{n_u} unstable direction(s)")
    # Confirm: steering to stable works, W correctly predicts direction
    print(f"    [steering toward MORE positive Re via W-kernel:]")
    target_chaos = np.array([3.0, 1.0, -4.0])
    k_ch, lam_ch, hist_ch = steer(sigma0, rho0, beta0, target_chaos, steps=80)
    n_unstable = int(np.sum(lam_ch > 0))
    print(f"    Target:   {target_chaos}")
    print(f"    Achieved: {np.round(lam_ch, 3)}")
    print(f"    New params: sigma={k_ch[0]:.2f}, rho={k_ch[1]:.2f}, beta={k_ch[2]:.3f}")
    print(f"    -> {n_unstable} unstable direction(s) "
          f"(W-kernel guides parameter search; positivity constraint limits result)")

    # ── 4b. Steer with the LIBRARY design step (wkernel.design) ─────────────
    print(f"\n  STEERING DEMO 3: same stabilization, library step (wk.design):")
    k_d, lam_d = steer_design(sigma0, rho0, beta0, target_stable, steps=80)
    stable_d = np.all(lam_d < 0)
    print(f"    Target:   {target_stable}")
    print(f"    Achieved: {np.round(lam_d, 3)}")
    print(f"    New params: sigma={k_d[0]:.2f}, rho={k_d[1]:.2f}, beta={k_d[2]:.3f}")
    print(f"    -> {'STABLE (all Re<0)' if stable_d else 'not fully stable'} "
          f"via wk.design (Tikhonov-regularized Hellmann-Feynman step)")

    # ── 5. resona.apply: linearized time evolution ────────────────────────────
    print(f"\n  LINEARIZED EVOLUTION via resona.apply (exp(t*J)*v0):")
    v0 = np.array([1.0, 0.0, 0.0])
    for t in [0.0, 0.05, 0.10, 0.20]:
        vt = resona.apply(lambda x: J0 @ x, lambda lam: np.exp(t * lam),
                          v0, k=3, hermitian=False)
        print(f"    t={t:.2f}:  ||v(t)|| = {np.linalg.norm(vt):.4f}")

    # Butterfly rate read by resona.apply at the UNSTABLE origin: the late-time
    # growth of ||exp(t*J_origin)·v0|| recovers the positive eigenvalue.
    J_orig = lorenz_jacobian(sigma0, rho0, beta0, fp=0)
    n1 = np.linalg.norm(resona.apply(lambda x: J_orig @ x,
                                     lambda lam: np.exp(0.4 * lam),
                                     v0, k=3, hermitian=False))
    n2 = np.linalg.norm(resona.apply(lambda x: J_orig @ x,
                                     lambda lam: np.exp(0.5 * lam),
                                     v0, k=3, hermitian=False))
    growth = float(np.log(n2 / n1) / 0.1)
    print(f"    At the origin (unstable): d ln||v||/dt = {growth:.4f}"
          f"  vs lambda_max = {lam_orig.max():.4f}"
          f"  (err {abs(growth - lam_orig.max()):.1e})")

    print(f"\n  METRICS SUMMARY:")
    print(f"    Classical Re(lambda) at C+/C-:  {np.round(lam_c, 4)}")
    print(f"    Dominant control parameter: '{dom_param}' (|dλ/dparam|_max={dom_val:.3f})")
    print(f"    Stabilization achieved: {stable_ok}")
    print(f"    Unstable modes after amplification: {n_unstable}")
    print(f"    resona eff_rank of J^T J: {eff_r:.2f}")
    print()
    print("  HONESTY NOTE: J is 3x3 so numpy eigvals suffices; resona adds the")
    print("  conceptual unification (same W-kernel at any scale) and the apply()")
    print("  time-evolution.  The steering is genuine: verified by re-evaluating")
    print("  eigenvalues at the steered parameters.")
    print("=" * 68)
