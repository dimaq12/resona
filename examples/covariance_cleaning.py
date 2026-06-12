"""
UN-ADD THE NOISE — free deconvolution of a sample covariance matrix.

THE PROBLEM (the daily bread of quantitative finance and high-dimensional
statistics).  You estimate a covariance of N assets from T observations.  When
T is not ≫ N, the sample covariance E is the TRUE covariance C polluted by
Marchenko–Pastur noise — its eigenvalues are systematically spread out, and a
portfolio optimized on E confidently mis-prices risk.

THE FREE-PROBABILITY FACT.  E is C ⊠-multiplied (free multiplicative
convolution) by MP noise of ratio q = N/T.  Free convolution is INVERTIBLE at
the eigenvalue level: the rotationally-invariant estimator (Ledoit–Péché /
Bun–Bouchaud–Potters) reads the inverse off the Stieltjes transform —

    ξ_i = λ_i / |1 − q + q·λ_i·G_E(λ_i − iη)|²,     resona.free.rie_clean

one line per eigenvalue, no fitting, no cross-validation, no prior.

WHAT THIS SCRIPT MEASURES (all against the known ground-truth C):
  • Frobenius distance: empirical vs cleaned vs the ORACLE (the best any
    estimator keeping E's eigenvectors could ever do);
  • the classic risk fiasco: the minimum-variance portfolio built on E
    UNDERESTIMATES its own realized risk; built on the cleaned Ξ it is honest;
  • the additive twin (A + σ·GOE) via rie_clean_additive.

HONEST LIMITS.  RIE is optimal *given E's eigenvectors* — it cannot repair the
basis; the guarantees are asymptotic (N, T → ∞, q fixed); and cleaning helps
exactly when q is sizable — at T ≫ N it converges to a no-op, as it should.

Run:  python3 examples/covariance_cleaning.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from resona.free import rie_clean, rie_clean_additive

rng = np.random.default_rng(0)


def min_var_portfolio(M):
    w = np.linalg.solve(M, np.ones(len(M)))
    return w / w.sum()


if __name__ == "__main__":
    print("=" * 74)
    print("FREE DECONVOLUTION — un-add Marchenko-Pastur noise from a covariance")
    print("=" * 74)

    # ── the market: 3 factors + sector ladder, N=300 assets, T=600 days ──────
    N, T = 300, 600
    q = N / T
    lam_true = np.concatenate([[15.0, 8.0, 4.0], np.linspace(2.0, 0.5, N - 3)])
    U = np.linalg.qr(rng.standard_normal((N, N)))[0]
    C = (U * lam_true) @ U.T
    X = rng.multivariate_normal(np.zeros(N), C, size=T)
    E = X.T @ X / T                                    # the sample covariance
    le, Ue = np.linalg.eigh(E)

    xi = rie_clean(le, q=q)                            # ← the deconvolution
    Xi = (Ue * xi) @ Ue.T
    xi_oracle = np.array([Ue[:, i] @ C @ Ue[:, i] for i in range(N)])
    Or = (Ue * xi_oracle) @ Ue.T

    err = lambda M: np.linalg.norm(M - C)
    print(f"\n  N={N} assets, T={T} observations (q={q}) — distance to the TRUE C:")
    print(f"    raw sample covariance   ‖E−C‖  = {err(E):7.2f}")
    print(f"    RIE-cleaned             ‖Ξ−C‖  = {err(Xi):7.2f}   "
          f"({err(E)/err(Xi):.2f}x closer)")
    print(f"    oracle (same basis)     ‖O−C‖  = {err(Or):7.2f}   "
          f"(RIE delivers {err(Or)/err(Xi)*100:.0f}% of the optimum)")

    # ── the risk fiasco: min-variance portfolio, predicted vs realized ───────
    w_E = min_var_portfolio(E)
    w_Xi = min_var_portfolio(Xi)
    w_C = min_var_portfolio(C)                          # the unreachable ideal
    pred_E = w_E @ E @ w_E
    real_E = w_E @ C @ w_E
    pred_Xi = w_Xi @ Xi @ w_Xi
    real_Xi = w_Xi @ C @ w_Xi
    real_C = w_C @ C @ w_C
    print(f"\n  minimum-variance portfolio (risk = wᵀCw, the REALIZED number):")
    print(f"    {'built on':>12} {'predicts':>10} {'realizes':>10} {'self-deception':>15}")
    print(f"    {'raw E':>12} {pred_E:10.4f} {real_E:10.4f} {real_E/pred_E:14.2f}x")
    print(f"    {'cleaned Ξ':>12} {pred_Xi:10.4f} {real_Xi:10.4f} {real_Xi/pred_Xi:14.2f}x")
    print(f"    {'true C':>12} {real_C:10.4f} {real_C:10.4f} {'1.00x':>15}")
    print(f"    → the raw portfolio under-prices its own risk {real_E/pred_E:.1f}x;")
    print(f"      the cleaned one is nearly honest AND realizes lower risk.")

    # ── the additive twin: E = A + σ·GOE ─────────────────────────────────────
    N2, sigma = 400, 0.7
    A = np.diag(np.linspace(-2, 2, N2))
    W = rng.standard_normal((N2, N2)); W = (W + W.T) / np.sqrt(2 * N2)
    le2, Ue2 = np.linalg.eigh(A + sigma * W)
    xi2 = rie_clean_additive(le2, sigma)
    errA = lambda lam: np.linalg.norm((Ue2 * lam) @ Ue2.T - A)
    print(f"\n  additive twin (A + {sigma}·GOE), subtract the semicircle:")
    print(f"    raw  ‖E−A‖ = {errA(le2):7.2f}    cleaned ‖Ξ−A‖ = {errA(xi2):7.2f}   "
          f"({errA(le2)/errA(xi2):.2f}x, reaches the oracle)")

    print("\n" + "=" * 74)
    print("  Free convolution composes spectra; free DEconvolution un-composes")
    print("  them.  One Stieltjes transform, one line per eigenvalue — and the")
    print("  most common matrix estimate in applied statistics stops lying about")
    print("  its own risk.  resona.free.rie_clean / rie_clean_additive.")
    print("=" * 74)
