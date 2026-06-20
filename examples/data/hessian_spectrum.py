"""
THE SHAPE OF A LOSS LANDSCAPE — the Hessian spectrum, matrix-free.

WHAT:  the eigenvalue structure of a model's loss Hessian — how many directions
       carry real curvature ("sharp") vs how many are flat — read with resona from
       Hessian-vector products alone, never forming the d×d Hessian.

WHY:   the curvature of the loss is the Hessian H = ∂²L/∂θ².  Its spectrum is the
       famous picture of modern optimization (Sagun, Ghorbani–Krishnan–Xiao): a
       big BULK of near-zero eigenvalues (flat directions the optimizer ignores)
       plus a FEW large outliers (the sharp directions that set the step size and
       the conditioning).  For d parameters the dense Hessian is d×d — O(d²) memory,
       O(d³) to diagonalize — but a Hessian-VECTOR product Hv is cheap (one extra
       backward pass, or here one pair of data matvecs).  That is all resona needs.

RESONA's role:  resona.of(Hv, d) turns the HVP into a Spectral object and reads the
       curvature structure matrix-free: effective_rank (the real curvature
       dimension), the sharpest eigenvalue, the flat/sharp split — verified against
       a dense eigendecomposition at a size where that is still affordable.

MODEL:  logistic regression, H = (1/n) Xᵀ diag(p(1−p)) X + λI (the exact loss
        Hessian, no autodiff needed); HVP = (1/n) Xᵀ(w ⊙ (X v)) + λ v.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import time
import numpy as np
import resona


def make_logreg(n, d, n_factors=12, lam=1e-2, seed=0):
    """Correlated features via a LATENT-FACTOR model (no d×d matrix, O(n·d)):
    X = Z·Fᵀ + noise, Z latent (n×k), F loadings (d×k) — a few shared factors give
    the Hessian its handful of sharp directions."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((d, n_factors)) / np.sqrt(n_factors)   # loadings
    Z = rng.standard_normal((n, n_factors))                        # latent factors
    X = Z @ F.T + 0.3 * rng.standard_normal((n, d))                # correlated features
    w_true = rng.standard_normal(d) / np.sqrt(d)
    p = 1.0 / (1.0 + np.exp(-(X @ w_true)))
    reweight = p * (1.0 - p)                       # diag of the loss curvature

    def hvp(v):                                    # H v, matrix-free
        return (X.T @ (reweight * (X @ v))) / n + lam * v

    return hvp, X, reweight, lam


def hline(c="="):
    print(c * 76)


def main():
    hline()
    print("  THE SHAPE OF A LOSS LANDSCAPE — Hessian spectrum, matrix-free")
    hline()

    # ── (A) GROUND TRUTH at d=600 (dense Hessian still affordable) ────────────
    n, d, lam = 8000, 600, 1e-2
    hvp, X, rw, lam = make_logreg(n, d, lam=lam, seed=0)
    s = resona.of(hvp, d, k=90, probes=30, seed=1)
    H = (X.T * rw) @ X / n + lam * np.eye(d)
    ev = np.linalg.eigvalsh(H)
    er_true = (ev.sum() ** 2) / (ev ** 2).sum()
    sharp = int((ev > 0.1).sum())

    print(f"\n[A] logistic Hessian, d={d} params, n={n} samples   (λ={lam})")
    print(f"    effective_rank (curvature dimension): resona {s.effective_rank():5.1f}  "
          f"vs dense {er_true:5.1f}")
    print(f"    sharpest curvature λ_max            : resona {s.extreme()[1]:7.3f}  "
          f"vs dense {ev.max():7.3f}")
    print(f"    sharp directions (λ > 0.1)          : {sharp} of {d}")
    print(f"    ⇒ {100 * (ev <= 0.1).mean():.0f}% of the {d}-dimensional loss landscape is FLAT;")
    print(f"      the curvature lives in ~{sharp}–{round(er_true)} directions — read from HVPs, no d×d matrix.")

    # counting function N(t)=#{λ<t} — the ROBUST density read (the spiky histogram
    # is meaningless for a near-degenerate bulk; the COUNT past the bulk is exact).
    eps = 0.03
    def N_mf(t):                                   # Σ smooth-step(λ_i < t) = Tr of f(H)
        return s.trace(lambda x, t=t: 1.0 / (1.0 + np.exp((x - t) / eps)))
    flat_mf, flat_d = N_mf(0.3), int((ev < 0.3).sum())
    sharp_mf, sharp_d = d - N_mf(1.0), d - int((ev < 1.0).sum())
    print(f"    counting function N(t)=#{{λ<t}}, matrix-free vs dense:")
    print(f"      flat bulk  (λ<0.3): resona {flat_mf:6.1f}  vs dense {flat_d}   (the spike — counted, not resolved)")
    print(f"      outliers   (λ>1.0): resona {sharp_mf:6.1f}  vs dense {sharp_d}    (the sharp directions — exact)")

    # ── (B) SCALE: d where the dense Hessian is infeasible ────────────────────
    print(f"\n[B] SCALE — the same read where a dense Hessian is out of reach")
    for d2 in [20_000]:
        hvp2, X2, rw2, lam2 = make_logreg(3000, d2, lam=lam, seed=2)
        t = time.time()
        s2 = resona.of(hvp2, d2, k=70, probes=12, seed=3)
        er = s2.effective_rank(); lo, hi = s2.extreme()
        gb = d2 * d2 * 8 / 1e9
        print(f"    d={d2:>6,}: curvature dim ≈ {er:5.1f}, λ_max ≈ {hi:6.2f}, "
              f"read in {time.time()-t:4.1f}s  (a dense {d2}×{d2} Hessian would be {gb:.0f} GB)")

    hline()
    print("  The loss landscape is mostly flat with a few sharp directions — the")
    print("  modern picture of optimization — read matrix-free from Hessian-vector")
    print("  products, exact on the sharpest curvature, at a width no dense eig"
          "decomposition reaches.")
    hline()


if __name__ == "__main__":
    main()
