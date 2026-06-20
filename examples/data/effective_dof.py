"""
HOW MANY PARAMETERS ARE YOU REALLY USING — the effective degrees of freedom.

WHAT:  a ridge / kernel regression with regularization λ does not use all its
       nominal parameters.  Its EFFECTIVE degrees of freedom is

           DoF(λ) = Tr( K (K + λI)⁻¹ ) = Σ_i σ_i / (σ_i + λ),

       (σ_i = eigenvalues of the kernel/Gram K) — the real model complexity, the
       quantity that sits in AIC/GCV, the bias–variance dial.  resona reads it
       matrix-free, never forming or factoring the n×n kernel.

WHY:   DoF(λ) is a trace of a function of K.  Computed densely it needs the
       eigenvalues of K — O(n³).  But it is exactly a spectral-functional trace,
       Tr f(K) with f(σ)=σ/(σ+λ), so stochastic Lanczos quadrature reads it from
       kernel matvecs alone.  Sweep λ and you get the whole complexity curve for
       the price of a few matvecs — on a kernel too big to diagonalize.

RESONA's role:  resona.of(Kv, n).trace(lambda σ: σ/(σ+λ)) = DoF(λ), matrix-free,
       verified against the dense Σ σ_i/(σ_i+λ) at a size where that is affordable.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import time
import numpy as np
import resona


def make_kernel(n, p, strong=10, seed=0):
    """Linear/Gram kernel K = XXᵀ/p with a few dominant feature directions
    (so DoF has a clear knee).  Kv = X(Xᵀv)/p is matrix-free, O(n·p)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X[:, :strong] *= 3.0                       # a handful of strong directions
    return (lambda v: (X @ (X.T @ v)) / p), X


def hline(c="="):
    print(c * 74)


def main():
    hline()
    print("  HOW MANY PARAMETERS ARE YOU REALLY USING — effective DoF, matrix-free")
    hline()

    # ── (A) GROUND TRUTH at n=2000 (dense kernel eig still affordable) ────────
    n, p = 2000, 300
    Kv, X = make_kernel(n, p, seed=0)
    K = X @ X.T / p
    ev = np.linalg.eigvalsh(K)
    s = resona.of(Kv, n, k=80, probes=24, seed=1)

    print(f"\n[A] kernel K = XXᵀ/p,  n={n} points, p={p} features   (verify vs dense)")
    print(f"    {'λ (ridge)':>10} {'DoF resona':>11} {'DoF dense':>10} {'rel-err':>8}")
    for lam in (10.0, 1.0, 0.1, 0.01):
        dof_mf = s.trace(lambda x, l=lam: x / (x + l))
        dof_d = float((ev / (ev + lam)).sum())
        print(f"    {lam:>10} {dof_mf:>11.1f} {dof_d:>10.1f} {abs(dof_mf-dof_d)/dof_d:>8.1%}")
    print(f"    ⇒ DoF is the effective # of parameters: λ→0 gives ~rank({K.shape[0]} kernel),")
    print(f"      λ→∞ gives ~0.  The knee is the regularization that matters.")

    # ── (B) SCALE: a kernel too big to diagonalize ───────────────────────────
    print(f"\n[B] SCALE — the same DoF curve where a dense n×n kernel is infeasible")
    n2, p2 = 50_000, 400
    Kv2, X2 = make_kernel(n2, p2, seed=2)
    s2 = resona.of(Kv2, n2, k=70, probes=16, seed=3)
    gb = n2 * n2 * 8 / 1e9
    t = time.time()
    print(f"    n={n2:,} points  (a dense {n2}×{n2} kernel would be {gb:.0f} GB, O(n³) to eig)")
    for lam in (10.0, 1.0, 0.1):
        dof = s2.trace(lambda x, l=lam: x / (x + l))
        print(f"      λ={lam:<5}: effective DoF ≈ {dof:7.1f}")
    print(f"    read in {time.time()-t:.1f}s, from matvecs only.")

    hline()
    print("  The real complexity of a regularized model — its effective degrees of")
    print("  freedom — is a spectral-functional trace, read matrix-free from kernel")
    print("  matvecs, on a kernel far too large to diagonalize.")
    hline()


if __name__ == "__main__":
    main()
