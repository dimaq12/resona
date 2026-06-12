"""
CERTIFIED, NOT ESTIMATED — Gauss–Radau brackets for the numbers that matter.

Stochastic spectral estimates are honest about being estimates (`with_err`).
This stand is about the rarer thing: numbers with a mathematical GUARANTEE,
matrix-free.  Golub–Meurant: for f with sign-definite high derivatives (log,
1/x, √, exp — the money functions), the Gauss quadrature of a Lanczos
tridiagonal is one-sided, and the Gauss–Radau rule with a prescribed endpoint
is one-sided the other way.  Together: a rigorous bracket.

  ACT 1  GP POSTERIOR VARIANCE, FULLY CERTIFIED.  σ²(x*) = k** − kᵀ(K+σ²I)⁻¹k
         is a QUADRATIC FORM — no stochastic probes anywhere — so the bracket
         is a true certificate: the dense answer PROVABLY lies inside.
         `resona.quadform(mv, "inv", k_vec, certified=True, support=(σ², None))`
         — note the support endpoint is known STRUCTURALLY (the regularizer),
         not estimated: the certificate's condition is exact.

  ACT 2  LOG-DET WITH THE TWO ERROR SOURCES SEPARATED.  Tr log(K) via SLQ has
         (i) a k-truncation error — CERTIFIED by the bracket below, and
         (ii) Monte-Carlo probe scatter — reported by `with_err`.
         The bracket collapses exponentially in k (printed); the scatter
         shrinks as 1/√probes.  Honesty = knowing which error you still have.

Run:  python3 examples/certified_logdet.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import resona
from resona.spectral import quadform

rng = np.random.default_rng(0)


def rbf_kernel(X, ell=0.7):
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * ell ** 2))


if __name__ == "__main__":
    print("=" * 74)
    print("CERTIFIED BRACKETS — Gauss–Radau: the answer is PROVABLY inside")
    print("=" * 74)

    # ── ACT 1: GP posterior variance, a true certificate ─────────────────────
    N, sig2 = 800, 0.1
    X = rng.uniform(-3, 3, (N, 2))
    K = rbf_kernel(X) + sig2 * np.eye(N)
    xstar = np.array([[0.3, -0.7]])
    kvec = rbf_kernel(np.vstack([X, xstar]))[:N, -1]
    mv = lambda v: K @ v

    exact_q = float(kvec @ np.linalg.solve(K, kvec))
    print(f"\n  ACT 1 — GP posterior variance at x*: σ² = k** − kᵀ(K+σ²I)⁻¹k")
    print(f"  (quadratic form, zero stochastics; support a = σ² = {sig2} is known")
    print(f"   structurally — the certificate is unconditional)\n")
    print(f"    {'k':>4} {'certified bracket [lo, hi]':>38} {'width':>10}  dense inside?")
    for k in (8, 12, 16, 24):
        lo, hi = quadform(mv, "inv", kvec, k=k, certified=True, support=(sig2, None))
        inside = "YES" if lo <= exact_q <= hi else "NO !!"
        print(f"    {k:>4} [{lo:.12f}, {hi:.12f}] {hi - lo:10.2e}  {inside}")
    print(f"    dense truth: {exact_q:.12f} — inside every bracket; width falls")
    print(f"    exponentially in k.  GP uncertainty with a PROOF, matrix-free.")

    # ── ACT 2: log-det — certify the truncation, report the scatter ──────────
    truth = float(np.linalg.slogdet(K)[1])
    print(f"\n  ACT 2 — log|K| (dense truth {truth:.4f}): two error sources, separated")
    print(f"\n    {'k':>4} {'quadrature bracket':>26} {'width':>9}   {'± MC scatter':>12}")
    for k in (8, 16, 32):
        s = resona.of(mv, N, k=k, probes=16)
        lo, hi = s.trace_certified("log", support=(sig2, None))
        _, se = s.trace(np.log, with_err=True)
        print(f"    {k:>4} [{lo:10.4f}, {hi:10.4f}] {hi - lo:9.2e}   {se:12.3f}")
    print(f"    → the k-truncation is CERTIFIED (bracket → 0 exponentially);")
    print(f"      what remains is the probe scatter — honest, separate, 1/√probes.")

    print("\n" + "=" * 74)
    print("  Estimates come with error bars; brackets come with theorems.  resona")
    print("  now hands out both — and tells you which error source each one covers.")
    print("=" * 74)
