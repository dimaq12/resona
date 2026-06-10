"""
hardness_exponents.py — is the cost of an answer quantized by singularity TYPE?

The Extraction-Law conjecture (FRONTIER.md §3): the cost of extracting an answer
near the non-removable singular set Σ* scales as dist^{-b}, with b set by the
ANALYTIC TYPE of the singularity.  Two clean, complementary measurements:

  (A) NON-HERMITIAN, exceptional point of order q.  The eigenvalue CONDITION
      number (an algorithm-INDEPENDENT lower bound on extraction cost) diverges as
      dist^{-(1-1/q)} as the EP is approached.  Companion family λ^q = s realizes
      an exact order-q EP at s=0; splitting ~ s^{1/q} confirms the order.

  (B) HERMITIAN, continuum edge.  The ACTUAL Lanczos iteration count to resolve an
      eigenvalue sitting a gap g above a DENSE bulk scales as g^{-1/2} — the
      square-root spectral edge (the q=2 class), a genuine algorithmic cost.

Result: b_q = 1 − 1/q  (q=2→½, 3→⅔, 4→¾, …) for EP conditioning, and ½ for the
Krylov edge cost — the same q=2 exponent by two different mechanisms.

HONEST STATUS.  Neither exponent is new: the EP splitting s^{1/q} and conditioning
s^{-(1-1/q)} are classical perturbation theory (Vishik–Lyusternik, Lidskii; the
basis of EP sensing); the Krylov g^{-1/2} edge rate is Kaniel–Paige–Saad /
potential theory.  The CONTRIBUTION is the unification — one cost exponent set by
the singularity type, spanning Hermitian/non-Hermitian and conditioning/iteration
count, quantized into classes.  What is NOT yet done: the full multi-parameter
Arnold catastrophe stratification (fold vs cusp on one discriminant), and a
rigorous algorithm-independent lower bound.

Run:  python3 theory/hardness_exponents.py
"""
import numpy as np

rng = np.random.default_rng(0)


# ── (A) exceptional point of order q: conditioning exponent ──
def companion_EP(q, s):
    """q×q: superdiagonal ones + s in the corner → char. poly λ^q = s (order-q EP)."""
    H = np.diag(np.ones(q - 1), 1).astype(complex)
    H[q - 1, 0] = s
    return H


def eig_conditioning(H):
    lam, X = np.linalg.eig(H)
    Xi = np.linalg.inv(X)
    kappa = np.array([np.linalg.norm(X[:, i]) * np.linalg.norm(Xi[i, :])
                      for i in range(len(lam))])      # Wilkinson cond. (y_i^H x_i = 1)
    return lam, kappa


# ── (B) Hermitian continuum edge: Lanczos iterations to resolve the edge eigenvalue ──
def lanczos_iters(d, target, eps=1e-8, kmax=6000):
    N = len(d); v = rng.standard_normal(N); v /= np.linalg.norm(v)
    V = np.zeros((N, kmax)); al = np.zeros(kmax); be = np.zeros(kmax)
    q = v.copy(); V[:, 0] = q; qp = np.zeros(N); b = 0.0
    for j in range(kmax):
        w = d * q - b * qp; al[j] = q @ w; w = w - al[j] * q
        w -= V[:, :j + 1] @ (V[:, :j + 1].T @ w)
        m = j + 1
        T = np.diag(al[:m]) + np.diag(be[:m - 1], 1) + np.diag(be[:m - 1], -1)
        if abs(np.linalg.eigvalsh(T)[-1] - target) < eps:
            return m
        if j < kmax - 1:
            b = np.linalg.norm(w)
            if b < 1e-13:
                return m
            be[j] = b; qp, q = q, w / b; V[:, j + 1] = q
    return kmax


if __name__ == "__main__":
    print("=" * 72)
    print("HARDNESS EXPONENTS — is the cost of an answer quantized by singularity type?")
    print("=" * 72)

    print("\n  (A) order-q exceptional point — eigenvalue conditioning (algorithm-free):")
    print(f"      {'q':>3} {'splitting exp':>14} {'(1/q)':>7} {'cond. exp b':>13} {'(1−1/q)':>9}")
    ss = np.geomspace(1e-3, 1e-9, 13)
    for q in (2, 3, 4, 5):
        sp = [np.max(np.abs((l := eig_conditioning(companion_EP(q, s)))[0] - l[0].mean()))
              for s in ss]
        cd = [eig_conditioning(companion_EP(q, s))[1].max() for s in ss]
        a = np.polyfit(np.log(ss), np.log(sp), 1)[0]
        b = -np.polyfit(np.log(ss), np.log(cd), 1)[0]
        print(f"      {q:>3} {a:>14.4f} {1/q:>7.4f} {b:>13.4f} {1-1/q:>9.4f}")

    print("\n  (B) Hermitian continuum edge — actual Lanczos iterations vs gap g:")
    N = 3000
    gaps = np.geomspace(0.2, 2e-3, 8)
    its = np.array([lanczos_iters(np.concatenate([np.linspace(0, 1 - g, N - 1), [1.0]]), 1.0)
                    for g in gaps], float)
    bedge = -np.polyfit(np.log(gaps), np.log(its), 1)[0]
    for g, n in zip(gaps, its):
        print(f"      gap={g:.1e}  iters={int(n)}")
    print(f"      → iters ~ gap^(−{bedge:.3f})   (square-root edge: 1/2)")

    print("\n" + "=" * 72)
    print("  b_q = 1 − 1/q for EP conditioning; 1/2 for the Krylov edge (the q=2 class)")
    print("  — one cost exponent set by the singularity type, two mechanisms.  The")
    print("  exponents are classical; the unification/quantization is the claim.  NOT")
    print("  yet done: multi-parameter catastrophe strata, a rigorous lower bound.")
    print("=" * 72)
