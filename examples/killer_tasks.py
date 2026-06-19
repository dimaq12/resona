"""
resona — killer tasks, one primitive, matrix-free, with metrics vs ground truth.

Every task below is: give an operator as a matvec → read a spectral functional
(or compose first) → never form the matrix, never call eig.

Run:  python3 examples/killer_tasks.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy import linalg
from resona import Spectral

rng = np.random.default_rng(0)


def hdr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# 1 ── GP log-determinant  (the classic SLQ killer app) ───────────────────────
def task_gp_logdet(N=700):
    hdr("1. Gaussian-process log-det  log|K|  (hyperparameter learning at scale)")
    X = rng.standard_normal((N, 3))
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    K = np.exp(-0.5 * D2) + 1e-2 * np.eye(N)                # RBF + jitter (PD)
    # deflate=64: the RBF kernel's spiked top — its largest 64 eigenpairs become
    # EXACT atoms (Hutch++ at the measure level); SLQ carries only the flat rest
    s = Spectral.of(lambda v: K @ v, N, k=70, probes=24, deflate=64)
    est = s.trace(lambda x: np.log(np.maximum(x, 1e-12)))
    true = np.linalg.slogdet(K)[1]
    print(f"   log|K|   resona = {est:>12.2f}   true = {true:>12.2f}   "
          f"rel.err = {abs(est-true)/abs(true):.2%}   (no Cholesky, matvec only)")
    lo, hi = s.trace_certified("log", support=(1e-2, None))
    print(f"   certified k-truncation bracket [{lo:.2f}, {hi:.2f}] (Gauss-Radau, "
          f"width {hi-lo:.1e}; support = the jitter, known structurally)")


# 2 ── Deep-learning Hessian spectrum  (sharpness, no Hessian formed) ──────────
def task_hessian_spectrum(d=600, n=900):
    hdr("2. Loss-Hessian spectrum  (sharpness & curvature from HVPs, no H formed)")
    A = rng.standard_normal((n, d)) / np.sqrt(n)
    HVP = lambda v: A.T @ (A @ v)                          # Gauss–Newton Hessian-vector product
    s = Spectral.of(HVP, d, k=80, probes=16)
    H = A.T @ A
    top_est = s.extreme()[1]
    top_true = linalg.eigvalsh(H)[-1]
    tr_est, tr_true = s.moment(1), float(np.trace(H))
    print(f"   sharpness (λ_max)  resona = {top_est:>9.4f}   true = {top_true:>9.4f}   "
          f"err = {abs(top_est-top_true)/top_true:.2%}")
    print(f"   total curvature Tr resona = {tr_est:>9.2f}   true = {tr_true:>9.2f}   "
          f"err = {abs(tr_est-tr_true)/tr_true:.2%}")


# 3 ── Spectrum of A+B at scale  (compose without forming or rediagonalizing) ──
def task_compose(N=1500):
    hdr("3. Spectrum of A+B  (composed, matrix-free — Horn's problem in practice)")
    from scipy import sparse
    A0 = sparse.diags([-np.ones(N-1), 2*np.ones(N), -np.ones(N-1)], [-1,0,1]).tocsr()
    wA, wB = 0.7*rng.standard_normal(N), 0.7*rng.standard_normal(N)
    sA = Spectral.of(lambda x: A0@x + wA*x, N)
    sB = Spectral.of(lambda x: A0@x + wB*x, N)
    lo, hi = (sA + sB).extreme()                           # A+B never formed
    # TRUTH: only the max/min eigenvalue of the same operator A+B = 2*A0 + diag(wA+wB)
    # are needed — read them matrix-free with eigsh(k=1) instead of a dense eigvalsh
    # (byte-identical extremes, hundreds of × faster than forming the 1500×1500 matrix).
    from scipy.sparse.linalg import eigsh
    AplusB = (2 * A0 + sparse.diags(wA + wB)).tocsr()
    true_hi = eigsh(AplusB, k=1, which='LA', return_eigenvectors=False)[0]
    true_lo = eigsh(AplusB, k=1, which='SA', return_eigenvectors=False)[0]
    print(f"   eig(A+B) max  resona = {hi:>8.3f}   true = {true_hi:>8.3f}   err = {abs(hi-true_hi):.1e}")
    print(f"   eig(A+B) min  resona = {lo:>8.3f}   true = {true_lo:>8.3f}   err = {abs(lo-true_lo):.1e}")


# 4 ── Deep-net trainability from init  (S-transform / dynamical isometry) ─────
def task_trainability(N=200):
    hdr("4. Deep-net trainability  cond(W_L…W_1)  (predict from init, no fwd/bwd)")
    def cond_product(make_W, L):
        P = np.eye(N)
        for _ in range(L):
            P = make_W() @ P
        s = Spectral.of(lambda v: P.T @ (P @ v), N, k=100, probes=4)
        lo, hi = s.extreme()
        return np.sqrt(max(hi, 0) / max(lo, 1e-30))
    gaussian = lambda: rng.standard_normal((N, N)) / np.sqrt(N)
    orthog = lambda: linalg.qr(rng.standard_normal((N, N)))[0]
    print(f"   {'depth L':>8} {'Gaussian cond':>16} {'Orthogonal cond':>17}")
    for L in [2, 8, 16]:
        print(f"   {L:>8} {cond_product(gaussian, L):>16.2e} {cond_product(orthog, L):>17.3f}")
    print("   Gaussian explodes → untrainable deep;  orthogonal ≈1 → trainable (isometry).")


# 5 ── Effective rank / capacity dial  (structured vs full) ────────────────────
def task_effrank(N=800):
    hdr("5. Effective rank Φ₁  (the cost dial: structured/cheap vs full/frontier)")
    U = rng.standard_normal((N, 8)); low = U @ U.T          # PSD, rank 8
    M = rng.standard_normal((N, N)); full = (M @ M.T) / N    # PSD, full rank
    s_low = Spectral.of(lambda v: low @ v, N, k=60, probes=12)
    s_full = Spectral.of(lambda v: full @ v, N, k=60, probes=12)
    print(f"   low-rank operator   Φ₁ = {s_low.effective_rank():>6.1f}   (≈ 8 → cheap, liftable)")
    print(f"   full operator       Φ₁ = {s_full.effective_rank():>6.1f}   (high → genuine frontier)")


if __name__ == "__main__":
    t0 = time.perf_counter()
    task_gp_logdet()
    task_hessian_spectrum()
    task_compose()
    task_trainability()
    task_effrank()
    print(f"\n{'='*70}\nall killer tasks, one primitive (Spectral), matrix-free, "
          f"{time.perf_counter()-t0:.1f}s\n{'='*70}")
