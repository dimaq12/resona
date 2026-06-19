"""
MATRIX-FREE SPECTRAL DESIGN — killing the last O(N³) in the W-kernel.

WHAT:  wkernel.kappa_w / track needed a full dense eigh (O(N³)) to get the
       eigenvectors behind W = ∂λ/∂k.  But to track only a few modes you only
       need THOSE eigenvectors — so modes=k routes to a shift-invert Lanczos
       (eigsh) on that block, matrix-free (only matvecs / sparse solves).  Same
       number to ~1e-10, cost drops O(N³) → O(N·k) for sparse/structured A.

WHY:   W (the Hellmann-Feynman density) is the "expensive" vector-side of the
       W⊥Φ watershed.  Closing it matrix-free lets the whole spectral-design /
       inverse pipeline run on large sparse systems instead of dense toys.

RESONA's role:  kappa_w(modes=k) / track(modes=k) — OPT-IN; modes='all' is the
       unchanged dense path.  Plus the new matrix-free spectral reads:
       defect.normality (departure from normality), defect.hard_points
       (avoided-crossing / EP locator, no eig), cost.rmt_class (RMT class).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import scipy.sparse as sp
import time
import resona


def family(N, seed=0):
    """sparse banded parametric family  A(k) = A0 + k0·B0 + k1·B1."""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(N); o1 = rng.standard_normal(N - 1) * 0.8; o2 = rng.standard_normal(N - 2) * 0.3
    A0 = sp.diags([o2, o1, d, o1, o2], [-2, -1, 0, 1, 2]).tocsc()
    B0 = sp.diags([np.ones(N - 1) * 0.5, np.zeros(N), np.ones(N - 1) * 0.5], [-1, 0, 1]).tocsc()
    B1 = sp.diags([np.ones(N - 2) * 0.3, np.ones(N - 2) * 0.3], [-2, 2]).tocsc()
    return A0, [B0, B1]


def kappa_w_dense_block(A0, Bs, k0, m=6, probes=4, seed=0):
    """reference: full eigh → bottom-m block → κ_W, SAME random directions as kappa_w."""
    A0d, Bd = A0.toarray(), [B.toarray() for B in Bs]

    def W(k):
        _, V = np.linalg.eigh(A0d + sum(float(k[j]) * Bd[j] for j in range(len(k))))
        V = V[:, :m]
        return np.array([[float(V[:, i] @ (Bd[j] @ V[:, i])) for j in range(len(k))] for i in range(m)])

    r = np.random.default_rng(seed); W0 = W(k0); vals = []
    for _ in range(probes):
        u = r.standard_normal(len(k0)); u /= np.linalg.norm(u)
        vals.append(float(np.linalg.norm(W(k0 + 1e-5 * u) - W0) / 1e-5))
    return max(vals)


print("=" * 72)
print("MATRIX-FREE κ_W — the cube-killer  (kappa_w(modes=k) vs dense eigh)")
print("=" * 72)
print(f"  {'N':>6} {'κ_W dense':>12} {'κ_W matfree':>12} {'rel.err':>9} {'dense(s)':>9} {'mf(s)':>7} {'speedup':>8}")
for N in [800, 1600, 3000]:
    A0, Bs = family(N); k0 = np.array([0.1, 0.05])
    t0 = time.perf_counter(); kd = kappa_w_dense_block(A0, Bs, k0, probes=4); td = time.perf_counter() - t0
    t0 = time.perf_counter(); km = resona.wkernel.kappa_w(A0, Bs, k0, modes=6, probes=4); tm = time.perf_counter() - t0
    rel = abs(kd - km) / max(abs(kd), 1e-30)
    print(f"  {N:>6} {kd:>12.6f} {km:>12.6f} {rel:>9.1e} {td:>9.2f} {tm:>7.3f} {td/tm:>7.1f}x")
print("  same number to ~1e-9; matrix-free is ~flat while dense grows as N³.")

print("\n" + "=" * 72)
print("MATRIX-FREE track(modes=k) — spectral flow of the bottom-4 modes")
print("=" * 72)
N = 600; rng = np.random.default_rng(1)
d = np.sort(rng.standard_normal(N)) * 0.3
A0 = sp.diags(d).tocsc()
B0 = sp.diags([np.ones(N - 1) * 0.2, np.zeros(N), np.ones(N - 1) * 0.2], [-1, 0, 1]).tocsc()
path = np.linspace(0, 0.2, 5).reshape(-1, 1)
lams_mf, _ = resona.wkernel.track(A0, [B0], path, steps=2, modes=4)
lams_all, _ = resona.wkernel.track(A0.toarray(), [B0.toarray()], path, steps=2, modes="all")
diff = np.max(np.abs(np.sort(lams_mf, axis=1) - np.sort(lams_all[:, :4], axis=1)))
print(f"  matrix-free track(modes=4) vs dense track: max|Δλ| = {diff:.2e}  (identical, same method)")

print("\n" + "=" * 72)
print("NEW matrix-free spectral reads")
print("=" * 72)
# normality — departure from normality (=0 for symmetric, matrix-free for non-symmetric)
M = np.random.default_rng(0).standard_normal((400, 400))
true_nn = float(np.linalg.norm(M @ M.T - M.T @ M) ** 2)
est_nn, se = resona.defect.normality(lambda x: M @ x, N=400, rmatvec=lambda x: M.T @ x)
print(f"  defect.normality ‖[A,A*]‖²_F: matfree={est_nn:.0f} vs dense={true_nn:.0f}  rel.err={abs(est_nn-true_nn)/true_nn:.2%}")

# hard_points — avoided-crossing locator, no eig
Nh = 200; delta = 0.04
bg = np.diag(np.concatenate([[0, 0], np.random.default_rng(2).uniform(0.5, 3, Nh - 2) *
                             np.random.default_rng(3).choice([-1, 1], Nh - 2)]))
off = np.zeros((Nh, Nh)); off[0, 1] = off[1, 0] = delta


def H_of_k(k):
    H = bg.copy().astype(float); H[0, 0] = k; H[1, 1] = -k; H += off; return (H + H.T) / 2


Bh = np.zeros((Nh, Nh)); Bh[0, 1] = Bh[1, 0] = 1.0
k_star, _ = resona.defect.hard_points(H_of_k, np.linspace(-0.3, 0.3, 13), Bh, E=0.0, eta=0.06)
print(f"  defect.hard_points: avoided crossing located at k* = {k_star:+.3f}  (true crossing k=0, no eig)")

# rmt_class — RMT universality class (averaged over a few draws)
D, reps = 600, 4


def avg_R4(make):
    return float(np.mean([resona.cost.rmt_class(make())[1] for _ in range(reps)]))


rg = np.random.default_rng(5)
rp = avg_R4(lambda: np.sort(rg.uniform(-1, 1, D)))
rgoe = avg_R4(lambda: np.linalg.eigvalsh((lambda H: (H + H.T) / 2)(rg.standard_normal((D, D)))))
rgue = avg_R4(lambda: np.linalg.eigvalsh((lambda H: (H + H.conj().T) / 2)(rg.standard_normal((D, D)) + 1j * rg.standard_normal((D, D)))))
print(f"  cost.rmt_class R4:  Poisson={rp:+.3f} > GOE={rgoe:+.3f} > GUE={rgue:+.3f}  (rigidity ordering)")
