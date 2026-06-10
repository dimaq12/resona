"""
inverse_graph_design.py — choose edge weights so the Laplacian's first K
eigenvalues match a target, using the spectral Jacobian (Hellmann-Feynman).
=============================================================================
WHAT.  Given a graph topology and a target spectrum [λ₁*,...,λ_K*], find edge
weights w such that the weighted Laplacian's first K eigenvalues match the
target.  The key ingredient is the spectral Jacobian:

    dλ_k/dw_e = ⟨φ_k, (dL/dw_e) φ_k⟩

where dL/dw_e is the "edge Laplacian" for edge e (a rank-2 update), and φ_k
is the k-th eigenvector.  This is the Hellmann-Feynman theorem: once you have
the eigenvectors, the gradient is free — no extra eigensolves.

WHY (speedup).  A naive finite-difference Jacobian costs K+1 eigensolves per
gradient step (one for the residual, K for each parameter perturbation in the
general case; or ≥2 per parameter for 2-point FD).  The analytical Jacobian
costs exactly 1 eigensolver call (for the K eigenpairs) plus O(K·E) cheap
inner products.  On a graph with E edges and K modes, this is a speedup of
roughly E/K eigensolves per gradient step — significant when E >> K.

resona's ROLE.  resona.of is called on the final weighted Laplacian to report
the spectral effective_rank and support as a "health check" on the designed
graph — confirming the optimised weights produce a well-conditioned operator
rather than a degenerate one.

Run:  python3 examples/graphs/inverse_graph_design.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import resona
from resona import wkernel as wk  # spectral Jacobian primitive: W[k,e] = v_k^T B_e v_k

rng = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# Graph: path graph (N nodes, N-1 edges)
# ---------------------------------------------------------------------------

def path_laplacian(N: int, weights: np.ndarray) -> sparse.csr_matrix:
    """Weighted Laplacian of the path graph P_N with edge weights `weights`."""
    # edges: (0,1),(1,2),...,(N-2,N-1) — exactly N-1 of them
    assert len(weights) == N - 1
    rows = list(range(N - 1)) + list(range(1, N))
    cols = list(range(1, N)) + list(range(N - 1))
    off = np.concatenate([-weights, -weights])
    L_off = sparse.csr_matrix((off, (rows, cols)), shape=(N, N))
    deg = np.zeros(N)
    for i, w in enumerate(weights):
        deg[i] += w; deg[i + 1] += w
    L = sparse.diags(deg) + L_off
    return L.tocsr()


def edge_laplacians(N: int) -> list:
    """Return the M=N-1 edge Laplacians dL/dw_e (each is a 4-element rank-2 matrix)."""
    edge_laps = []
    for i in range(N - 1):
        # dL/dw_e has entries: (+1 at (i,i), +1 at (i+1,i+1), -1 at (i,i+1), -1 at (i+1,i))
        r = [i, i + 1, i, i + 1]
        c = [i, i + 1, i + 1, i]
        v = [1.0, 1.0, -1.0, -1.0]
        edge_laps.append(sparse.csr_matrix((v, (r, c)), shape=(N, N)))
    return edge_laps


# ---------------------------------------------------------------------------
# Jacobian-based gradient step (Hellmann-Feynman)
# ---------------------------------------------------------------------------

def spectral_jacobian(eigvecs: np.ndarray, edge_laps: list, K: int) -> np.ndarray:
    """J[k, e] = dλ_k/dw_e — routed through resona.wkernel (Hellmann-Feynman)."""
    return wk.wkernel(eigvecs, edge_laps)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

N = 60      # nodes
K = 30      # modes to match (out of N-1=59 non-zero; K/M ≈ 0.51)
M = N - 1   # edges (path graph: exactly N-1)

print("=" * 72)
print(f"INVERSE GRAPH DESIGN — Hellmann-Feynman Jacobian (N={N}, K={K}, M={M})")
print(f"  (K/M = {K/M:.2f}: well-determined when K ≈ M)")
print("=" * 72)

# True weights and target spectrum (larger perturbation so optimisation is non-trivial)
w_true = 1.0 + 0.5 * rng.standard_normal(M)
w_true = np.clip(w_true, 0.05, 3.0)
L_true = path_laplacian(N, w_true)
# For small N use dense eigsh (all non-zero eigenvalues)
lam_target_all = np.linalg.eigvalsh(L_true.toarray())
lam_target_all.sort()
lam_target = lam_target_all[1:K + 1]  # skip λ₁=0 (Laplacian null space)

edge_laps = edge_laplacians(N)

def get_eigpairs(w, k):
    """Compute k eigenpairs of the weighted path Laplacian (dense, small N)."""
    L = path_laplacian(N, w).toarray()
    vals, vecs = np.linalg.eigh(L)
    idx = np.argsort(vals)
    return vals[idx][1:k + 1], vecs[:, idx][:, 1:k + 1]

# ---------- Hellmann-Feynman Gauss-Newton --------------------------------

w_hf = np.ones(M)
HF_STEPS = 20
hf_eig_count = 0
hf_errors = []
t_hf = time.perf_counter()

for step in range(HF_STEPS):
    lam_cur, phi_cur = get_eigpairs(w_hf, K)
    hf_eig_count += 1
    residual = lam_cur - lam_target
    err = float(np.max(np.abs(residual)))
    hf_errors.append(err)

    J = spectral_jacobian(phi_cur, edge_laps, K)      # O(K·M) — one eigsolve
    # Gauss-Newton step: (J^T J + λI) dw = J^T r
    lam_reg = 1e-3 * np.trace(J.T @ J) / M
    dw = np.linalg.solve(J.T @ J + lam_reg * np.eye(M), J.T @ residual)
    w_hf = np.clip(w_hf - dw, 0.05, 5.0)

hf_ms = (time.perf_counter() - t_hf) * 1000

# ---------- Naive finite-difference Gauss-Newton baseline ---------------

w_fd = np.ones(M)
FD_STEPS = 20
EPS = 1e-5
fd_eig_count = 0
fd_errors = []
t_fd = time.perf_counter()

for step in range(FD_STEPS):
    lam_cur, _ = get_eigpairs(w_fd, K)
    fd_eig_count += 1
    residual = lam_cur - lam_target
    err = float(np.max(np.abs(residual)))
    fd_errors.append(err)

    # Finite-difference Jacobian: M additional eigsolves
    J_fd = np.zeros((K, M))
    for e in range(M):
        w_plus = w_fd.copy(); w_plus[e] += EPS
        lam_plus, _ = get_eigpairs(w_plus, K)
        fd_eig_count += 1
        J_fd[:, e] = (lam_plus - lam_cur) / EPS

    lam_reg = 1e-3 * np.trace(J_fd.T @ J_fd) / M
    dw = np.linalg.solve(J_fd.T @ J_fd + lam_reg * np.eye(M), J_fd.T @ residual)
    w_fd = np.clip(w_fd - dw, 0.05, 5.0)

fd_ms = (time.perf_counter() - t_fd) * 1000

# ---------- Results -----------------------------------------------------

hf_final_err = hf_errors[-1]
fd_final_err = fd_errors[-1]
speedup_time = fd_ms / max(hf_ms, 1e-9)
speedup_eigs = fd_eig_count / max(hf_eig_count, 1)

print(f"\n  {'Method':<30} {'Steps':>6} {'Eigsolves':>10} {'Time ms':>9} {'Final err':>12}")
print("  " + "-" * 72)
print(f"  {'HF Jacobian (analytical)':<30} {HF_STEPS:>6} {hf_eig_count:>10} {hf_ms:>9.1f} {hf_final_err:>12.3e}")
print(f"  {'FD Jacobian (naive)':<30} {FD_STEPS:>6} {fd_eig_count:>10} {fd_ms:>9.1f} {fd_final_err:>12.3e}")
print(f"\n  Eigensolve reduction : {speedup_eigs:.1f}x  ({fd_eig_count} vs {hf_eig_count})")
print(f"  Wall-time speedup    : {speedup_time:.1f}x  ({fd_ms:.1f}ms vs {hf_ms:.1f}ms)")
print(f"  Expected reduction   : ≈ M+1 = {M+1}x per step (FD costs M+1 eigsolves; HF costs 1)")

print(f"\n  Convergence (max |Δλ|):")
print(f"    HF  step 1→{HF_STEPS}: {hf_errors[0]:.3e} → {hf_errors[-1]:.3e}")
print(f"    FD  step 1→{FD_STEPS}: {fd_errors[0]:.3e} → {fd_errors[-1]:.3e}")

# ---------- resona health-check on the designed Laplacian ---------------

print(f"\n  resona spectral health-check on the HF-designed Laplacian:")
L_hf = path_laplacian(N, w_hf)
mv_hf = lambda v: L_hf @ v
s_hf = resona.of(mv_hf, N, k=48, probes=12)
eff_r = s_hf.effective_rank()
lo, hi = s_hf.extreme()
print(f"    support=[{lo:.4f}, {hi:.4f}], eff_rank={eff_r:.1f} / {N}")
print(f"    (eff_rank > 1 confirms non-degenerate designed operator)")

print()
print("=" * 72)
print(f"  HF Jacobian: 1 eigsolve/step + O(K·M) dot products = {hf_eig_count} total.")
print(f"  FD Jacobian: {M+1} eigsolves/step = {fd_eig_count} total. Ratio: {speedup_eigs:.0f}x.")
print(f"  Hellmann-Feynman is exact — no finite-difference noise in the gradient.")
print("=" * 72)
