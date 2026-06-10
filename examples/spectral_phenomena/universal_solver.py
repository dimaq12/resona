"""
UNIVERSAL SOLVER — precompute once, harvest instantly.

WHAT:  Solve  (A + σ I) u = b  for thousands of right-hand-sides b and
       parameter values σ WITHOUT re-factoring A each time.
       Strategy A (oracle baseline): eigendecompose A once → O(N) harvest per
       solve.  Strategy B (resona.apply): matrix-free Krylov per RHS, no matrix
       ever formed.  Both are demonstrated and compared to naive per-problem
       np.solve.

WHY:   In the FA program, the "harvest principle" says: precompute a spectral
       response field ONCE from the operator's action, then read off any
       functional — solutions, moments, energies — without re-running the
       solver.  One Krylov run per RHS (resona.apply) OR one eigendecomposition
       (dense, oracle) both embody this idea; the resona path is matrix-free
       and applies to operators too large to store.

RESONA's role:
       (1)  resona.of(matvec, N)  — probe the operator's spectrum in one
            matrix-free pass: effective_rank, moments, extreme eigenvalues.
       (2)  resona.apply(matvec, f=lambda lam: 1/(lam+sigma), b, hermitian=True)
            — resolvent action (A+σI)⁻¹b for a given b, matrix-free, k Krylov steps.

Honest caveat:  The oracle harvest (eigenbasis precomputed) is exact to
machine precision but costs O(N³) upfront and requires storing V.  The
resona.apply path is O(Nk) per RHS (no stored matrix) and is accurate to
Krylov precision.  For moderate-N well-conditioned SPD operators (as here),
both paths achieve 14+ digit accuracy.

Run:   python3 examples/spectral_phenomena/universal_solver.py

Note on resona.defect (Richardson/defect calculus):
  This file does not hand-roll a Richardson or defect step.  The two harvest
  paths — (1) eigenbasis precompute + O(N) per-solve, (2) resona.apply Krylov
  per RHS — both achieve machine precision directly.  Richardson extrapolation
  (resona.defect.richardson) is for the pattern P_n → P_{2n} at two spatial
  resolutions; the resolvent harvest here uses a single precomputed eigenbasis
  and a single Krylov run, not a resolution doubling.  No refactoring needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import time
import resona

rng = np.random.default_rng(42)

# ── operator: random SPD covariance matrix, well-conditioned ────────────────
# We use a random Wishart-type SPD so resona.apply (Lanczos/Krylov) converges
# rapidly (smooth spectrum, low condition number).
N = 256
raw = rng.standard_normal((N, N))
A_dense = raw @ raw.T / N + 0.5 * np.eye(N)   # condition ~ 10

def matvec(v):
    return A_dense @ v

# ── PRECOMPUTE: full eigendecomposition (oracle baseline) ───────────────────
t_pre = time.perf_counter()
lam_true, V = np.linalg.eigh(A_dense)
t_pre = time.perf_counter() - t_pre


def harvest_solve(b, sigma):
    """O(N) solve from precomputed eigenbasis — the 'harvest' step."""
    c = V.T @ b
    c /= (lam_true + sigma)
    return V @ c


def naive_solve(b, sigma):
    """Baseline: O(N³) factorisation per problem."""
    return np.linalg.solve(A_dense + sigma * np.eye(N), b)


# ── spectral probe via resona (matrix-free) ──────────────────────────────────
t_probe = time.perf_counter()
s = resona.of(matvec, N, k=64, probes=10)
t_probe = time.perf_counter() - t_probe

lam_min_r, lam_max_r = s.extreme()
er = s.effective_rank()
tr1 = s.moment(1)   # Tr(A)

# ── generate test problems ────────────────────────────────────────────────────
n_harvest = 10_000
sigmas = rng.uniform(0.05, 0.50, n_harvest)   # small positive shifts
bs = rng.standard_normal((n_harvest, N))

# ── timed harvest run ────────────────────────────────────────────────────────
t_harvest = time.perf_counter()
sols_harvest = np.array([harvest_solve(bs[i], sigmas[i]) for i in range(n_harvest)])
t_harvest = time.perf_counter() - t_harvest

# ── naive solve on a subsample ───────────────────────────────────────────────
n_check = 100
t_naive = time.perf_counter()
sols_naive = np.array([naive_solve(bs[i], sigmas[i]) for i in range(n_check)])
t_naive = time.perf_counter() - t_naive

# ── residuals ────────────────────────────────────────────────────────────────
rel_errors = np.array([
    np.linalg.norm(sols_harvest[i] - sols_naive[i]) /
    (np.linalg.norm(sols_naive[i]) + 1e-300)
    for i in range(n_check)
])
speedup_harvest = (t_naive / n_check * n_harvest) / max(t_harvest, 1e-15)

# ── resona.apply resolvent (matrix-free, no eigenvectors stored) ─────────────
k_krylov = 80
resona_errs = []
t_resona_start = time.perf_counter()
for i in range(50):
    sig_i = float(sigmas[i])
    r = resona.apply(matvec, lambda lam, s=sig_i: 1.0 / (lam + s),
                     bs[i], k=k_krylov, hermitian=True)
    ref = sols_naive[min(i, n_check - 1)]
    resona_errs.append(
        np.linalg.norm(r - harvest_solve(bs[i], sig_i)) /
        (np.linalg.norm(harvest_solve(bs[i], sig_i)) + 1e-300)
    )
t_resona_50 = time.perf_counter() - t_resona_start

# ── report ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("UNIVERSAL SOLVER — one precompute, then harvest instantly")
    print("=" * 70)
    print(f"\n  Operator: random SPD N={N}  λ∈[{lam_true[0]:.3f}, {lam_true[-1]:.3f}]"
          f"  cond≈{lam_true[-1]/lam_true[0]:.1f}")

    print(f"\n  PRECOMPUTE (paid once):")
    print(f"    Eigendecomposition:         {t_pre*1000:.1f} ms   (dense, O(N³))")
    print(f"    resona matrix-free probe:   {t_probe*1000:.1f} ms   (O(probes·k·N))")
    print(f"    resona λ range:   [{lam_min_r:.3f}, {lam_max_r:.3f}]  (exact: [{lam_true[0]:.3f}, {lam_true[-1]:.3f}])")
    print(f"    resona Tr(A):     {tr1:.3f}  (exact: {np.trace(A_dense):.3f})")
    print(f"    resona eff.rank:  {er:.1f}  of {N} modes")

    print(f"\n  HARVEST: {n_harvest:,} solves (A+σI)u=b, varying b and σ:")
    print(f"    Harvest  ({n_harvest:,} solves): {t_harvest*1000:.1f} ms  "
          f"({t_harvest/n_harvest*1e6:.2f} µs/solve)")
    print(f"    Naive    ({n_check} solves):  {t_naive*1000:.1f} ms  "
          f"({t_naive/n_check*1e3:.2f} ms/solve)")
    print(f"    Projected speedup ({n_harvest:,} solves):  {speedup_harvest:.0f}x")
    print(f"    Harvest residual — median:   {np.median(rel_errors):.2e}")
    print(f"    Harvest residual — max:      {np.max(rel_errors):.2e}")
    frac = np.mean(rel_errors < 1e-10) * 100
    print(f"    Fraction with err < 1e-10:   {frac:.0f}%")

    print(f"\n  resona.apply matrix-free resolvent (50 random pairs, k={k_krylov}):")
    print(f"    Time for 50 solves:          {t_resona_50*1000:.1f} ms  "
          f"({t_resona_50/50*1e3:.2f} ms/solve)")
    print(f"    Relative error — median:     {np.median(resona_errs):.2e}")
    print(f"    Relative error — max:        {np.max(resona_errs):.2e}")

    print("\n" + "=" * 70)
    print(f"  Precompute A's eigenbasis ONCE ({t_pre*1000:.0f} ms).")
    print(f"  {n_harvest:,} resolvent solves run at {t_harvest/n_harvest*1e6:.1f} µs each")
    print(f"  — {speedup_harvest:.0f}x faster than re-solving; residual {np.median(rel_errors):.1e} (machine prec.).")
    print(f"  resona.apply gives matrix-free resolvent without forming or storing")
    print(f"  eigenvectors: each RHS costs O(N·k) matvecs, error {np.median(resona_errs):.1e}.")
    print(f"  Trade-off: oracle harvest is faster (µs), resona.apply is memory-free.")
    print("=" * 70)
