"""
EXP 3 — "Professors' Wall" epic.

  SCULPTING EIGENVALUES: the inverse spectral problem at scale.

Given an ARBITRARY target spectrum, DESIGN the parameters of a large parametric
operator

        A(k) = A0 + Σ_e k_e B_e

so that a chosen set of m of its eigenvalues HITS the target — to high precision.

The lever is the matrix-free Hellmann–Feynman Jacobian:

        W[i,e] = ∂λ_i/∂k_e = v_iᵀ B_e v_i          (resona.wkernel.wkernel)

and its regularized SVD inverse step

        dk = design(W, λ_target − λ_current, reg)   (resona.wkernel.design)

The m tracked eigenpairs come from scipy.sparse.linalg.eigsh on the SPARSE
A(k): O(N·m·nnz) per solve, never the dense O(N³) eigh.  Gauss–Newton iterates
(recompute W at the new k) until the m eigenvalues land on the target.

The family here is a discrete SCHRÖDINGER operator A(k) = T0 + V_harm + diag(k)
(fixed hopping + a fixed confining well + a TUNABLE on-site potential at M sites);
each B_e is a single-site projector, so W[i,e] = v_i[e]² — strictly positive and
well-conditioned, every tracked mode controllable.  (A bare graph Laplacian, the
first family tried, has a structural 0 eigenvalue whose CONSTANT eigenvector
gives ∂λ/∂k ≡ 0 — that mode is uncontrollable and the inverse blows up; the
Schrödinger family removes that pathology.)

HONESTY — SCOPE.  We design the m TRACKED eigenvalues, not all N.  The family
has many more tunable parameters than targets, so the inverse is UNDER-DETERMINED:
infinitely many k hit the target.  `design`'s Tikhonov reg picks the minimum-norm
(smoothed) solution.  The untracked N−m eigenvalues are free to move; we do NOT
claim to control them.  The win is: a matrix-free Jacobian INVERSE that hits a
chosen target spectrum at a scale where the dense Jacobian + dense eigh is O(N³).

Run:  PYTHONPATH=/home/dima/resona python3 experiments/exp3_sculpting_eigenvalues.py

Result side: ONLY resona primitives (wkernel.wkernel / wkernel.design) + sparse
eigsh.  Dense numpy/scipy is used ONLY for ground-truth verification (and only at
small N).
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from resona.wkernel import wkernel, design


# ────────────────────────────────────────────────────────────────────────────
# The SPARSE parametric family:  a 1-D discrete SCHRÖDINGER operator with a
# TUNABLE ON-SITE POTENTIAL.  (The classic discrete inverse-eigenvalue problem.)
#
#   A(k) = T0 + Σ_i k_i B_i ,   B_i = diag(e_i)  (the i-th site projector)
#
#   T0   : FIXED hopping — the discrete Laplacian  tridiag(−1, 2, −1).
#   k_i  : the tunable on-site potential at site i  (M = N parameters).
#
# Why THIS family (not a pure graph Laplacian): a graph Laplacian is forced to
# carry an exact 0 eigenvalue whose eigenvector is CONSTANT, for which every
# Hellmann–Feynman row ∂λ/∂k_e = (v_e − v_{e+1})² is identically 0 — that mode
# is structurally UNCONTROLLABLE and the inverse blows up.  The Schrödinger
# potential has NO such null mode and W[mode,i] = v_mode[i]² > 0 for every site:
# every tracked mode is controllable and the inverse is well-conditioned.
#
# Each B_i is a single-nonzero diagonal projector → A(k) is a sparse tridiagonal,
# assembled in O(N).  The B_i are used UNCHANGED by resona's wkernel (matvecs).
# ────────────────────────────────────────────────────────────────────────────
def site_projector(N, i):
    """Sparse single-site diagonal projector B_i = diag(e_i)."""
    return sp.csr_matrix(([1.0], ([i], [i])), shape=(N, N))


# Fixed harmonic confining potential — a discrete quantum oscillator.  Its low
# modes are EQUALLY SPACED (≈ Ω(n+½)) and WELL-GAPPED, so eigsh resolves the
# bottom band fast (no pathological 1e-8 clustering of a bare chain), and the
# modes are localized near the centre (oscillator length ℓ = Ω^{-1/2}).
_OMEGA = 0.02


def background_potential(N):
    c = (N - 1) / 2.0
    x = np.arange(N) - c
    return 0.5 * _OMEGA ** 2 * x ** 2


def build(N, sites, k):
    """Assemble A(k) = T0 + V_harm + Σ_i k_i diag(e_{site_i}) as a tridiagonal.

    T0 = tridiag(−1, 2, −1) (fixed hopping); V_harm = ½Ω²(i−c)² (fixed confining
    well, makes the bottom band well-gapped); the M tunable parameters k put
    on-site potentials at the chosen `sites`.  O(N) per assembly — not a sum of
    M sparse-matrix objects.  The B_i (site projectors) are used UNCHANGED by
    resona's wkernel (matvecs only)."""
    diag = 2.0 + background_potential(N)
    diag[sites] += np.asarray(k, float)
    off = -np.ones(N - 1)
    return sp.diags([off, diag, off], [-1, 0, 1], format="csc")


def tracked_modes_matfree(A, m, sigma):
    """The m bottom eigenpairs of sparse A, MATRIX-FREE (shift-invert Lanczos).

    The bottom band sits just above the constant +2 hopping floor; SHIFT-INVERT
    at `sigma` (just below the baseline band bottom) makes ARPACK converge in a
    handful of iterations even though the band is only ~0.02-gapped — plain 'SA'
    Lanczos crawls on this clustered low band.  sigma stays valid throughout the
    solve because the design only LIFTS the tracked modes (they move up, away
    from sigma).  Returns (vals[m], vecs[N,m]) sorted ascending."""
    vals, vecs = eigsh(A.tocsc(), k=m, sigma=sigma, which="LM")
    o = np.argsort(vals)
    return vals[o], vecs[:, o]


# ────────────────────────────────────────────────────────────────────────────
# Gauss–Newton inverse-spectral solve — ALL matrix-free.
# ────────────────────────────────────────────────────────────────────────────
def sculpt(N, sites, Bs, lam_target, k0, sigma, reg=1e-6, iters=40, tol=1e-12):
    """Drive the m bottom eigenvalues of A(k) onto lam_target.

    Each step:
      1. eigsh → (λ_current, V)         [matrix-free, the m tracked modes]
      2. W = wkernel(V, Bs)             [resona: Hellmann–Feynman Jacobian]
      3. dk = design(W, λ_target − λ_current, reg)   [resona: regularized inverse]
      4. k += dk  (with a back-tracking line search guarding against overshoot —
         W is only the FIRST-ORDER response; for a large first step the linearity
         can over-predict, so accept the largest fraction of dk that decreases
         the residual)
    Returns (k, history, lam) where history[i] = max|λ − λ_target| after step i.
    """
    m = len(lam_target)
    k = np.array(k0, float)
    history = []
    A = build(N, sites, k)
    lam, V = tracked_modes_matfree(A, m, sigma)
    for _ in range(iters):
        resid = lam_target - lam
        rnorm = float(np.max(np.abs(resid)))
        history.append(rnorm)
        if rnorm < tol:
            break
        W = wkernel(V, Bs)               # (m, M) spectral Jacobian — resona
        dk = design(W, resid, reg=reg)   # regularized SVD inverse step — resona
        # back-tracking line search on the matrix-free residual
        step = 1.0
        for _bt in range(30):
            A_try = build(N, sites, k + step * dk)
            lam_t, V_t = tracked_modes_matfree(A_try, m, sigma)
            if np.max(np.abs(lam_target - lam_t)) < rnorm:
                break
            step *= 0.5
        k = k + step * dk
        lam, V = lam_t, V_t
    history.append(float(np.max(np.abs(lam_target - lam))))
    return k, np.array(history), lam


def main():
    rc = 0
    print("=" * 74)
    print("EXP 3 — Sculpting eigenvalues: the inverse spectral problem at scale")
    print("=" * 74)

    m = 6

    # ── 2. THE SPARSE PARAMETRIC FAMILY ──────────────────────────────────────
    N = 100000                           # full operator dimension (eigsolve scale)
    sigma = -0.05                        # shift just below the baseline band bottom
    # The bottom modes of the confining well are localised in a ~210-site window
    # around the centre (oscillator length ℓ = Ω^{-1/2}); the tunable on-site
    # potentials must SIT WHERE THE MODES LIVE to control them, so we place M
    # parameters in that central window.  (Sites elsewhere have ~0 mode overlap →
    # ~0 Hellmann–Feynman row → uncontrollable; honest physics.)
    c = (N - 1) // 2
    half = 130
    sites = np.arange(c - half, c + half + 1)      # central window
    M = len(sites)
    Bs = [site_projector(N, int(i)) for i in sites]
    A0 = build(N, sites, np.zeros(M))   # base discrete Schrödinger op (k=0)
    print(f"\nFAMILY  A(k) = T0 + V_harm + Σ k_i diag(e_i)   (sparse discrete")
    print(f"        Schrödinger; T0 = tridiag(−1,2,−1), V_harm = ½Ω²(i−c)² confines)")
    print(f"   N = {N}   parameters M = {M} on-site potentials in the central window")
    print(f"   (where the bottom modes live)   under-determined: M ≫ m={m}")
    print(f"   A(k) nnz = {A0.nnz} (tridiagonal)   each B_i nnz = 1   → genuinely sparse")

    # ── 1. THE TARGET ────────────────────────────────────────────────────────
    # An ARBITRARY chosen target for the m=6 bottom eigenvalues — a "tune".  The
    # baseline well gives a nearly-uniform ladder; we DESIGN a deliberately
    # NON-UNIFORM spectrum (an ascending melody of chosen gaps) that the bare
    # well does NOT have.
    lam0, _ = tracked_modes_matfree(A0, m, sigma)
    lam_target = lam0[0] + np.cumsum([0.0, 0.030, 0.060, 0.045, 0.090, 0.075])
    print(f"\nTARGET spectrum (m={m} chosen bottom eigenvalues — the 'tune'):")
    print("   ", np.array2string(lam_target, precision=6))
    print(f"\nBaseline (k=0) bottom-{m} eigenvalues:")
    print("   ", np.array2string(lam0, precision=6))
    print(f"   start residual max|λ0 − target| = {np.max(np.abs(lam0 - lam_target)):.4e}")

    # ── 3. GAUSS–NEWTON via resona.wkernel.design (matrix-free) ──────────────
    print("\nGAUSS–NEWTON  (matrix-free shift-invert eigsh + resona wkernel/design):")
    t0 = time.time()
    k, hist, lam_final = sculpt(N, sites, Bs, lam_target, np.zeros(M), sigma,
                                reg=1e-8, iters=60, tol=1e-13)
    t1 = time.time()
    for i, h in enumerate(hist):
        print(f"   step {i:2d}:  max|λ − target| = {h:.3e}")
    print(f"   ({t1 - t0:.1f}s wall, N={N}, M={M})")

    # ── 4a. GROUND TRUTH — achieved vs target ────────────────────────────────
    final_resid = float(np.max(np.abs(lam_final - lam_target)))
    print(f"\nACHIEVED bottom-{m} eigenvalues (matrix-free eigsh on A(k)):")
    print("   ", np.array2string(lam_final, precision=8))
    print(f"FINAL residual  max|λ_achieved − λ_target| = {final_resid:.3e}")

    # ── 4b. GROUND TRUTH — matrix-free tracked modes vs DENSE eigvalsh ────────
    # Cross-check the matrix-free tracked eigenvalues against a DENSE eigh at a
    # SMALL N where O(N³) is affordable.  Same family, same target, same solver.
    print("\n" + "-" * 74)
    print("CROSS-CHECK at small N — matrix-free tracked modes vs dense eigvalsh:")
    Ns = 400
    cs = (Ns - 1) // 2
    sites_s = np.arange(cs - half, cs + half + 1)   # same central window
    Ms = len(sites_s)
    Bss = [site_projector(Ns, int(i)) for i in sites_s]
    A0s = build(Ns, sites_s, np.zeros(Ms))
    lam0s, _ = tracked_modes_matfree(A0s, m, sigma)
    lam_target_s = lam0s[0] + np.cumsum([0.0, 0.030, 0.060, 0.045, 0.090, 0.075])
    ks, hists, lam_s = sculpt(Ns, sites_s, Bss, lam_target_s, np.zeros(Ms), sigma,
                              reg=1e-8, iters=60, tol=1e-13)
    As = build(Ns, sites_s, ks).toarray()        # DENSE — ground truth only
    lam_dense_all = np.linalg.eigvalsh(As)        # O(Ns³) reference
    lam_dense = np.sort(lam_dense_all)[:m]
    matfree_vs_dense = float(np.max(np.abs(lam_s - lam_dense)))
    resid_small = float(np.max(np.abs(lam_s - lam_target_s)))
    print(f"   N = {Ns}  (dense eigvalsh affordable here)")
    print(f"   matrix-free tracked : {np.array2string(lam_s, precision=8)}")
    print(f"   dense eigvalsh      : {np.array2string(lam_dense, precision=8)}")
    print(f"   max|matfree − dense| (tracked block) = {matfree_vs_dense:.3e}")
    print(f"   small-N final residual max|λ − target| = {resid_small:.3e}")

    # ── HONESTY: scope / non-identifiability ─────────────────────────────────
    print("\n" + "-" * 74)
    print("SCOPE (honest):")
    n_untracked_moved = int(np.sum(lam_dense_all[m:] >
                                   np.sort(np.linalg.eigvalsh(A0s.toarray()))[m:] + 1e-9))
    print(f"   We designed the {m} TRACKED eigenvalues, NOT all {Ns}.")
    print(f"   M={Ms} parameters ≫ m={m} targets → the inverse is UNDER-DETERMINED;")
    print(f"   design()'s Tikhonov reg returns the minimum-norm k.  The other")
    print(f"   {Ns - m} eigenvalues are FREE and move uncontrolled (≈{n_untracked_moved} shifted up).")

    # ── VERDICT ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    hit_large = final_resid < 1e-9
    agree = matfree_vs_dense < 1e-7
    if hit_large and agree:
        verdict = "🟢"
    elif final_resid < 1e-5 and matfree_vs_dense < 1e-4:
        verdict = "🟡"
    else:
        verdict = "🔴"
        rc = 1
    print(f"VERDICT {verdict}")
    print(f"   target hit at N={N}: max|λ_achieved − λ_target| = {final_resid:.3e}")
    print(f"   matrix-free vs dense (tracked, N={Ns})        = {matfree_vs_dense:.3e}")
    print(f"   steps to converge (large N)                   = {len(hist) - 1}")
    print(f"   dense full eigh at N={N} would be O(N³) ≈ {N**3:.1e} flops — avoided")
    print("=" * 74)
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(main())
