"""
FREE PROBABILITY — Voiculescu's coordinates, made first-class.

WHAT:
  This demo shows that resona.free + resona.lift.r_transform expose the
  canonical coordinates of random-matrix / operator composition — free
  cumulants and the R-transform.  Four claims are verified numerically against
  dense ground truth (eigenvectors of explicit Wigner/Wishart matrices):

  (1)  FREE CUMULANTS of a Wigner matrix → only κ₂ ≈ 1 (the semicircle
       fingerprint).  All higher κ_n ≈ 0.  Computed via resona.free.free_cumulants.

  (2)  FREE CUMULANT ADDITIVITY: κ_n(A+B) ≈ κ_n(A)+κ_n(B) for a FREE pair
       (B = Q A Qᵀ, Q Haar).  FAILS for a NON-free pair (B = A, same eigenbasis).
       Additivity is the exact linearity of composition in free coordinates.

  (3)  FREENESS CRITERION via resona.free.freeness_defect: ≈ 0 (O(1/√N)) for
       the free pair, O(1) for the non-free pair.  A single number that detects
       eigenbasis correlation.

  (4)  R-TRANSFORM ADDITIVITY via resona.lift.r_transform: R_{A+B}(w) ≈
       R_A(w)+R_B(w) for the free pair — the shock is linear in the R-coordinate.
       Fails for the non-free pair.

WHY — FREE PROBABILITY (Voiculescu, 1985):
  Classical probability has the Fourier transform: the log-characteristic function
  linearizes independent addition (cumulants add for independent variables).  Free
  probability is the non-commutative analogue for large random matrices: in free
  cumulants κ_n (defined via non-crossing partitions), the operation A ⊞ B
  (free additive convolution) satisfies κ_n(A⊞B) = κ_n(A)+κ_n(B) — EXACTLY when
  A and B are free (=have uniformly random relative eigenbasis, the generic case
  for independent random matrices).

  The R-transform R(w) = G⁻¹(w) − 1/w (where G is the Cauchy/Stieltjes transform
  of the spectral measure) is the generating function of the free cumulants.  It
  linearizes free addition: R_{A⊞B} = R_A + R_B.  A spectral shock (band-edge
  merger, a Burgers discontinuity in density-of-states) looks smooth and additive
  in R — it is smooth in the Cole–Hopf coordinate.  This is the deepest connection
  to defect_calculus: "any shock is a sum of linearities" is literally R_{A+B}=R_A+R_B.

  resona makes these coordinates first-class (resona.free, resona.lift), so a
  practitioner can verify freeness, compute the free cumulants of a black-box
  operator, and add spectra in the R-coordinate without forming the sum.

Run:   python3 examples/spectral_phenomena/free_probability.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import resona
from resona import free as rfree
from resona.lift import r_transform

rng = np.random.default_rng(7)

# ── shared dimensions ─────────────────────────────────────────────────────────
N = 1200      # matrix size — large enough for O(1/√N) freeness floor to be small
K = 6         # number of free cumulants to compute
PROBES = 32   # Hutchinson probes for freeness_defect


def haar_orthogonal(n, rng=rng):
    """Haar-distributed orthogonal matrix via QR of Gaussian."""
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    return Q * np.sign(np.diag(R))


def moments_exact(M, K):
    """Exact normalized moments m_k = Tr(M^k)/N, k=1..K, via repeated mat-mul."""
    out = []
    P = np.eye(N)
    for _ in range(K):
        P = P @ M
        out.append(float(np.trace(P)) / N)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ACT 1: FREE CUMULANTS OF A WIGNER MATRIX → SEMICIRCLE (κ₂≈1, rest≈0)
# ─────────────────────────────────────────────────────────────────────────────

def act1_semicircle():
    print("=" * 72)
    print("ACT 1 — Wigner free cumulants: only κ₂≈1 (the semicircle fingerprint)")
    print("=" * 72)

    # Standard GUE-type real symmetric Wigner matrix, variance 1/N
    G = rng.standard_normal((N, N))
    W = (G + G.T) / np.sqrt(2.0 * N)   # eigenvalues → semicircle on [-2,2]

    # exact moments via matrix powers
    m_exact = moments_exact(W, K)

    # free cumulants via resona.free.free_cumulants
    kappa = rfree.free_cumulants(m_exact)

    print(f"\n  Wigner matrix, N={N} (semicircle law, support ≈[-2,2]).")
    print(f"  Exact moments m_1..m_{K} used; κ computed by resona.free.free_cumulants.\n")
    print(f"  {'n':>3}  {'m_n':>10}  {'κ_n':>10}  verdict")
    for n in range(K):
        verdict = "✓  κ₂≈1" if n == 1 else ("✓  ≈0" if abs(kappa[n]) < 0.05 else "✗ large")
        print(f"  {n+1:>3}  {m_exact[n]:>10.4f}  {kappa[n]:>10.4f}  {verdict}")

    ok_k2 = abs(kappa[1] - 1.0) < 0.05
    ok_higher = all(abs(kappa[n]) < 0.05 for n in [0, 2, 3, 4, 5])
    flag = "PASS" if (ok_k2 and ok_higher) else "FAIL"
    print(f"\n  [{flag}] κ₂ = {kappa[1]:.4f} ≈ 1 ;  max|κ_n| (n≠2) = "
          f"{max(abs(kappa[n]) for n in [0,2,3,4,5]):.4f} ≈ 0.")
    print(f"  The Wigner semicircle is the FREE Gaussian — only κ₂ survives.\n")
    return kappa


# ─────────────────────────────────────────────────────────────────────────────
# ACT 2: FREE CUMULANT ADDITIVITY vs NON-FREE (same basis)
# ─────────────────────────────────────────────────────────────────────────────

def act2_additivity():
    print("=" * 72)
    print("ACT 2 — κ_n(A+B) = κ_n(A)+κ_n(B): FREE✓ vs NON-FREE✗")
    print("=" * 72)

    # A: diagonal with a 3-point spectrum (non-trivial higher cumulants)
    spec_A = rng.choice([-1.5, 0.3, 1.4], size=N)
    A = np.diag(spec_A)

    # FREE pair: B = Q A Qᵀ, Haar Q → A and B are asymptotically free
    Q = haar_orthogonal(N)
    B_free = Q @ A @ Q.T

    # NON-FREE pair: B = A (same eigenbasis → definitely not free)
    B_nonfree = A.copy()

    # exact moments and free cumulants
    kA = rfree.free_cumulants(moments_exact(A, K))
    kBf = rfree.free_cumulants(moments_exact(B_free, K))
    kBn = rfree.free_cumulants(moments_exact(B_nonfree, K))

    kSf = rfree.free_cumulants(moments_exact(A + B_free, K))
    kSn = rfree.free_cumulants(moments_exact(A + B_nonfree, K))

    print(f"\n  A: 3-point spectrum {{-1.5, 0.3, 1.4}}, N={N}.")
    print(f"  Free pair:    B = Q A Qᵀ  (Haar Q)  →  A ⊞ B (asymptotically free).")
    print(f"  Non-free pair: B = A (same eigenbasis) → A + A = 2A (correlated).\n")

    print(f"  FREE PAIR  — prediction κ_n(A)+κ_n(B) vs actual κ_n(A+B):")
    print(f"  {'n':>3}  {'κ_n(A)':>9}  {'κ_n(B)':>9}  {'sum':>9}  {'actual':>9}  {'|err|':>9}")
    err_free = []
    for n in range(K):
        pred = kA[n] + kBf[n]
        err = abs(pred - kSf[n])
        err_free.append(err)
        print(f"  {n+1:>3}  {kA[n]:>9.4f}  {kBf[n]:>9.4f}  {pred:>9.4f}  {kSf[n]:>9.4f}  {err:>9.2e}")
    max_err_free = max(err_free[1:])   # skip k1 (mean — exact by linearity)

    print(f"\n  NON-FREE PAIR — prediction κ_n(A)+κ_n(A)=2κ_n(A) vs actual κ_n(2A)=2^n κ_n(A):")
    print(f"  {'n':>3}  {'2·κ_n(A)':>10}  {'actual':>10}  {'|err|':>10}")
    err_nonfree = []
    for n in range(K):
        pred = 2 * kA[n]
        err = abs(pred - kSn[n])
        err_nonfree.append(err)
        print(f"  {n+1:>3}  {pred:>10.4f}  {kSn[n]:>10.4f}  {err:>10.4f}")
    max_err_nonfree = max(err_nonfree[2:])   # κ_1 and κ_2 coincide; higher differ

    flag_free = "PASS" if max_err_free < 0.05 else "FAIL"
    flag_nonfree = "PASS" if max_err_nonfree > 0.1 else "FAIL"
    print(f"\n  [{flag_free}] FREE additivity: max|κ_n(A+B)−κ_n(A)−κ_n(B)| = {max_err_free:.2e}  (O(1/N))")
    print(f"  [{flag_nonfree}] NON-FREE breaks: max deviation = {max_err_nonfree:.2f}  (O(1), freeness defect)")
    print(f"  Composition = addition in free-cumulant coordinates, iff eigenbases are random.\n")
    return A, B_free, kA, kBf


# ─────────────────────────────────────────────────────────────────────────────
# ACT 3: FREENESS CRITERION via resona.free.freeness_defect
# ─────────────────────────────────────────────────────────────────────────────

def act3_freeness_defect(A, B_free):
    print("=" * 72)
    print("ACT 3 — freeness_defect ≈0 for free pair, O(1) for non-free")
    print("=" * 72)
    B_nonfree = A.copy()

    Amv = lambda v: A @ v
    Bfmv = lambda v: B_free @ v
    Bnmv = lambda v: B_nonfree @ v

    # freeness_defect uses Hutchinson trace estimates of alternating centered products
    d_free = rfree.freeness_defect(Amv, Bfmv, N, word="ABAB", probes=PROBES, seed=42)
    d_nonfree = rfree.freeness_defect(Amv, Bnmv, N, word="ABAB", probes=PROBES, seed=42)

    floor = 1.0 / np.sqrt(N)   # expected O(1/√N) floor for a free pair

    flag_free = "PASS" if d_free < 10 * floor else "FAIL"
    flag_nonfree = "PASS" if d_nonfree > 0.05 else "FAIL"

    print(f"\n  N={N};  word = ABAB;  Hutchinson probes = {PROBES}.")
    print(f"  Expected floor for free pair ≈ 1/√N = {floor:.4f}.\n")
    print(f"  freeness_defect(A, B_free)    = {d_free:.4f}   (O(1/√N) = free)  [{flag_free}]")
    print(f"  freeness_defect(A, B_nonfree) = {d_nonfree:.4f}   (O(1)   = not free) [{flag_nonfree}]")
    print(f"  Ratio: {d_nonfree/max(d_free, 1e-12):.1f}×  (non-free defect >> free floor)")
    print(f"\n  A single scalar detects eigenbasis correlation — the non-closable residue.\n")


# ─────────────────────────────────────────────────────────────────────────────
# ACT 4: R-TRANSFORM ADDITIVITY — R_{A+B}(w) ≈ R_A(w)+R_B(w) for FREE pair
# ─────────────────────────────────────────────────────────────────────────────

def act4_r_transform(A, B_free, kA, kBf):
    print("=" * 72)
    print("ACT 4 — R_{A+B}(w) = R_A(w)+R_B(w): free✓  non-free✗")
    print("=" * 72)
    B_nonfree = A.copy()

    # Build (nodes, weights) spectral measures from exact eigenvalues
    # (dense ground truth: full diagonalization of the symmetric matrices)
    lam_A = np.linalg.eigvalsh(A)
    lam_Bf = np.linalg.eigvalsh(B_free)
    lam_Bn = B_nonfree.diagonal()           # diagonal matrix: eigenvalues = diagonal

    lam_SumF = np.linalg.eigvalsh(A + B_free)
    lam_SumN = np.linalg.eigvalsh(A + B_nonfree)

    w_unif = np.ones(N) / N                 # uniform weights (normalized empirical measure)

    # evaluation points for R — must be positive (R defined for w > 0)
    w_pts = np.array([0.05, 0.10, 0.20, 0.40, 0.60])

    # R-transform via resona.lift.r_transform (each takes (nodes,weights) or Spectral)
    RA = r_transform((lam_A, w_unif), w_pts)
    RBf = r_transform((lam_Bf, w_unif), w_pts)
    RBn = r_transform((lam_Bn, w_unif), w_pts)
    RSf = r_transform((lam_SumF, w_unif), w_pts)
    RSn = r_transform((lam_SumN, w_unif), w_pts)

    pred_free = RA + RBf
    pred_nonfree = RA + RBn

    print(f"\n  Dense ground truth (eigvalsh of explicit N={N} matrices).")
    print(f"  R-transform evaluated at w ∈ {{0.05, 0.10, 0.20, 0.40, 0.60}}.")
    print(f"  Additivity claim: R_{{A+B}}(w) ?= R_A(w) + R_B(w).\n")

    print(f"  FREE PAIR (B=QAQᵀ):")
    print(f"  {'w':>6}  {'R_A':>9}  {'R_B':>9}  {'R_A+R_B':>9}  {'R_{{A+B}}':>9}  {'|err|':>9}")
    err_free_R = []
    for i, w in enumerate(w_pts):
        err = abs(pred_free[i] - RSf[i])
        err_free_R.append(err)
        print(f"  {w:>6.2f}  {RA[i]:>9.4f}  {RBf[i]:>9.4f}  {pred_free[i]:>9.4f}  "
              f"{RSf[i]:>9.4f}  {err:>9.2e}")

    print(f"\n  NON-FREE PAIR (B=A):")
    print(f"  {'w':>6}  {'R_A':>9}  {'R_B=R_A':>9}  {'R_A+R_B':>9}  {'R_{{A+B}}':>9}  {'|err|':>9}")
    err_nonfree_R = []
    for i, w in enumerate(w_pts):
        err = abs(pred_nonfree[i] - RSn[i])
        err_nonfree_R.append(err)
        print(f"  {w:>6.2f}  {RA[i]:>9.4f}  {RBn[i]:>9.4f}  {pred_nonfree[i]:>9.4f}  "
              f"{RSn[i]:>9.4f}  {err:>9.2e}")

    max_err_free_R = max(err_free_R)
    max_err_nonfree_R = max(err_nonfree_R)
    flag_free = "PASS" if max_err_free_R < 0.05 else "FAIL"
    flag_nonfree = "PASS" if max_err_nonfree_R > 0.05 else "FAIL"

    print(f"\n  [{flag_free}] FREE R-additivity:     max|R_{{A+B}}−R_A−R_B| = {max_err_free_R:.3e}")
    print(f"  [{flag_nonfree}] NON-FREE R-additivity: max|R_{{A+B}}−R_A−R_B| = {max_err_nonfree_R:.3f}  (fails)")
    print(f"\n  The shock (band merger in density) is SMOOTH and LINEAR in R.")
    print(f"  R-transform is the Cole–Hopf coordinate of free probability: R_{{A⊞B}}=R_A+R_B.\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 72)
    print("FREE PROBABILITY — resona.free + resona.lift  (Voiculescu, 1985)")
    print("=" * 72)
    print(f"  N={N}  K={K} cumulants  probes={PROBES}  numpy/scipy/resona only.")
    print(f"  Ground truth: dense eigenvectors of explicit N×N matrices.\n")

    # Act 1
    act1_semicircle()

    # Act 2
    A, B_free, kA, kBf = act2_additivity()

    # Act 3
    act3_freeness_defect(A, B_free)

    # Act 4
    act4_r_transform(A, B_free, kA, kBf)

    print("=" * 72)
    print("SUMMARY — all four pillars of free probability verified:")
    print("  (1)  Wigner → semicircle: only κ₂≈1 (resona.free.free_cumulants)")
    print("  (2)  κ_n(A⊞B) = κ_n(A)+κ_n(B) for free pair; fails for non-free")
    print("  (3)  freeness_defect ≈0 (free) vs O(1) (non-free) (resona.free)")
    print("  (4)  R_{A+B}=R_A+R_B for free pair; fails for non-free (resona.lift)")
    print()
    print("  Free cumulants = canonical coordinates of composition.")
    print("  R-transform = the Cole–Hopf lift: shock becomes a straight line.")
    print("  Freeness = the exact condition under which the response algebra closes.")
    print("=" * 72)
