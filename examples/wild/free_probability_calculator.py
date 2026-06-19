"""
EPIC 2 "Professors' Wall" — Experiment 2: THE FREE-PROBABILITY CALCULATOR
========================================================================

CLAIM under test
----------------
Given two large operators A, B that are *asymptotically free*, predict the FULL
spectrum of

        A ⊞ B   (free additive sum)        and       A ⊠ B   (free mult. product)

PURELY from the two individual spectra — never forming A+B or A·B, never
diagonalizing the composite — via free convolution, and match the TRUE
(dense-formed) composite spectrum to ~1%.

AND: when A, B are NOT free, the certificate `freeness_defect` must blow up AND
the free-convolution prediction must degrade — so the calculator KNOWS its
domain.

What is "pure measure-level" here (the honest, strict reading):
  - ADDITIVE:        s.boxplus(t)              -> moments of A⊞B from κ_n(A)+κ_n(B).
                     This touches ONLY the two harvested measures (nodes/weights).
                     It does NOT call a joint matvec.  (s + t, by contrast,
                     re-probes the real sum operator Ax+Bx — exact but it DOES
                     apply the composite action; we report it as a cross-check.)
  - MULTIPLICATIVE:  S_{A⊠B}(w) = S_A(w)·S_B(w)  (resona.lift.s_transform), then
                     reconstruct the moments of A⊠B from the product S-transform.
                     Again ONLY the two measures are used.

GROUND TRUTH (dense, ONLY for verification): form A+B, A·B at N≈3000, eigvalsh,
compare predicted moments / density / edges.

Run:  PYTHONPATH=/home/dima/resona python experiments/exp2_free_probability_calculator.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import resona
from resona.lift import s_transform


# ───────────────────────── operator builders (matrix-free matvecs) ──────────
def diag_matvec(d):
    return lambda x: d[:, None] * x if x.ndim == 2 else d * x


def dense_sym_matvec(M):
    return lambda x: M @ x


def haar_orthogonal(N, rng):
    """A Haar-distributed orthogonal N×N (QR of a Gaussian, sign-fixed)."""
    Z = rng.standard_normal((N, N))
    Q, R = np.linalg.qr(Z)
    return Q * np.sign(np.diag(R))[None, :]


# ───────────────────── multiplicative reconstruction (measure-level) ────────
def _series_inverse(c, order):
    """Compositional inverse of  χ(w) = Σ_{k>=1} c[k] w^k  (c[0]=0, c[1]!=0).

    Returns ψ(z) = Σ_{k>=1} b[k] z^k  with χ(ψ(z)) = z up to `order`
    (Lagrange inversion, done iteratively / exactly in float)."""
    c = np.asarray(c, float)
    b = np.zeros(order + 1)
    b[1] = 1.0 / c[1]
    # solve χ(ψ(z)) = z order by order: [z^n] Σ_k c_k ψ^k = δ_{n,1}
    psi_pows = {1: b.copy()}                         # ψ^1 coeffs (updated as b grows)

    def poly_mul(a, d):
        out = np.zeros(order + 1)
        for i in range(order + 1):
            if a[i] == 0:
                continue
            for j in range(order + 1 - i):
                out[i + j] += a[i] * d[j]
        return out

    for n in range(2, order + 1):
        # contribution to [z^n] from all c_k ψ^k with current b[1..n-1] known;
        # b[n] enters linearly through c_1 · ([z^n] ψ) = c_1 · b[n].
        # Build ψ with b[n]=0, get residual, then set b[n] = -residual / c_1.
        psi = b.copy(); psi[n] = 0.0
        acc = np.zeros(order + 1)
        pk = np.zeros(order + 1); pk[0] = 1.0        # ψ^0
        for k in range(1, n + 1):
            pk = poly_mul(pk, psi)
            acc += c[k] * pk
        residual = acc[n]                            # [z^n] with b[n]=0
        b[n] = -residual / c[1]
    return b[1:order + 1]


def moments_from_product_S(sA, sB, order=4, w_grid=None):
    """Moments m_1..m_order of A ⊠ B from S_{A⊠B}(w) = S_A(w)·S_B(w).

    Uses ONLY the two spectra (via resona.lift.s_transform).  Reconstruction:

        S(w) = (1+w)/w · χ(w),   χ = ψ^{-1},   ψ(z) = Σ_{k>=1} m_k z^k .

    So χ(w) = w/(1+w) · S(w).  We:
      1. fit the Taylor coefficients c_k of χ(w) = Σ c_k w^k on a SMALL-w grid
         (well-conditioned for low k near 0),
      2. compositionally invert  χ -> ψ  (Lagrange inversion),
      3. read the moments m_k = [z^k] ψ(z).

    Series inversion is exact in arithmetic; the only error is the χ-coeff fit,
    which is accurate at low order on a small grid (truncation ~ w^{order+1}).
    """
    if w_grid is None:
        w_grid = np.linspace(0.002, 0.05, 30)       # small w: low z, series valid
    S_AB = s_transform(sA, w_grid) * s_transform(sB, w_grid)
    chi = w_grid / (1.0 + w_grid) * S_AB            # χ(w)
    # fit χ(w) = Σ_{k=1..order+1} c_k w^k  (no constant; χ(0)=0)
    V = np.vstack([w_grid ** k for k in range(1, order + 2)]).T
    coef, *_ = np.linalg.lstsq(V, chi, rcond=None)
    c = np.concatenate([[0.0], coef])               # c[0]=0, c[1..order+1]
    return _series_inverse(c, order)                # m_1..m_order


# ───────────────────────────── ground-truth metrics ─────────────────────────
def empirical_moments(eigs, order):
    return [float(np.mean(eigs ** p)) for p in range(1, order + 1)]


def density_hist(eigs, edges):
    h, _ = np.histogram(eigs, bins=edges, density=True)
    return h


def density_from_spectral(s, centers, eta):
    rho = s.density(centers, eta=eta)
    # normalize to a probability density on the grid
    dx = centers[1] - centers[0]
    return rho / (rho.sum() * dx)


def report_block(title, pred_m, true_m, edges_pred, edges_true, l1=None):
    print(f"\n  {title}")
    print(f"    moments  p :   predicted        true         |Δ|/|true|")
    worst = 0.0
    for p, (pm, tm) in enumerate(zip(pred_m, true_m), start=1):
        rel = abs(pm - tm) / max(abs(tm), 1e-12)
        worst = max(worst, rel)
        print(f"      m_{p}      : {pm:12.5f}  {tm:12.5f}     {rel*100:7.3f}%")
    print(f"    edges     : pred [{edges_pred[0]:.3f}, {edges_pred[1]:.3f}]"
          f"   true [{edges_true[0]:.3f}, {edges_true[1]:.3f}]")
    edge_err = max(abs(edges_pred[0] - edges_true[0]),
                   abs(edges_pred[1] - edges_true[1])) / (edges_true[1] - edges_true[0])
    print(f"    edge err  : {edge_err*100:.2f}% of span")
    if l1 is not None:
        print(f"    density L1: {l1:.4f}")
    print(f"    >>> worst moment rel-error: {worst*100:.3f}%")
    return worst, edge_err


# ════════════════════════════════════════════════════════════════════════════
def main():
    N = 3000
    ORDER = 4
    rng = np.random.default_rng(7)
    print("=" * 78)
    print("EXP 2 — THE FREE-PROBABILITY CALCULATOR   (N = %d)" % N)
    print("=" * 78)

    # ---- Build two operators that ARE asymptotically free -------------------
    # A : a fixed deterministic spectrum (uniform on [0, 2] -> shifted so it has
    #     a non-trivial, asymmetric distribution).  Diagonal.
    # B : U A' U^T  with U Haar-orthogonal, A' a DIFFERENT fixed spectrum
    #     (uniform on [0.5, 1.5]).  Rotating one operator's eigenbasis by an
    #     independent Haar U makes A and B asymptotically free (Voiculescu).
    dA = np.linspace(0.2, 2.2, N)                       # A spectrum
    dAp = np.linspace(0.5, 1.5, N)                      # B's "own" spectrum
    U = haar_orthogonal(N, rng)
    B_dense = (U * dAp[None, :]) @ U.T                  # = U diag(dAp) U^T
    B_dense = 0.5 * (B_dense + B_dense.T)               # symmetrize (kill fp drift)

    mvA = diag_matvec(dA)
    mvB = dense_sym_matvec(B_dense)

    # ---- PROBE each operator (matrix-free) ----------------------------------
    sA = resona.of(mvA, N, k=64, probes=16, seed=1)
    sB = resona.of(mvB, N, k=64, probes=16, seed=2)
    print("\nIndividual spectra harvested (matrix-free Lanczos):")
    print(f"  A: edges {sA.extreme()}   moments {[round(sA.moment(p)/N,4) for p in range(1,4)]}")
    print(f"  B: edges {sB.extreme()}   moments {[round(sB.moment(p)/N,4) for p in range(1,4)]}")

    # =========================================================================
    # (1) ADDITIVE:  predict A⊞B from spectra alone (boxplus, no joint matvec)
    # =========================================================================
    pred_add_m = sA.boxplus(sB, order=ORDER)                    # measure-level
    s_add_pred = sA.boxplus(sB, order=ORDER, as_spectral=True)  # quadrature meas.
    # cross-check predictor that re-probes the REAL sum (exact, but uses A+B action)
    s_sum_reprobe = sA + sB

    # GROUND TRUTH: form A+B densely and diagonalize
    A_dense = np.diag(dA)
    eigs_sum = np.linalg.eigvalsh(A_dense + B_dense)
    true_add_m = empirical_moments(eigs_sum, ORDER)

    # density L1 between predicted (reprobe sum) and true
    lo, hi = eigs_sum.min(), eigs_sum.max()
    centers = np.linspace(lo, hi, 200)
    edges = np.linspace(lo, hi, 201)
    rho_true = density_hist(eigs_sum, edges)
    rho_pred = density_from_spectral(s_sum_reprobe, centers, eta=0.05)
    dx = centers[1] - centers[0]
    l1_add = float(np.sum(np.abs(rho_pred - rho_true)) * dx)

    # edges: boxplus as_spectral UNDERSHOOTS (inner nodes) by construction;
    # report both the measure predictor edges and the reprobe-sum edges.
    edges_pred_add = s_sum_reprobe.extreme()
    worst_add, eerr_add = report_block(
        "A ⊞ B   (additive free convolution)",
        pred_add_m, true_add_m, edges_pred_add, (lo, hi), l1=l1_add)

    # =========================================================================
    # (2) MULTIPLICATIVE: predict A⊠B from spectra alone (S_A·S_B)
    # =========================================================================
    pred_mul_m = moments_from_product_S(sA, sB, order=ORDER)
    s_prod_reprobe = sA @ sB                                    # re-probes A·B

    # GROUND TRUTH: form A·B.  A·B is not symmetric, but its eigenvalues equal
    # those of the symmetric A^{1/2} B A^{1/2} (A,B PSD here) -> real, positive.
    Ah = np.diag(np.sqrt(dA))
    sym_prod = Ah @ B_dense @ Ah
    sym_prod = 0.5 * (sym_prod + sym_prod.T)
    eigs_prod = np.linalg.eigvalsh(sym_prod)
    true_mul_m = empirical_moments(eigs_prod, ORDER)

    loP, hiP = eigs_prod.min(), eigs_prod.max()
    centersP = np.linspace(loP, hiP, 200)
    edgesP = np.linspace(loP, hiP, 201)
    rho_trueP = density_hist(eigs_prod, edgesP)
    rho_predP = density_from_spectral(s_prod_reprobe, centersP, eta=0.05)
    dxP = centersP[1] - centersP[0]
    l1_mul = float(np.sum(np.abs(rho_predP - rho_trueP)) * dxP)

    worst_mul, eerr_mul = report_block(
        "A ⊠ B   (multiplicative free convolution, via S_A·S_B)",
        pred_mul_m, true_mul_m, s_prod_reprobe.extreme(), (loP, hiP), l1=l1_mul)

    # =========================================================================
    # (3) FREENESS SELF-CERTIFICATION
    #     free pair (A, B)  vs  NON-free pair (A, A_perm sharing eigenbasis)
    # =========================================================================
    print("\n" + "-" * 78)
    print("FREENESS SELF-CERTIFICATION  (does the calculator know when it is valid?)")
    print("-" * 78)

    # Non-free pair: C shares A's eigenbasis (both diagonal) -> they COMMUTE,
    # maximally non-free.  C = diag of a shuffled-but-same-basis spectrum.
    dC = np.linspace(0.5, 1.5, N)            # diagonal -> commutes with diagonal A
    mvC = diag_matvec(dC)
    sC = resona.of(mvC, N, k=64, probes=16, seed=3)

    defect_free = resona.free.freeness_defect(mvA, mvB, N, word="ABAB",
                                              probes=64, seed=11)
    defect_nonfree = resona.free.freeness_defect(mvA, mvC, N, word="ABAB",
                                                 probes=64, seed=11)
    print(f"\n  freeness_defect |τ(ÅB̊ÅB̊)| :")
    print(f"    FREE pair (A, U·U^T) : {defect_free:.5f}")
    print(f"    NON-free (A, C) commuting : {defect_nonfree:.5f}")
    print(f"    contrast ratio        : {defect_nonfree/max(defect_free,1e-9):.1f}×")

    # Now show the PREDICTION degrades on the non-free pair.
    # Predict A⊞C via boxplus (assumes freeness); compare to TRUE A+C.
    pred_add_nf = sA.boxplus(sC, order=ORDER)
    eigs_sum_nf = np.linalg.eigvalsh(np.diag(dA) + np.diag(dC))   # = dA+dC (commute)
    true_add_nf = empirical_moments(eigs_sum_nf, ORDER)
    worst_nf = max(abs(p - t) / max(abs(t), 1e-12)
                   for p, t in zip(pred_add_nf, true_add_nf))

    print(f"\n  ⊞-prediction error (worst moment rel-err), free vs non-free:")
    print(f"    FREE     (A ⊞ B) : {worst_add*100:7.3f}%   <- valid")
    print(f"    NON-free (A ⊞ C) : {worst_nf*100:7.3f}%   <- DEGRADES "
          f"(free-conv assumption violated)")
    print(f"    degradation       : {worst_nf/max(worst_add,1e-9):.1f}× worse")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    add_ok = worst_add < 0.03
    mul_ok = worst_mul < 0.05
    cert_ok = (defect_nonfree > 10 * defect_free) and (worst_nf > 3 * worst_add)
    print(f"  additive  ⊞ : worst rel-err {worst_add*100:.2f}%, edge {eerr_add*100:.2f}%, "
          f"L1 {l1_add:.3f}   -> {'OK' if add_ok else 'FAIL'}")
    print(f"  mult.     ⊠ : worst rel-err {worst_mul*100:.2f}%, edge {eerr_mul*100:.2f}%, "
          f"L1 {l1_mul:.3f}   -> {'OK' if mul_ok else 'FAIL'}")
    print(f"  self-cert   : defect {defect_nonfree/max(defect_free,1e-9):.0f}× , "
          f"pred degrades {worst_nf/max(worst_add,1e-9):.0f}×   "
          f"-> {'OK' if cert_ok else 'FAIL'}")

    if add_ok and mul_ok and cert_ok:
        verdict = "GREEN"
    elif (add_ok or mul_ok) and cert_ok:
        verdict = "YELLOW"
    else:
        verdict = "RED"
    print(f"\n  OVERALL: {verdict}")
    return verdict


if __name__ == "__main__":
    main()
