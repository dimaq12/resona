"""
quantum/laptop_quantum.py
==============================================================================
J1-J2 HEISENBERG SPIN CHAIN AT SCALE — ground energy and trace monomials,
no diagonalization, no dense matrix.

THE PROBLEM.  The frustrated J1-J2 Heisenberg chain

    H = J1 Σ Sᵢ·Sᵢ₊₁  +  J2 Σ Sᵢ·Sᵢ₊₂       (J1=1, J2=0.5)

at L=16 has D = 2^16 = 65536 states.  Full diagonalization requires O(D³) ≈
2.8×10¹⁴ floating-point operations and O(D²) ≈ 32 GB memory — infeasible on
any single machine.  The Hilbert-space matrix itself has ~4.3×10⁹ elements.

WHAT WE DO INSTEAD.  We never store H.  A matrix-free matvec applies H to a
vector in O(D·L) time.  resona.of(matvec, N) uses stochastic Lanczos quadrature
(SLQ) to read:
  • E₀  = s.extreme()[0]          ground energy,  O(probes·k·D) matvecs
  • Tr(H^p) = s.moment(p)         spectral monomials, same cost

Calibration note: The ground energy from SLQ/Lanczos is a rigorous UPPER BOUND
(variational principle), and at k=48 probes=8 it is typically within ~1% of the
true E₀ for these system sizes.  The trace monomials Tr(H^p)/D are unbiased
stochastic estimators with statistical noise ∝ 1/√(probes).

WHY IT MATTERS.  At L=16 the dimensional hierarchy flips completely:
  eigh: 2.8×10¹⁴ FLOP, ~32 GB RAM — impossible.
  matvec: O(D·L) ≈ 10⁶ FLOP per apply — laptop in seconds.

Run:  python3 examples/quantum/laptop_quantum.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from scipy import sparse as sp
import resona


def build_j1j2(L, J1=1.0, J2=0.5):
    """Sparse J1-J2 Heisenberg H = J1 Σ Sᵢ·Sᵢ₊₁ + J2 Σ Sᵢ·Sᵢ₊₂ (periodic).

    Sᵢ·Sⱼ = (1/4) ZᵢZⱼ  +  (1/2)(S⁺ᵢS⁻ⱼ + S⁻ᵢS⁺ⱼ)
    ZᵢZⱼ is diagonal; S⁺S⁻ + S⁻S⁺ flips both bits i,j when they differ.
    """
    D = 1 << L
    states = np.arange(D, dtype=np.int64)
    diag = np.zeros(D)
    row_all, col_all, val_all = [], [], []

    for J, dist in [(J1, 1), (J2, 2)]:
        for i in range(L):
            j = (i + dist) % L
            zi = 1 - 2 * ((states >> i) & 1).astype(float)
            zj = 1 - 2 * ((states >> j) & 1).astype(float)
            diag += (J / 4) * zi * zj                        # ZᵢZⱼ diagonal
            # S⁺ᵢS⁻ⱼ + S⁻ᵢS⁺ⱼ: nonzero when bits i,j differ
            diff = ((states >> i) & 1) != ((states >> j) & 1)
            s_diff = states[diff]
            flipped = s_diff ^ ((1 << i) | (1 << j))
            row_all.append(s_diff)
            col_all.append(flipped)
            val_all.append(np.full(diff.sum(), J / 2))

    rows = np.concatenate(row_all)
    cols = np.concatenate(col_all)
    vals = np.concatenate(val_all)
    Hoff = sp.coo_matrix((vals, (rows, cols)), shape=(D, D)).tocsr()
    return (sp.diags(diag) + Hoff).tocsr()


if __name__ == "__main__":
    print("=" * 74)
    print("J1-J2 HEISENBERG SPIN CHAIN — ground energy + trace monomials, no eigh")
    print("=" * 74)
    print("  H = J1 Σ Sᵢ·Sᵢ₊₁ + J2 Σ Sᵢ·Sᵢ₊₂  (J1=1, J2=0.5, periodic)")
    print("  resona.extreme() → E₀ (matrix-free Lanczos upper bound)")
    print("  resona.moment(p) → Tr(H^p) (Hutchinson stochastic estimator)\n")

    # Small-L exact comparison table
    print(f"  {'L':>3}  {'D=2^L':>7}  {'E0/L exact':>12}  {'E0/L resona':>12}  "
          f"{'err%':>6}  {'Tr(H²)/D':>10}  {'resona ms':>10}")
    print("  " + "─" * 70)

    for L in [8, 10, 12]:
        H = build_j1j2(L)
        D = H.shape[0]
        matvec = lambda v, H=H: H @ v

        exact_e0 = np.linalg.eigvalsh(H.toarray())[0]

        t0 = time.perf_counter()
        s = resona.of(matvec, D, k=64, probes=12)
        E0_res = s.extreme()[0]
        m2 = s.moment(2) / D
        t_ms = (time.perf_counter() - t0) * 1e3

        err_pct = 100 * abs(E0_res - exact_e0) / abs(exact_e0)
        print(f"  {L:>3}  {D:>7}  {exact_e0/L:>12.5f}  {E0_res/L:>12.5f}  "
              f"{err_pct:>5.2f}%  {m2:>10.5f}  {t_ms:>9.1f}")

    # Scale-up: L=14 and L=16 — eigh impossible
    print()
    print(f"  {'L':>3}  {'D':>7}  {'eigh FLOP':>14}  {'E0/L resona':>12}  "
          f"{'Tr(H²)/D':>10}  {'Tr(H⁴)/D':>10}  {'time s':>7}")
    print("  " + "─" * 74)

    for L in [14, 16]:
        D = 1 << L
        eigh_flop = D ** 3
        H = build_j1j2(L)
        matvec = lambda v, H=H: H @ v

        t0 = time.perf_counter()
        s = resona.of(matvec, D, k=48, probes=8)
        E0_res = s.extreme()[0]
        m2 = s.moment(2) / D
        m4 = s.moment(4) / D
        dt = time.perf_counter() - t0

        eigh_str = f"{eigh_flop:.1e}"
        print(f"  {L:>3}  {D:>7}  {eigh_str:>14}  {E0_res/L:>12.5f}  "
              f"{m2:>10.5f}  {m4:>10.5f}  {dt:>6.1f}s")

    print()
    print("  Calibration: E0/L resona is a Lanczos upper bound, typically <1% above")
    print("  the true ground energy.  Tr(H^p)/D are unbiased stochastic estimates.")
    print()
    print(f"  L=16: eigh needs ~{(1<<16)**3:.1e} FLOP + ~{(1<<16)**2*8/1e9:.0f} GB RAM — infeasible.")
    print(f"        resona needs O(D·L·k·probes) ≈ {(1<<16)*16*48*8:.2e} FLOP,  O(D) memory.")
    print("=" * 74)
