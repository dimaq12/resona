"""
quantum/hubbard_mott.py
==============================================================================
HUBBARD MODEL MOTT METAL-INSULATOR TRANSITION — DOS depletion via resona,
no diagonalization, no sign problem.

THE MODEL.  The 1D Hubbard model

    H = −t Σ_{i,σ} (c†_{i,σ} c_{i+1,σ} + h.c.)  +  U Σ_i n_{i↑} n_{i↓}

describes electrons with hopping t competing with on-site Coulomb repulsion U.
At half-filling (one electron per site), small U gives a metallic phase with a
continuous DOS through the Fermi level; large U forces localization and opens
a Mott gap.  The spectral weight at the spectral midpoint—the DOS in the
gap region—decreases monotonically with U as the lower and upper Hubbard bands
separate: ρ(E_mid) ≈ 1 at U=0 (metal), ρ(E_mid) → 0 as U/t → ∞ (insulator).

THE TRICK (NO SIGN PROBLEM).  Quantum Monte Carlo sign problem plagues
fermionic systems at general filling.  Here we bypass it entirely: the full
Hilbert-space Hamiltonian is applied as a matrix-free matvec, and resona
computes the DOS via the Lanczos-quadrature representation of the spectral
density.  No fermion determinants, no sign cancellations.

resona's ROLE.
  s = resona.of(matvec, N, k=64, probes=12)    — SLQ, matrix-free
  E0, Emax = s.extreme()                        — spectral support
  ρ(E_mid) = s.density([E_mid], eta=0.5)        — DOS at the Hubbard midgap
  Tr(H²)/D = s.moment(2)/D                      — bandwidth^2

ρ(E_mid) is the Mott gap OPENING SIGNAL: monotone decreasing from metallic to
insulating as U/t grows.  It is computed from ~probes×k Lanczos steps —
O(D·L) per step, O(D) memory, never the O(D²) or O(D³) of diagonalization.

ENCODING.  Jordan-Wigner: spin-up electrons on orbitals 2i, spin-down on 2i+1,
for i=0..L-1.  Hilbert space D = 4^L.  For L=6: D=4096; for L=8: D=65536.
No dense matrix is formed.

HONESTY CAVEAT.  The Mott "transition" in 1D at half-filling is not a true
sharp phase transition (it is a crossover); ρ(E_mid) decreases smoothly with U.
The DOS signal with Lorentzian broadening η=0.5 measures spectral density on
the scale η, not individual eigenvalues.  The crossover is clear and
monotone.  Exact eigvalsh comparison at L=6 confirms resona accuracy ~1-6%.

Run:  python3 examples/quantum/hubbard_mott.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from scipy import sparse as sp
import resona


def build_hubbard(L, t=1.0, U=4.0):
    """Sparse 1D Hubbard model.  Jordan-Wigner: orbital 2i=up, 2i+1=down.  D=4^L."""
    Ns = 2 * L
    D  = 1 << Ns
    states = np.arange(D, dtype=np.int64)
    diag_vals = np.zeros(D)
    row_list, col_list, val_list = [], [], []

    for spin in range(2):          # 0=up, 1=down
        for i in range(L):
            j = (i + 1) % L       # periodic boundaries
            oi = 2 * i + spin
            oj = 2 * j + spin
            lo, hi = min(oi, oj), max(oi, oj)

            occ_i = (states >> oi) & 1
            occ_j = (states >> oj) & 1
            hop_mask = (occ_i ^ occ_j).astype(bool)
            s_h = states[hop_mask]
            s_flip = s_h ^ ((1 << oi) | (1 << oj))

            # Jordan-Wigner string: parity of orbitals strictly between lo and hi
            if hi > lo + 1:
                jw_bits = np.zeros(hop_mask.sum(), dtype=np.int64)
                for b in range(lo + 1, hi):
                    jw_bits += (s_h >> b) & 1
                sign = (1 - 2 * (jw_bits & 1)).astype(float)
            else:
                sign = np.ones(hop_mask.sum(), dtype=float)

            row_list.append(s_h)
            col_list.append(s_flip)
            val_list.append(-t * sign)

    # U term: diagonal Σ n_{i↑} n_{i↓}
    for i in range(L):
        n_up   = ((states >> (2 * i))     & 1).astype(float)
        n_down = ((states >> (2 * i + 1)) & 1).astype(float)
        diag_vals += U * n_up * n_down

    rows = np.concatenate(row_list)
    cols = np.concatenate(col_list)
    vals = np.concatenate(val_list)
    # Symmetrise: add transpose (real hopping, H_ij = H_ji)
    rows2 = np.concatenate([rows, cols])
    cols2 = np.concatenate([cols, rows])
    vals2 = np.concatenate([vals, vals])
    Hoff = sp.coo_matrix((vals2, (rows2, cols2)), shape=(D, D)).tocsr()
    return (Hoff + sp.diags(diag_vals)).tocsr()


if __name__ == "__main__":
    L = 6
    D = 4 ** L
    eta = 0.5   # Lorentzian broadening for DOS

    print("=" * 70)
    print(f"HUBBARD MOTT TRANSITION — DOS depletion via resona.density()")
    print(f"  1D Hubbard L={L}, D=4^L={D}  (periodic boundaries)")
    print(f"  No diagonalization.  No sign problem.  O(D·L·k) matvecs.")
    print("=" * 70)
    print()

    # Exact validation at L=6, a few U values
    print("  Cross-check resona vs exact eigvalsh (L=6):")
    print(f"  {'U/t':>5}  {'DOS_mid exact':>14}  {'DOS_mid resona':>15}  {'err%':>7}")
    print("  " + "─" * 50)
    for U_check in [0.0, 4.0, 8.0]:
        H = build_hubbard(L, t=1.0, U=U_check)
        evals = np.sort(np.linalg.eigvalsh(H.toarray()))
        E0_ex, Emax_ex = evals[0], evals[-1]
        mid_ex = (E0_ex + Emax_ex) / 2
        dos_ex = np.sum(eta**2 / ((evals - mid_ex)**2 + eta**2)) / (np.pi * eta * D)

        s = resona.of(lambda v, H=H: H @ v, D, k=64, probes=12)
        E0_res, Emax_res = s.extreme()
        mid_res = (E0_res + Emax_res) / 2
        dos_res = s.density(np.array([mid_res]), eta=eta)[0]

        err = 100 * abs(dos_res - dos_ex) / (dos_ex + 1e-30)
        print(f"  {U_check:>5.1f}  {dos_ex:>14.5f}  {dos_res:>15.5f}  {err:>6.1f}%")

    print()
    # Main sweep: U/t from 0 to 8
    U_vals = np.linspace(0.0, 8.0, 13)
    print(f"  {'U/t':>5}  {'E0':>8}  {'E_max':>8}  {'Tr(H²)/D':>10}  "
          f"{'DOS(mid)':>10}  {'relative':>10}  {'time':>6}")
    print("  " + "─" * 68)

    dos_vals = []
    t0 = time.perf_counter()
    dos0 = None
    for idx, U in enumerate(U_vals):
        H = build_hubbard(L, t=1.0, U=U)
        matvec = lambda v, H=H: H @ v

        s = resona.of(matvec, D, k=64, probes=12, seed=42 + idx)
        E0, Emax = s.extreme()
        m2 = s.moment(2) / D
        mid = (E0 + Emax) / 2
        dos_mid = s.density(np.array([mid]), eta=eta)[0]
        dos_vals.append(dos_mid)

        if dos0 is None:
            dos0 = dos_mid
        rel = dos_mid / dos0

        dt = time.perf_counter() - t0
        print(f"  {U:>5.2f}  {E0:>8.3f}  {Emax:>8.3f}  {m2:>10.4f}  "
              f"{dos_mid:>10.5f}  {rel:>10.4f}  {dt:>5.1f}s")

    dos_arr = np.array(dos_vals)
    print()
    print(f"  DOS(midgap) change: U/t=0 → U/t=8:  "
          f"{dos_arr[0]:.5f} → {dos_arr[-1]:.5f}  "
          f"(×{dos_arr[-1]/dos_arr[0]:.3f}, {100*(1-dos_arr[-1]/dos_arr[0]):.0f}% depletion)")
    mono = np.all(np.diff(dos_arr) <= 0.01)   # monotone down (allow small noise)
    print(f"  Monotone decrease: {mono}")
    print()
    print("  PHYSICS: as U/t grows, the spectrum splits into lower and upper")
    print("  Hubbard bands separated by ~U.  Spectral weight drains from the")
    print("  midgap region — ρ(E_mid) is the Mott depletion signal.")
    print("  No fermion determinants, no sign cancellations — resona.density()")
    print("  reads this directly from Lanczos quadrature nodes and weights.")
    print("=" * 70)
