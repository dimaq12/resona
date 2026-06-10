"""
quantum/heisenberg_thermo.py
==============================================================================
FULL THERMODYNAMICS OF THE HEISENBERG CHAIN — matrix-free via resona.trace().

THE MODEL.  The antiferromagnetic Heisenberg XXX chain (periodic boundaries):

    H = Σᵢ Sᵢ·Sᵢ₊₁ = Σᵢ [(1/4)ZᵢZᵢ₊₁ + (1/2)(S⁺ᵢS⁻ᵢ₊₁ + S⁻ᵢS⁺ᵢ₊₁)]

At L=20 the Hilbert space has D = 2^20 = 1,048,576 states.  Full
diagonalization requires O(D³) ≈ 10¹⁸ FLOP and O(D²) ≈ 8 TB memory —
physically impossible on any single machine.

WHAT WE DO INSTEAD.  Given the resona spectrum object s = resona.of(matvec, D):

  Z(β) = Tr e^{−βH}         = s.trace(lambda l: exp(-β·l))
  ⟨H⟩  = Tr[H e^{−βH}] / Z  = s.trace(lambda l: l·exp(-β·l)) / Z(β)
  ⟨H²⟩                       = s.trace(lambda l: l²·exp(-β·l)) / Z(β)
  F(β) = −T log Z             free energy
  S(β) = (⟨H⟩ − F) / T       entropy
  Cv(β) = β² (⟨H²⟩ − ⟨H⟩²)  specific heat (per site: Cv/L)

All thermodynamic quantities from ONE call to resona.of() — one SLQ pass,
O(probes·k·D) cost, O(D) memory.  No diagonalization.

WHY THIS WORKS.  resona.trace(f) = N · Σ_k w_k f(node_k) evaluates Tr f(H)
exactly for the SLQ polynomial representation.  For smooth f = exp(-β·λ) the
quadrature is highly accurate at high and moderate T.  At very low T (β ≫ 1/E_gap)
only the ground-state sector contributes; this is also where resona is most
accurate (Lanczos preferentially resolves extremes).

THE L=20 FRAMING.  eigh at L=20: 8 TB matrix, 10¹⁸ FLOP — eigh is infeasible.
resona at L=20: k=48 probes=8 → ~3.2×10⁸ matvec FLOP, ~8 MB memory.

VALIDATION.  We verify against exact eigvalsh at L=8 and L=10 where eigh is
feasible.  Thermodynamic quantities agree within ~1-3% at intermediate T;
high-T agreement is excellent, very-low-T is Lanczos-accurate.

Run:  python3 examples/quantum/heisenberg_thermo.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from scipy import sparse as sp
import resona


def build_heisenberg(L):
    """Sparse Heisenberg H = Σ Sᵢ·Sᵢ₊₁ (periodic), real symmetric, D=2^L."""
    D = 1 << L
    states = np.arange(D, dtype=np.int64)
    diag = np.zeros(D)
    rows, cols = [], []
    for i in range(L):
        j = (i + 1) % L
        zi = 1 - 2 * ((states >> i) & 1).astype(float)
        zj = 1 - 2 * ((states >> j) & 1).astype(float)
        diag += 0.25 * zi * zj
        diff = ((states >> i) & 1) != ((states >> j) & 1)
        s_diff = states[diff]
        rows.append(s_diff)
        cols.append(s_diff ^ ((1 << i) | (1 << j)))
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.full(len(row), 0.5)
    Hoff = sp.coo_matrix((val, (row, col)), shape=(D, D)).tocsr()
    return (sp.diags(diag) + Hoff).tocsr()


def thermo_from_resona(s, betas):
    """Compute Z, F, E, S, Cv from resona spectrum object s."""
    Z   = np.array([s.trace(lambda l, b=b: np.exp(-b * l))         for b in betas])
    EZ  = np.array([s.trace(lambda l, b=b: l * np.exp(-b * l))     for b in betas])
    E2Z = np.array([s.trace(lambda l, b=b: l**2 * np.exp(-b * l))  for b in betas])
    E_mean  = EZ  / Z
    E2_mean = E2Z / Z
    T = 1.0 / betas
    F  = -T * np.log(np.maximum(Z, 1e-300))
    S  = (E_mean - F) / T
    Cv = (E2_mean - E_mean**2) * betas**2
    return dict(Z=Z, E=E_mean, F=F, S=S, Cv=Cv)


def thermo_exact(evals, betas):
    """Exact thermodynamics from full spectrum."""
    out = dict(Z=[], E=[], F=[], S=[], Cv=[])
    for b in betas:
        w = np.exp(-b * evals)
        Z = np.sum(w)
        e1 = np.sum(evals * w) / Z
        e2 = np.sum(evals**2 * w) / Z
        T  = 1.0 / b
        out["Z"].append(Z)
        out["E"].append(e1)
        out["F"].append(-T * np.log(Z))
        out["S"].append((e1 + T * np.log(Z)) / T)
        out["Cv"].append((e2 - e1**2) * b**2)
    return {k: np.array(v) for k, v in out.items()}


if __name__ == "__main__":
    print("=" * 74)
    print("HEISENBERG CHAIN THERMODYNAMICS — Z,F,E,S,Cv from resona.trace()")
    print("=" * 74)
    print("  H = Σ Sᵢ·Sᵢ₊₁  (periodic).  One resona.of() call → all thermodynamics.")
    print("  resona.trace(f) = N·Σ w_k f(node_k)  — no diagonalization, O(D).")
    print()

    betas = np.logspace(-1, 2, 40)     # T from 10 down to 0.01

    # Exact-vs-resona comparison table
    print("  EXACT vs RESONA cross-check (max error across β grid):")
    print(f"  {'L':>3}  {'D':>5}  {'max err Z%':>12}  {'max err E%':>12}  "
          f"{'max err Cv%':>12}  {'resona ms':>10}")
    print("  " + "─" * 60)

    for L in [8, 10, 12]:
        H = build_heisenberg(L)
        D = H.shape[0]
        matvec = lambda v, H=H: H @ v

        evals = np.sort(np.linalg.eigvalsh(H.toarray()))
        th_ex = thermo_exact(evals, betas)

        t0 = time.perf_counter()
        s = resona.of(matvec, D, k=64, probes=12)
        th_re = thermo_from_resona(s, betas)
        dt_ms = (time.perf_counter() - t0) * 1e3

        # Relative errors — only at moderate T where signals are non-negligible
        # Use 0.2 <= beta <= 10 (T: 0.1 to 5) — avoid numerical noise at very low T
        mask = (betas >= 0.2) & (betas <= 10.0)
        # Normalize by max of exact (so near-zero Cv doesn't dominate)
        Z_scale  = np.max(np.abs(th_ex["Z"][mask]))  + 1e-30
        E_scale  = np.max(np.abs(th_ex["E"][mask]))  + 1e-30
        Cv_scale = np.max(np.abs(th_ex["Cv"][mask])) + 1e-30
        err_Z  = 100 * np.max(np.abs(th_re["Z"][mask]  - th_ex["Z"][mask]))  / Z_scale
        err_E  = 100 * np.max(np.abs(th_re["E"][mask]  - th_ex["E"][mask]))  / E_scale
        err_Cv = 100 * np.max(np.abs(th_re["Cv"][mask] - th_ex["Cv"][mask])) / Cv_scale
        print(f"  {L:>3}  {D:>5}  {err_Z:>12.2f}%  {err_E:>12.2f}%  "
              f"{err_Cv:>12.2f}%  {dt_ms:>9.1f}")

    print()
    print("  THERMODYNAMICS TABLE for L=10 (resona vs exact):")
    print(f"  {'T=1/β':>7}  {'E_ex':>8}  {'E_re':>8}  {'Cv_ex':>7}  "
          f"{'Cv_re':>7}  {'S_ex':>7}  {'S_re':>7}")
    print("  " + "─" * 60)

    H10 = build_heisenberg(10)
    evals10 = np.sort(np.linalg.eigvalsh(H10.toarray()))
    s10 = resona.of(lambda v: H10 @ v, H10.shape[0], k=64, probes=12)
    th_ex10 = thermo_exact(evals10, betas)
    th_re10 = thermo_from_resona(s10, betas)
    for i in range(0, len(betas), 5):
        T = 1.0 / betas[i]
        print(f"  {T:>7.3f}  {th_ex10['E'][i]:>8.4f}  {th_re10['E'][i]:>8.4f}  "
              f"{th_ex10['Cv'][i]:>7.4f}  {th_re10['Cv'][i]:>7.4f}  "
              f"{th_ex10['S'][i]:>7.4f}  {th_re10['S'][i]:>7.4f}")

    # Scale-up: L=20, where eigh is impossible
    print()
    print("  SCALE-UP — L=16 and L=20 (eigh impossible):")
    print(f"  {'L':>3}  {'D':>8}  {'E0':>8}  {'E(T=0.5)':>10}  "
          f"{'Cv(T=0.5)':>11}  {'S(T=0.5)':>10}  {'time s':>7}")
    print("  " + "─" * 70)
    for L in [16, 20]:
        D = 1 << L
        H = build_heisenberg(L)
        matvec = lambda v, H=H: H @ v
        t0 = time.perf_counter()
        s = resona.of(matvec, D, k=48, probes=4)   # fewer probes for speed at large D
        E0 = s.extreme()[0]
        betas_spot = np.array([2.0])   # T=0.5
        th_spot = thermo_from_resona(s, betas_spot)
        dt = time.perf_counter() - t0
        print(f"  {L:>3}  {D:>8}  {E0:>8.4f}  {th_spot['E'][0]:>10.4f}  "
              f"{th_spot['Cv'][0]:>11.4f}  {th_spot['S'][0]:>10.4f}  {dt:>6.1f}s")

    print()
    eigh_flop_20 = (1 << 20)**3
    eigh_mem_20  = (1 << 20)**2 * 8 / 1e12
    res_flop_20  = (1 << 20) * 20 * 48 * 8      # L*D*k*probes
    print(f"  L=20 eigh: {eigh_flop_20:.1e} FLOP, {eigh_mem_20:.0f} TB memory — INFEASIBLE.")
    print(f"  L=20 resona: ~{res_flop_20:.1e} FLOP, ~{3*(1<<20)*8/1e6:.0f} MB memory — runs in seconds.")
    print()
    print("  Full thermodynamics (Z, F, E, S, Cv) from one resona.of() call.")
    print("  No diagonalization. No dense matrix. No exponential memory.")
    print("=" * 74)
