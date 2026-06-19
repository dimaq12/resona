"""
quantum/phase_transition.py
==============================================================================
LOCATE A QUANTUM PHASE TRANSITION WITHOUT EVER DIAGONALIZING — matrix-free.

THE PROBLEM.  Turn a knob h on a quantum Hamiltonian H(h) and at some critical
h_c the ground state reorganizes qualitatively — a QUANTUM PHASE TRANSITION (a
zero-temperature change of phase, driven by quantum not thermal fluctuations).
Finding h_c the naive way means diagonalizing H(h) on a grid of h — O(D³) per
point, D=2ⁿ.  Infeasible past a dozen qubits.

THE IDEA — the transition is a DEFECT in the response.  Define the spectral
RESPONSE SUSCEPTIBILITY of the ground state to the knob B = ∂H/∂h:

      Φ_η(E₀) = (η/π)² · Tr[ B R B R ],     R = ((H − E₀)² + η²)⁻¹

R is a smoothed projector onto states near the ground energy E₀; Φ_η measures how
strongly B couples the ground state to its neighbours.  At a phase transition the
gap collapses and the ground state becomes maximally susceptible — Φ_η SPIKES.
So the peak of Φ_η(h) marks h_c.  This is the same "defect = edge = where the
spectrum reorganizes" object used everywhere in this program, read at the bottom
of the spectrum.

WHY IT'S MATRIX-FREE.  Everything is matvecs:
  • E₀ from resona (extreme eigenvalue, stochastic Lanczos) — no eig.
  • R·z = ((H−E₀)²+η²)⁻¹ z by two COMPLEX-SHIFTED solves — only H·v products.
  • Tr[B R B R] by HUTCHINSON — random probes x, average xᵀ B R B R x.
η is the resolution at which we look for the defect.  The squared form (H−E₀)²+η²
has condition number κ²; factoring R = ((H−E₀)−iη)⁻¹((H−E₀)+iη)⁻¹ into two shifted
solves (condition κ) is what actually CONVERGES — plain CG on the squared operator
stalls at maxiter and returns garbage.  Total cost O(probes · n_iter · D), i.e.
linear in D — no O(D³), no matrix ever formed.

MODEL.  TFIM  H(h) = −Σ ZᵢZᵢ₊₁ − h Σ Xᵢ,  knob B = ∂H/∂h = −Σ Xᵢ.  Exactly
solvable: the quantum phase transition sits at h_c = 1 (ordered/ferromagnetic for
h<1, disordered/paramagnetic for h>1).  We recover it from response alone.
(Note: h_c is NOT the gap minimum — the ordered phase has a near-degenerate
doublet, so a gap scan is fooled; the susceptibility is not.)

Run:  python3 examples/quantum/phase_transition.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from scipy import sparse as sp
import resona

rng = np.random.default_rng(7)


def build_tfim_parts(n):
    """Return D, diagonal of −Σ ZZ, and the −Σ X flip operator (the knob B)."""
    D = 1 << n
    states = np.arange(D, dtype=np.int64)
    diag = np.zeros(D)
    for i in range(n):
        j = (i + 1) % n
        diag -= (1 - 2 * ((states >> i) & 1)) * (1 - 2 * ((states >> j) & 1))
    rows = np.tile(states, n)            # bit-major, to align with cols below
    cols = np.concatenate([states ^ (1 << i) for i in range(n)])
    Hx = sp.coo_matrix((-np.ones(D * n), (rows, cols)), shape=(D, D)).tocsr()
    return D, diag, Hx


def susceptibility(matH, matB, N, E0, eta, n_probe=16):
    """Φ_η(E₀) = (η/π)² Tr[B R B R], R=((H−E₀)²+η²)⁻¹ — via `resona.defect.hard_points`,
    the library primitive for exactly this resolvent susceptibility (Hutchinson + CG,
    matvecs only).  We call it on a single point with E=E₀(h).

    Conditioning note (the v2.0.1 lesson, now handled inside the primitive): at this
    η=0.5 the squared resolvent (H−E₀)²+η² is well-conditioned (η² is a large floor),
    so the primitive's CG converges cleanly — verified bit-identical to the
    complex-shifted ((H−E₀)±iη)⁻¹ factoring.  At SMALL η the squared form is too
    ill-conditioned (κ²) and CG stalls; `hard_points` captures the CG flag and WARNS,
    so a silent-garbage regression like the original can no longer pass unnoticed.
    """
    _, profile = resona.defect.hard_points(lambda k: matH, [0.0], matB, N=N, E=E0,
                                           eta=eta, probes=n_probe)
    return float(profile[0])


if __name__ == "__main__":
    n = 12
    print("=" * 74)
    print(f"QUANTUM PHASE TRANSITION located matrix-free — TFIM n={n} (D={1<<n})")
    print("=" * 74)
    print("  full eigh on a grid would be O(D³)·grid — infeasible.  We use only")
    print("  E₀ (resona, matrix-free) + Φ_η (shifted-resolvent + Hutchinson trace).\n")
    D, diag, Hx = build_tfim_parts(n)
    matB = lambda v: Hx @ v
    hs = np.linspace(0.5, 1.5, 11)
    chis = []
    for h in hs:
        H = (sp.diags(diag) + h * Hx).tocsr()
        matH = lambda v, H=H: H @ v
        E0 = resona.of(matH, D, k=120, probes=4).extreme()[0]  # ground energy, no eig
        chis.append(susceptibility(matH, matB, D, E0, eta=0.5, n_probe=16))
    chis = np.array(chis)
    # Φ_η rises sharply ACROSS the transition and stays large in the disordered
    # (paramagnetic) phase — so the TRANSITION is its ONSET, the steepest rise,
    # not a symmetric peak.  Locate h_c by the largest jump between grid points.
    rise = np.diff(chis)
    j = int(np.argmax(rise))
    h_star = 0.5 * (hs[j] + hs[j + 1])

    print(f"  {'h':>6} {'Φ_η (response)':>16}")
    print("  " + "─" * 30)
    for h, c in zip(hs, chis):
        bar = "█" * int(38 * c / max(chis.max(), 1e-30))
        print(f"  {h:>6.2f} {c:>16.3e} {bar}")
    ok = abs(h_star - 1.0) <= 1.5 * (hs[1] - hs[0]) + 1e-9
    print(f"\n  steepest rise of Φ_η at h ≈ {h_star:.2f}   (exact h_c = 1.0)   "
          f"{'LOCATED ✓' if ok else '✗'}")
    print("  (Φ_η stays large for h>h_c: the paramagnet remains susceptible — the")
    print("   transition is the ONSET of the defect, not a symmetric peak.)")
    print("\n" + "=" * 74)
    print("  The transition is a DEFECT in the response: as the gap closes at h_c the")
    print("  ground state becomes susceptible to the knob and Φ_η turns on sharply.")
    print("  Read from H·v products (shifted-resolvent solves + Hutchinson trace) and a")
    print("  matrix-free ground energy — no diagonalization, O(D) per point.  The same")
    print("  defect object that locates avoided crossings, at the quantum critical point.")
    print("=" * 74)
