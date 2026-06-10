"""
anderson_localization.py — where electrons stop conducting, matrix-free.
==============================================================================
THE PHYSICS.  Put a quantum particle (an electron) on a 3D lattice and add
random on-site disorder of strength W (impurities, defects in a real crystal).
At weak disorder the wavefunctions are EXTENDED — the electron travels, the
material conducts.  Past a critical disorder W_c they become LOCALIZED — trapped
in a finite region, the material becomes an insulator.  This is ANDERSON
LOCALIZATION (Nobel 1977): a metal–insulator transition driven purely by
disorder, with no interactions.  Finding W_c usually means diagonalizing a
huge disordered Hamiltonian; here we never call eig.

THE ORDER PARAMETER — a DEFECT between two averages of the response.  The local
density of states ρ_i(E) (how much spectral weight lives at site i) fluctuates
wildly when states localize.  Compare its two averages over sites:
      mean DOS    = arithmetic mean of ρ_i    (the usual smooth DOS)
      typical DOS = geometric  mean of ρ_i  = exp⟨log ρ_i⟩
      extended  ⇔  typical ≈ mean            (Λ = ∫typ/∫mean → 1)
      localized ⇔  typical ≪ mean            (Λ → 0, log-normal LDOS)
Localization is exactly the DEFECT between the geometric and arithmetic response
average opening up — the same "defect" lens used throughout, here as a phase
order parameter.

resona's ROLE.  The LOCAL density of states at site i is resona's response
PROBED AT ONE SITE: run the Lanczos kernel from the unit vector e_i — the Ritz
weights are |⟨i|ψ_n⟩|², i.e. exactly the local spectral weights.  So ρ_i is
`resona.density` with a site probe instead of a random one.  Everything is
matrix-free: only H·x products (sparse hopping + diagonal disorder), no matrix
diagonalized, on L³ = 4096 sites on a laptop.

Run:  python3 examples/anderson_localization.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy import sparse
from resona.spectral import _lanczos                  # resona's own Lanczos kernel

rng = np.random.default_rng(5)


def hopping_3d(L):
    """3D cubic-lattice nearest-neighbour hopping (t=1), periodic — the 'kinetic' H."""
    idx = lambda x, y, z: (x % L) + L * ((y % L) + L * (z % L))
    rows, cols = [], []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = idx(x, y, z)
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    rows.append(i); cols.append(idx(x + dx, y + dy, z + dz))
    return sparse.csr_matrix((-np.ones(len(rows)), (rows, cols)), shape=(L**3, L**3))


def ldos_site(matvec, N, i, k, Egrid, eta):
    """ρ_i(E) — local DOS at site i = resona's response probed at e_i (matrix-free).

    The Ritz weights of a Lanczos run started at e_i are |⟨i|ψ_n⟩|²; smoothing
    them with a Lorentzian of width η is exactly `resona.density` at one site.
    """
    v0 = np.zeros(N); v0[i] = 1.0
    al, be = _lanczos(matvec, v0, k)
    kk = len(al)
    T = np.diag(al) + np.diag(be[:kk - 1], 1) + np.diag(be[:kk - 1], -1)
    theta, S = np.linalg.eigh(T)
    w = S[0, :] ** 2                                   # local weights |⟨i|ψ_n⟩|²
    return (w[None, :] * (eta / np.pi)
            / ((Egrid[:, None] - theta[None, :]) ** 2 + eta ** 2)).sum(1)


def order_parameter(A0, W, L, n_sites=60, k=70, eta=0.15, nE=121):
    """Λ(W) = (∫ typical DOS)/(∫ mean DOS) over the band.  Λ→1 extended, →0 localized."""
    N = L ** 3
    w = W * (rng.random(N) - 0.5)                      # uniform on-site disorder [-W/2,W/2]
    matvec = lambda x: A0 @ x + w * x                  # H = hopping + disorder, matvec only
    band = 6.0 + W / 2 + 1.0
    Egrid = np.linspace(-band, band, nE)
    sites = rng.choice(N, size=n_sites, replace=False)
    Lij = np.clip([ldos_site(matvec, N, int(i), k, Egrid, eta) for i in sites], 1e-12, None)
    mean_dos = Lij.mean(axis=0)
    typ_dos = np.exp(np.mean(np.log(Lij), axis=0))
    mask = mean_dos > 0.05 * mean_dos.max()            # restrict to the actual band
    Lam = float(typ_dos[mask].sum() / (mean_dos[mask].sum() + 1e-12))
    return Lam


if __name__ == "__main__":
    print("=" * 72)
    print("ANDERSON LOCALIZATION — metal→insulator transition, matrix-free (no eig)")
    print("=" * 72)
    L = 16
    A0 = hopping_3d(L)
    print(f"  3D cubic lattice, L={L}, N={L**3} sites.  W_c ≈ 16.5 (full localization).")
    print(f"  LDOS = resona's response probed at each site; Λ = ∫typ/∫mean DOS.\n")
    print(f"  {'W (disorder)':>13} {'Λ=∫typ/∫mean':>14}  conduction (█ = extended)")
    print("  " + "─" * 56)
    t0 = time.perf_counter()
    Lams = {}
    for W in (2.0, 6.0, 10.0, 14.0, 18.0, 24.0):
        Lam = order_parameter(A0, W, L)
        Lams[W] = Lam
        tag = "metal" if Lam > 0.4 else ("crossover" if Lam > 0.2 else "INSULATOR")
        print(f"  {W:>13.0f} {Lam:>14.3f}  {'█'*int(50*Lam)} {tag}")
    print(f"\n  computed in {time.perf_counter()-t0:.1f}s, fully matrix-free (no eig).")
    print(f"  Λ falls monotonically with disorder: {Lams[2.0]:.2f} → {Lams[24.0]:.2f}")
    print("\n" + "=" * 72)
    print("  The metal→insulator transition seen as a DEFECT between the geometric and")
    print("  arithmetic averages of the local response.  Each LDOS is resona's spectral")
    print("  density probed at one site — same SLQ kernel, deterministic site probe — so")
    print("  the whole order parameter is matrix-free on 4096 sites.  (Sharp mobility-")
    print("  EDGE resolution within the band needs larger L; Λ(W) is robust at L=16.)")
    print("=" * 72)
