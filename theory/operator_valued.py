"""
resolvent_algebra/operator_valued.py
=============================================================================
OPERATOR-VALUED free probability — the right frame for STRUCTURED disorder.

Scalar free probability (one self-consistent g) assumes homogeneity.  For
HETEROGENEOUS disorder (site-dependent variance σ_i²) it FAILS.  The correct
object is the operator-valued (matrix Dyson) subordination: a VECTOR of local
Green's functions g_i solving

      g_i(z) = [ ( z·I − A − diag(σ²· g) )^{-1} ]_{ii}.

We compare, for a 1D chain A with block-heterogeneous disorder:
  - TRUE averaged DOS (simulate many disorders, exact eig),
  - SCALAR prediction (single mean σ̄²),           ← wrong
  - OPERATOR-VALUED matrix Dyson (per-site g_i).    ← right

Run:  python3 resolvent_algebra/operator_valued.py
"""
import numpy as np
from scipy import linalg

rng = np.random.default_rng(0)


def true_dos(A, sig2, xs, eta, K=150):
    """Disorder-averaged DOS, Lorentzian-broadened (η), by simulation."""
    N = A.shape[0]
    acc = np.zeros_like(xs)
    sd = np.sqrt(sig2)
    for _ in range(K):
        lam = linalg.eigvalsh(A + np.diag(sd * rng.standard_normal(N)))
        for l in lam:
            acc += (eta / np.pi) / ((xs - l) ** 2 + eta ** 2)
    return acc / (K * N)


def scalar_dos(a_eig, sbar2, xs, eta):
    """Scalar self-consistent: g = G_A(z − σ̄² g).  DOS = −Im g/π."""
    N = len(a_eig)
    out = np.zeros_like(xs)
    for i, x in enumerate(xs):
        z = x + 1j * eta; g = -0.3j
        for _ in range(2000):
            g_new = np.mean(1.0 / (z - sbar2 * g - a_eig))
            if abs(g_new - g) < 1e-11:
                break
            g = g_new
        out[i] = max(-g.imag / np.pi, 0.0)
    return out


def operator_valued_dos(A, sig2, xs, eta):
    """Matrix Dyson: vector g_i = [(zI − A − diag(σ²g))^{-1}]_ii.  DOS = mean(−Im g_i)/π."""
    N = A.shape[0]
    out = np.zeros_like(xs)
    for i, x in enumerate(xs):
        z = x + 1j * eta; g = -0.3j * np.ones(N)
        for _ in range(400):
            G = linalg.inv(z * np.eye(N) - A - np.diag(sig2 * g))
            g_new = np.diag(G).copy()
            if np.max(np.abs(g_new - g)) < 1e-10:
                break
            g = g_new
        out[i] = max(np.mean(-g.imag) / np.pi, 0.0)
    return out


if __name__ == "__main__":
    N = 300
    A = np.diag(-np.ones(N - 1), 1) + np.diag(-np.ones(N - 1), -1)   # 1D chain hopping
    sig2 = np.where(np.arange(N) < N // 2, 0.1, 1.5)                  # block-heterogeneous
    a_eig = linalg.eigvalsh(A)
    eta = 0.12
    xs = np.linspace(-4, 4, 41)

    print("=" * 72)
    print("OPERATOR-VALUED free probability — structured disorder needs the matrix Dyson")
    print("=" * 72)
    print(f"  1D chain N={N}, block disorder σ²: half=0.1, half=1.5 (mean σ̄²={sig2.mean():.2f}).\n")

    dt = true_dos(A, sig2, xs, eta)
    ds = scalar_dos(a_eig, sig2.mean(), xs, eta)
    do = operator_valued_dos(A, sig2, xs, eta)

    rms_s = np.sqrt(np.mean((ds - dt) ** 2))
    rms_o = np.sqrt(np.mean((do - dt) ** 2))
    print(f"  {'E':>6} {'true DOS':>9} {'scalar':>8} {'op-valued':>10}")
    for j in range(0, len(xs), 3):
        print(f"  {xs[j]:>6.2f} {dt[j]:>9.3f} {ds[j]:>8.3f} {do[j]:>10.3f}")
    print(f"\n  RMS error vs TRUE:   scalar = {rms_s:.4f}   operator-valued = {rms_o:.4f}")
    print(f"  operator-valued is {rms_s/max(rms_o,1e-9):.1f}× closer to the truth.")
    print("\n" + "=" * 72)
    print("  Scalar free probability assumes one homogeneous medium and gets the shape")
    print("  wrong.  The operator-valued matrix Dyson tracks per-site Green's functions —")
    print("  the correct free-probability frame for STRUCTURED / correlated disorder")
    print("  (Anderson, real materials).  Same subordination idea, lifted to a matrix.")
    print("=" * 72)
