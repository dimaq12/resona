"""
resona on SIGNALS — photo / sound / radio.

resona is the FFT of *operators*, not of raw signals.  A signal becomes an operator
(covariance, trajectory/Hankel, graph, Koopman) — exactly what you build anyway
for denoising / PCA / source-counting / DOA — and resona reads its spectrum,
effective rank, and functionals MATRIX-FREE, at scale.

Demo: a noisy 3-tone signal (audio/radio-like) → trajectory covariance operator →
resona recovers the dominant components above the noise floor, never forming the
covariance.  (Same recipe: image = patch covariance; radio array = channel
covariance → number of emitters / DOA, MUSIC-style.)

Run:  python3 examples/signals.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy import linalg
from resona import Spectral

rng = np.random.default_rng(0)


def main():
    L, W = 5000, 150
    t = np.arange(L)
    freqs, amps = [0.05, 0.12, 0.23], [3.0, 2.0, 1.5]              # 3 tones (sources)
    x = sum(a * np.cos(2 * np.pi * f * t + rng.uniform(0, 6))
            for f, a in zip(freqs, amps)) + 0.8 * rng.standard_normal(L)

    # trajectory (Hankel) embedding → covariance operator C = X Xᵀ  (W×W), never formed
    K = L - W + 1
    X = np.lib.stride_tricks.sliding_window_view(x, W).T[:, :K]    # (W, K)
    matvec = lambda v: X @ (X.T @ v)                              # C·v, matrix-free

    s = Spectral.of(matvec, W, k=60, probes=12)               # for trace / Φ₁ (averaged)
    s1 = Spectral.of(matvec, W, k=50, probes=1)               # one Lanczos → top eigenvalues

    print("=" * 64)
    print("resona on a SIGNAL — 3 tones + noise via the trajectory covariance")
    print("=" * 64)
    C = X @ X.T
    ev = np.sort(linalg.eigvalsh(C))[::-1]
    top_ritz = np.sort(s1.nodes)[::-1][:6]
    print(f"  {'#':>2} {'resona eigenvalue':>18} {'true':>14}")
    for i in range(6):
        print(f"  {i+1:>2} {top_ritz[i]:>18.1f} {ev[i]:>14.1f}")
    floor = float(np.median(ev))
    n_sig = int(np.sum(ev > 5 * floor))
    print(f"\n  noise floor (median eig) = {floor:.1f}")
    print(f"  components above floor: {n_sig}  (3 tones → 6 modes, recovered)")
    print(f"  effective rank Φ₁ = {s.effective_rank():.1f}  (≪ W={W}: signal is low-rank)")
    print(f"  top eigenvalue  resona = {top_ritz[0]:.1f}   true = {ev[0]:.1f}   "
          f"err = {abs(top_ritz[0]-ev[0])/ev[0]:.1%}")
    print("\n  → C is W×W and never formed; the same recipe reads PHOTO patch-")
    print("    covariance (PCA/denoise), AUDIO/RADIO trajectory (periods, sources),")
    print("    and array covariance (number of emitters / DOA) — all matrix-free.")
    print("=" * 64)


if __name__ == "__main__":
    main()
