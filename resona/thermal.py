"""
resona.thermal — finite-temperature quantum reads on top of `apply`.

Quantum typicality: ONE random vector evolved in imaginary time,
|ψ_β⟩ = e^{−βH/2}|r⟩, already carries thermal expectation values with error
~ 1/√(D_eff) (D_eff = the thermal participation dimension) — no
diagonalization, no density matrix, no sign problem.  Three functions:

    state(Hmv, beta, N)          →  (ψ_β normalized, weight ‖e^{−βH/2}r‖²)
    expect(Hmv, Omv, beta, N)    →  ⟨O⟩_β  (probe-averaged, with stderr)
    correlator(Hmv, Omv, beta, ts, N) → C(t) = ⟨O(t) O⟩_β  (complex array)

C(t)'s Fourier transform is the spectral function S(ω) — the bread of
condensed-matter response theory, matrix-free at sizes where dense
diagonalization is dead.

HONEST LIMITS.  Typicality error decays with the EFFECTIVE thermal dimension,
not raw 2^L — at very low temperature (β large, few occupied states) the
estimator degrades toward a ground-state projector and needs more probes (the
stderr reports this).  Krylov imaginary/real-time steps inherit `apply`'s k
budget: large β·‖H‖ or t·‖H‖ may need larger k — verified against dense
ground truth in tests on small chains.
"""
import numpy as np
from .spectral import apply as _apply

__all__ = ["state", "expect", "correlator"]


def state(Hmv, beta, N, seed=0, k=64):
    """|ψ_β⟩ = e^{−βH/2}|r⟩ (normalized) and its weight ‖e^{−βH/2}r‖².

    The weight is the probe's share of the partition function:
    Z ≈ N · E_r[weight]."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(N) / np.sqrt(N)
    psi = _apply(Hmv, lambda lam: np.exp(-0.5 * beta * lam), r, k=k)
    w = float(np.real(np.vdot(psi, psi)))
    return psi / np.sqrt(w), w * N


def expect(Hmv, Omv, beta, N, probes=8, k=64, seed=0):
    """⟨O⟩_β = Tr[O e^{−βH}]/Z by typicality, probe-averaged.

    Returns (value, stderr) — the honest pair, like `trace(with_err=True)`."""
    vals, wts = [], []
    for p in range(probes):
        psi, w = state(Hmv, beta, N, seed=seed + p, k=k)
        vals.append(float(np.real(np.vdot(psi, Omv(psi)))))
        wts.append(w)
    vals, wts = np.array(vals), np.array(wts)
    mean = float(np.sum(vals * wts) / np.sum(wts))
    if probes < 2:
        return mean, float("nan")
    dev = (vals - mean) * wts / wts.mean()
    return mean, float(np.std(dev, ddof=1) / np.sqrt(probes))


def correlator(Hmv, Omv, beta, ts, N, probes=4, k=64, seed=0):
    """C(t) = ⟨O(t) O⟩_β = ⟨ψ_β| e^{iHt} O e^{−iHt} O |ψ_β⟩, typicality-averaged.

    Per probe and time: two real-time Krylov evolutions (`apply`,
    hermitian=False) — O(len(ts) · probes · k) matvecs.  FFT C(t) for the
    spectral function S(ω)."""
    ts = np.asarray(ts, float)
    dts = np.diff(ts)
    uniform = len(ts) > 1 and np.allclose(dts, dts[0])
    out = np.zeros((probes, len(ts)), complex)
    wts = np.zeros(probes)
    for p in range(probes):
        psi, w = state(Hmv, beta, N, seed=seed + p, k=k)
        wts[p] = w
        phi = np.asarray(Omv(psi), complex)
        if uniform:
            # incremental stepping: ONE Δt-evolution per time point instead of
            # evolving from t=0 each time — O(len(ts)) instead of O(len(ts)²)
            # matvecs; per-step Krylov error ~1e-12 accumulates linearly and
            # stays far below the typicality noise
            dt = float(dts[0])
            step = lambda v: _apply(Hmv, lambda lam: np.exp(-1j * dt * lam),
                                   v, k=k, hermitian=False)
            psi_t, phi_t = psi.astype(complex), phi
            for i, t in enumerate(ts):
                if i > 0:
                    psi_t, phi_t = step(psi_t), step(phi_t)
                out[p, i] = np.vdot(psi_t, np.asarray(Omv(phi_t), complex))
        else:
            for i, t in enumerate(ts):
                if t == 0.0:
                    psi_t, phi_t = psi.astype(complex), phi
                else:
                    psi_t = _apply(Hmv, lambda lam: np.exp(-1j * t * lam),
                                  psi.astype(complex), k=k, hermitian=False)
                    phi_t = _apply(Hmv, lambda lam: np.exp(-1j * t * lam),
                                  phi, k=k, hermitian=False)
                out[p, i] = np.vdot(psi_t, np.asarray(Omv(phi_t), complex))
    return (out * wts[:, None]).sum(0) / wts.sum()
