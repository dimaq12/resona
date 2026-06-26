"""
resona.cost — the Extraction Law: how hard is the answer to extract?

The answer pre-exists in the resolvent field G(z)=(z−A)⁻¹; a solver only EXTRACTS
from it.  The cost of extracting to accuracy ε near a query z follows

        Cost(ε, z)  ~  ε^{-a} · dist(z, Σ*)^{-b},

where Σ* is the NON-removable singular set (edges / branch points / shocks /
continua — NOT isolated poles, which deflate away).  Φ₁ = Tr(A)²/Tr(A²) is the
GLOBAL summary (low ⇒ structured/cheap, high ⇒ genuine frontier).

The operational test — REMOVABLE vs GENUINE — is lift-rank SATURATION: lift a
signal to its trajectory (Hankel) operator at growing window k.  If the effective
rank SATURATES, the singularity is removable → a finite chart linearizes it →
EXTRACTABLE.  If the rank keeps GROWING with k, there is no finite chart →
a genuine wall (structureless, e.g. aˣ mod N).  This is the dial that says
dequantizable-vs-quantum, easy-vs-hard, from the signal itself.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.linalg import hankel as _hankel, svdvals as _svdvals


def lift_rank(signal: Sequence, k: int = 60) -> float:
    """Effective rank of the trajectory (Hankel) operator of a signal at window k.
    = (Σσ²)²/Σσ⁴ of the Hankel singular values — the size of the linearizing chart.

    Needs at least ``2*k - 1`` samples to fill the (k × k) Hankel window; a
    shorter signal builds a degenerate/empty operator and is rejected.  A
    constant or all-zero signal has no trajectory structure → NaN."""
    s = np.asarray(signal, float)
    if len(s) < 2 * k - 1:
        raise ValueError(f"lift_rank needs len(signal) >= 2*k-1 = {2 * k - 1} "
                         f"for window k={k}, got {len(s)}")
    s = (s - s.mean()) / (s.std() + 1e-12)
    H = _hankel(s[:k], s[k - 1:2 * k - 1])
    s2 = _svdvals(H) ** 2
    if (s2 ** 2).sum() < 1e-30:                      # constant/zero signal → no chart
        return float("nan")
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def is_extractable(signal: Sequence, windows: Sequence[int] = (20, 40, 80, 120),
                   grow: float = 0.25) -> tuple[bool, list]:
    """The REMOVABLE-vs-GENUINE dichotomy, read by lift-rank saturation.

    REMOVABLE singularity (shock, exceptional point, pole): a FINITE lift
    linearizes it — the trajectory rank SATURATES with the window; the
    apparent wall is a coordinate artifact and the answer extracts through
    it.  GENUINE wall (structureless, e.g. aˣ mod N): rank GROWS ~ window —
    no finite chart exists, extraction stays extensive.

    Returns (extractable: bool, ranks: list).  The binary cutoff (`grow`)
    was calibrated on sharp walls; for borderline dynamics (e.g. Lorenz at
    coarse sampling) read the RANK CURVE itself — see
    examples/science/koopman_dynamics.py, where the curve is unambiguous
    while the binary is lenient.
    """
    n = len(np.asarray(signal))
    # a window k needs 2*k-1 samples (see lift_rank) — drop windows the signal
    # cannot fill rather than building a degenerate Hankel.
    usable = [k for k in windows if n >= 2 * k - 1]
    if len(usable) < 2:
        raise ValueError(f"is_extractable needs >=2 windows the signal can fill "
                         f"(each needs 2*k-1 samples); signal length {n} fills "
                         f"only {usable} of windows {tuple(windows)}")
    ranks = [lift_rank(signal, k) for k in usable]
    # genuine if the rank is still climbing with the window (chart never closes)
    growth = (ranks[-1] - ranks[-2]) / (usable[-1] - usable[-2])
    extractable = growth < grow and ranks[-1] < 0.4 * usable[-1]
    return bool(extractable), ranks


def level_spacing_ratio(eigenvalues: Sequence, eps: float = 1e-9) -> float:
    """The integrability detector: mean consecutive level-spacing ratio
    ⟨r⟩ = ⟨min(s_i, s_{i+1}) / max(s_i, s_{i+1})⟩ of a spectrum.

        ⟨r⟩ ≈ 0.386  → Poisson statistics → integrable (a lift EXISTS)
        ⟨r⟩ ≈ 0.531  → GOE level repulsion → chaotic (no lift)
        (GUE 0.600, GSE 0.676; rigid picket-fence → 1.)

    Unfolding-free (ratios cancel the local density), 3 lines, O(N log N).

    LOUD CAVEAT (a real numerics trap): RESOLVE ALL SYMMETRY SECTORS FIRST.
    Levels from different sectors do not repel; mixing sectors overlays
    independent series and FAKES Poisson — an un-projected chaotic Hamiltonian
    will read "integrable".  Project H into one sector (one reflection /
    momentum / magnetization sector) before calling this.
    """
    s = np.diff(np.sort(np.asarray(eigenvalues, float)))
    s = s[s > eps]
    if s.size < 2:                                   # need two spacings for a ratio
        return float("nan")
    return float(np.mean(np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])))


def rmt_class(eigenvalues: Sequence, deg: int = 8) -> tuple[str, float]:
    """Random-matrix UNIVERSALITY CLASS of a spectrum — Poisson / GOE / GUE / GSE —
    from the rigidity meter  R4 = ω₄(sorted UNFOLDED spacings),
        ω_k(s) = (s_max − s_{[M/k]}) / (|s_max| + |s_min|) − (1 − 1/k).

    The companion to `level_spacing_ratio` (which returns ⟨r⟩): R4 reads the same
    physics from the spacing-tail rigidity and resolves all FOUR Dyson classes —
    measured strictly ordered against the calibrated reference centres
    Poisson(+0.199) > GOE(+0.009) > GUE(−0.080) > GSE(−0.230),
    Spearman(R4, β) ≈ −0.95 (matrix-free-friendly: works on a window of interior
    eigenvalues, not just the full spectrum).  Returns (class, R4); a spectrum
    too short to unfold (< 5 levels) returns ("undetermined", nan).

    Unfolds internally (smooth empirical CDF → the local density flattened).

    HONEST CAVEATS.  (1) needs an UNFOLDABLE spectrum — a single smooth density;
    a multi-band / multi-sector spectrum must be split first (the same trap as
    `level_spacing_ratio` — resolve symmetry sectors).  (2) R4 is a POSITIVELY-BIASED
    rigidity proxy, not a β-on-the-nose readout — trust the class, not the decimals.
    (3) a SINGLE realization is NOISY near class boundaries (GOE↔GUE are only ~2σ
    apart at D≈600) — average R4 over realizations, or use large D, for a confident
    class; Poisson-vs-chaotic is robust per-draw.
    """
    lam = np.sort(np.asarray(eigenvalues, float))
    D = len(lam)
    if D < 5:                                       # too short to unfold meaningfully
        return "undetermined", float("nan")
    coef = np.polyfit(lam, np.arange(D) + 0.5, deg=min(deg, max(2, D // 8)))   # smooth CDF
    s = np.diff(np.polyval(coef, lam))                          # unfolded spacings
    s = np.sort(s[s > 0])
    if s.size < 2:                                  # degenerate spacings → undetermined
        return "undetermined", float("nan")
    idx = max(0, len(s) // 4 - 1)                   # never wrap to s[-1] on a short tail
    R4 = (s[-1] - s[idx]) / (abs(s[-1]) + abs(s[0]) + 1e-15) - 0.75
    refs = {"Poisson": 0.199, "GOE": 0.009, "GUE": -0.080, "GSE": -0.230}
    cls = min(refs, key=lambda c: abs(R4 - refs[c]))
    return cls, float(R4)


def extraction_cost(eps, dist, a: float = 1.0, b: float = 1.0,
                    c: float = 1.0) -> np.ndarray:
    """The law value  Cost = c · ε^{-a} · dist^{-b}."""
    return c * np.asarray(eps, float) ** (-a) * np.asarray(dist, float) ** (-b)


def fit_law(costs, eps, dist) -> tuple[float, float, float]:
    """Fit (a, b, c) of Cost ~ c·ε^{-a}·dist^{-b} by log-log least squares."""
    L = np.log(np.asarray(costs, float))
    A = np.column_stack([-np.log(eps), -np.log(dist), np.ones_like(L)])
    a, b, logc = np.linalg.lstsq(A, L, rcond=None)[0]
    return float(a), float(b), float(np.exp(logc))
