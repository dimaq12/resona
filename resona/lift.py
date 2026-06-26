"""
resona.lift — the LIFT: make a nonlinear / composed map LINEAR in a lifted basis.

"A shock / nonlinearity is a SUM OF LINEARITIES."  Three concrete lifts:

• R-TRANSFORM  R(w) = G⁻¹(w) − 1/w   (G = Cauchy transform of the spectrum).
  It linearizes FREE ADDITION: R_{A⊞B} = R_A + R_B — the Cole–Hopf of free
  probability.  A spectral SHOCK (band edge) is smooth and additive in R.
• S-TRANSFORM  S(w)  linearizes FREE MULTIPLICATION: S_{A⊠B} = S_A · S_B
  (products of operators — e.g. deep-net weight products).
• CARLEMAN lift — for a polynomial vector field ẋ = Σ c_k xᵏ, the monomials
  z_j = xʲ evolve LINEARLY ż = M z (truncated): solve a nonlinear ODE by one
  matrix exponential exp(tM)·z0 (resona.apply).  Over a finite field GF(p) the
  same lift makes ANY logic function an EXACT linear polynomial (x^p ≡ x).

And one BRIDGE: `koopman` — data → operator (the data-driven Carleman): a
trajectory matrix becomes the ACTION of its own propagator, so the whole
library's spectral reads land on dynamics data.
"""
from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np

Spectrum = Union["object", tuple]
ArrayLike = Union[float, Sequence[float], np.ndarray]


def _nw(s):
    if hasattr(s, "nodes"):
        return np.asarray(s.nodes, float), np.asarray(s.weights, float)
    nodes, weights = s
    return np.asarray(nodes, float), np.asarray(weights, float)


def cauchy(s: Spectrum, z: complex) -> complex:
    """Stieltjes/Cauchy transform G(z) = Σ w_i/(z − λ_i) of a spectrum."""
    nodes, w = _nw(s); w = w / w.sum()
    return np.sum(w / (z - nodes))


def r_transform(s: Spectrum, w: ArrayLike, tol: float = 1e-7) -> Union[float, np.ndarray]:
    """R-transform R(w) = G⁻¹(w) − 1/w (scalar or array w>0). R_{A⊞B}=R_A+R_B.

    Vectorized bisection on the monotone G (55 halvings → machine-tight for
    float64), all query points solved at once.  After the bisection the root
    residual |G(z*) − w| is checked against `tol·max(1,|w|)`; a target that no
    z* in the domain (z > λmax) can realize raises ValueError instead of
    silently returning a bracket endpoint.

    PRECISION NOTE: the root z* ≈ 1/w is found in absolute coordinates and
    1/w subtracted — for w ≪ 1 this cancels ~|log₁₀ w| digits (R(1e-10)
    carries ~6 significant digits, not 16).  w ≤ 0 returns R(0) = mean (the
    analytic limit at 0⁺; strictly negative w is outside the domain)."""
    nodes, wt = _nw(s); wt = wt / wt.sum(); lam = float(nodes.max())
    wq = np.atleast_1d(np.asarray(w, float))
    out = np.empty(len(wq))
    pos = wq > 0
    out[~pos] = float(np.sum(wt * nodes))                   # R(0) = mean
    if pos.any():
        wp = wq[pos]
        G = lambda z: (wt[None, :] / (z[:, None] - nodes[None, :])).sum(1)
        lo = np.full(len(wp), lam + 1e-12)
        hi = np.full(len(wp), lam + 1.0)
        for _ in range(200):                                # bracket: G↓ from +∞ to 0
            mask = G(hi) > wp
            if not mask.any():
                break
            hi[mask] = lam + (hi[mask] - lam) * 2.0 + 1.0
        for _ in range(55):                                 # bisection, machine-tight
            mid = 0.5 * (lo + hi)
            high = G(mid) > wp                              # root right of mid
            lo = np.where(high, mid, lo)
            hi = np.where(high, hi, mid)
        mid = 0.5 * (lo + hi)
        if np.any(np.abs(G(mid) - wp) > tol * np.maximum(1.0, np.abs(wp))):
            raise ValueError("r_transform: target out of achievable range "
                             "(no z > λmax realizes this w)")
        out[pos] = mid - 1.0 / wp
    return float(out[0]) if np.isscalar(w) else out


def s_transform(s: Spectrum, w: ArrayLike, tol: float = 1e-7) -> Union[float, np.ndarray]:
    """S-transform (positive spectrum). S_{A⊠B}=S_A·S_B. scalar or array w>0.

    Vectorized bisection on the monotone ψ (55 halvings → machine-tight for
    float64).  The bracket (lo, hi) hugs the pole at z = 1/λmax RELATIVELY
    (lo = inv·1e-12, hi = inv·(1−1e-12)) so it never inverts even when λmax is
    huge (inv ≈ 1e-13); the old absolute `inv − 1e-12` flipped lo>hi once
    1/λmax ≤ 1e-12 and returned a silent 2×-wrong answer.  After bisection the
    residual |ψ(z*) − w| is checked against `tol·max(1,|w|)`; an unreachable
    target raises ValueError instead of returning a wrong value."""
    nodes, wt = _nw(s); wt = wt / wt.sum(); inv = 1.0 / float(nodes.max())
    wq = np.atleast_1d(np.asarray(w, float))
    psi = lambda z: (wt[None, :] * nodes[None, :] * z[:, None]
                     / (1 - nodes[None, :] * z[:, None])).sum(1)
    lo = np.full(len(wq), inv * 1e-12)
    hi = np.full(len(wq), inv * (1 - 1e-12))
    assert np.all(hi > lo), "s_transform: degenerate bracket (λmax ≤ 0?)"
    for _ in range(55):                                     # ψ: 0→∞ on (0, 1/λmax)
        mid = 0.5 * (lo + hi)
        low = psi(mid) < wq
        lo = np.where(low, mid, lo)
        hi = np.where(low, hi, mid)
    mid = 0.5 * (lo + hi)
    if np.any(np.abs(psi(mid) - wq) > tol * np.maximum(1.0, np.abs(wq))):
        raise ValueError("s_transform: target out of achievable range")
    out = (1 + wq) / wq * mid
    return float(out[0]) if np.isscalar(w) else out


def r_inverse(s: Spectrum, value: ArrayLike, w_max: float = 1.0,
              n: int = 2001) -> Union[float, np.ndarray]:
    """w such that R(w) = value — the dual of `r_transform`, for spectral
    DESIGN (choose the coordinate w that realizes a target R).  Scalar or
    array `value`.

    Samples R on (0, w_max], requires monotonicity there (raises otherwise —
    the inverse is ill-posed; shrink `w_max`), then bisects to machine
    tightness.  R is analytic with R'(0) = κ₂ > 0, so a monotone window
    always exists near 0."""
    ws = np.linspace(w_max / n, w_max, n)
    rs = r_transform(s, ws)
    return _monotone_invert(lambda w: r_transform(s, w), ws, rs, value,
                            "R", w_max)


def s_inverse(s: Spectrum, value: ArrayLike, w_max: float = 1.0,
              n: int = 2001) -> Union[float, np.ndarray]:
    """w such that S(w) = value — the dual of `s_transform` (positive
    spectra), same contract as `r_inverse`."""
    ws = np.linspace(w_max / n, w_max, n)
    rs = s_transform(s, ws)
    return _monotone_invert(lambda w: s_transform(s, w), ws, rs, value,
                            "S", w_max)


def _monotone_invert(fn, ws, rs, value, name, w_max):
    d = np.diff(rs)
    if not (np.all(d > 0) or np.all(d < 0)):
        raise ValueError(f"{name} is not monotone on (0, {w_max}]: the "
                         "inverse is ill-posed there — shrink w_max")
    sign = 1.0 if d[0] > 0 else -1.0
    v = np.atleast_1d(np.asarray(value, float))
    lo_r, hi_r = min(rs[0], rs[-1]), max(rs[0], rs[-1])
    if v.min() < lo_r or v.max() > hi_r:
        raise ValueError(f"value outside {name}'s range [{lo_r:.6g}, "
                         f"{hi_r:.6g}] on (0, {w_max}]")
    idx = np.searchsorted(sign * rs, sign * v).clip(1, len(ws) - 1)
    lo, hi = ws[idx - 1].astype(float), ws[idx].astype(float)
    for _ in range(60):                                     # machine-tight
        mid = 0.5 * (lo + hi)
        below = sign * np.atleast_1d(fn(mid)) < sign * v
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    out = 0.5 * (lo + hi)
    return float(out[0]) if np.isscalar(value) else out


def free_convolution(sA: Spectrum, sB: Spectrum, order: int = 6) -> np.ndarray:
    """Moments of  A ⊞ B  (free additive convolution) from the two spectra ALONE.

    Composition linearizes in the free cumulants: κ_n(A⊞B) = κ_n(A) + κ_n(B).  So
    the spectrum of the sum is read off WITHOUT a joint matvec — just the two
    measures.  (This is the free-probability theorem behind `Spectral.__add__`,
    here at the measure level instead of re-probing the combined operator.)
    Returns the moments m_1..m_order of A⊞B; feed to resona.beta for a spectrum.

    Accuracy is set by the input moments: exact-in / exact-out, but HIGH orders
    are sensitive — with noisy SLQ input keep order ≲ 4 (or pass accurate moments).
    """
    from .free import free_cumulants, moments_from_cumulants
    nA, wA = _nw(sA); wA = wA / wA.sum()
    nB, wB = _nw(sB); wB = wB / wB.sum()
    mA = [float(np.sum(wA * nA ** n)) for n in range(1, order + 1)]
    mB = [float(np.sum(wB * nB ** n)) for n in range(1, order + 1)]
    kAB = np.array(free_cumulants(mA)) + np.array(free_cumulants(mB))
    return moments_from_cumulants(kAB)


def carleman_scalar(coeffs: ArrayLike, order: int) -> np.ndarray:
    """Carleman matrix M of the scalar polynomial ODE  ẋ = Σ_k coeffs[k]·xᵏ.

    Basis z = (x⁰, x¹, …, x^order) — index j IS the exponent (an (order+1)²
    matrix).  The CONSTANT coordinate z_0 = 1 is carried so the constant term
    c_0·x⁰ of the field is not lost:  ż_0 = 0 (row 0 is zero) and for j ≥ 1
        ż_j = j·xʲ⁻¹·ẋ = j·Σ_k c_k x^{j-1+k} = Σ_k c_k·j·z_{j-1+k}
    (truncated at `order`).  Evolve x(t) by  exp(tM)·z0  (resona.apply,
    hermitian=False) with z0 = (x⁰, …, x^order), reading **x = z_1** (index 1).

    The earlier basis (x¹…x^order) dropped c_0 (col 0 fell under a `1≤col`
    guard), so ż_1 silently lost its constant term; the augmented basis fixes it.

    The truncation is exact only as the order →∞; in practice use it as a small-
    STEP integrator — re-lift z0 = (x^j) from the current x each step dt and take
    exp(dt·M)·z0.  Re-lifting keeps the high modes tiny, so a bounded solution
    (logistic, Riccati, Bernoulli) tracks the exact trajectory to machine
    precision (a Carleman/ETD scheme).
    """
    c = np.asarray(coeffs, float)
    M = np.zeros((order + 1, order + 1))
    for j in range(1, order + 1):                           # row z_j (index j = exponent)
        for k, ck in enumerate(c):
            col = j - 1 + k                                 # z_{j-1+k}, index = exponent
            if 0 <= col <= order and ck != 0.0:
                M[j, col] += ck * j
    return M


def conserved_charge(H: np.ndarray, basis: Sequence[np.ndarray],
                     tol: float = 1e-7) -> tuple:
    """BLIND search for (quasi-)conserved charges: over the span of candidate
    operators {O_a}, find the Q that most nearly commutes with H.

    The constructive side of the lift principle (a lift exists ⟺ enough
    conserved charges ⟺ integrability): the commutator-Gram eigenproblem
        G_ab = Tr([H,O_a]†[H,O_b]),   S_ab = Tr(O_a† O_b),
        G c = μ S c
    has its near-zero eigenvalues exactly at the conserved charges — found with
    NO prior knowledge of what they are.  For an integrable H the search FINDS
    the charges (energy, total spin, free-fermion bilinears…); for a chaotic H
    it honestly reports none beyond H itself.

    H     : dense/sparse square matrix (the Hamiltonian).
    basis : list of candidate operators {O_a} (e.g. k-local Pauli strings).
    Returns (charges, comm_norms): charges[j] = Σ_a c_a O_a (matrices, unit
    Frobenius norm) sorted by comm_norms[j] = ‖[H,Q_j]‖/‖Q_j‖ ascending;
    comm_norms[j] < tol marks a genuine conserved charge.

    HONEST LIMIT: needs the user-supplied basis and forms the commutators —
    O(|basis|·dim²) memory/time; resolve symmetry sectors for spectral use.
    """
    from scipy.linalg import eigh as _geigh
    H = np.asarray(H)
    B = [np.asarray(O) for O in basis]
    C = np.stack([(H @ O - O @ H).ravel() for O in B])      # rows = [H, O_a]
    M = np.stack([O.ravel() for O in B])
    G = (C.conj() @ C.T)                                    # Tr([H,O_a]†[H,O_b])
    S = (M.conj() @ M.T)                                    # Tr(O_a†O_b)
    G = (G + G.conj().T) / 2; S = (S + S.conj().T) / 2
    mu, U = _geigh(G, S + 1e-12 * np.eye(len(B)))
    comm_norms = np.sqrt(np.clip(mu.real, 0.0, None))       # ‖[H,Q]‖/‖Q‖ per optimal Q
    charges = []
    for j in range(len(B)):
        Q = sum(U[a, j] * B[a] for a in range(len(B)))
        n = np.linalg.norm(Q)
        charges.append(Q / n if n > 0 else Q)
    return charges, comm_norms


def _gf_solve(V, y, p):
    """Solve V x = y over GF(p) (p prime) by vectorized Gauss–Jordan (~20× the
    pure-Python loop; the O(M³) work stays in numpy, only the M-pivot loop is Python)."""
    M = len(y)
    A = np.concatenate([np.asarray(V, np.int64) % p,
                        (np.asarray(y, np.int64) % p).reshape(-1, 1)], axis=1)
    for c in range(M):
        nz = np.nonzero(A[c:, c] % p)[0]
        if nz.size == 0:
            continue
        pr = c + nz[0]
        if pr != c:
            A[[c, pr]] = A[[pr, c]]
        A[c] = (A[c] * pow(int(A[c, c] % p), -1, p)) % p
        m = (A[:, c] % p) != 0; m[c] = False
        if m.any():
            A[m] = (A[m] - np.outer(A[m, c], A[c])) % p
    return A[:, -1] % p


def carleman_gf(p: int, n: int, func: Callable) -> tuple:
    """Exact GF(p) Carleman lift of ANY logic map f:{0..p-1}ⁿ→{0..p-1}.

    Since x^p ≡ x (mod p), the monomial basis has exponents in {0..p-1}ⁿ; the
    function becomes an EXACT linear combination f(x) = c·φ(x) over GF(p).
    Returns (coeffs, evaluate) where evaluate(x) reproduces f on every input.
    (The Vandermonde build + solve are numpy-vectorized — pure-Python would be
    ~20× slower; for n beyond ~10 a numba/C++ kernel is a further ~1.7×, rarely
    worth the build — the O(pⁿ·³) work dominates regardless of language.)
    """
    from itertools import product
    exps = np.array(list(product(range(p), repeat=n)), dtype=np.int64)   # (M, n)
    V = np.ones((len(exps), len(exps)), np.int64)                        # V[i,j]=∏ pt_i^e_j
    for d in range(n):
        V = (V * (exps[:, d][:, None] ** exps[:, d][None, :])) % p
    y = np.array([func(tuple(int(v) for v in pt)) for pt in exps], np.int64)
    coeffs = _gf_solve(V, y, p)

    def evaluate(x):
        x = np.asarray(x)
        return int(np.sum(coeffs * np.prod(x[None, :] ** exps, axis=1)) % p)
    return coeffs, evaluate


def koopman(snapshots: np.ndarray, rank: int | None = None,
            rtol: float = 1e-10) -> tuple:
    """DATA → OPERATOR: the action of the Koopman/DMD propagator, never formed.

    `snapshots`: (n_features, n_times) trajectory matrix of a dynamical
    system.  One thin SVD of X₀ = snapshots[:, :-1] gives the least-squares
    propagator  K = X₁ X₀⁺  in its rank-r POD basis:  K̃ = UᵀX₁ V Σ⁻¹  (r×r).
    Returns (matvec, rmatvec, r) — the action of K̃ and K̃ᵀ on R^r — so the
    WHOLE library falls onto data: `cloud(matvec, r)` reads the dynamics'
    frequencies/damping, `cost.is_extractable` grades it, etc.

    THE LIFT VIEW: for nonlinear dynamics, feed lifted observables (delay
    stacks, monomials) as rows — Koopman is the data-driven Carleman.

    HONEST LIMITS: the rank cutoff (singular values below rtol·σ₁ dropped) is
    REPORTED via r — a hard low-rank truncation, not a free lunch; short or
    noisy data make K̃ a least-squares fit, not the true propagator; continuous
    frequencies follow as log(λ)/Δt with the usual aliasing caveats.
    """
    X = np.asarray(snapshots, float)
    X0, X1 = X[:, :-1], X[:, 1:]
    U, sv, Vt = np.linalg.svd(X0, full_matrices=False)
    r = int(np.sum(sv > rtol * sv[0]))
    if rank is not None:
        r = min(r, int(rank))
    U, sv, Vt = U[:, :r], sv[:r], Vt[:r]
    M = U.T @ X1 @ Vt.T / sv[None, :]          # r×r reduced propagator (formed:
    mv = lambda v: M @ v                       # r is DATA-rank, tiny by design)
    rmv = lambda v: M.T @ v
    return mv, rmv, r
