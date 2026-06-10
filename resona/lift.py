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
"""
import numpy as np
from scipy.optimize import brentq


def _nw(s):
    if hasattr(s, "nodes"):
        return np.asarray(s.nodes, float), np.asarray(s.weights, float)
    nodes, weights = s
    return np.asarray(nodes, float), np.asarray(weights, float)


def cauchy(s, z):
    """Stieltjes/Cauchy transform G(z) = Σ w_i/(z − λ_i) of a spectrum."""
    nodes, w = _nw(s); w = w / w.sum()
    return np.sum(w / (z - nodes))


def r_transform(s, w):
    """R-transform R(w) = G⁻¹(w) − 1/w (scalar or array w>0). R_{A⊞B}=R_A+R_B."""
    nodes, wt = _nw(s); wt = wt / wt.sum(); lam = float(nodes.max())

    def R1(wi):
        if wi <= 0:
            return float(np.sum(wt * nodes))                # R(0) = mean
        g = lambda z: float(np.sum(wt / (z - nodes))) - wi
        hz = lam + 1.0
        while float(np.sum(wt / (hz - nodes))) > wi:        # bracket: G↓ from +∞ to 0
            hz = lam + (hz - lam) * 2.0 + 1.0
        z = brentq(g, lam + 1e-12, hz)
        return z - 1.0 / wi
    return R1(float(w)) if np.isscalar(w) else np.array([R1(float(x)) for x in w])


def s_transform(s, w):
    """S-transform (positive spectrum). S_{A⊠B}=S_A·S_B. scalar or array w>0."""
    nodes, wt = _nw(s); wt = wt / wt.sum(); inv = 1.0 / float(nodes.max())

    def S1(wi):
        psi = lambda z: float(np.sum(wt * nodes * z / (1 - nodes * z))) - wi
        z = brentq(psi, 1e-12, inv - 1e-12)                 # ψ: 0→∞ on (0,1/λmax)
        return (1 + wi) / wi * z
    return S1(float(w)) if np.isscalar(w) else np.array([S1(float(x)) for x in w])


def moments_from_cumulants(kappa):
    """Inverse of free.free_cumulants: moments m_1..m_N from free cumulants κ.

    Solves M(z) = 1 + Σ κ_n zⁿ M(z)ⁿ order by order.
    """
    from .free import _trunc_pow
    kap = list(kappa); N = len(kap); m = [1.0]
    for j in range(1, N + 1):
        s = 0.0
        mm = m + [0.0] * (N - len(m) + 1)
        for n in range(1, j + 1):
            s += kap[n - 1] * _trunc_pow(mm, n, N)[j - n]
        m.append(s)
    return m[1:]


def free_convolution(sA, sB, order=6):
    """Moments of  A ⊞ B  (free additive convolution) from the two spectra ALONE.

    Composition linearizes in the free cumulants: κ_n(A⊞B) = κ_n(A) + κ_n(B).  So
    the spectrum of the sum is read off WITHOUT a joint matvec — just the two
    measures.  (This is the free-probability theorem behind `Spectral.__add__`,
    here at the measure level instead of re-probing the combined operator.)
    Returns the moments m_1..m_order of A⊞B; feed to resona.beta for a spectrum.

    Accuracy is set by the input moments: exact-in / exact-out, but HIGH orders
    are sensitive — with noisy SLQ input keep order ≲ 4 (or pass accurate moments).
    """
    from .free import free_cumulants
    nA, wA = _nw(sA); wA = wA / wA.sum()
    nB, wB = _nw(sB); wB = wB / wB.sum()
    mA = [float(np.sum(wA * nA ** n)) for n in range(1, order + 1)]
    mB = [float(np.sum(wB * nB ** n)) for n in range(1, order + 1)]
    kAB = np.array(free_cumulants(mA)) + np.array(free_cumulants(mB))
    return moments_from_cumulants(kAB)


def carleman_scalar(coeffs, order):
    """Carleman matrix M of the scalar polynomial ODE  ẋ = Σ_k coeffs[k]·xᵏ.

    On z = (x¹, …, x^order):  ż_j = j·Σ_k c_k x^{j-1+k} = Σ_k c_k·j·z_{j-1+k}
    (truncated at `order`).  Evolve x(t) by  exp(tM)·z0  (resona.apply,
    hermitian=False), reading x = z_1.

    The truncation is exact only as the order →∞; in practice use it as a small-
    STEP integrator — re-lift z0 = (x^j) from the current x each step dt and take
    exp(dt·M)·z0.  Re-lifting keeps the high modes tiny, so a bounded solution
    (logistic, Riccati, Bernoulli) tracks the exact trajectory to machine
    precision (a Carleman/ETD scheme).
    """
    c = np.asarray(coeffs, float)
    M = np.zeros((order, order))
    for j in range(1, order + 1):                           # row z_j (index j-1)
        for k, ck in enumerate(c):
            col = j - 1 + k                                 # z_{j-1+k} (index col-1)
            if 1 <= col <= order and ck != 0.0:
                M[j - 1, col - 1] += ck * j
    return M


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


def carleman_gf(p, n, func):
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
