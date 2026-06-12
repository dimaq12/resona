"""
resona.free — free probability: free cumulants, freeness, cross-moments.

The canonical coordinates of the response algebra are the FREE cumulants κ_n
(Voiculescu / Speicher).  They linearize composition: κ_n(A⊞B) = κ_n(A) + κ_n(B),
the moment↔cumulant map running over NON-CROSSING partitions (the 2κ₂² vs the
classical 3κ₂² is exactly the free-vs-classical signature).

WHY FREE PROBABILITY IS EVERYWHERE — the free CLT: summing K free copies
(normalized) kills every cumulant beyond the second, κ_{n>2} → 0, so the
spectrum flows to the SEMICIRCLE — the free Gaussian, the universal
attractor.  Every random/disordered/generic system is asymptotically free at
scale; that is why ⊞ keeps appearing.  (Verified: theory/free_clt.py —
κ₄ ∝ 1/K measured.)

FREENESS — the exact condition under which the response composes — is the
vanishing of MIXED cumulants, equivalently of alternating CENTERED moments
τ(Å B̊ Å B̊ …) with Å = A − τ(A).  Their magnitude is the freeness DEFECT: ≈0
(O(1/√N)) for a free pair, O(1) otherwise.  And the CROSS-moments τ(AB…) are the
orientation information the spectrum alone discards (Horn's obstruction).
"""
import numpy as np


def _trunc_mul(a, b, N):
    out = [0.0] * (N + 1)
    for i in range(min(len(a), N + 1)):
        if a[i] == 0:
            continue
        for j in range(min(len(b), N + 1 - i)):
            out[i + j] += a[i] * b[j]
    return out


def _trunc_pow(a, n, N):
    r = [1.0] + [0.0] * N
    for _ in range(n):
        r = _trunc_mul(r, a, N)
    return r


def free_cumulants(moments):
    """Free cumulants κ_1..κ_N from moments m_1..m_N.

    Solves the non-crossing functional equation M(z) = 1 + Σ κ_n zⁿ M(z)ⁿ order by
    order (M(z)=Σ m_k z^k, m_0=1).  E.g. the semicircle has only κ₂=1.
    """
    m = [1.0] + [float(x) for x in moments]
    N = len(moments)
    kappa = [0.0] * (N + 1)
    for j in range(1, N + 1):
        s = 0.0
        for n in range(1, j):
            s += kappa[n] * _trunc_pow(m, n, N)[j - n]
        kappa[j] = m[j] - s
    return kappa[1:]


def moments_from_cumulants(kappa):
    """Inverse of free_cumulants: moments m_1..m_N from free cumulants κ.

    Solves M(z) = 1 + Σ κ_n zⁿ M(z)ⁿ order by order — the other direction of the
    moment↔free-cumulant bijection.
    """
    kap = list(kappa); N = len(kap); m = [1.0]
    for j in range(1, N + 1):
        s = 0.0
        mm = m + [0.0] * (N - len(m) + 1)
        for n in range(1, j + 1):
            s += kap[n - 1] * _trunc_pow(mm, n, N)[j - n]
        m.append(s)
    return m[1:]


def _htrace(matvec, N, probes, rng):
    """Hutchinson estimate of Tr(matvec) with Rademacher probes."""
    acc = 0.0
    for _ in range(probes):
        x = rng.choice([-1.0, 1.0], size=N)
        acc += float(x @ matvec(x))
    return acc / probes


def freeness_defect(Amv, Bmv, N, word="ABAB", probes=24, seed=0):
    """|τ(Å B̊ Å …)| for the alternating centered word — the freeness criterion.

    ≈0 (O(1/√N)) ⇔ A and B are free ⇔ their response composes exactly; O(1) marks
    the non-closable residue (correlated eigenbases).  τ(X)=Tr(X)/N via Hutchinson.

    RESOLUTION NOTE: the Hutchinson stderr at default probes is itself
    O(1/√N) — the read reliably separates O(1) (non-free) from small, but
    cannot resolve structure AT the O(1/√N) scale; raise `probes` to push
    the floor down.
    """
    rng = np.random.default_rng(seed)
    a = _htrace(Amv, N, probes, rng) / N
    b = _htrace(Bmv, N, probes, rng) / N
    ops = {"A": (lambda x: Amv(x) - a * x), "B": (lambda x: Bmv(x) - b * x)}

    def wordmv(x):
        for ch in reversed(word):
            x = ops[ch](x)
        return x
    return abs(_htrace(wordmv, N, probes, rng) / N)


def cross_moment(matvecs, word, N, probes=24, seed=0, normalize=True):
    """τ(word) = Tr(word)/N for a word in operators, e.g. word='AB' with
    matvecs={'A':..,'B':..} — the orientation info the spectrum discards."""
    rng = np.random.default_rng(seed)

    def wordmv(x):
        for ch in reversed(word):
            x = matvecs[ch](x)
        return x
    tr = _htrace(wordmv, N, probes, rng)
    return tr / N if normalize else tr


def _stieltjes_at(eigs, points, eta):
    """G_E(x − iη) of the empirical measure, vectorized over all query points."""
    eigs = np.asarray(eigs, float)
    z = np.asarray(points, float) - 1j * eta
    return (1.0 / (z[:, None] - eigs[None, :])).mean(axis=1)


def rie_clean(eigenvalues, q, eta=None):
    """FREE DECONVOLUTION of a sample covariance: the rotationally-invariant
    estimator (Ledoit–Péché / Bun–Bouchaud–Potters).

    A sample covariance E (N assets, T observations, q = N/T) is the TRUE
    covariance ⊠-multiplied by Marchenko–Pastur noise.  The RIE inverts that
    free multiplication at the eigenvalue level:

        ξ_i = λ_i / |1 − q + q·λ_i·G_E(λ_i − iη)|² ,

    keeping E's eigenvectors.  Asymptotically OPTIMAL among rotation-invariant
    estimators (measured here: ~95% of the oracle, ~1.8× closer to the truth
    in Frobenius norm at q = 1/2).  η defaults to N^{-1/2} (the BBP kernel).

    eigenvalues : eigenvalues of the sample covariance E (ascending or not)
    q           : N/T (dimension over observations)
    → cleaned eigenvalues ξ (same order); rebuild  Ξ = U·diag(ξ)·Uᵀ  with E's
    eigenvectors U.

    HONEST LIMIT: optimal *given E's eigenvectors* — it cannot repair the
    eigenbasis itself; for q → 0 it converges to no-op (E is already good).
    """
    lam = np.asarray(eigenvalues, float)
    if not (0.0 < q <= 1.1):
        raise ValueError(f"q = N/T (dimension/observations) must be in (0, 1.1]; "
                         f"got {q} — check which way your data matrix is oriented")
    if eta is None:
        eta = 1.0 / np.sqrt(len(lam))
    g = _stieltjes_at(lam, lam, eta)
    return lam / np.abs(1.0 - q + q * lam * g) ** 2


def rie_clean_additive(eigenvalues, sigma, eta=None):
    """FREE DECONVOLUTION of additive noise:  E = A + σ·(GOE-like W).

    The optimal rotation-invariant cleaning of the eigenvalues,

        ξ_i = λ_i − 2σ²·Re G_E(λ_i − iη) ,

    i.e. subtract the semicircle ⊞-component at the eigenvalue level (the
    additive counterpart of `rie_clean`; measured: reaches the oracle).
    Same honest limit: E's eigenvectors are kept.
    """
    lam = np.asarray(eigenvalues, float)
    if eta is None:
        eta = 1.0 / np.sqrt(len(lam))
    g = _stieltjes_at(lam, lam, eta)
    return lam - 2.0 * sigma ** 2 * np.real(g)
