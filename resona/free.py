"""
resona.free — free probability: free cumulants, freeness, cross-moments.

The canonical coordinates of the response algebra are the FREE cumulants κ_n
(Voiculescu / Speicher).  They linearize composition: κ_n(A⊞B) = κ_n(A) + κ_n(B),
the moment↔cumulant map running over NON-CROSSING partitions (the 2κ₂² vs the
classical 3κ₂² is exactly the free-vs-classical signature).

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
