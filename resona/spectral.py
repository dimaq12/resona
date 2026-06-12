"""
resona.spectral — the Response transform: the "FFT of operators".

`fft(x)` takes a signal to the basis where convolution becomes pointwise multiply.
`Spectral.of(A)` takes an operator (black-box matvec) to the representation where
COMPOSITION becomes ADDITION (free convolution) — matrix-free — and from which
every spectral functional is read.

One object.  Three verbs:
    s = Spectral.of(matvec, N)     # PROBE   — harvest the field   (like fft)
    s + t   ;   s @ t              # COMPOSE — A+B, A·B  (never formed, never eig'd)
    s.trace(f) ; s.density(x) ; s.extreme() ; s.moment(p)   # READ   (like ifft)

Everything else is read off the same object:
    s.boxplus(t)                   # A ⊞ B at the measure level (no joint matvec)
    s.cauchy(z) ; s.r(w) ; s.s(w) ; s.cumulants()   # the lifted coordinates
    s.flow(t, xs) ; s.shock_time()                  # free heat flow / disorder average
    s.levels(N)                    # whole spectrum from 4 numbers (Beta closure)
    s.effective_rank() ; s.condition()              # the honest cost dials
"""
from __future__ import annotations
import numpy as np

__all__ = ["Spectral", "apply", "quadform", "local_spectrum", "local_density", "from_measure", "from_eigenbasis"]


def _supports_block(matvec, N):
    """True iff `matvec` correctly maps an (N, m) block column-wise.

    Verified, not assumed: one block call on a 2-column test block must return
    shape (N, 2) with column 0 equal to the single-vector result.  Costs 2 extra
    matvecs — negligible next to the k·probes harvest, and it buys BLAS-3.
    """
    x = np.cos(np.arange(N) * 0.7) + 0.1            # deterministic, generic test vectors
    X = np.column_stack([x, np.sin(np.arange(N) * 0.3) - 0.2])
    try:
        Y = np.asarray(matvec(X.copy()))
        if Y.shape != X.shape or not np.issubdtype(Y.dtype, np.number):
            return False
        y0 = np.asarray(matvec(x.copy()))
        return y0.shape == (N,) and np.allclose(Y[:, 0], y0, rtol=1e-10, atol=1e-300)
    except Exception:
        return False


def _lanczos_block(matvec, V0, k):
    """`probes` independent Lanczos recurrences advanced together — one BLOCK
    matvec per step (BLAS-3) instead of `probes` single matvecs.  Identical math
    per probe (same probe vectors, full reorthogonalization); returns the list
    of (alpha, beta) pairs, one per probe."""
    N, p = V0.shape
    Q = V0 / np.linalg.norm(V0, axis=0)
    V = np.zeros((p, k, N)); V[:, 0, :] = Q.T                    # (p, k, N): each probe's
    al = np.zeros((p, k)); be = np.zeros((p, max(k - 1, 1)))     # basis rows CONTIGUOUS
    Qprev = np.zeros((N, p)); b = np.zeros(p)
    m = np.full(p, k); active = np.ones(p, bool)
    for j in range(k):
        W = matvec(Q) - b * Qprev
        a_j = np.einsum('np,np->p', Q, W)
        al[:, j] = np.where(active, a_j, al[:, j])
        W = W - a_j * Q
        for i in range(p):                                       # full reorth against
            Vi = V[i, :j + 1]                                    # each probe's OWN basis
            W[:, i] -= Vi.T @ (Vi @ W[:, i])                     # (contiguous BLAS-2)
        if j < k - 1:
            nb = np.linalg.norm(W, axis=0)
            newly_done = active & (nb < 1e-12)
            m[newly_done] = j + 1
            active = active & ~newly_done
            be[:, j] = np.where(active, nb, 0.0)
            Qn = W / np.where(nb < 1e-12, 1.0, nb)
            Qn[:, ~active] = 0.0
            Qprev, Q = Q, Qn
            b = be[:, j]
            V[:, j + 1, :] = Q.T
    return [(al[i, :m[i]], be[i, :m[i] - 1]) for i in range(p)]


def _lanczos(matvec, v0, k):
    """k-step Lanczos with full reorthogonalization. Returns (alpha, beta)."""
    N = len(v0)
    V = np.zeros((N, k)); alpha = np.zeros(k); beta = np.zeros(max(k - 1, 0))
    q = v0 / np.linalg.norm(v0); V[:, 0] = q; qprev = np.zeros(N); b = 0.0
    for j in range(k):
        w = matvec(q) - b * qprev
        alpha[j] = float(q @ w)
        w = w - alpha[j] * q
        w -= V[:, :j + 1] @ (V[:, :j + 1].T @ w)
        if j < k - 1:
            b = float(np.linalg.norm(w))
            beta[j] = b
            if b < 1e-12:
                return alpha[:j + 1], beta[:j]
            qprev, q = q, w / b
            V[:, j + 1] = q
    return alpha, beta


def _lanczos_herm(matvec, v0, k):
    """k-step Lanczos for a complex HERMITIAN operator, full reorthogonalization.

    The recurrence runs in complex arithmetic, but Hermiticity makes the Jacobi
    tridiagonal REAL (α real by ⟨q, Hq⟩ ∈ ℝ, β ≥ 0) — so everything downstream
    (eigh of T, nodes/weights) is unchanged.  Returns (alpha, beta), real.
    """
    N = len(v0)
    V = np.zeros((N, k), complex)
    alpha = np.zeros(k); beta = np.zeros(max(k - 1, 0))
    q = np.asarray(v0, complex); q = q / np.linalg.norm(q)
    V[:, 0] = q; qprev = np.zeros(N, complex); b = 0.0
    for j in range(k):
        w = np.asarray(matvec(q), complex) - b * qprev
        alpha[j] = float(np.real(np.vdot(q, w)))
        w = w - alpha[j] * q
        w -= V[:, :j + 1] @ (V[:, :j + 1].conj().T @ w)
        if j < k - 1:
            b = float(np.linalg.norm(w))
            beta[j] = b
            if b < 1e-12:
                return alpha[:j + 1], beta[:j]
            qprev, q = q, w / b
            V[:, j + 1] = q
    return alpha, beta


def _is_complex_operator(matvec, N):
    """One deterministic matvec: does the operator live on C^N?"""
    x = np.cos(np.arange(N) * 0.7) + 0.1
    try:
        return bool(np.iscomplexobj(matvec(x)))
    except Exception:
        return True              # real test vector rejected → assume complex domain


# f-families with sign-definite high derivatives on (0, ∞) — the Golub–Meurant
# certificate set: (callable, which side plain Gauss lands on, which endpoint
# the Radau rule must pin).  log/sqrt: f^(2k)<0 → Gauss OVER, Radau(a) under.
# inv: f^(2k)>0, f^(2k+1)<0 → Gauss UNDER, Radau(a) over.  exp: all derivatives
# positive → Gauss UNDER, Radau(b) over.
_FAMS = {
    "log":  (np.log,            "over",  "a"),
    "sqrt": (np.sqrt,           "over",  "a"),
    "inv":  (lambda x: 1.0 / x, "under", "a"),
    "exp":  (np.exp,            "under", "b"),
}


def _resolve_f(f):
    """Accept a callable or a family name ('log', 'inv', 'sqrt', 'exp')."""
    if callable(f):
        return f, None
    if f in _FAMS:
        return _FAMS[f][0], f
    raise ValueError(f"unknown f family {f!r}; use a callable or one of {list(_FAMS)}")


def _gauss_value(al, be, f):
    kk = len(al)
    T = np.diag(al) + np.diag(be[:kk - 1], 1) + np.diag(be[:kk - 1], -1)
    th, S = np.linalg.eigh(T)
    return float(np.sum(S[0, :] ** 2 * f(th))), float(th[0]), float(th[-1])


def _radau_value(al, be, f, end):
    """Gauss–Radau with the node pinned at `end`, built from the STORED
    recurrence: T_{k-1} extended by one row via β_{k-1} (Golub–Meurant) —
    zero extra matvecs."""
    kk = len(al)
    T = np.diag(al[:kk - 1]) + np.diag(be[:kk - 2], 1) + np.diag(be[:kk - 2], -1)
    rhs = np.zeros(kk - 1); rhs[-1] = be[kk - 2] ** 2
    delta = np.linalg.solve(T - end * np.eye(kk - 1), rhs)
    al_ext = np.append(al[:kk - 1], end + delta[-1])
    T2 = np.diag(al_ext) + np.diag(be[:kk - 1], 1) + np.diag(be[:kk - 1], -1)
    th, S = np.linalg.eigh(T2)
    return float(np.sum(S[0, :] ** 2 * f(th)))


def _bracket(al, be, fam, support):
    """Certified [lo, hi] for ∫f dμ of ONE unit-mass tridiagonal measure."""
    func, side, which = _FAMS[fam]
    if len(al) < 3:
        raise ValueError("certified bounds need k >= 3 Lanczos steps")
    g, th_lo, th_hi = _gauss_value(al, be, func)
    a, b = (support if support is not None else (None, None))
    if which == "a":
        if a is None:
            raise ValueError(f"f={fam!r} needs support=(a, ...) with a <= λ_min "
                             "(e.g. a known shift/regularizer)")
        if a >= th_lo or (fam in ("log", "sqrt", "inv") and a <= 0):
            raise ValueError(f"support endpoint a={a} is not strictly below the "
                             f"observed spectrum (Ritz min {th_lo:.6g}) and positive")
        r = _radau_value(al, be, func, a)
    else:
        if b is None:
            raise ValueError(f"f={fam!r} needs support=(..., b) with b >= λ_max")
        if b <= th_hi:
            raise ValueError(f"support endpoint b={b} is not strictly above the "
                             f"observed spectrum (Ritz max {th_hi:.6g})")
        r = _radau_value(al, be, func, b)
    return (min(g, r), max(g, r))


def quadform(matvec, f, v, k: int = 48, certified: bool = False, support=None):
    """vᵀ f(A) v — the quadratic-form read (one Lanczos from v, matrix-free).

    The certified twin of `apply`: with ``certified=True`` and f given as a
    family name ('log', 'inv', 'sqrt', 'exp'), returns a RIGOROUS bracket
    (lo, hi) via Gauss–Radau (Golub–Meurant) — and since a quadratic form has
    NO stochastic probes, the bracket is an unconditional certificate, subject
    only to the stated support endpoint (a <= λ_min for log/inv/sqrt — often
    known structurally, e.g. a regularizer; b >= λ_max for exp).

    Money case: GP posterior variance kᵀ(K+σ²I)⁻¹k with support=(σ², None).
    """
    func, fam = _resolve_f(f)
    v = np.asarray(v)
    nv2 = float(np.real(np.vdot(v, v)))
    if np.iscomplexobj(v) or _is_complex_operator(matvec, len(v)):
        al, be = _lanczos_herm(matvec, np.asarray(v, complex), k)
    else:
        al, be = _lanczos(matvec, np.asarray(v, float), k)
    if not certified:
        val, _, _ = _gauss_value(al, be, func)
        return nv2 * val
    if fam is None:
        raise ValueError("certified=True needs f as a family name "
                         f"({list(_FAMS)}), not a callable — derivative signs "
                         "must be known for the bracket to be a theorem")
    lo, hi = _bracket(al, be, fam, support)
    return nv2 * lo, nv2 * hi


class Spectral:
    """The response of an operator: Ritz nodes (the 'frequencies') and quadrature
    weights (the 'amplitudes'), harvested matrix-free by stochastic Lanczos
    quadrature.  Carries an optional matvec so compositions stay matrix-free.
    """

    def __init__(self, nodes, weights, matvec=None, N=None, probe_sizes=None):
        self.nodes = np.asarray(nodes, float)
        self.weights = np.asarray(weights, float)
        self.matvec = matvec
        self.N = N
        self.probe_sizes = probe_sizes      # per-probe node counts (error bars)
        self._tridiags = None               # per-probe (α, β) when built by of()

    # ── PROBE (forward transform) ────────────────────────────────────────────
    @classmethod
    def of(cls, matvec, N, k=48, probes=8, seed=0):
        """Harvest the response of the operator given by `matvec` (acts on R^N,
        or on C^N for a complex HERMITIAN operator — detected automatically;
        complex probes are used and the read side is unchanged).

        Stochastic Lanczos quadrature: `probes` random vectors, `k` Lanczos steps
        each.  Cost O(probes * k) matvecs.  No matrix is formed, no eig is called.

        If `matvec` also maps (N, m) blocks column-wise (verified automatically,
        e.g. a dense/sparse ``A @ x``), all probes advance in ONE block matvec
        per step — BLAS-3, ~3× faster, the same probe vectors and the same math.
        """
        rng = np.random.default_rng(seed)
        if _is_complex_operator(matvec, N):
            # complex HERMITIAN operator: complex Gaussian probes, complex
            # Lanczos — the tridiagonal (and the whole read side) stays real
            tridiags = [_lanczos_herm(matvec,
                                      (rng.standard_normal(N)
                                       + 1j * rng.standard_normal(N)) / np.sqrt(2), k)
                        for _ in range(probes)]
        # block pays when the matvec dominates the per-step bookkeeping: measured
        # crossover ~N=1000 (4× faster at N≥2000, ~even below)
        elif probes > 1 and N >= 1000 and _supports_block(matvec, N):
            V0 = rng.standard_normal((probes, N)).T          # same draws as the loop
            tridiags = _lanczos_block(matvec, V0, k)
        else:
            tridiags = [_lanczos(matvec, rng.standard_normal(N), k)
                        for _ in range(probes)]
        nodes, weights = [], []
        for al, be in tridiags:
            kk = len(al)
            T = np.diag(al) + np.diag(be[:kk - 1], 1) + np.diag(be[:kk - 1], -1)
            theta, S = np.linalg.eigh(T)
            nodes.append(theta)
            weights.append(S[0, :] ** 2 / probes)
        obj = cls(np.concatenate(nodes), np.concatenate(weights), matvec, N,
                  probe_sizes=[len(th) for th in nodes])
        obj._tridiags = tridiags          # per-probe (α, β): certificates, zoom
        return obj

    # ── COMPOSE (the free-convolution theorem: hard op → linear, matrix-free) ──
    def __add__(self, other: "Spectral") -> "Spectral":
        """A + B — never forms A+B, never diagonalizes; uses (A+B)x = Ax + Bx."""
        self._require_matvec(other)
        return Spectral.of(lambda x: self.matvec(x) + other.matvec(x), self.N)

    def __matmul__(self, other: "Spectral") -> "Spectral":
        """A · B — composition of the actions."""
        self._require_matvec(other)
        return Spectral.of(lambda x: self.matvec(other.matvec(x)), self.N)

    def reprobe(self, g_matvec) -> "Spectral":
        """f(A): supply a matvec for f(A) (shift, power, etc.) — the pushforward.

        (Distinct from ``resona.apply(matvec, f, v)``, which computes f(A)·v on a
        single vector; this re-harvests the whole response of the new operator.)
        """
        return Spectral.of(g_matvec, self.N)

    apply = reprobe          # back-compat alias; prefer `reprobe`

    def boxplus(self, other: "Spectral", order: int = 6, as_spectral: bool = False):
        """A ⊞ B — free additive convolution at the MEASURE level: moments of the
        sum from the two spectra alone, no joint matvec (κ_n add).  Exact when
        A, B are free; with noisy SLQ moments keep order ≲ 4.  Returns moments
        m_1..m_order; with ``as_spectral=True``, the Gauss-quadrature `Spectral`
        those moments determine (Golub–Welsch: an ⌊order/2⌋-point measure that
        reproduces m_1..m_{order-1} exactly, the top moment entering only the
        construction — the honest content of the data; its `extreme()` are
        inner quadrature nodes, which UNDERSHOOT the true support edges).
        For the matvec-level composition use
        ``s + t`` (re-probes the actual sum — exact regardless of freeness)."""
        from .lift import free_convolution
        m = free_convolution(self, other, order=order)
        if not as_spectral:
            return m
        nodes, weights = _gauss_from_moments(m)
        return Spectral(nodes, weights, matvec=None, N=self.N)

    # ── READ (inverse transform: any spectral functional) ─────────────────────
    def trace(self, f, with_err: bool = False, certified: bool = False,
              support=None):
        """Tr f(A) = N · E[f(λ)] = N · Σ w·f(node).

        ``with_err=True`` → (value, stderr): the standard error of the
        stochastic estimate, read from the scatter of the independent probes
        (free — no extra matvecs).  The honest number, with its honest ±.

        ``certified=True`` (f as a family name: 'log', 'inv', 'sqrt', 'exp';
        plus ``support=(a, b)`` with the needed endpoint) → (lo, hi), the
        Gauss–Radau bracket (Golub–Meurant) of the K-TRUNCATION error: the
        fully-converged SLQ value of these same probes provably lies inside.
        It does NOT certify the Monte-Carlo probe scatter — that is a separate
        error source, reported by ``with_err``; the two are stated apart by
        design.  For an unconditional certificate of a probe-free quantity,
        see `quadform`.
        """
        f, fam = _resolve_f(f)
        if certified:
            if fam is None:
                raise ValueError("certified=True needs f as a family name "
                                 f"({list(_FAMS)}), not a callable")
            if not self._tridiags:
                raise ValueError("certificates need the probe recurrences: "
                                 "build this object with Spectral.of")
            los, his = zip(*(_bracket(al, be, fam, support)
                             for al, be in self._tridiags))
            return float(self.N) * float(np.mean(los)), \
                   float(self.N) * float(np.mean(his))
        vals = self.weights * f(self.nodes)
        total = float(self.N) * float(np.sum(vals))
        if not with_err:
            return total
        if not self.probe_sizes or len(self.probe_sizes) < 2:
            raise ValueError("error bars need the probe structure from "
                             "Spectral.of with probes >= 2")
        p = len(self.probe_sizes)
        ests, i = [], 0
        for sz in self.probe_sizes:
            ests.append(float(self.N) * p * float(np.sum(vals[i:i + sz])))
            i += sz
        stderr = float(np.std(ests, ddof=1) / np.sqrt(p))
        return total, stderr

    def moment(self, p: int, with_err: bool = False):
        """Tr(A^p).  ``with_err=True`` → (value, stderr), see `trace`."""
        return self.trace(lambda x: x ** p, with_err=with_err)

    def density(self, xs, eta: float = 0.1) -> np.ndarray:
        """Density of states ρ(x), Lorentzian-broadened by `eta`."""
        xs = np.atleast_1d(np.asarray(xs, float))
        return (self.weights[None, :] * (eta / np.pi)
                / ((xs[:, None] - self.nodes[None, :]) ** 2 + eta ** 2)).sum(1)

    def extreme(self) -> tuple[float, float]:
        """Extreme eigenvalues (Lanczos resolves these first / most reliably)."""
        return float(self.nodes.min()), float(self.nodes.max())

    # ── TRANSFORMS (the lifted coordinates, read off the measure) ─────────────
    def cauchy(self, z):
        """Stieltjes/Cauchy transform G(z) = Σ w_i/(z − λ_i)."""
        from .lift import cauchy
        return cauchy(self, z)

    def r(self, w):
        """R-transform R(w) = G⁻¹(w) − 1/w — linearizes ⊞: R_{A⊞B} = R_A + R_B
        (the Cole–Hopf of free probability).  Scalar or array w > 0."""
        from .lift import r_transform
        return r_transform(self, w)

    def s(self, w):
        """S-transform — linearizes ⊠: S_{A⊠B} = S_A · S_B (positive spectrum).
        Scalar or array w > 0.  (`s.s(w)` reads oddly on a variable named `s` —
        `s_transform` is the same method under its full name.)"""
        from .lift import s_transform
        return s_transform(self, w)

    r_transform = r          # full-name aliases (same methods; pick what reads
    s_transform = s          # better at the call site)

    def cumulants(self, order: int = 6):
        """Free cumulants κ_1..κ_order — the canonical coordinates in which
        composition is ADDITION (κ_n(A⊞B) = κ_n(A) + κ_n(B))."""
        from .free import free_cumulants
        m = [self.moment(p) / self.N for p in range(1, order + 1)]
        return free_cumulants(m)

    # ── FLOW / DISORDER (the resolvent fixed point, vectorized) ───────────────
    def flow(self, t, xs, eta: float = 1e-3, g0=None):
        """Density of μ_t = μ_0 ⊞ semicircle(variance t) — the free heat flow,
        equivalently the disorder average of A + √t·GOE.  Closed form via the
        Pastur fixed point; a shock = a spectral edge / band merger."""
        from .flow import burgers_density
        return burgers_density(self, t, xs, eta=eta, g0=g0)

    def shock_time(self, **kw):
        """The band-merger / shock time t_c of the free heat flow (None if the
        gap never fills before t_max)."""
        from .flow import shock_time
        return shock_time(self, **kw)

    def zoom(self, a, b, k: int = 48, probes: int = 8, degree: int = 200,
             seed: int = 0) -> "Spectral":
        """The spectral measure INSIDE the window [a, b] — interior eigenvalues
        at full resolution, matrix-free (needs the carried matvec).

        Chebyshev spectrum slicing: probes are filtered by a Jackson-damped
        polynomial approximation p(A) of the window indicator (`degree`
        matvecs per probe, applied by the bare three-term recurrence — no
        inner Lanczos), then the standard Lanczos harvest runs from the
        filtered probes, so its k-resolution is spent INSIDE the window
        instead of on the global extremes.

        Returns a `Spectral` whose weights carry the window measure, normalized
        by the filter's exactly-known in-window p² gain — so
        ``N * z.weights[(a<=z.nodes)&(z.nodes<=b)].sum()`` estimates the
        EIGENVALUE COUNT in the window.  Polish individual nodes with
        `solve.rayleigh_polish` for machine precision.

        HONEST LIMIT: the filter has a transition band of width
        ~(λmax−λmin)·π/degree around each edge — eigenvalues there enter with
        partial weight; raise `degree` to sharpen.
        """
        if self.matvec is None:
            raise ValueError("zoom needs the carried matvec")
        lo, hi = self.extreme()
        pad = 0.05 * (hi - lo) + 1e-12
        lo, hi = lo - pad, hi + pad
        c, w_half = 0.5 * (hi + lo), 0.5 * (hi - lo)
        # Chebyshev coefficients of the indicator of [a, b], Jackson-damped
        th = np.linspace(0.0, np.pi, 4096)
        x = np.cos(th)
        ind = ((c + w_half * x >= a) & (c + w_half * x <= b)).astype(float)
        j = np.arange(degree + 1)
        coef = np.trapezoid(ind[None, :] * np.cos(j[:, None] * th[None, :]),
                            th, axis=1) * (2.0 / np.pi)
        coef[0] *= 0.5
        g = ((degree - j + 1) * np.cos(np.pi * j / (degree + 1))
             + np.sin(np.pi * j / (degree + 1)) / np.tan(np.pi / (degree + 1)))
        coef *= g / (degree + 1)                                  # Jackson kernel
        p_grid = np.polynomial.chebyshev.chebval(x, coef)         # the filter, exactly
        in_win = ind > 0
        gain = float(np.mean(p_grid[in_win] ** 2))                # in-window p² mass

        Amv = self.matvec
        tilde = lambda v: (Amv(v) - c * v) / w_half               # map to [-1, 1]
        rng = np.random.default_rng(seed)
        nodes, weights, sizes, tris = [], [], [], []
        for _ in range(probes):
            v = rng.standard_normal(self.N)
            t0, t1 = v, tilde(v)
            wf = coef[0] * t0 + coef[1] * t1
            for jj in range(2, degree + 1):                       # T_{j+1} = 2ÃT_j − T_{j−1}
                t0, t1 = t1, 2.0 * tilde(t1) - t0
                wf = wf + coef[jj] * t1
            mass = float(wf @ wf) / float(v @ v)                  # ≈ ∫p² dμ
            al, be = _lanczos(Amv, wf, k)
            kk = len(al)
            T = np.diag(al) + np.diag(be[:kk - 1], 1) + np.diag(be[:kk - 1], -1)
            theta, S = np.linalg.eigh(T)
            nodes.append(theta)
            weights.append(S[0, :] ** 2 * (mass / gain) / probes)
            sizes.append(len(theta)); tris.append((al, be))
        out = Spectral(np.concatenate(nodes), np.concatenate(weights),
                       self.matvec, self.N, probe_sizes=sizes)
        out._tridiags = tris
        return out

    # ── CLOSURE (whole spectrum from a few numbers) ────────────────────────────
    def levels(self, N: int = None):
        """All N eigenvalues from FOUR numbers — support + first two moments —
        via the maximum-entropy Beta closure (smooth/local operators).  O(N),
        two factors of N below dense diagonalization."""
        from .beta import beta_from
        return beta_from(self, N)

    # ── COST dial (the harvestable structure) ─────────────────────────────────
    def condition(self, polish: bool = False) -> float:
        """Condition number κ = λ_max/λ_min for a POSITIVE-definite operator.

        Read from `extreme()` — so it is a LOWER BOUND on the true κ, and the
        gap can be large: when λ_min hides in a dense near-singular tail the
        Ritz value lands far above the true edge (a 1e9-conditioned Wishart
        reads as ~1e3), and NO local refinement can recover an edge Lanczos
        never saw.  Trustworthy for well-separated edges.  ``polish=True``
        (needs the carried matvec) refines both Ritz edges to true nearby
        eigenvalues via `solve.rayleigh_polish` — exact for an isolated small
        edge, still a lower bound for a tail.  Returns inf if the spectrum
        touches 0.
        """
        lo, hi = self.extreme()
        if polish:
            if self.matvec is None:
                raise ValueError("condition(polish=True) needs the carried matvec")
            from .solve import rayleigh_polish
            hi = max(hi, rayleigh_polish(self.matvec, hi, N=self.N))
            lo_p = rayleigh_polish(self.matvec, lo, N=self.N)
            if 0.0 < lo_p:
                lo = min(lo, lo_p)
        return float(hi / lo) if lo > 0 else float("inf")

    def effective_rank(self) -> float:
        """Φ₁ = Tr(A)² / Tr(A²) — participation ratio / effective number of modes,
        for POSITIVE-SEMIDEFINITE operators (covariance, kernel, Hessian).  Low Φ₁
        ⇒ structured/cheap; high ⇒ near the genuine frontier.  (Computed from the
        trace moments, which SLQ estimates robustly.)"""
        m1, m2 = self.moment(1), self.moment(2)
        return float(m1 * m1 / m2) if m2 > 0 else 1.0

    # ── internals ─────────────────────────────────────────────────────────────
    def _require_matvec(self, other):
        if self.matvec is None or other.matvec is None or self.N != other.N:
            raise ValueError("composition needs both operands' matvecs on the same R^N")

    def __repr__(self):
        lo, hi = self.extreme()
        return f"Spectral(N={self.N}, support=[{lo:.3g}, {hi:.3g}], eff_rank={self.effective_rank():.1f})"


def _gauss_from_moments(moments):
    """Golub–Welsch: raw moments m_1..m_2n (m_0=1) → the n-point Gauss
    quadrature (nodes, weights) reproducing them exactly.

    Hankel Cholesky → three-term recurrence (α, β) → Jacobi eigh.  If the
    Hankel matrix is not numerically PD (noisy moments), the point count is
    reduced until it is — the honest resolution the data supports.
    """
    m = [1.0] + [float(x) for x in moments]
    n = len(moments) // 2
    while n >= 1:
        H = np.array([[m[i + j] for j in range(n + 1)] for i in range(n + 1)])
        try:
            R = np.linalg.cholesky(H).T
        except np.linalg.LinAlgError:
            n -= 1
            continue
        alpha = np.empty(n); beta = np.empty(max(n - 1, 0))
        for k in range(n):
            alpha[k] = (R[k, k + 1] / R[k, k]
                        - (R[k - 1, k] / R[k - 1, k - 1] if k > 0 else 0.0))
        for k in range(1, n):
            beta[k - 1] = R[k, k] / R[k - 1, k - 1]
        T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
        theta, S = np.linalg.eigh(T)
        return theta, S[0, :] ** 2
    raise ValueError("moments do not determine a positive measure")


# ── APPLY (matrix function on a vector — the universal solve/evolve engine) ───
def apply(matvec, f, v, k: int = 48, hermitian: bool = True):
    """f(A) · v, matrix-free — the universal engine (not just spectra).

    Any scalar function of an operator applied to a vector, from matvecs only:
        f = 1/λ        →  A⁻¹·v          (SOLVE a linear system, matrix-free)
        f = exp(tλ)    →  exp(tA)·v      (EVOLVE: heat / diffusion / dynamics)
        f = exp(-itλ)  →  exp(-itA)·v    (quantum / wave propagation; complex f)
        f = a low-pass →  filter(A)·v    (DENOISE a signal on any operator)
        f = √λ, sign…  →  √A·v, projectors (whiten, sample, split spectrum)
    Lift a NONLINEAR problem to a linear operator K (Carleman / Koopman /
    Cole–Hopf) and evolve ``u(t)=exp(tK)·u0`` — so this also solves nonlinear
    ODE/PDE on the manifold where they linearize.

    hermitian=True : symmetric/self-adjoint A — Lanczos (cheapest, real f).
    hermitian=False: GENERAL (non-symmetric / non-normal) A — Arnoldi, and f may
                     be COMPLEX (e.g. exp(-itλ)).  The result is returned complex
                     if f or A drives it complex.  This is what makes resona a
                     general solver, not a spectra-only tool.
    """
    v = np.asarray(v, complex if not hermitian else float)
    nv = np.linalg.norm(v)
    if nv == 0:
        return v.copy()
    N = len(v)

    if hermitian:                                       # ── Lanczos (symmetric) ──
        V = np.zeros((N, k)); al = np.zeros(k); be = np.zeros(k)
        q = v.real / nv; V[:, 0] = q; qprev = np.zeros(N); b = 0.0; m = k
        for j in range(k):
            w = matvec(q) - b * qprev
            al[j] = float(q @ w)
            w = w - al[j] * q
            w -= V[:, :j + 1] @ (V[:, :j + 1].T @ w)    # full reorth
            if j < k - 1:
                b = float(np.linalg.norm(w))
                if b < 1e-12:
                    m = j + 1; break
                be[j] = b; qprev, q = q, w / b; V[:, j + 1] = q
        T = np.diag(al[:m]) + np.diag(be[:m - 1], 1) + np.diag(be[:m - 1], -1)
        theta, S = np.linalg.eigh(T)
        return nv * (V[:, :m] @ (S @ (np.asarray(f(theta), float) * S[0, :])))

    # ── Arnoldi (general / non-symmetric, possibly complex) ──
    Q = np.zeros((N, k + 1), complex); H = np.zeros((k + 1, k), complex)
    Q[:, 0] = v / nv; m = k
    for j in range(k):
        w = np.asarray(matvec(Q[:, j]), complex)
        for i in range(j + 1):                          # modified Gram–Schmidt
            H[i, j] = np.vdot(Q[:, i], w); w -= H[i, j] * Q[:, i]
        h = np.linalg.norm(w); H[j + 1, j] = h
        if h < 1e-12:
            m = j + 1; break
        Q[:, j + 1] = w / h
    Hm = H[:m, :m]
    vals, W = np.linalg.eig(Hm)                          # f(Hm)·e1 via eig of small Hm
    c = np.linalg.solve(W, np.eye(m, 1)[:, 0].astype(complex))
    fHe1 = W @ (np.asarray(f(vals), complex) * c)
    out = nv * (Q[:, :m] @ fHe1)
    return out


# ── LOCAL response (probe the operator from a CHOSEN vector, not random) ──────
def local_spectrum(matvec, v, k: int = 48):
    """The local spectral measure seen from vector v:  μ_v = Σ_i |⟨v|ψ_i⟩|² δ(λ_i).

    One Lanczos from v → Ritz nodes + first-component² weights.  This is the
    vector-resolved response; with v = e_i it is the LOCAL density of states
    (LDOS) at site i.  ``Spectral.of`` averages this over random v (→ the trace);
    here you choose v.  Returns (nodes, weights), weights summing to ‖v‖²-norm 1.
    """
    if np.iscomplexobj(v) or _is_complex_operator(matvec, len(v)):
        al, be = _lanczos_herm(matvec, np.asarray(v, complex), k)
    else:
        al, be = _lanczos(matvec, np.asarray(v, float), k)
    kk = len(al)
    T = np.diag(al) + np.diag(be[:kk - 1], 1) + np.diag(be[:kk - 1], -1)
    theta, S = np.linalg.eigh(T)
    return theta, S[0, :] ** 2


def local_density(matvec, v, xs, k: int = 48, eta: float = 0.1):
    """LDOS / vector-resolved density  ρ_v(x) = Σ_i |⟨v|ψ_i⟩|² · L_η(x − λ_i),
    a Lorentzian-smoothed local_spectrum (matrix-free, one Lanczos from v)."""
    theta, w = local_spectrum(matvec, v, k)
    xs = np.asarray(xs, float)
    return (w[None, :] * (eta / np.pi)
            / ((xs[:, None] - theta[None, :]) ** 2 + eta ** 2)).sum(1)


def from_measure(nodes, weights, k=None):
    """The INVERSE response transform: a spectral measure (nodes, weights) → the
    Jacobi (tridiagonal) operator whose e₀-measure it is, as (α, β) — diagonal and
    POSITIVE off-diagonal.

    Run Lanczos on diag(nodes) from the start vector √weights — the exact inverse
    of `Spectral.of`'s matrix→measure direction (the Stieltjes / Gauss-quadrature
    construction; `of` ∘ `from_measure` = identity on the (α,β)).  Recovers a
    well-conditioned / smooth operator to ~machine precision; the long recurrence
    is genuinely ill-conditioned when the weights span many orders of magnitude
    (sharp operators) — the inverse spectral problem's intrinsic difficulty.
    """
    lam = np.asarray(nodes, float); w = np.asarray(weights, float)
    return _lanczos(lambda x: lam * x, np.sqrt(np.clip(w, 0.0, None)), k or len(lam))


def from_eigenbasis(eigenvalues, eigenvectors):
    """Reconstruct a Jacobi (tridiagonal) operator's band from its FULL
    eigendecomposition A = V·diag(λ)·Vᵀ — EXACT (machine precision) for ANY
    operator, sharp or smooth, because it reads the tridiagonal entries of VΛVᵀ
    directly:

        diag_j = Σ_i λ_i v_i[j]² ,    off_j = Σ_i λ_i v_i[j] v_i[j+1].

    Unlike `from_measure` (ONE boundary probe — compressed, but ill-conditioned
    for sharp operators because far-from-boundary modes are invisible), this uses
    the FULL eigenbasis (every eigenvector component), so it is well-conditioned
    everywhere — at the cost of needing the whole spectral data, not just the
    boundary measure.  Returns (diag, off-diagonal).
    """
    lam = np.asarray(eigenvalues, float); V = np.asarray(eigenvectors, float)
    return (V * V) @ lam, (V[:-1, :] * V[1:, :]) @ lam
