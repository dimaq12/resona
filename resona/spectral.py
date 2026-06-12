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
    xp = _xp(V0)
    if xp is not None:                  # device block → device recurrence
        return _lanczos_block_xp(matvec, V0, k, xp)
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


def _lanczos_core(matvec, q, k, cplx):
    """THE Lanczos loop (full reorthogonalization), consolidated in 2.0 —
    one implementation behind `_lanczos`, `_lanczos_herm` AND `apply`'s
    Hermitian branches (formerly three inline copies; the gallery ratchet
    is the bit-parity certificate of the merge).

    `q` must be UNIT (callers own the normalization — that is what keeps
    each call site bit-identical to its pre-merge inline loop).
    Returns (alpha[:m], beta[:m−1], V[:, :m])."""
    N = len(q)
    V = np.zeros((N, k), complex if cplx else float)
    al = np.zeros(k); be = np.zeros(k)
    V[:, 0] = q
    qprev = np.zeros(N, complex if cplx else float); b = 0.0; m = k
    for j in range(k):
        if cplx:
            w = np.asarray(matvec(q), complex) - b * qprev
            al[j] = float(np.real(np.vdot(q, w)))
        else:
            w = matvec(q) - b * qprev
            al[j] = float(q @ w)
        w = w - al[j] * q
        w -= V[:, :j + 1] @ ((V[:, :j + 1].conj().T if cplx else V[:, :j + 1].T) @ w)
        if j < k - 1:
            b = float(np.linalg.norm(w))
            if b < 1e-12:
                m = j + 1
                break
            be[j] = b
            qprev, q = q, w / b
            V[:, j + 1] = q
    return al[:m], be[:m - 1], V[:, :m]


def _lanczos(matvec, v0, k):
    """k-step Lanczos with full reorthogonalization. Returns (alpha, beta)."""
    xp = _xp(v0)
    if xp is not None:                  # device vector → device recurrence
        return _lanczos_xp(matvec, v0, k, xp)
    al, be, _ = _lanczos_core(matvec, v0 / np.linalg.norm(v0), k, False)
    return al, be


def _lanczos_herm(matvec, v0, k):
    """k-step Lanczos for a complex HERMITIAN operator, full reorthogonalization.

    The recurrence runs in complex arithmetic, but Hermiticity makes the Jacobi
    tridiagonal REAL (α real by ⟨q, Hq⟩ ∈ ℝ, β ≥ 0) — so everything downstream
    (eigh of T, nodes/weights) is unchanged.  Returns (alpha, beta), real.
    """
    xp = _xp(v0)
    if xp is not None:                  # device vector → device recurrence
        return _lanczos_xp(matvec, xp.as_complex(v0), k, xp)
    q = np.asarray(v0, complex)
    al, be, _ = _lanczos_core(matvec, q / np.linalg.norm(q), k, True)
    return al, be


def _topk_pairs(matvec, N, K, extra=None, seed=0):
    """Top-K Ritz pairs (values, vectors) from one Lanczos with a stored basis
    (extremes converge first — Kaniel–Paige).  Used by `of(deflate=K)`."""
    extra = extra if extra is not None else max(16, K // 2)
    k = K + extra
    rng = np.random.default_rng(seed + 7919)
    V = np.zeros((N, k)); al = np.zeros(k); be = np.zeros(k)
    q = rng.standard_normal(N); q /= np.linalg.norm(q)
    V[:, 0] = q; qp = np.zeros(N); b = 0.0
    for j in range(k):
        w = matvec(q) - b * qp
        al[j] = float(q @ w); w = w - al[j] * q
        w -= V[:, :j + 1] @ (V[:, :j + 1].T @ w)
        if j < k - 1:
            b = float(np.linalg.norm(w))
            if b < 1e-12:
                k = j + 1; break
            be[j] = b; qp, q = q, w / b; V[:, j + 1] = q
    T = np.diag(al[:k]) + np.diag(be[:k - 1], 1) + np.diag(be[:k - 1], -1)
    th, S = np.linalg.eigh(T)
    idx = np.argsort(th)[-K:]
    return th[idx], V[:, :k] @ S[:, idx]


def _kpm_tridiags(matvec, N, k, probes, rng, block):
    """KPM harvest: Chebyshev moments (NO reorthogonalization — the O(N·k²)
    term vanishes, stable at thousands of moments), Jackson-damped density per
    probe, Gauss nodes via the measure's own recurrence (`from_measure`).
    Returns the per-probe (α, β) list, same contract as the Lanczos engines."""
    M = max(4 * k, 256)                                   # moment count
    s0_tris = [_lanczos(matvec, rng.standard_normal(N), 12) for _ in range(2)]
    ths = np.concatenate([np.linalg.eigvalsh(np.diag(al) + np.diag(be[:len(al) - 1], 1)
                                             + np.diag(be[:len(al) - 1], -1))
                          for al, be in s0_tris])
    lo, hi = ths.min(), ths.max()
    pad = 0.06 * (hi - lo) + 1e-12
    lo, hi = lo - pad, hi + pad
    c, h = 0.5 * (hi + lo), 0.5 * (hi - lo)
    jj = np.arange(M + 1)
    g = ((M - jj + 1) * np.cos(np.pi * jj / (M + 1))
         + np.sin(np.pi * jj / (M + 1)) / np.tan(np.pi / (M + 1))) / (M + 1)
    G = 2048
    xs = np.cos((np.arange(G) + 0.5) * np.pi / G)
    Tj = np.cos(jj[:, None] * np.arccos(xs[None, :]))
    Vp = rng.standard_normal((N, probes))
    Vp /= np.linalg.norm(Vp, axis=0)
    MU = np.empty((M + 1, probes))
    if block:
        tilde = lambda X: (matvec(X) - c * X) / h
        T0, T1 = Vp, tilde(Vp)
        MU[0] = np.einsum('np,np->p', Vp, T0)
        MU[1] = np.einsum('np,np->p', Vp, T1)
        for m in range(2, M + 1):
            T0, T1 = T1, 2.0 * tilde(T1) - T0
            MU[m] = np.einsum('np,np->p', Vp, T1)
    else:
        tilde = lambda v: (matvec(v) - c * v) / h
        for p in range(probes):
            v = Vp[:, p]
            t0, t1 = v, tilde(v)
            MU[0, p] = v @ t0; MU[1, p] = v @ t1
            for m in range(2, M + 1):
                t0, t1 = t1, 2.0 * tilde(t1) - t0
                MU[m, p] = v @ t1
    tridiags = []
    for p in range(probes):
        coef = MU[:, p] * g; coef[1:] *= 2.0
        w = np.clip((coef[:, None] * Tj).sum(0), 0.0, None) / G
        w /= w.sum()
        tridiags.append(from_measure(c + h * xs, w, k=k))
    return tridiags


def _is_complex_operator(matvec, N):
    """One deterministic matvec: does the operator live on C^N?"""
    return _probe_operator(matvec, N)[0]


# ── array-namespace dispatch: invisible GPU/device support ───────────────────
# A matvec that consumes and returns torch / cupy / array-API arrays runs the
# SAME recurrences on its own device — zero new API.  Dispatch happens strictly
# BEFORE any arithmetic: numpy inputs take the exact code paths elsewhere in
# this file, bit-identically.  torch/cupy are referenced only here, and only
# when the user's arrays already are torch/cupy (so the import is a dict
# lookup, never a new dependency).

def _xp(x):
    """The array-namespace adapter for x, or None when x is numpy's business."""
    if isinstance(x, np.ndarray) or not hasattr(x, "shape"):
        return None
    ns = getattr(x, "__array_namespace__", None)
    if ns is not None:
        mod = ns()
        return None if mod is np else _XP(mod)   # numpy scalars carry it too
    mod = type(x).__module__.partition(".")[0]
    if mod in ("torch", "cupy"):                 # tensors predate the standard
        return _XP(__import__(mod))
    return None


class _XP:
    """The handful of operations the device-side recurrences need, expressed
    once over any array-API-style namespace (torch, cupy, array_api_strict…).
    Vectors live on the device; each method returning a Python scalar is a
    deliberate, tiny device→host sync — exactly the (α, β) coefficients."""

    def __init__(self, mod):
        self.mod = mod

    def _make(self, fn, x, dtype, like):
        try:
            return fn(x, dtype=dtype, device=like.device)
        except TypeError:                        # namespaces without device=
            return fn(x, dtype=dtype)

    def iscomplex(self, x):
        m = self.mod
        return x.dtype == getattr(m, "complex64", None) or x.dtype == m.complex128

    def cdtype(self, like):
        m = self.mod
        if self.iscomplex(like):
            return like.dtype
        return m.complex64 if like.dtype == m.float32 else m.complex128

    def asarray(self, x, like):
        """Host (numpy) → like's device; dtype follows `like` (fp32 stays fp32)."""
        dt = self.cdtype(like) if np.iscomplexobj(x) else like.dtype
        return self._make(self.mod.asarray, x, dtype=dt, like=like)

    def zeros(self, shape, like):
        return self._make(self.mod.zeros, shape, dtype=like.dtype, like=like)

    def as_complex(self, x):
        if self.iscomplex(x):
            return x
        dt, astype = self.cdtype(x), getattr(self.mod, "astype", None)
        if astype is not None:                   # array API
            return astype(x, dt)
        return x.to(dt) if hasattr(x, "to") else x.astype(dt)   # torch / cupy

    def conj(self, x):
        return self.mod.conj(x) if self.iscomplex(x) else x

    def norm(self, x):
        """‖x‖₂ → Python float."""
        try:
            return float(self.mod.linalg.vector_norm(x))
        except (AttributeError, TypeError):
            return float(self.mod.sum(self.mod.abs(x) ** 2)) ** 0.5

    def rdot(self, q, w):
        """Re⟨q, w⟩ → Python float."""
        s = self.mod.sum(self.conj(q) * w)
        return float(self.mod.real(s)) if self.iscomplex(s) else float(s)

    def vdot(self, q, w):
        """⟨q, w⟩ (conjugate-linear in q) → Python complex."""
        return complex(self.mod.sum(self.conj(q) * w))

    def host(self, x):
        """Small array → host numpy (the k×k eigh and (α, β) live on the host)."""
        if hasattr(x, "get"):                    # cupy
            return np.asarray(x.get())
        if hasattr(x, "detach"):                 # torch (also under autograd)
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        try:
            return np.asarray(x)
        except Exception:
            return np.from_dlpack(x)

    def colsum(self, X):
        """Σ over axis 0 → host numpy (one (p,)-sized sync per Lanczos step)."""
        return self.host(self.mod.sum(X, axis=0))

    def colnorms(self, X):
        return np.sqrt(self.colsum(X * X))


def _probe_operator(matvec, N):
    """One deterministic matvec: is the operator complex, and on which array
    namespace does it live?  Returns (is_complex, xp, ref); xp is None on the
    numpy path (then everything downstream is today's exact code) and ref is a
    sample output carrying the operator's device and dtype."""
    x = np.cos(np.arange(N) * 0.7) + 0.1
    try:
        y = matvec(x)
    except Exception:
        y = _device_probe(matvec, x)
        xp = None if y is None else _xp(y)
        if xp is None:
            return True, None, None  # real test vector rejected → assume complex domain
        return xp.iscomplex(y), xp, y
    xp = _xp(y)
    if xp is None:
        return bool(np.iscomplexobj(y)), None, None
    # confirm on-device: a numpy probe can be silently PROMOTED (a float32
    # torch matrix @ float64 numpy vector yields float64 — but the same matrix
    # rejects a float64 tensor), so re-run the test vector as a device array
    # and let the output fix the operator's true dtype/device.
    try:
        y = matvec(xp.asarray(x, like=y))
    except Exception:
        y = _device_probe(matvec, x)
        if y is None or _xp(y) is None:
            return True, None, None
        xp = _xp(y)
    return xp.iscomplex(y), xp, y


def _device_probe(matvec, x):
    """The numpy test vector was rejected: if torch/cupy is ALREADY loaded the
    matvec may simply insist on its own array type — retry on that device."""
    import sys
    for name in ("torch", "cupy"):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        for dt in (None, getattr(mod, "float32", None),
                   getattr(mod, "complex128", None), getattr(mod, "complex64", None)):
            try:
                return matvec(mod.asarray(x) if dt is None else mod.asarray(x, dtype=dt))
            except Exception:
                pass
    return None


def _lanczos_xp(matvec, v0, k, xp):
    """`_lanczos`/`_lanczos_herm` with the vectors living on v0's device.
    Same recurrence (full reorthogonalization), real or complex-Hermitian by
    dtype; only the scalar Jacobi coefficients (α, β) cross to the host, where
    the small k×k eigh already lives.  Returns host (alpha, beta), real."""
    N = v0.shape[0]
    V = xp.zeros((k, N), like=v0)                # device basis, rows contiguous
    alpha = np.zeros(k); beta = np.zeros(max(k - 1, 0))
    q = v0 / xp.norm(v0)
    V[0, :] = q
    qprev = xp.zeros((N,), like=v0); b = 0.0
    for j in range(k):
        w = matvec(q) - b * qprev
        a = xp.rdot(q, w); alpha[j] = a
        w = w - a * q
        Vj = V[:j + 1, :]
        w = w - Vj.T @ (xp.conj(Vj) @ w)         # full reorth, on-device
        if j < k - 1:
            b = xp.norm(w); beta[j] = b
            if b < 1e-12:
                return alpha[:j + 1], beta[:j]
            qprev, q = q, w / b
            V[j + 1, :] = q
    return alpha, beta


def _supports_block_xp(matvec, N, xp, ref):
    """`_supports_block` with the test block living on ref's device."""
    x = np.cos(np.arange(N) * 0.7) + 0.1
    X = np.column_stack([x, np.sin(np.arange(N) * 0.3) - 0.2])
    try:
        Y = matvec(xp.asarray(X, like=ref))
        if tuple(getattr(Y, "shape", ())) != X.shape:
            return False
        y0 = matvec(xp.asarray(x, like=ref))
        return tuple(y0.shape) == (N,) and \
            xp.norm(Y[:, 0] - y0) <= 1e-10 * max(1.0, xp.norm(y0))
    except Exception:
        return False


def _lanczos_block_xp(matvec, V0, k, xp):
    """`_lanczos_block` with the block on V0's device: one block matvec per
    step, per-probe reorthogonalization on-device; only the per-step (p,)
    coefficient rows sync to the host.  Same math per probe."""
    N, p = V0.shape
    Q = V0 / xp.asarray(xp.colnorms(V0), like=V0)
    Vd = xp.zeros((p, k, N), like=V0); Vd[:, 0, :] = Q.T
    al = np.zeros((p, k)); be = np.zeros((p, max(k - 1, 1)))
    Qprev = xp.zeros((N, p), like=V0); b = np.zeros(p)
    m = np.full(p, k); active = np.ones(p, bool)
    for j in range(k):
        W = matvec(Q) - xp.asarray(b, like=V0) * Qprev
        a_j = xp.colsum(Q * W)
        al[:, j] = np.where(active, a_j, al[:, j])
        W = W - xp.asarray(a_j, like=V0) * Q
        for i in range(p):                       # reorth against each probe's
            Vi = Vd[i, :j + 1, :]                # OWN basis (contiguous rows)
            W[:, i] = W[:, i] - Vi.T @ (Vi @ W[:, i])
        if j < k - 1:
            nb = xp.colnorms(W)
            newly_done = active & (nb < 1e-12)
            m[newly_done] = j + 1
            active = active & ~newly_done
            be[:, j] = np.where(active, nb, 0.0)
            Qn = (W / xp.asarray(np.where(nb < 1e-12, 1.0, nb), like=V0)) \
                * xp.asarray(active.astype(float), like=V0)
            Qprev, Q = Q, Qn
            b = be[:, j]
            Vd[:, j + 1, :] = Q.T
    return [(al[i, :m[i]], be[i, :m[i] - 1]) for i in range(p)]


def _harvest_xp(matvec, N, k, probes, rng, is_c, xp, ref):
    """`of`'s harvest on ref's device: the SAME numpy probe draws as the cpu
    path, moved across once; the block matvec is used when the device matvec
    maps (N, m) blocks (verified on-device)."""
    if is_c:
        return [_lanczos_xp(matvec,
                            xp.asarray((rng.standard_normal(N)
                                        + 1j * rng.standard_normal(N)) / np.sqrt(2),
                                       like=ref), k, xp)
                for _ in range(probes)]
    Z = rng.standard_normal((probes, N))         # rows = the sequential draws
    if probes > 1 and _supports_block_xp(matvec, N, xp, ref):
        return _lanczos_block_xp(matvec, xp.asarray(Z.T, like=ref), k, xp)
    return [_lanczos_xp(matvec, xp.asarray(Z[i], like=ref), k, xp)
            for i in range(probes)]


def _complexified(matvec, xp):
    """The matvec on complex device vectors.  Some backends (torch) refuse a
    real operator on a complex vector; linearity fixes it at the cost of two
    matvecs per step: A(x + iy) = A·x + i·A·y.  Probed once, on first use."""
    accepts_complex = []
    def mvc(u):
        if not accepts_complex:
            try:
                w = matvec(u)
                accepts_complex.append(True)
                return w
            except Exception:
                accepts_complex.append(False)
        elif accepts_complex[0]:
            return matvec(u)
        return matvec(xp.mod.real(u)) + 1j * matvec(xp.mod.imag(u))
    return mvc


def _apply_xp(matvec, f, v, k, hermitian, xp):
    """`apply` with v on its own device: the Krylov basis and every O(N) op
    stay on the device; only the small projected eigenproblem visits the host.
    Returns an array on v's device (complex for the Arnoldi path)."""
    nv = xp.norm(v)
    if nv == 0:
        return v * 1.0
    N = v.shape[0]
    if hermitian:                                # ── Lanczos (self-adjoint) ──
        q = v / nv
        if xp.iscomplex(v):                      # complex v on a possibly-real
            mv = _complexified(matvec, xp)       # A: linearity if A rejects C^N
            w0 = xp.as_complex(mv(q))
        else:
            mv = matvec                          # the first step doubles as the
            try:                                 # complex-HERMITIAN probe: a
                w0 = mv(q)                       # complex A may reject a real q
            except Exception:
                v = xp.as_complex(v); q = xp.as_complex(q)
                w0 = mv(q)
            if xp.iscomplex(w0) and not xp.iscomplex(v):  # …or promote it —
                v = xp.as_complex(v); q = xp.as_complex(q)  # complex Lanczos,
            if xp.iscomplex(v):                  # REAL tridiagonal (the host
                mv = _complexified(matvec, xp)   # contract, on the device)
                w0 = xp.as_complex(w0)
        V = xp.zeros((k, N), like=v)
        al = np.zeros(k); be = np.zeros(k); m = k
        V[0, :] = q
        qprev = xp.zeros((N,), like=v); b = 0.0
        cplx = xp.iscomplex(v)
        for j in range(k):
            w = w0 if j == 0 else (xp.as_complex(mv(q)) if cplx else mv(q))
            w = w - b * qprev
            a = xp.rdot(q, w); al[j] = a
            w = w - a * q
            Vj = V[:j + 1, :]
            w = w - Vj.T @ (xp.conj(Vj) @ w)     # full reorth
            if j < k - 1:
                b = xp.norm(w)
                if b < 1e-12:
                    m = j + 1; break
                be[j] = b; qprev, q = q, w / b; V[j + 1, :] = q
        T = np.diag(al[:m]) + np.diag(be[:m - 1], 1) + np.diag(be[:m - 1], -1)
        theta, S = np.linalg.eigh(T)
        # complex f on a real spectrum (e^{-iHt}) is fine when v or A is
        # complex — exactly the host contract; the real path casts to float
        fth = np.asarray(f(theta)) if xp.iscomplex(v) \
            else np.asarray(f(theta), float)
        y = S @ (fth * S[0, :])
        return nv * (V[:m, :].T @ xp.asarray(y, like=v))
    v = xp.as_complex(v)                         # ── Arnoldi (general) ──
    Q = xp.zeros((k + 1, N), like=v)
    H = np.zeros((k + 1, k), complex); m = k
    Q[0, :] = v / nv
    mvc = _complexified(matvec, xp)
    for j in range(k):
        w = xp.as_complex(mvc(Q[j, :]))
        for i in range(j + 1):                   # modified Gram–Schmidt
            hij = xp.vdot(Q[i, :], w); H[i, j] = hij
            w = w - hij * Q[i, :]
        h = xp.norm(w); H[j + 1, j] = h
        if h < 1e-12:
            m = j + 1; break
        Q[j + 1, :] = w / h
    Hm = H[:m, :m]
    vals, W = np.linalg.eig(Hm)                  # f(Hm)·e1 via eig of small Hm
    c = np.linalg.solve(W, np.eye(m, 1)[:, 0].astype(complex))
    fHe1 = W @ (np.asarray(f(vals), complex) * c)
    return nv * (Q[:m, :].T @ xp.asarray(fHe1, like=v))


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


def _as_operator(A, N=None):
    """(matvec, N) from whatever the caller has: a bare matvec callable, a
    scipy.sparse.linalg.LinearOperator, a scipy sparse matrix, or an ndarray.

    The ONE interop shim (1.4.x): anything with a `.shape` is treated as an
    operator object and applied via `.matvec`/`@`; a bare callable stays the
    bare callable it always was (bit-identical path) and needs N."""
    if callable(A) and not hasattr(A, "shape"):
        if N is None:
            raise ValueError("a bare matvec callable needs N (the dimension); "
                             "operator objects (LinearOperator / sparse / "
                             "ndarray) carry it in .shape")
        return A, int(N)
    n = int(A.shape[0])
    if N is not None and int(N) != n:
        raise ValueError(f"N={N} contradicts the operator's shape {A.shape}")
    mv = A.matvec if hasattr(A, "matvec") else (lambda v, A=A: A @ v)
    return mv, n


def quadform(matvec, f, v, k: int = 48, certified: bool = False, support=None):
    """vᵀ f(A) v — the quadratic-form read (one Lanczos from v, matrix-free).

    `matvec` may be a callable, a scipy LinearOperator, a sparse matrix, or a
    dense array — anywhere resona takes an operator, all four work.

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
    matvec, _ = _as_operator(matvec, len(v))
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
        self._slq_scale = 1.0               # complement mass (deflate)
        self._n_atoms = 0                   # exact deflated atoms at the tail

    # ── PROBE (forward transform) ────────────────────────────────────────────
    @classmethod
    def of(cls, matvec, N=None, k=48, probes=8, seed=0, engine="lanczos", deflate=0):
        """Harvest the response of the operator given by `matvec` (acts on R^N,
        or on C^N for a complex HERMITIAN operator — detected automatically;
        complex probes are used and the read side is unchanged).

        `matvec` may equally be a scipy LinearOperator, a sparse matrix, or a
        dense array — then N is read off `.shape` and may be omitted
        (``resona.of(L)``).  A bare callable still needs N.

        Stochastic Lanczos quadrature: `probes` random vectors, `k` Lanczos steps
        each.  Cost O(probes * k) matvecs.  No matrix is formed, no eig is called.

        If `matvec` also maps (N, m) blocks column-wise (verified automatically,
        e.g. a dense/sparse ``A @ x``), all probes advance in ONE block matvec
        per step — BLAS-3, ~3× faster, the same probe vectors and the same math.

        ``deflate=K`` (Hutch++ at the measure level): the top-K Ritz pairs are
        captured EXACTLY (one padded Lanczos) and enter as atoms; the probes are
        projected onto the complement.  For spiked operators and top-weighted f
        the trace variance collapses (measured: 63× on Tr A², 724× on Tr e^A);
        for f that flattens the top (log) the gain is ~1 — `effective_rank` is
        the dial that predicts it.  Real symmetric operators only.

        ``engine="kpm"``: Chebyshev/Jackson harvest — no reorthogonalization
        (the O(N·k²) term vanishes), 4·k moments, the same `Spectral` out (the
        measure's own recurrence via `from_measure`).  Wins at HIGH resolution
        (k ≳ 200, measured 2.7× at k=256 with equal-or-better moments); edges
        are smoothed by ~span/(4k) (Jackson) — certificates and `extreme` near
        sharp edges want the default engine.  Real symmetric only.

        GPU / device arrays are invisible: if `matvec` consumes and returns
        torch / cupy / array-API arrays (detected from its output on one test
        vector), the same probe draws are moved to that device once and the
        whole recurrence runs there — zero new API, and the returned object is
        the usual host-side Spectral (so `trace(certified=)` works as usual).
        DEVICE COVERAGE TODAY: this default path (engine="lanczos", deflate=0)
        and `apply`; `engine="kpm"` / `deflate` raise on device operators, and
        `zoom` / `quadform` are host-only for now.  HONEST PRECISION NOTE:
        float32-default backends (torch's default dtype) give SLQ only ~2–3
        significant digits; build the operator on float64 tensors for the
        usual accuracy (measured: a float64 torch matvec agrees with the numpy
        path to ~1e-12).  The numpy path itself is untouched, bit-identically.
        """
        matvec, N = _as_operator(matvec, N)
        rng = np.random.default_rng(seed)
        is_c, xp, ref = _probe_operator(matvec, N)
        if xp is not None and (deflate or engine != "lanczos"):
            raise ValueError(
                "device (torch/cupy/array-API) operators cover the default "
                "path only — engine='lanczos', deflate=0; kpm and deflate "
                "are host-only for now")
        mv, atoms, scale, proj = matvec, None, 1.0, None
        if deflate:
            if engine != "lanczos":
                raise ValueError("deflate requires the lanczos engine")
            if is_c:
                raise ValueError("deflate: real symmetric operators only (for now)")
            lam, W = _topk_pairs(matvec, N, deflate, seed=seed)
            proj = lambda X: X - W @ (W.T @ X)
            mv = lambda X: proj(matvec(proj(X)))
            atoms, scale = lam, (N - deflate) / N
        certifiable = True
        if xp is not None:
            # device operator (torch/cupy/array-API): same draws, on-device;
            # the harvested (α, β) live on the host → certificates work
            tridiags = _harvest_xp(matvec, N, k, probes, rng, is_c, xp, ref)
        elif engine == "kpm":
            if is_c:
                raise ValueError("engine='kpm': real symmetric operators only")
            tridiags = _kpm_tridiags(mv, N, k, probes, rng,
                                     block=_supports_block(mv, N))
            certifiable = False                # KPM recurrences describe the
        elif engine != "lanczos":              # SMOOTHED measure — no theorems
            raise ValueError(f"unknown engine {engine!r}: 'lanczos' or 'kpm'")
        elif is_c:
            # complex HERMITIAN operator: complex Gaussian probes, complex
            # Lanczos — the tridiagonal (and the whole read side) stays real
            tridiags = [_lanczos_herm(mv,
                                      (rng.standard_normal(N)
                                       + 1j * rng.standard_normal(N)) / np.sqrt(2), k)
                        for _ in range(probes)]
        # block pays when the matvec dominates the per-step bookkeeping: measured
        # crossover ~N=1000 (4× faster at N≥2000, ~even below)
        elif probes > 1 and N >= 1000 and _supports_block(mv, N):
            V0 = rng.standard_normal((probes, N)).T          # same draws as the loop
            if proj is not None:
                V0 = proj(V0)                  # probes live in the complement
            tridiags = _lanczos_block(mv, V0, k)
        else:
            draws = (rng.standard_normal(N) for _ in range(probes))
            tridiags = [_lanczos(mv, proj(v) if proj is not None else v, k)
                        for v in draws]
        nodes, weights = [], []
        for al, be in tridiags:
            kk = len(al)
            T = np.diag(al) + np.diag(be[:kk - 1], 1) + np.diag(be[:kk - 1], -1)
            theta, S = np.linalg.eigh(T)
            w = S[0, :] ** 2 * (scale / probes)
            if deflate:
                # rounding leaks ~1e-16 of each probe back into the deflated
                # span; it surfaces as massless nodes near 0 that can poison
                # f (log of a stray negative).  Mass dropped here: < k·1e-14.
                keep = w > 1e-14
                theta, w = theta[keep], w[keep]
            nodes.append(theta)
            weights.append(w)
        sizes = [len(th) for th in nodes]
        if atoms is not None:
            nodes.append(np.sort(atoms))
            weights.append(np.full(len(atoms), 1.0 / N))     # exact top-K atoms
        obj = cls(np.concatenate(nodes), np.concatenate(weights), matvec, N,
                  probe_sizes=sizes)
        obj._tridiags = tridiags if certifiable else None    # certificates, zoom
        obj._slq_scale = scale
        obj._n_atoms = 0 if atoms is None else len(atoms)
        return obj

    # ── COMPOSE (the free-convolution theorem: hard op → linear, matrix-free) ──
    def __add__(self, other: "Spectral") -> "Spectral":
        """A + B — never forms A+B, never diagonalizes; uses (A+B)x = Ax + Bx.

        What makes this legitimate is the CLOSURE THEOREM of the response
        algebra: every moment of A+B is a finite combination of joint moments
        of (A, B) — the algebra of responses is closed under +, verified to
        machine precision (1.6e-14) in the source program.  The spectrum
        alone does NOT compose (Horn's problem); the response does.
        """
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
    def trace(self, f, with_err: bool = False):
        """Tr f(A) = N · E[f(λ)] = N · Σ w·f(node).

        f may be a CALLABLE (``s.trace(np.log)``) or a family NAME
        (``s.trace("log")`` — identical here; family names matter for
        `trace_certified`, where derivative signs must be theorems).

        ``with_err=True`` → (value, stderr): the standard error of the
        stochastic estimate, read from the scatter of the independent probes
        (free — no extra matvecs).  The honest number, with its honest ±.
        For the PROVABLE k-truncation bracket see `trace_certified` (2.0:
        the old ``trace(certified=True)`` spelling is removed — one name
        per return shape).
        """
        f, _ = _resolve_f(f)
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
        atom_part = float(self.N) * float(np.sum(vals[i:]))   # deflate atoms:
        ests = [e + atom_part for e in ests]                  # exact, zero scatter
        stderr = float(np.std(ests, ddof=1) / np.sqrt(p))
        return total, stderr

    def trace_certified(self, f, support=None):
        """(lo, hi) — the Gauss–Radau bracket (Golub–Meurant) of Tr f(A)'s
        K-TRUNCATION error: the fully-converged SLQ value of these same
        probes provably lies inside.

        f must be a family NAME ('log', 'inv', 'sqrt', 'exp') — derivative
        signs must be known for the bracket to be a theorem — plus
        ``support=(a, b)`` with the needed endpoint (a ≤ λ_min for
        log/inv/sqrt, b ≥ λ_max for exp; often known structurally, e.g. a
        jitter).  It does NOT certify the Monte-Carlo probe scatter — that
        is a separate error source, reported by ``trace(f, with_err=True)``;
        the two are stated apart by design.  For an unconditional
        certificate of a probe-free quantity, see `quadform`."""
        f, fam = _resolve_f(f)
        if fam is None:
            raise ValueError("trace_certified needs f as a family name "
                             f"({list(_FAMS)}), not a callable — derivative "
                             "signs must be known for the bracket to be a "
                             "theorem")
        if not self._tridiags:
            raise ValueError("certificates need the probe recurrences: "
                             "build this object with Spectral.of "
                             "(lanczos engine — KPM's recurrences describe "
                             "the smoothed measure, no theorems there)")
        los, his = zip(*(_bracket(al, be, fam, support)
                         for al, be in self._tridiags))
        n_at = getattr(self, "_n_atoms", 0)
        scale = getattr(self, "_slq_scale", 1.0)
        atom_part = (float(self.N) * float(np.sum(
            self.weights[-n_at:] * f(self.nodes[-n_at:]))) if n_at else 0.0)
        return (float(self.N) * scale * float(np.mean(los)) + atom_part,
                float(self.N) * scale * float(np.mean(his)) + atom_part)

    def moment(self, p: int, with_err: bool = False):
        """Tr(A^p).  ``with_err=True`` → (value, stderr), see `trace`."""
        return self.trace(lambda x: x ** p, with_err=with_err)

    def _probe_blocks(self):
        """(sizes, n_atoms) — the per-probe node-block structure, validated."""
        if not self.probe_sizes or len(self.probe_sizes) < 2:
            raise ValueError("error bars need the probe structure from "
                             "Spectral.of with probes >= 2")
        used = int(np.sum(self.probe_sizes))
        return list(self.probe_sizes), len(self.nodes) - used

    def density(self, xs, eta: float = 0.1, with_err: bool = False):
        """Density of states ρ(x), Lorentzian-broadened by `eta`.

        ``with_err=True`` → (rho, stderr): the per-x standard error from the
        independent-probe scatter (same epistemics as `trace`; deflate atoms
        are exact and contribute no scatter)."""
        xs = np.atleast_1d(np.asarray(xs, float))
        ker = (self.weights[None, :] * (eta / np.pi)
               / ((xs[:, None] - self.nodes[None, :]) ** 2 + eta ** 2))
        rho = ker.sum(1)
        if not with_err:
            return rho
        sizes, _ = self._probe_blocks()
        p = len(sizes)
        ests, i = [], 0
        for sz in sizes:
            ests.append(p * ker[:, i:i + sz].sum(1))
            i += sz
        atom = ker[:, i:].sum(1)                              # exact, zero scatter
        ests = np.stack([e + atom for e in ests])
        return rho, np.std(ests, axis=0, ddof=1) / np.sqrt(p)

    def extreme(self, with_err: bool = False):
        """Extreme eigenvalues (Lanczos resolves these first / most reliably).

        Always an INNER estimate — Ritz values lie in the spectral hull, so
        deterministically lo ≥ λ_min and hi ≤ λ_max (Rayleigh–Ritz); the
        approach is from inside as k grows.

        ``with_err=True`` → ((lo, hi), (lo_err, hi_err)): the standard error
        of the per-probe extreme reads — a REPRODUCIBILITY bar (how much the
        read moves probe to probe), not a certified enclosure."""
        lo, hi = float(self.nodes.min()), float(self.nodes.max())
        if not with_err:
            return lo, hi
        sizes, _ = self._probe_blocks()
        p = len(sizes)
        atoms = self.nodes[int(np.sum(sizes)):]
        mins, maxs, i = [], [], 0
        for sz in sizes:
            blk = self.nodes[i:i + sz]
            cand = np.concatenate([blk, atoms]) if len(atoms) else blk
            mins.append(float(cand.min())); maxs.append(float(cand.max()))
            i += sz
        return (lo, hi), (float(np.std(mins, ddof=1) / np.sqrt(p)),
                          float(np.std(maxs, ddof=1) / np.sqrt(p)))

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

    def cumulants(self, order: int = 6, with_err: bool = False):
        """Free cumulants κ_1..κ_order — the canonical coordinates in which
        composition is ADDITION (κ_n(A⊞B) = κ_n(A) + κ_n(B)).

        ``with_err=True`` → (kappa, stderr): per-cumulant standard error from
        the independent-probe scatter (each probe's moments → its cumulants;
        the nonlinearity is propagated by recomputation, not linearization)."""
        from .free import free_cumulants
        m = [self.moment(p) / self.N for p in range(1, order + 1)]
        kappa = free_cumulants(m)
        if not with_err:
            return kappa
        sizes, _ = self._probe_blocks()
        p = len(sizes)
        per_probe = []
        for r in range(p):
            m_r = []
            for q in range(1, order + 1):
                vals = self.weights * (self.nodes ** q)
                i = int(np.sum(sizes[:r]))
                blk = float(np.sum(vals[i:i + sizes[r]])) * p
                atom = float(np.sum(vals[int(np.sum(sizes)):]))
                m_r.append(blk + atom)
            per_probe.append(free_cumulants(m_r))
        return kappa, np.std(np.stack(per_probe), axis=0, ddof=1) / np.sqrt(p)

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
        if not (a < b):
            raise ValueError(f"zoom window needs a < b (got a={a}, b={b})")
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

    def effective_rank(self, with_err: bool = False):
        """Φ₁ = Tr(A)² / Tr(A²) — participation ratio / effective number of modes,
        for POSITIVE-SEMIDEFINITE operators (covariance, kernel, Hessian).  Low Φ₁
        ⇒ structured/cheap; high ⇒ near the genuine frontier.  (Computed from the
        trace moments, which SLQ estimates robustly.)

        ``with_err=True`` → (value, stderr): Φ₁ is a STOCHASTIC estimate like
        any trace read — the bar is the scatter of per-probe Φ₁ values."""
        m1, m2 = self.moment(1), self.moment(2)
        val = float(m1 * m1 / m2) if m2 > 0 else 1.0
        if not with_err:
            return val
        if not self.probe_sizes or len(self.probe_sizes) < 2:
            raise ValueError("error bars need the probe structure from "
                             "Spectral.of with probes >= 2")
        p = len(self.probe_sizes)
        ests, i = [], 0
        for sz in self.probe_sizes:
            nd, wt = self.nodes[i:i + sz], self.weights[i:i + sz] * p
            mm1 = float(self.N) * float(np.sum(wt * nd))
            mm2 = float(self.N) * float(np.sum(wt * nd ** 2))
            ests.append(mm1 * mm1 / mm2 if mm2 > 0 else 1.0)
            i += sz
        return val, float(np.std(ests, ddof=1) / np.sqrt(p))

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

    If v is a torch/cupy/array-API array, the whole Krylov loop runs on its
    device and the result is returned there (float32 caveat: see `Spectral.of`).
    `matvec` may be a callable, LinearOperator, sparse matrix, or ndarray.
    """
    if hasattr(matvec, "shape"):        # LinearOperator / sparse / ndarray
        matvec, _ = _as_operator(matvec)
    xp = _xp(v)
    if xp is not None:                  # device vector → device Krylov
        return _apply_xp(matvec, f, v, k, hermitian, xp)
    v = np.asarray(v, complex if not hermitian else None)
    nv = np.linalg.norm(v)
    if nv == 0:
        return v.copy()
    N = len(v)

    if hermitian and (np.iscomplexobj(v) or _is_complex_operator(matvec, N)):
        # complex HERMITIAN: complex Lanczos, REAL tridiagonal — f(θ) real-
        # valued on a real spectrum, the result reconstructed in C^N.  (For
        # complex f on a Hermitian operator — e^{-iHt} — pass hermitian=True
        # too: θ are real, f(θ) complex is fine.)  One _lanczos_core call —
        # the 2.0 consolidation; q = v/nv preserved so the merge is
        # bit-identical to the former inline loop.
        vc = np.asarray(v, complex)
        al, be, Vb = _lanczos_core(matvec, vc / nv, k, True)
        m = len(al)
        T = np.diag(al) + np.diag(be[:m - 1], 1) + np.diag(be[:m - 1], -1)
        theta, S = np.linalg.eigh(T)
        return nv * (Vb @ (S @ (np.asarray(f(theta)) * S[0, :])))

    if hermitian:                                       # ── Lanczos (symmetric) ──
        v = np.asarray(v, float)
        al, be, V = _lanczos_core(matvec, v / nv, k, False)
        m = len(al)
        T = np.diag(al) + np.diag(be[:m - 1], 1) + np.diag(be[:m - 1], -1)
        theta, S = np.linalg.eigh(T)
        return nv * (V @ (S @ (np.asarray(f(theta), float) * S[0, :])))

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


def grad_trace(matvec, dmatvecs, fprime, N=None, k: int = 48, probes: int = 16,
               seed: int = 0, with_err: bool = False):
    """∂/∂θ_j Tr f(A(θ)) — DIFFERENTIABLE spectral reads, matrix-free.

    For a smooth parametric symmetric family A(θ), the exact identity
        d/dθ_j Tr f(A) = Tr( f'(A) · ∂A/∂θ_j )
    is estimated by Hutchinson probes shared across ALL parameters: per probe
    z, ONE Krylov chain u = f'(A)z (`apply`), then every component is the
    cheap inner product u·(∂A/∂θ_j z).  Cost: probes × (one apply + one
    matvec per parameter).

    matvec    : A(θ) at the current θ (callable / LinearOperator / sparse /
                dense — the usual four).
    dmatvecs  : ∂A/∂θ_j as ONE operator or a LIST of them (same four forms).
    fprime    : the DERIVATIVE f' as a callable on eigenvalues — you state
                what you differentiate (e.g. f=log → fprime=lambda x: 1/x:
                the log-det gradient; f=x² → 2x: a sharpness regularizer).
    with_err=True → (grad, stderr) per component, the probe scatter.

    This is the socket that makes spectral REGULARIZERS trainable: penalize
    log-det / sharpness / effective rank during optimization by feeding this
    gradient to the optimizer (wrap in torch.autograd.Function for autograd;
    the estimator itself is framework-agnostic).

    HONEST NOTE: stochastic like any trace read — the SAME probes are reused
    across parameters, so component errors are correlated (a common scale,
    harmless for descent directions); raise `probes` to tighten.
    """
    matvec, N = _as_operator(matvec, N)
    single = callable(dmatvecs) or hasattr(dmatvecs, "shape")
    dms = [dmatvecs] if single else list(dmatvecs)
    dms = [_as_operator(d, N)[0] for d in dms]
    rng = np.random.default_rng(seed)
    ests = np.empty((probes, len(dms)))
    for r in range(probes):
        z = rng.standard_normal(N)
        u = apply(matvec, fprime, z, k=k)
        for j, dm in enumerate(dms):
            ests[r, j] = float(u @ np.asarray(dm(z)))
    g = ests.mean(axis=0)
    if with_err:
        se = ests.std(axis=0, ddof=1) / np.sqrt(probes)
        return (float(g[0]), float(se[0])) if single else (g, se)
    return float(g[0]) if single else g


# ── LOCAL response (probe the operator from a CHOSEN vector, not random) ──────
def local_spectrum(matvec, v, k: int = 48):
    """The local spectral measure seen from vector v:  μ_v = Σ_i |⟨v|ψ_i⟩|² δ(λ_i).

    One Lanczos from v → Ritz nodes + first-component² weights.  This is the
    vector-resolved response; with v = e_i it is the LOCAL density of states
    (LDOS) at site i.  ``Spectral.of`` averages this over random v (→ the trace);
    here you choose v.  Returns (nodes, weights), weights summing to ‖v‖²-norm 1.
    """
    if hasattr(matvec, "shape"):        # LinearOperator / sparse / ndarray
        matvec, _ = _as_operator(matvec)
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

    Run Lanczos on diag(nodes) from the start vector √weights — the exact
    inverse of the e₀-MEASURE read (the Stieltjes / Gauss-quadrature
    construction: `from_measure(*local_spectrum(J, e0))` recovers (α, β)).
    `Spectral.of`'s random-probe average recovers the NODES of J but pushes
    weights toward 1/N — it is not the inverse for non-uniform weights;
    `local_spectrum` is.  Recovers a
    well-conditioned / smooth operator to ~machine precision; the long recurrence
    is genuinely ill-conditioned when the weights span many orders of magnitude
    (sharp operators) — the inverse spectral problem's intrinsic difficulty.
    """
    lam = np.asarray(nodes, float); w = np.asarray(weights, float)
    return _lanczos(lambda x: lam * x, np.sqrt(np.clip(w, 0.0, None)), k or len(lam))


def synthesize(nodes, weights, k=None):
    """CONSTRUCT an operator with the prescribed spectral measure — the
    discoverable verb (1.4+) for `from_measure` (the same function; both
    names stay, `from_measure` remains the precise synonym)."""
    return from_measure(nodes, weights, k=k)


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
