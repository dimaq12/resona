"""Array-namespace dispatch (invisible GPU, EPIC Phase C).

A matvec that consumes/returns torch (or any array-API) arrays must take the
device code path with ZERO new API and agree with the numpy path; the numpy
path itself dispatches before any arithmetic (bit-parity is exercised by every
other test in this suite, which all run pure numpy).

Two backends are tested:
  * a minimal array-API shim defined here (always runs, pins the dispatch
    logic without any extra dependency);
  * torch CPU tensors (skipped cleanly when torch is not installed).
"""
import numpy as np
import pytest

from resona.spectral import Spectral, apply, _xp

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _sym(N, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N))
    return (A + A.T) / (2 * np.sqrt(N))


# ── dispatch happens before any arithmetic: numpy stays numpy ─────────────────

def test_xp_is_none_for_numpy_business():
    assert _xp(np.ones(3)) is None
    assert _xp(np.ones((2, 2))) is None
    assert _xp(3.0) is None
    assert _xp([1.0, 2.0]) is None
    assert _xp(np.float64(1.0)) is None


# ── a minimal array-API shim (no torch needed) ───────────────────────────────

def _u(x):
    return x._a if isinstance(x, ShimArray) else x


class ShimArray:
    """The smallest array-API citizen: wraps numpy, owns __array_namespace__."""

    def __init__(self, a):
        self._a = np.asarray(a)

    def __array_namespace__(self, api_version=None):
        return _SHIM_NS

    # introspection
    shape = property(lambda self: self._a.shape)
    dtype = property(lambda self: self._a.dtype)
    device = property(lambda self: "shim")
    T = property(lambda self: ShimArray(self._a.T))

    # indexing
    def __getitem__(self, ix):
        return ShimArray(self._a[ix])

    def __setitem__(self, ix, val):
        self._a[ix] = _u(val)

    # arithmetic
    def __add__(self, o):  return ShimArray(self._a + _u(o))
    def __radd__(self, o): return ShimArray(_u(o) + self._a)
    def __sub__(self, o):  return ShimArray(self._a - _u(o))
    def __rsub__(self, o): return ShimArray(_u(o) - self._a)
    def __mul__(self, o):  return ShimArray(self._a * _u(o))
    def __rmul__(self, o): return ShimArray(_u(o) * self._a)
    def __truediv__(self, o):  return ShimArray(self._a / _u(o))
    def __rtruediv__(self, o): return ShimArray(_u(o) / self._a)
    def __matmul__(self, o):   return ShimArray(self._a @ _u(o))
    def __pow__(self, o):      return ShimArray(self._a ** _u(o))
    def __neg__(self):         return ShimArray(-self._a)

    # host crossings
    def __float__(self):   return float(self._a)
    def __complex__(self): return complex(self._a)
    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._a, dtype)


class _ShimLinalg:
    @staticmethod
    def vector_norm(x):
        return ShimArray(np.linalg.norm(_u(x)))


class _ShimNS:
    float32 = np.dtype(np.float32)
    float64 = np.dtype(np.float64)
    complex64 = np.dtype(np.complex64)
    complex128 = np.dtype(np.complex128)
    linalg = _ShimLinalg

    @staticmethod
    def asarray(x, dtype=None, device=None):
        return ShimArray(np.asarray(_u(x), dtype))

    @staticmethod
    def zeros(shape, dtype=None, device=None):
        return ShimArray(np.zeros(shape, dtype))

    @staticmethod
    def sum(x, axis=None):
        return ShimArray(np.sum(_u(x), axis=axis))

    @staticmethod
    def conj(x):
        return ShimArray(np.conj(_u(x)))

    @staticmethod
    def real(x):
        return ShimArray(np.real(_u(x)))

    @staticmethod
    def imag(x):
        return ShimArray(np.imag(_u(x)))

    @staticmethod
    def abs(x):
        return ShimArray(np.abs(_u(x)))

    @staticmethod
    def astype(x, dtype):
        return ShimArray(_u(x).astype(dtype))


_SHIM_NS = _ShimNS


def test_shim_of_matches_numpy():
    N = 200
    A = _sym(N)
    s_np = Spectral.of(lambda v: A @ v, N, k=24, probes=4)
    s_sh = Spectral.of(lambda v: ShimArray(A @ _u(v)), N, k=24, probes=4)
    # host-side object out, device recurrence in: same draws, same math
    assert isinstance(s_sh.nodes, np.ndarray)
    assert np.allclose(s_np.nodes, s_sh.nodes, atol=1e-10)
    assert np.allclose(s_np.weights, s_sh.weights, atol=1e-10)


def test_shim_apply_hermitian_and_arnoldi():
    N = 150
    A = _sym(N)
    rng = np.random.default_rng(3)
    x = rng.standard_normal(N)
    y_np = apply(lambda v: A @ v, np.exp, x, k=32)
    y_sh = apply(lambda v: ShimArray(A @ _u(v)), np.exp, ShimArray(x.copy()), k=32)
    assert isinstance(y_sh, ShimArray)
    assert np.allclose(y_np, _u(y_sh), atol=1e-10)

    B = rng.standard_normal((N, N)) / np.sqrt(N)
    f = lambda lam: np.exp(-1j * lam)
    z_np = apply(lambda v: B @ v, f, x, k=32, hermitian=False)
    z_sh = apply(lambda v: ShimArray(B @ _u(v)), f, ShimArray(x.copy()), k=32,
                 hermitian=False)
    assert np.allclose(z_np, _u(z_sh), atol=1e-10)


def test_shim_complex_hermitian_of():
    N = 120
    rng = np.random.default_rng(5)
    H = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (H + H.conj().T) / (2 * np.sqrt(N))
    s_np = Spectral.of(lambda v: H @ v, N, k=24, probes=3)
    s_sh = Spectral.of(lambda v: ShimArray(H @ _u(v)), N, k=24, probes=3)
    assert np.allclose(s_np.nodes, s_sh.nodes, atol=1e-10)
    assert np.allclose(s_np.weights, s_sh.weights, atol=1e-10)


def test_shim_apply_complex_hermitian():
    # the Phase D contract on the device: hermitian=True with a complex
    # HERMITIAN operator (real v promoted) and complex f on a real spectrum
    N = 120
    rng = np.random.default_rng(7)
    H = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (H + H.conj().T) / (2 * np.sqrt(N))
    x = rng.standard_normal(N)
    f = lambda lam: np.exp(-1j * lam)                   # e^{-iHt}, t=1
    y_np = apply(lambda v: H @ v, f, x, k=32)
    y_sh = apply(lambda v: ShimArray(H @ _u(v)), f, ShimArray(x.copy()), k=32)
    assert isinstance(y_sh, ShimArray)
    assert np.allclose(y_np, _u(y_sh), atol=1e-10)


def test_device_kpm_and_deflate_refuse_honestly():
    # device coverage today: engine="lanczos", deflate=0 — anything else raises
    N = 64
    A = _sym(N)
    mv = lambda v: ShimArray(A @ _u(v))
    with pytest.raises(ValueError):
        Spectral.of(mv, N, engine="kpm")
    with pytest.raises(ValueError):
        Spectral.of(mv, N, deflate=4)


# ── torch CPU tensors (skipped when torch is absent) ─────────────────────────

def test_torch_of_float64_matches_numpy():
    torch = pytest.importorskip("torch")
    N = 400
    A = _sym(N)
    At = torch.from_numpy(A)
    s_np = Spectral.of(lambda v: A @ v, N, k=32, probes=4)
    s_t = Spectral.of(lambda v: At @ v, N, k=32, probes=4)
    assert isinstance(s_t.nodes, np.ndarray)            # host-side object out
    assert np.allclose(s_np.nodes, s_t.nodes, atol=1e-10)
    assert np.allclose(s_np.weights, s_t.weights, atol=1e-10)
    assert abs(s_np.trace("exp") - s_t.trace("exp")) < 1e-8


def test_torch_only_matvec_rejecting_numpy():
    torch = pytest.importorskip("torch")
    N = 300
    A = _sym(N)
    At = torch.from_numpy(A)

    def strict_mv(v):
        assert isinstance(v, torch.Tensor)              # refuses the numpy probe
        return At @ v

    s_np = Spectral.of(lambda v: A @ v, N, k=24, probes=4)
    s_t = Spectral.of(strict_mv, N, k=24, probes=4)
    assert np.allclose(s_np.nodes, s_t.nodes, atol=1e-10)


def test_torch_apply_stays_on_device():
    torch = pytest.importorskip("torch")
    N = 250
    A = _sym(N) @ _sym(N).T + np.eye(N)                 # SPD
    At = torch.from_numpy(A)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(N)
    y_np = apply(lambda v: A @ v, np.exp, x)
    y_t = apply(lambda v: At @ v, np.exp, torch.from_numpy(x.copy()))
    assert isinstance(y_t, torch.Tensor)                # result stays a tensor
    assert np.allclose(y_np, y_t.numpy(), atol=1e-10)
    # dense ground truth
    lam, V = np.linalg.eigh(A)
    truth = V @ (np.exp(lam) * (V.T @ x))
    assert np.allclose(y_t.numpy(), truth, atol=1e-8 * np.linalg.norm(truth))


def test_torch_arnoldi_real_operator_complex_f():
    torch = pytest.importorskip("torch")
    N = 200
    rng = np.random.default_rng(4)
    B = rng.standard_normal((N, N)) / np.sqrt(N)
    Bt = torch.from_numpy(B)
    x = rng.standard_normal(N)
    f = lambda lam: np.exp(-1j * lam)
    z_np = apply(lambda v: B @ v, f, x, hermitian=False)
    # torch refuses real-matrix @ complex-vector; the dispatch must route
    # around it (A(x+iy) = Ax + i·Ay) without any user-side change
    z_t = apply(lambda v: Bt @ v, f, torch.from_numpy(x.copy()), hermitian=False)
    assert isinstance(z_t, torch.Tensor)
    assert np.allclose(z_np, z_t.numpy(), atol=1e-10)


def test_torch_complex_hermitian_of():
    torch = pytest.importorskip("torch")
    N = 200
    rng = np.random.default_rng(6)
    H = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (H + H.conj().T) / (2 * np.sqrt(N))
    Ht = torch.from_numpy(H)
    s_np = Spectral.of(lambda v: H @ v, N, k=24, probes=3)
    s_t = Spectral.of(lambda v: Ht @ v, N, k=24, probes=3)
    assert np.allclose(s_np.nodes, s_t.nodes, atol=1e-10)
    assert np.allclose(s_np.weights, s_t.weights, atol=1e-10)


def test_torch_apply_complex_hermitian():
    torch = pytest.importorskip("torch")
    N = 150
    rng = np.random.default_rng(8)
    H = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (H + H.conj().T) / (2 * np.sqrt(N))
    Ht = torch.from_numpy(H)
    x = rng.standard_normal(N)
    f = lambda lam: np.exp(-1j * lam)
    y_np = apply(lambda v: H @ v, f, x, k=32)
    # torch's complex matrix REJECTS the real tensor — the hermitian device
    # path must promote v to complex itself, exactly like the host branch
    y_t = apply(lambda v: Ht @ v, f, torch.from_numpy(x.copy()), k=32)
    assert isinstance(y_t, torch.Tensor)
    assert np.allclose(y_np, y_t.numpy(), atol=1e-10)


def test_torch_float32_works_with_honest_precision():
    torch = pytest.importorskip("torch")
    N = 300
    A = _sym(N)
    At32 = torch.from_numpy(A).to(torch.float32)        # torch's default world
    s64 = Spectral.of(lambda v: torch.from_numpy(A) @ v, N, k=24, probes=4)
    s32 = Spectral.of(lambda v: At32 @ v, N, k=24, probes=4)
    rel = abs(s32.moment(2) - s64.moment(2)) / abs(s64.moment(2))
    assert rel < 1e-2                                    # ~2-3 digits: the honest fp32 story
    # and apply keeps the user's dtype
    x32 = torch.from_numpy(np.random.default_rng(0).standard_normal(N)).to(torch.float32)
    y32 = apply(lambda v: At32 @ v, np.exp, x32, k=24)
    assert y32.dtype == torch.float32
