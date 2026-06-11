# Solving & evolving (the `apply` engine)

`resona.apply(matvec, f, v)` computes **f(A)·v** from matvecs only — the universal
solve / evolve / filter engine.  Pick `f`, get the operation:

```python
import numpy as np, resona
resona.apply(matvec, f, v, k=48, hermitian=True)
```

| you want | `f` | meaning |
|----------|-----|---------|
| solve `A x = b` | `lambda l: 1/l` | `A⁻¹·b` (A SPD) |
| evolve `u_t = A u` to time `t` | `lambda l: np.exp(t*l)` | `exp(tA)·u₀` |
| quantum / wave `ψ_t = -iHψ` | `lambda l: np.exp(-1j*t*l)` | `exp(-iHt)·ψ` (needs `hermitian=False`) |
| diffuse / low-pass a signal | `lambda l: np.exp(-t*l)` | heat-kernel smoothing on a graph Laplacian |
| Tikhonov / ridge solve | `lambda l: 1/(l+α)` | `(A+αI)⁻¹·b` |
| whiten / sample a Gaussian | `np.sqrt` | `√A·v` |
| spectral projector | `lambda l: (l>c)*1.0` | project onto modes above `c` |

## Recipes

```python
# 1) solve a (symmetric positive-definite) linear system, matrix-free
x = resona.apply(A_matvec, lambda l: 1/l, b, k=80)

# 2) evolve a linear PDE  u_t = A u   (heat, diffusion, …)
u_t = resona.apply(A_matvec, lambda l: np.exp(t*l), u0)

# 3) quantum dynamics  ψ(t) = e^{-iHt} ψ0   (unitary, norm-preserving)
psi_t = resona.apply(H_matvec, lambda l: np.exp(-1j*t*l), psi0, hermitian=False)

# 4) denoise a signal living on a graph/grid Laplacian L (heat-kernel low-pass)
clean = resona.apply(L_matvec, lambda l: np.exp(-tau*l), noisy)
```

## How it works

`apply` builds a `k`-dimensional Krylov subspace (Lanczos if `hermitian`, Arnoldi
if not), evaluates `f` on the tiny `k×k` projected operator, and lifts back.  For a
smooth `f` and well-behaved spectrum, `k≈40–80` reaches machine precision.  It is
the matrix-function counterpart of `of`: same Krylov machinery, different readout.

## hermitian = True vs False

- **`hermitian=True`** (default): self-adjoint A, real `f` — cheapest (Lanczos).
- **`hermitian=False`**: non-symmetric A **or** complex `f` (e.g. `e^{-iHt}`) —
  Arnoldi; returns complex when `f` or A is complex (quantum dynamics is unitary,
  norm preserved to ~1e-15).

## Nonlinear PDEs

`apply` is linear, but a nonlinear PDE often **lifts** to a linear one — then
evolve with `apply`.  Viscous Burgers → (Cole–Hopf) → heat equation; a polynomial
ODE → (Carleman) → a linear system.  See [lifting-nonlinear](lifting-nonlinear.md)
and [`examples/nonlinear_pde.py`](../examples/nonlinear_pde.py).

## Gotchas (honest)

- **Conditioning sets `k`.** `A⁻¹·b` on an ill-conditioned A needs more steps
  (like CG); near a singularity the cost diverges — see
  [measuring-difficulty](measuring-difficulty.md).
- **`f` is evaluated on the Ritz values**, so `f` must be defined across the
  spectrum (no `1/l` if 0 is an eigenvalue).
- **Sharp shocks** (e.g. Burgers at tiny viscosity) make the lift ill-conditioned —
  machine precision only where the lift is well-conditioned.

## Worked examples

- **Burgers via Cole–Hopf lift** — [`examples/nonlinear_pde.py`](../examples/nonlinear_pde.py)
- **10⁴ instant solves from one precompute** — [`examples/spectral_phenomena/universal_solver.py`](../examples/spectral_phenomena/universal_solver.py)
- **Stiff/non-normal ODE via exact exponential flow** — [`examples/spectral_phenomena/affine_flow.py`](../examples/spectral_phenomena/affine_flow.py)
- **Quantum dynamics** — [`examples/quantum/`](../examples/quantum/)

Cost: `O(k·matvec + k²·N)` — flat in N for fixed k. See [`COMPLEXITY.md`](../COMPLEXITY.md).
