# Lifting nonlinear problems to linear ones

The core trick: *a shock / nonlinearity / hard composition is a **sum of
linearities** in the right coordinates.*  Lift to a linear operator, solve there
with [`apply`](solving-and-evolving.md), read back.  One move, three faces.

## 1. Nonlinear ODE → linear (Carleman)

A polynomial vector field `ẋ = Σ cₖ xᵏ` becomes linear on the monomials
`z = (x, x², x³, …)`:

```python
import numpy as np, resona
M = resona.lift.carleman_scalar([0, 1, -1], order=12)   # ẋ = x − x²  (logistic)
# evolve as a small-step integrator: re-lift each step, exp(dt·M)·z0
x, dt = 0.2, 0.1
for _ in range(20):
    z0 = np.array([x**(j+1) for j in range(12)])
    x  = float(resona.apply(lambda v: M@v, lambda l: np.exp(dt*l), z0,
                            k=12, hermitian=False)[0].real)   # → logistic, ~1e-15
```

Re-lifting each small step keeps the high monomials tiny, so a bounded solution
(logistic, Riccati, Bernoulli) tracks the exact trajectory to machine precision.

## 2. Nonlinear PDE → linear (Cole–Hopf / Carleman)

Viscous Burgers `u_t + u·u_x = ν·u_xx` becomes the **linear heat equation** under
`u = -2ν (ln φ)_x`; evolve `φ` with `apply`, map back.  Full recipe in
[`examples/nonlinear_pde.py`](../examples/nonlinear_pde.py).

## 3. Logic over GF(p) → linear polynomial (exact)

Any function `f:{0..p-1}ⁿ → {0..p-1}` becomes an **exact** linear combination of
monomials (since `x^p ≡ x`):

```python
coeffs, evaluate = resona.lift.carleman_gf(3, 2, lambda x: max(x))   # ternary max over GF(3)
# evaluate(x) reproduces f on every input — 0 errors
```

This is the same lift as above, over a finite field (Reed–Muller / algebraic
normal form for `p=2`).  Powers instant SAT / truth-table engines.

## 4. Free convolution → addition (the R / S transform)

Composition of operators linearizes too: `+` is linear in the **R-transform**
(`R_{A⊞B} = R_A + R_B`), `×` in the **S-transform**.  The R-transform is the
Cole–Hopf of free probability.

```python
resona.lift.r_transform(sA, w_grid)     # the additive linearizer
resona.lift.s_transform(sA, w_grid)     # the multiplicative one
```

See [composing-operators](composing-operators.md) and [`THEORY.md`](../THEORY.md) §4.

## How it works

Each instance turns a hard operation into addition in a basis of **powers /
monomials**: Cole–Hopf (analysis), Carleman/Koopman (dynamics), R/S-transform
(free probability), Reed–Muller (GF(p) logic).  The lift exists *finitely* exactly
when an associated rank saturates — see
[measuring-difficulty](measuring-difficulty.md).

## Gotchas (honest)

- **Carleman is a truncation** for general polynomial ODEs — use it as a
  small-step integrator (re-lift each step); valid for bounded trajectories.
- **`carleman_gf` is `O(pⁿ·³)`** — exponential in the number of variables; fine for
  `n ≲ 10`.
- **Cole–Hopf** is exact for Burgers but ill-conditioned at tiny viscosity (sharp
  shocks).

## Does a lift exist? — find the charges, blind

```python
charges, comm_norms = resona.lift.conserved_charge(H, basis)
# comm_norms[j] = ‖[H,Q_j]‖/‖Q_j‖ ascending;  < 1e-7 ⇒ a genuine conserved charge
```

A lift exists ⟺ enough conserved charges.  The commutator-Gram search FINDS
them with no prior knowledge (energy, total Z, free-fermion bilinears for an
integrable chain) and honestly returns none beyond H for a chaotic one.
Heavy tool: it forms `[H, O_a]` — `O(|basis|·dim²)`; see
[`examples/quantum/integrability_detector.py`](../examples/quantum/integrability_detector.py).

## Worked examples

- **Burgers (Cole–Hopf)** — [`examples/nonlinear_pde.py`](../examples/nonlinear_pde.py)
- **Exact GF(p) / Boolean logic lift** — [`examples/logic/`](../examples/logic/)
- **Exact affine flow for stiff ODEs** — [`examples/spectral_phenomena/affine_flow.py`](../examples/spectral_phenomena/affine_flow.py)
