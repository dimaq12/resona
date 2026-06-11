# Inverse problems — recover the operator from its spectrum

`of` goes operator → spectrum.  The inverses go back.  **Which inverse you can use
depends on how much spectral data you have — and that sets the conditioning.**

## The data ↔ conditioning hierarchy

```
  full eigenbasis (all v_i[j])     →  from_eigenbasis   →  EXACT on every operator (~1e-14)
  boundary measure (only v_i[0])   →  from_measure      →  exact (smooth) / ill-posed (sharp)
  eigenvalues only (+ a target)    →  wkernel.design    →  regularized: bounded but smoothed
```

*The inverse problem's difficulty is exactly how much spectral data you discard.*

## Recipes

### 1. Full eigenbasis → exact (machine precision on all)

```python
import numpy as np, resona
lam, V = np.linalg.eigh(A)                       # the full spectral data
diag, off = resona.from_eigenbasis(lam, V)       # the Jacobi band of V·diag(λ)·Vᵀ, ~1e-14
```

Exact for **any** operator, sharp or smooth — it reads the tridiagonal band of
`VΛVᵀ` directly.  The catch: it needs every eigenvector (`O(N²)` data, `O(N³)` to
obtain via `eig`).

### 2. Boundary measure → compressed, but fragile

```python
nodes, weights = resona.local_spectrum(A_matvec, e0)   # spectrum + boundary amplitudes
alpha, beta = resona.from_measure(nodes, weights)      # Jacobi (α,β) via inverse Lanczos
```

Uses only the spectral **measure** from one boundary probe `e₀` (`O(N)` data,
*matrix-free*).  Exact for smooth operators; **blows up for sharp ones** —
boundary weights span ~10⁵⁰ and far-from-boundary modes underflow.  This is the
classical "you can't hear the shape of a drum from one point."

### 3. Design parameters to hit a target spectrum (`wkernel`)

```python
W  = resona.wkernel.wkernel(eigvecs, perturbations)    # spectral Jacobian ∂λ/∂k
dk = resona.wkernel.design(W, target_shift, reg=1e-3)  # Tikhonov-regularized step
# iterate (recompute W at the new parameters) for a nonlinear target
```

`reg=0` is the exact least-squares step (machine-precise when well-posed, blows up
when under-determined); **`reg>0` is bounded and robust, recovering a smoothed
solution** — the bias–variance dial of an ill-posed inverse.

## How it works

A symmetric tridiagonal (Jacobi) operator is determined by its `e₀` spectral
measure (eigenvalues + boundary overtone amplitudes) — `from_measure` is the
inverse of `of`'s Lanczos (the Stieltjes/Gauss–quadrature construction).  With the
full eigenbasis the band of `VΛVᵀ` is read off directly (`from_eigenbasis`).  With
only eigenvalues the problem is under-determined and needs regularization
(`wkernel.design`).

## Gotchas (honest)

- **`from_measure` is genuinely ill-conditioned for sharp operators** — not a bug,
  the inverse spectral problem's intrinsic difficulty. Use `from_eigenbasis` if you
  have `V`, or regularize.
- **`from_measure`'s long recurrence also degrades past N≈40** for spread spectra.
- **Gauge:** the off-diagonals are recovered positive; the conductivity/parameter
  has an alternating null mode fixed by one boundary value or the mean.

## Worked example

- **All three inverses side by side, the full hierarchy** —
  [`examples/inverse_spectral.py`](../examples/inverse_spectral.py)
- **Inverse graph design (iterated `wkernel.design`)** —
  [`examples/graphs/inverse_graph_design.py`](../examples/graphs/inverse_graph_design.py),
  [`examples/graphs/edge_weight_recovery.py`](../examples/graphs/edge_weight_recovery.py)

Cost: `from_eigenbasis` `O(N²)` (+`eig`), `from_measure` `O(N²..N³)`,
`design` `O(N³)` SVD. See [`COMPLEXITY.md`](../COMPLEXITY.md).
