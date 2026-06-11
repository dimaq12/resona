# Reading spectra (matrix-free)

Read any spectral functional of an operator you can only multiply by a vector —
no matrix formed, no `eig` called.  One probe, then read whatever you need.

```python
import numpy as np, resona
s = resona.of(matvec, N)        # PROBE once  (matvec: v → A·v ; N = dimension)
```

`s` is the **spectral response** — Ritz nodes (the "frequencies") + weights (the
"amplitudes") from stochastic Lanczos quadrature.  Everything below reads `s`.

## Recipes

```python
s.trace(np.log)                 # log|A|  — log-determinant (A positive definite)
s.trace(lambda l: 1/l)          # Tr A⁻¹
s.trace(lambda l: 1/(1+l))      # Tr (I+A)⁻¹  (GP / ridge effective d.o.f.)
s.moment(2)                     # Tr A²   (Schatten, Frobenius², variance)
s.density(np.linspace(a,b,200)) # ρ(x) — density of states / spectrum shape
s.extreme()                     # (λ_min, λ_max)
s.effective_rank()              # Φ₁ = Tr(A)²/Tr(A²)  (see measuring-difficulty)
```

## How it works

`of` runs `probes` random Lanczos chains of `k` steps and turns each into a tiny
quadrature rule for the spectral measure.  `trace(f)` is then `N·Σ wᵢ f(λᵢ)`; it
estimates `Tr f(A)` to a few percent at `O(probes·k)` matvecs — independent of the
dimension.  (Golub–Meurant / Ubaru–Saad stochastic Lanczos quadrature.)

## Tuning

- **Accuracy:** raise `probes` (variance ∝ 1/probes) and `k` (resolves more of the
  spectrum). `resona.of(matvec, N, k=80, probes=16)`.
- **Extreme eigenvalues** converge first and most reliably — `extreme()` is good
  with `probes=4`. **Interior / smooth functionals** want more probes.
- **Density** has a broadening `eta`: `s.density(xs, eta=0.05)` (smaller = sharper,
  noisier).

## Gotchas (honest)

- `trace(np.log)` needs a **positive-definite** A (log of a negative eigenvalue is
  NaN). For `Tr A⁻¹` the spectrum must avoid 0.
- `effective_rank()` (Φ₁) is meaningful for **PSD** operators (covariance, kernel,
  Hessian); for an indefinite/traceless A it is not a rank.
- These are **stochastic estimates** — report a band, not a last digit. For exact
  individual eigenvalues you need a different tool (Lanczos with restarts, or the
  inverse guides).

## Worked examples

- **GP log-determinant, loss-Hessian spectrum, effective rank** —
  [`examples/killer_tasks.py`](../examples/killer_tasks.py)
- **Signal / image spectra** — [`examples/signals.py`](../examples/signals.py),
  [`examples/image_anomaly.py`](../examples/image_anomaly.py)
- **Detecting a planted signal (BBP threshold)** via `extreme()` —
  [`examples/spike_detection.py`](../examples/spike_detection.py)
- **Quantum many-body DOS / spectrum from 2 moments** —
  [`examples/quantum/many_body_spectrum.py`](../examples/quantum/many_body_spectrum.py)

Cost: `O(probes·k·matvec)` — flat in N. See [`COMPLEXITY.md`](../COMPLEXITY.md).
