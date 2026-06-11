# Composing operators

Get the spectrum of `A+B`, `A·B`, or a disorder-averaged operator — without ever
forming the composite or calling `eig` on it.

## Two ways to compose

```python
import numpy as np, resona
sA = resona.of(A_matvec, N);  sB = resona.of(B_matvec, N)
```

### 1. Exact, via the matvec (you can apply both)

```python
(sA + sB).extreme()             # spectrum of A+B — never forms A+B, uses (A+B)x = Ax+Bx
(sA @ sB).density(xs)           # spectrum of A·B
```

`sA + sB` re-probes the combined operator `x → A·x + B·x`.  Exact (it is the real
A+B), matrix-free.  Use when you can apply both A and B.

### 2. From the spectra alone (free convolution — no joint matvec)

```python
moments = resona.lift.free_convolution(sA, sB, order=4)   # moments of A⊞B from μ_A, μ_B
```

When you only have the two **spectra** (measured separately, different times/
machines, or A+B too big to instantiate), free probability composes them —
*valid when A and B are free* (generic relative position).  Check with
`resona.free.freeness_defect(A_matvec, B_matvec, N)` (≈0 ⇒ free ⇒ the prediction
holds).  See [`THEORY.md`](../THEORY.md) §3.

## Disorder averaging (free addition with a semicircle)

Average over a random ensemble `H = A + σ·W` in **closed form** — no realization
loop, no `eig`:

```python
xs = np.linspace(lo, hi, 400)
rho = resona.subordination.averaged_dos(sA, sigma, xs)    # ⟨DOS⟩ of A + σ·GOE (Pastur)
```

The averaged density solves the self-consistent `g = G_A(z − σ²g)`.  Exact in the
large-N / free limit (`m₂ = m₂(A) + σ²`, …).

## Free convolution as a flow (the shock)

Push the noise variance as time `t` and you get the complex Burgers flow; two
bands collide into one at a critical time:

```python
resona.flow.shock_time(sA)      # the band-merger time t_c (defect = spectral edge)
```

## How it works

Composition closes in the **free cumulants** (`κ_n(A⊞B)=κ_n(A)+κ_n(B)`) when the
operators are free — the spectrum itself does *not* compose (Horn's problem).
`sA + sB` sidesteps this by re-probing the true composite; `free_convolution`
uses the free-probability theorem on the measures.  See [`THEORY.md`](../THEORY.md).

## Gotchas (honest)

- **`free_convolution` needs the parts to be free** — `freeness_defect` tells you.
  High orders are sensitive to noisy input moments; keep `order ≲ 4` with SLQ input.
- **`sA + sB` requires both matvecs** on the same `Rᴺ`.
- **`averaged_dos`** is the semicircle (GOE-noise) case; structured/heterogeneous
  disorder needs the operator-valued (matrix-Dyson) extension.

## Worked examples

- **Spectrum of A+B at scale (Horn in practice)** — [`examples/killer_tasks.py`](../examples/killer_tasks.py)
- **Free convolution + Pastur DOS + the shock** — [`examples/spectral_phenomena/free_convolution_flow.py`](../examples/spectral_phenomena/free_convolution_flow.py)
- **Free cumulants & the freeness criterion** — [`examples/spectral_phenomena/free_probability.py`](../examples/spectral_phenomena/free_probability.py)
- **Disorder-averaged Anderson DOS** — [`examples/anderson_localization.py`](../examples/anderson_localization.py)
