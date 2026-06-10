# resona — the FFT of operators

> `fft(x)` takes a **signal** to the basis where convolution becomes pointwise
> multiply. `Spectral.of(A)` takes an **operator** (black-box matvec) to the
> representation where composition becomes addition — **matrix-free** — and from
> which every spectral functional is read.

One object. Three verbs. No matrix is ever formed; `eig` is never called.

```python
from resona import Spectral

s = Spectral.of(matvec, N)      # PROBE   — harvest the operator's response
t = Spectral.of(matvec2, N)
(s + t).extreme()               # COMPOSE — eig(A+B), A+B never formed
(s @ t)                         #           A·B
s.trace(np.log)                 # READ    — log|A|  (any spectral functional)
s.density(x) ; s.moment(2) ; s.extreme()
s.effective_rank()             # the honest cost dial (Φ₁)

resona.apply(matvec, f, v)      # APPLY   — f(A)·v  (exp(tA)·v, A⁻¹·v, …): evolve PDEs
```

## Install

```bash
pip install -e .            # from source
pip install -e ".[test]"   # with pytest
```

## What it solves (matrix-free, one primitive)

Verified against dense ground truth in [`examples/killer_tasks.py`](examples/killer_tasks.py):

| task | what | metric |
|------|------|--------|
| **GP log-determinant** | `log|K|` for hyperparameter learning at scale | 0.84% rel.err, no Cholesky |
| **Loss-Hessian spectrum** | sharpness & curvature from HVPs, no Hessian formed | λ_max 0.00%, Tr 0.30% |
| **Spectrum of A+B** | composed, matrix-free (Horn's problem in practice) | extreme eig to 1e-9 |
| **Deep-net trainability** | `cond(W_L…W_1)` predicted from init, no fwd/bwd | Gaussian explodes, orthogonal ≈1 |
| **Effective rank Φ₁** | the cost dial: structured/cheap vs full/frontier | 14 vs 466 |
| **Nonlinear PDE (Burgers)** | lift to linear (Cole–Hopf) → `exp(tK)·v` via `resona.apply` | residual 5e-9, matrix-free |
| **35 operators → spectra** | matrix-free Ritz seed → Rayleigh polish (the sft35 pipeline) | seed 1e-4 → **1e-16**, 99% machine-zero |
| **Signal in noise (BBP)** | the detection threshold — does a spike detach from the bulk? | `extreme()` tracks λ=θ+1/θ above θ_c=1 |
| **Anderson localization** | metal→insulator transition from disorder, matrix-free | Λ=∫typ/∫mean LDOS, 0.97→0.15 in 3.4s |
| **Tracy–Widom edge** | the universal fluctuation law of `extreme()` (λ_max) | std·N^⅔→1.27, exponent −0.65 (target −⅔) |

More broadly: density of states, `Tr f(A)` (log-det, `Tr A⁻¹`, partition
functions, Schatten norms), extreme eigenvalues & spectral gaps, disorder-averaged
spectra, phase-transition detection, spectral clustering — anything that is a
spectral functional of a (possibly composed) operator you can only matvec.

## The three verbs

- **PROBE** — `Spectral.of(matvec, N)`: stochastic Lanczos quadrature →
  Ritz nodes (the "frequencies") + weights (the "amplitudes"). Cost `O(probes·k)`
  matvecs.
- **COMPOSE** — `s + t`, `s @ t`: `A+B`, `A·B` without forming or diagonalizing
  them (the free-convolution theorem; exact closure `(A+B)x = Ax + Bx`).
- **READ** — `trace`, `moment`, `density`, `extreme`: any spectral functional.

Plus the **cost dial** `effective_rank()` (`Φ₁`) — low ⇒ structured/cheap;
high ⇒ near the genuine frontier — and the *Extraction Law* it comes from.

## Honesty

The underlying algorithms (SLQ, Lanczos, free probability) are **classical** and
credited in [`NOVELTY.md`](NOVELTY.md). `resona`'s contribution is the **single
primitive + matrix-free composition algebra + the built-in cost law** as one
object — the way FFT organizes signal processing. The unifying claims (the
Extraction Law, `Φ₁`-as-boundary) are research hypotheses, labelled as such; the
computations are verified.

## Theory

The unified picture behind the library — the response measure as a conjugate
pair, free probability (closure, the freeness boundary, the semicircle attractor),
the defect = shock = edge identity, and the Extraction Law — is in
[`THEORY.md`](THEORY.md), with reproducible verification scripts in
[`theory/`](theory/). The intellectual claims are stated honestly in
[`NOVELTY.md`](NOVELTY.md).

## License

MIT © 2026 Dmitry Sierikov. Attribution requested for the research contributions
in `NOVELTY.md`.
