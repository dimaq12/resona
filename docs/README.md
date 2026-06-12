# resona — how-to (the cookbook)

**First time here?** Take **[the tour](tour.md)** — ten stops, plain words
first, the math second, from "what is a matvec" to synthesizing operators.

Task-oriented guides: **find your task in the table, copy the recipe, follow the
guide, read the full example.**  Every recipe is matrix-free unless noted — you
supply a `matvec` (a function `v → A·v`), never a matrix.

```python
import numpy as np, resona
matvec = lambda v: A @ v          # your operator, however you can apply it
```

## I want to…

| I want to… | recipe | guide | full example |
|------------|--------|:-----:|--------------|
| compute `log\|A\|`, `Tr A⁻¹`, `Tr f(A)` without forming A | `resona.of(mv,N).trace(f)` | [reading-spectra](reading-spectra.md) | [`killer_tasks.py`](../examples/killer_tasks.py) |
| get the density of states / spectrum shape | `resona.of(mv,N).density(xs)` | [reading-spectra](reading-spectra.md) | [`signals.py`](../examples/signals.py) |
| get the largest/smallest eigenvalue | `resona.of(mv,N).extreme()` | [reading-spectra](reading-spectra.md) | [`spike_detection.py`](../examples/spike_detection.py) |
| grade a problem's structure / cost (Φ₁) | `resona.of(mv,N).effective_rank()` | [measuring-difficulty](measuring-difficulty.md) | [`killer_tasks.py`](../examples/killer_tasks.py) |
| solve `A x = b` matrix-free | `resona.apply(mv, lambda l: 1/l, b)` | [solving-and-evolving](solving-and-evolving.md) | [`spectral_phenomena/universal_solver.py`](../examples/spectral_phenomena/universal_solver.py) |
| evolve a (linear) PDE `u_t = A u` | `resona.apply(mv, lambda l: np.exp(t*l), u0)` | [solving-and-evolving](solving-and-evolving.md) | [`nonlinear_pde.py`](../examples/nonlinear_pde.py) |
| simulate quantum dynamics `e^{-iHt}ψ` | `resona.apply(mv, lambda l: np.exp(-1j*t*l), psi, hermitian=False)` | [solving-and-evolving](solving-and-evolving.md) | [`quantum/`](../examples/quantum/) |
| solve a NONLINEAR PDE | lift → `resona.apply` | [lifting-nonlinear](lifting-nonlinear.md) | [`nonlinear_pde.py`](../examples/nonlinear_pde.py) |
| denoise a signal / image | `resona.apply(L, lowpass, x)` | [solving-and-evolving](solving-and-evolving.md) | [`image_anomaly.py`](../examples/image_anomaly.py) |
| get the spectrum of `A+B` without forming it | `(sA + sB).extreme()` | [composing-operators](composing-operators.md) | [`killer_tasks.py`](../examples/killer_tasks.py) |
| compose two spectra you measured separately | `resona.lift.free_convolution(sA,sB)` | [composing-operators](composing-operators.md) | [`spectral_phenomena/free_convolution_flow.py`](../examples/spectral_phenomena/free_convolution_flow.py) |
| CLEAN a noisy sample covariance (free deconvolution / RIE) | `resona.free.rie_clean(eigs, q=N/T)` | [composing-operators](composing-operators.md) | [`covariance_cleaning.py`](../examples/covariance_cleaning.py) |
| subtract additive noise from a spectrum | `resona.free.rie_clean_additive(eigs, σ)` | [composing-operators](composing-operators.md) | [`covariance_cleaning.py`](../examples/covariance_cleaning.py) |
| get error bars on a stochastic trace | `s.trace(f, with_err=True)` | [reading-spectra](reading-spectra.md) | — |
| collapse the variance on a spiked operator (Hutch++) | `resona.of(mv, N, deflate=K)` | [reading-spectra](reading-spectra.md) | — |
| high-resolution density without reorthogonalization | `resona.of(mv, N, k=256, engine="kpm")` | [reading-spectra](reading-spectra.md) | — |
| get a CERTIFIED bracket (the answer provably inside) | `resona.quadform(mv, "inv", v, certified=True, support=(a,None))` | [precision-and-defects](precision-and-defects.md) | [`certified_logdet.py`](../examples/certified_logdet.py) |
| certify the k-truncation of a trace estimate | `s.trace("log", certified=True, support=(a,None))` | [precision-and-defects](precision-and-defects.md) | [`certified_logdet.py`](../examples/certified_logdet.py) |
| resolve INTERIOR eigenvalues (spectrum slicing) | `s.zoom(a, b)` → polish nodes | [reading-spectra](reading-spectra.md) | [`spectra_to_machine_precision.py`](../examples/spectra_to_machine_precision.py) |
| disorder-average a DOS (no realizations) | `resona.subordination.averaged_dos(sA,σ,xs)` | [composing-operators](composing-operators.md) | [`anderson_localization.py`](../examples/anderson_localization.py) |
| CONSTRUCT an operator with a prescribed spectrum | inverse-CDF levels → `resona.from_measure(levels, 1/N)` | [inverse-problems](inverse-problems.md) | [`spectral_phenomena/operator_synthesis.py`](../examples/spectral_phenomena/operator_synthesis.py) |
| probe a NON-HERMITIAN operator (Markov, Koopman, damping) | `resona.cloud(mv, N)` → `.radius() .abscissa() .nodes` | [reading-spectra](reading-spectra.md) | [`science/koopman_dynamics.py`](../examples/science/koopman_dynamics.py) |
| read a topological invariant matrix-free | P = `apply(H, step, v)` chains → Chern marker | [solving-and-evolving](solving-and-evolving.md) | [`quantum/chern_from_noise.py`](../examples/quantum/chern_from_noise.py) |
| turn a TIME SERIES into an operator | `mv, rmv, r = resona.lift.koopman(snapshots)` | [lifting-nonlinear](lifting-nonlinear.md) | [`science/koopman_dynamics.py`](../examples/science/koopman_dynamics.py) |
| thermal expectation ⟨O⟩_β matrix-free (typicality) | `resona.thermal.expect(Hmv, Omv, beta, N)` | [solving-and-evolving](solving-and-evolving.md) | — |
| dynamical correlator ⟨O(t)O⟩_β / spectral function | `resona.thermal.correlator(Hmv, Omv, beta, ts, N)` | [solving-and-evolving](solving-and-evolving.md) | [`quantum/thermal_response.py`](../examples/quantum/thermal_response.py) |
| recover the operator from its spectrum | `resona.from_eigenbasis(λ,V)` / `from_measure(λ,w)` | [inverse-problems](inverse-problems.md) | [`inverse_spectral.py`](../examples/inverse_spectral.py) |
| design parameters to hit a target spectrum | `resona.wkernel.design(W, Δλ, reg=…)` | [inverse-problems](inverse-problems.md) | [`graphs/inverse_graph_design.py`](../examples/graphs/inverse_graph_design.py) |
| linearize a nonlinear ODE / logic function | `resona.lift.carleman_scalar / carleman_gf` | [lifting-nonlinear](lifting-nonlinear.md) | [`logic/`](../examples/logic/) |
| tell if a problem is tractable or a wall | `resona.cost.is_extractable(signal)` | [measuring-difficulty](measuring-difficulty.md) | [`spectral_phenomena/extraction_law.py`](../examples/spectral_phenomena/extraction_law.py) |
| detect a planted signal in noise | `resona.of(mv,N).extreme()` (BBP) | [reading-spectra](reading-spectra.md) | [`spike_detection.py`](../examples/spike_detection.py) |
| polish ONE eigenvalue to machine precision | `resona.solve.rayleigh_polish(mv, seed, N=N)` | [precision-and-defects](precision-and-defects.md) | [`spectra_to_machine_precision.py`](../examples/spectra_to_machine_precision.py) |
| solve a CLUSTERED / near-degenerate root problem fully | `resona.solve.catastrophe_solve(coeffs)` | [precision-and-defects](precision-and-defects.md) | [`spectra_to_machine_precision.py`](../examples/spectra_to_machine_precision.py) |
| know if my eigenvalue is trustworthy (non-normal defect) | `resona.defect.pseudospectrum_radius(A, ε, z0=λ)` | [precision-and-defects](precision-and-defects.md) | [`spectral_phenomena/nonnormal_convergence.py`](../examples/spectral_phenomena/nonnormal_convergence.py) |
| predict whether GMRES will stall | `resona.defect.sigma_min(A, 0)` + the bloom | [precision-and-defects](precision-and-defects.md) | [`spectral_phenomena/nonnormal_convergence.py`](../examples/spectral_phenomena/nonnormal_convergence.py) |
| tell integrable from chaotic (is there a lift?) | `resona.cost.level_spacing_ratio(λ)` | [measuring-difficulty](measuring-difficulty.md) | [`quantum/integrability_detector.py`](../examples/quantum/integrability_detector.py) |
| find conserved charges blind | `resona.lift.conserved_charge(H, basis)` | [lifting-nonlinear](lifting-nonlinear.md) | [`quantum/integrability_detector.py`](../examples/quantum/integrability_detector.py) |
| get a condition number matrix-free (a LOWER bound; `polish=True` for exact edges) | `resona.of(mv,N).condition()` | [measuring-difficulty](measuring-difficulty.md) | [`killer_tasks.py`](../examples/killer_tasks.py) |

## The mental model

Three verbs on one object — the operator's **spectral response**:

```
        PROBE                      READ                       COMPOSE
   resona.of(matvec,N)  ─────▶  .trace/.density/.extreme   sA + sB ,  sA @ sB
        │  (matrix-free, the matvec is the only cost)         │
        │                                                     ▼
        └────────────▶  resona.apply(matvec,f,v)  ────▶  f(A)·v  (solve / evolve / filter)
```

Everything else reads off the same object — the `Spectral` hub:

```python
s.boxplus(t)                  # A ⊞ B at the measure level (no joint matvec)
s.cauchy(z); s.r(w); s.s(w)   # the lifted coordinates (R linearizes ⊞, S linearizes ⊠)
s.cumulants(); s.flow(t, xs); s.shock_time()
s.levels(N)                   # whole spectrum from 4 numbers (Beta closure)
s.effective_rank(); s.condition()
```

Plus the **theory modules** (`wkernel`, `lift`, `beta`, `defect`, `free`,
`subordination`, `cost`, `flow`, `solve`) and the **inverse** (`from_measure`,
`from_eigenbasis`) — see the guides.

## Conventions & gotchas (read once)

- **You supply a matvec**, not a matrix. Sparse/implicit operators are the point.
- **Forward is matrix-free and cheap** (cost = the matvec); **inverse is polynomial
  in N** — see [`COMPLEXITY.md`](../COMPLEXITY.md).
- **`hermitian=False`** in `apply` for non-symmetric operators or complex `f`
  (quantum `e^{-iHt}`); the default symmetric path is cheapest.
- **Honest limits** are stated in each guide (conditioning, regularization,
  approximation order) — resona reports the real number, never a headline.

The references for full, runnable, *verified* implementations are the
[`examples/`](../examples/) (see [`examples/README.md`](../examples/README.md));
the theory behind each is in [`THEORY.md`](../THEORY.md).
