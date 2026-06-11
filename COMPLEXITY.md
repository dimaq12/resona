# resona — complexity of each method

Cost of every public method, with the architectural point made explicit: the
**forward** transforms are *matrix-free* (cost set by the matvec, not by `N`), the
**inverses** are polynomial in `N`, and the difficulty mirrors how much spectral
data each consumes.  Notation: `N` = dimension, `C` = cost of one matvec (`O(N)`
dense, `O(nnz)` sparse), `k` = Lanczos/Arnoldi steps, `p` = probes.  Empirical
slopes (measured `time ∝ N^s` on tridiagonal operators, N=64…512) are in **bold**.

## Forward / read — MATRIX-FREE (the point of the library)

| method | time | data | matrix-free | notes |
|--------|------|------|:-----------:|-------|
| `of(matvec,N,k,p)` | `O(p·k·C + p·k²)` — **N^0.2 (flat)** | matvec | ✓ | SLQ; cost is the matvec, not N |
| `apply(matvec,f,v,k)` | `O(k·C + k²·N)` (+`k³` Arnoldi) — **N^0.2** | matvec | ✓ | f(A)·v; solve/evolve |
| `local_spectrum / local_density` | `O(k·C + k²·N)` | matvec, one probe | ✓ | one Lanczos from a chosen v |
| `moment / trace / extreme / effective_rank / density` | `O(p·k)` | the nodes/weights | ✓ | reads of's output |
| `free.freeness_defect / cross_moment` | `O(p·|word|·C)` | matvec | ✓ | Hutchinson |
| `subordination.pastur` | `O(iters·N)` per z | spectrum (nodes,w) | ✓* | scalar fixed point |
| `subordination.averaged_dos / flow.burgers_density` | `O(n_x·iters·N)` | spectrum | ✓* | over an x-grid |
| `flow.shock_time` | `O(n_t·n_x·iters·N)` | spectrum | ✓* | scans t |

`✓*` = matrix-free given the spectrum, which `of` reads matrix-free.

## Inverse — recover the operator (the DATA ↔ CONDITIONING hierarchy)

| method | time | data needed | matrix-free | conditioning |
|--------|------|-------------|:-----------:|--------------|
| `from_eigenbasis(λ,V)` | `O(N²)` — **N^1.9** | full eigenbasis `V` (`O(N²)`) | ✗ (needs eig) | EXACT, every operator (~1e-14) |
| `from_measure(λ,w)` | `O(N²)`–`O(N³)` — **N^1.5** | one boundary measure (`O(N)`) | ✓* | exact (smooth) / blows up (sharp) |
| `wkernel.design(W,y,reg)` | `O(N³)` SVD (reg) / `O(m·M·minⁿ)` lstsq | response `W` | depends | bounded (reg>0), smoothed |
| `wkernel.wkernel(V,B)` | `O(m·M·C)` | m eigenvectors + M perturbations | ✗ (needs eigvecs) | — |

The more data you keep, the better conditioned the inverse: full `V` (`O(N²)` data)
→ exact; boundary measure (`O(N)` data) → ill-posed for sharp operators; eigenvalues
only → needs regularization (bounded but biased).  *Difficulty = data discarded.*

## Theory transforms

| method | time | notes |
|--------|------|-------|
| `lift.r_transform / s_transform` | `O(n_w·N·root)` | per w: a root-find over `G(z)=Σw/(z−λ)` |
| `lift.free_convolution` | `O(order³)` | cumulants ↔ moments, small `order` |
| `lift.carleman_scalar` | `O(order²)` build + `apply` | polynomial-ODE lift |
| `lift.carleman_gf(p,n,·)` | `O(M³)`, `M=pⁿ` (numpy-vectorized, ~20× the Python loop) | exact GF(p) logic lift; the one genuinely heavy kernel |
| `beta.beta_spectrum / beta_from` | `O(N)` | inverse-CDF |
| `defect.richardson` | `O(N)`; `richardson_limit` `O(L²·N)` | extrapolation |
| `free.free_cumulants / moments_from_cumulants` | `O(n³)` (truncated poly powers) | small n (moments) |
| `cost.lift_rank` | `O(k³)` (SVD of a k×k Hankel); `is_extractable` `O(#windows·k³)` | the dial |
| `cost.extraction_cost / fit_law` | `O(#data)` | the law |

## The headline

- **Forward = matrix-free.** `of`/`apply` cost the matvec, not `O(N³)` — flat in the
  measurement. That is the whole library: never form the matrix, never call `eig`.
- **Inverse = polynomial, and conditioning ∝ data kept.** `from_eigenbasis` is exact
  (`O(N²)`, full eigenbasis); `from_measure` uses `O(N)` boundary data and is
  ill-posed for sharp operators; regularization buys robustness at the price of bias.
- **One heavy kernel:** `lift.carleman_gf` is `O(pⁿ·³)` — exponential in the number of
  variables (it builds the full GF(p) monomial table); fine for `n ≲ 10`, vectorized.

*Measured with the benchmark in the commit that added this file; reproduce by timing
the methods across `N` and fitting `log(time)` vs `log(N)`.*
