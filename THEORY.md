# resona — the theory

Condensed companion research behind the library. The classical foundations are
credited in [`NOVELTY.md`](NOVELTY.md); here is the unified picture, with the
numbers from the verification scripts in [`theory/`](theory/). The unifying
claims are **research hypotheses**, labelled as such; the computations are
verified against ground truth.

---

## 1. The field

The "field of already-solved information" is the **resolvent / Green's function**
`G(z) = Tr((z−A)^{-1}) = Σ_p Tr(A^p)/z^{p+1}`. The answer to a problem pre-exists
the moment it is posed (`x=A^{-1}b` exists before you compute it); `G` holds every
solution at once. **Solving is extraction, not creation.** The cost of extraction
is the geometry of the field's singularities (§5).

## 2. One operator, two resolutions (the conjugate pair)

An operator is one **response measure** `μ_B = Σ_i (v_iᵀ B v_i) δ(λ_i)` seen two ways:

| | **density** (`W`) | **moments** (`Φ`) |
|---|---|---|
| object | `W_{ij}=v_iᵀ B_j v_i = ∂λ_i/∂k_j` | `Tr(A^p B)` |
| cost | needs eigenvectors, `O(N³)` | matrix-free, `O(N·p)` |
| composes? | — | yes |
| resolves λ? | yes | no |

Bridge identity (exact): `Σ_i λ_i^p W_i = Tr(A^p B)`. The two are **Fourier
conjugates** through `φ(t)=Tr e^{-itA}`, giving the uncertainty
`(λ-resolution) × (moment order) ≳ 1` — the blind spot of moments is literal
Heisenberg. The transform between them is **Lanczos / stochastic Lanczos
quadrature** — exactly what `resona.of` computes.

**The W⊥Φ watershed.** `W` and `Φ` sit on opposite sides of a cost watershed:
`Φ` (moments) is **matrix-free** — `Tr(A^p B)` needs only the matvec — while
`W` (the Hellmann–Feynman density `∂λ_i/∂k_j = v_iᵀB_j v_i`) needs the
**eigenvectors**, classically `O(N³)`. Moments compose and are blind to λ;
`W` resolves λ but does not compose. The watershed is real, but it is not the
whole spectrum: for a **selected** set of modes the W-side is matrix-free too —
`wkernel.kappa_w(modes=k)` / `track(modes=k)` run `eigsh` on the tracked block,
giving `∂λ` and its curvature `κ_W` for the bottom/top-k modes in `O(N·k)`
(measured ~294× over the dense eigh at N=4000, rel.err `1e-10`). So the dense
`O(N³)` survives only when you genuinely want **all N** eigenvectors; the
conjugate pair is matrix-free on both sides for any fixed mode budget.

## 3. Free probability (the algebra underneath)

The response algebra **is** Voiculescu free probability.

- **Additive closure.** `Tr((A+B)^p) = Σ_words Tr(word)` — composition closed in
  response coordinates (verified `1.6e-14`). The spectrum, by contrast, does NOT
  compose (`eig(A+B)≠eig(A'+B)` for equal spectra — Horn).
- **The linearizing coordinates.** `+` is linear in the **R-transform**
  (`R_{A⊞B}=R_A+R_B`); `×` in the **S-transform** (`S_{A⊠B}=S_A·S_B`). Free
  cumulants are the canonical "defect coordinates that compose."
- **Attractor.** Free CLT: `(X_1+…+X_K)/√K → semicircle`; `κ_4 ∝ 1/K` (verified,
  `κ_4·K≈−1.00`). The semicircle is the free Gaussian — why generic/disordered
  systems gravitate to "free."
- **The boundary is a theorem.** Closure is exact ⟺ the operators are FREE ⟺ all
  **mixed free cumulants** (= alternating centered moments `φ(ȦḂȦḂ…)`) vanish
  (Speicher). Verified: free ≈`1e-3` (O(1/√N)) vs non-free ≈`28` — ratio `6143×`.
  The non-closable residue **is** the freeness defect.
- **Structured disorder.** Heterogeneous disorder needs the **operator-valued /
  matrix Dyson** subordination `g_i=[(zI−A−diag(σ²g))^{-1}]_{ii}` (verified `4.7×`
  closer to truth than scalar).

## 4. The flow — and why the defect is a shock

Adding free semicircle of variance `t` evolves the Cauchy transform by the
**complex inviscid Burgers equation** `∂_t G + G ∂_z G = 0` (subordination is its
characteristic solution). Its **shocks are the spectral edges**: a gap `½δ₋₁+½δ₊₁`
closes at `t_c = −1/G₀'(0) = 1` — a band-merger phase transition = a shock
collision (verified).

> **Any shock is a sum of linearities.** The R-transform is the **Cole–Hopf** of
> free probability: in `R`-coordinates the shock-forming flow is a straight line
> (`R_{μ_t}(w)=R_{μ_0}(w)+t·w`, verified `1e-14`); the shock lives only in the
> density. This bridges the defect-calculus Carleman/Cole–Hopf insight to
> Voiculescu.

And the same point is the **edge of chaos** of the subordination fixed point:
its contraction `|T'|` runs `≈0.2` in the bulk (≈17 iters) → `→1` at the edge
(485 iters, verified). So:

```
DEFECT  =  Burgers shock  =  spectral edge  =  edge of chaos of the fixed point
        =  spectral phase boundary
```

Five names, one place — and it is literally a PDE shock, closing the loop to
numerical-analysis defect calculus.

## 5. The Extraction Law

```
   Cost(extract to accuracy ε)  ~  ε^{-a} · dist(z, Σ*)^{-b}
```

`Σ*` = the field's **non-removable** singular set (edges/branch-points/shocks/
continua — *not* isolated poles, which deflate); `b` = the singularity *type*.
Verified exponents: isolated pole `b≈0` (deflatable — answer extractable cheaply);
spectral edge `b≈0.5–1` (critical slowing); structureless → extensive (Shor).

Over a parameter family the cost is a **scalar field, singular on the
discriminant** (the EP / gap-closing locus) — the **phase diagram of
computability**. Emergent: difficulty landscape; geodesics = optimal solve paths
that route around the discriminant; curvature = `κ_W`.

**Is the answer extractable even where it seems not?** YES whenever the
singularity is **removable** — a finite lift makes it linear (`shock→R`, `EP→λ^q`,
`pole→deflate`); most apparent walls are this kind. The **operational test**: does
the lift's effective rank *saturate* (apparent wall, extractable) or *grow*
(genuine wall)? Verified: a 3-tone covariance saturates at `Φ₁≈6`; `aˣ mod N`
(Shor) grows without bound.

**Φ₁ as the dial.** `Φ₁ = Tr(A)²/Tr(A²)` (effective rank of the response) grades a
problem: low ⇒ structured / cheap / dequantizable; high ⇒ genuine frontier,
including the classical↔quantum boundary. Demonstrated: low-rank quantum speedup
dequantizes by sampling (dimension-independent); `aˣ mod N` reads as structureless
(`Φ₁` high) — Shor's wall, marked honestly by our own dial.

## 6. The non-Hermitian extension (by hermitization)

A non-Hermitian `A` has no spectral measure on the line; the right object is the
**Brown measure** `μ_A` on the complex plane (Haagerup–Larsen). It is matrix-free
in the same currency as the rest of the library, via **hermitization**: the
log-potential
`S(z) = (1/2N) Tr log((A−z)*(A−z)) = (1/N) Tr log|A−z|` is one **SLQ log-det**
per grid point — `resona.of(matvec, N).trace(log)` on the Hermitian dilation —
and `μ_A = (1/2π) Δ S` (Laplacian on the grid). The free additive sum of two
Brown measures is the per-z Hermitian free convolution of the hermitizations
(`brown_boxplus`). So the plane-valued spectrum costs one matrix-free log-det
per grid point — no `eig`, no SVD (the exact-SVD path is the `O(N³)` ground
truth only). `S` is log-singular on `supp μ_A`, so the stochastic estimate is
read as a smoothed density, not pointwise.

The conjugate-pair `W` also extends. The Hellmann–Feynman derivative
`∂λ_i = v_iᵀ B v_i` is Hermitian-only (orthonormal eigenvectors); for a
non-Hermitian `A` the correct generalization is **biorthogonal perturbation
theory** — `∂λ_i = (u_i* B v_i)/(u_i* v_i)` with `u_i, v_i` the left/right
eigenvectors (Arnoldi / shift-invert for the targeted complex λ).
`resona.cloud_flow` computes exactly this, reducing to `wkernel` when `A` is
Hermitian (`u_i = v_i`, denominator `=1`). The denominator `u_i* v_i → 0`
precisely at an **exceptional point** — where left and right eigenvectors
become parallel — so the biorthogonal `W` *is* the EP locator: the divergence
is the read, not a failure.

## 7. Verification scripts ([`theory/`](theory/))

| script | claim verified |
|--------|----------------|
| `free_prob_bridge.py` | W=Φ identity; additive closure; freeness defect (free vs non-free) |
| `free_clt.py` | free CLT (semicircle attractor); multiplicative closure |
| `freeness_criterion.py` | residue = mixed cumulants (`6143×`) — the boundary as a theorem |
| `operator_valued.py` | structured disorder → matrix Dyson (`4.7×` over scalar) |
| `burgers_shock.py` | free convolution = Burgers; shock at `t_c=1` |
| `shock_is_linear.py` | shock = sum of linearities (R = Cole–Hopf), `1e-14` |
| `subordination_chaos.py` | defect = edge of chaos of the fixed point (`|T'|→1`) |
| `extraction_law.py` | cost ~ `dist(Σ*)^{-b}`; cost-field singular on the discriminant |
| `manifold_extraction.py` | removable (lift saturates) vs genuine (grows) |

```bash
cd theory && python3 free_prob_bridge.py   # etc.
```

---

*The classical theorems are the field's (Voiculescu, Speicher, Biane; Golub–
Meurant, Ubaru–Saad; Tang; Haagerup–Larsen for the Brown measure; biorthogonal /
non-Hermitian perturbation theory for `∂λ`). The cross-field synthesis — the
response measure as a conjugate pair, the defect=shock=edge identity, the
Extraction Law, and Φ₁ as the dial — is the research contribution, offered openly.
See `NOVELTY.md`.*
