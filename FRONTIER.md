# resona — the frontier (what building it revealed)

A research log, in the same calibrated-honesty spirit as [`NOVELTY.md`](NOVELTY.md):
what emerged from *implementing* the whole theory as one library, the open
conjectures, and — including the failures — what is actually verified. Nothing
here is new mathematics; the candidates are **cross-field syntheses**, each
labelled with its honest status.

---

## 0. Milestones (the build arc)

1. **Scattered folders → one object.** Every branch of the program computes the
   same thing: the response measure `μ_B = Σ_i (v_iᵀ B v_i) δ(λ_i)`.
2. **The conjugate pair.** `W` (density, `∂λ/∂k`, expensive, *resolves* λ) ⟂ `Φ`
   (moments, matrix-free, *composes*); bridged by Lanczos, with a literal
   Heisenberg bound `(λ-resolution)·(moment order) ≳ 1`.
3. **Free probability underneath.** Composition closes ⟺ the parts are FREE ⟺
   mixed free cumulants vanish (Speicher). Semicircle = the free-Gaussian attractor.
4. **One locus, five names:** defect = Burgers shock = spectral edge = edge of
   chaos of the subordination fixed point = spectral phase boundary.
5. **The lift:** `R`-transform = the Cole–Hopf of free probability; "a shock is a
   sum of linearities."
6. **`Φ₁` as the dial:** cost / dequantizability / the classical↔quantum boundary;
   the lift-rank saturation test for removable-vs-genuine walls.
7. **The library.** `resona` v0.3 — 8 modules absorbing the whole theory, an
   example gallery, C++ profiling (BLAS-bound everywhere except `carleman_gf` →
   20× by numpy), API hygiene (36 functions, none dead), the `grand_tour`
   capstone, and a public `local_density` primitive (added when a tail exposed a
   real gap).

---

## 1. What implementation revealed

### ① "Tractable" is *literally a rank* — always (sharp synthesis)

Building the examples side by side made it concrete that the same dial reappears
as an actual rank, only the operator and the field change:

| domain | the rank that decides tractability |
|--------|------------------------------------|
| measurement-induced transition (`mipt`) | entanglement entropy = **GF(2) rank** across the cut |
| low-rank quantum ML (`dequantize`, Tang) | **rank of the matrix** |
| Koopman / realization (`cost.is_extractable`) | **Hankel rank** of the trajectory (Kronecker/Prony) |
| structured disorder (`subordination`) | rank of the **matrix-valued** resolvent |

*Ours:* the observation that these are one computation, exposed uniformly by
`effective_rank` / `lift_rank` / `local_density`. *Classical:* each rank is
standard (von Neumann entropy, Tang, Kronecker, operator-valued free prob).
*Status:* **sharp synthesis** — it is the same dial, not a metaphor.

### ② The dial is *soft*, not an exact dimension counter (tested, downgraded)

We were tempted to claim `lift_rank` reads the **exact** linear dimension (a
"Kronecker thermometer"). **Tested it and it is false:** `lift_rank` of a sum of
`r` sinusoids grows monotonically with `r` but is *not* `2r` (r=8 → 4.9, r=12 →
13.3). It is the participation ratio `(Σσ²)²/Σσ⁴` — a **soft** rank that
under-weights small/clustered modes. What *is* razor-sharp is the **binary**
split (saturates ⇒ extractable; grows ⇒ wall: periodic/3-tone vs aˣ mod N/noise).
*Status:* honest correction — soft dial + sharp classifier, **not** an exact counter.

### ③ The lift is ONE functor across three unrelated categories

Written next to each other, these are the same move — *a hard composition becomes
addition in a basis of monomials/powers*:

- `carleman_gf` — logic over GF(p) → linear polynomial (= Reed–Muller / ANF);
- `carleman_scalar` — nonlinear ODE → linear (= Koopman / Carleman / Cole–Hopf);
- `r_transform` / `s_transform` — free `⊞`/`⊠` → `+`/`×` (Voiculescu).

*Ours:* the identification of all four as one *linearizing lift*. *Classical:*
every instance (Cole–Hopf, Koopman, R-transform, ANF). *Status:* **organizing
principle**, not a theorem.

### ④ The response is matrix-valued, read at three levels

Adding `local_density` exposed a ladder the library had only half-used:

```
trace  μ_B (random probes)  ⊂  local μ_v (one vector = LDOS)  ⊂  operator-valued G_ii (matrix Dyson)
```

A problem's cost is, in part, **how much of the matrix structure cannot be
collapsed to a scalar** — exactly why structured disorder needs the
operator-valued Dyson (4.7× over the scalar Pastur). *Status:* clean structural
realization; the operator-valued level is the genuinely richer, less-exploited object.

---

## 2. Experimental log (including what failed)

Calibrated honesty means recording the negative results too.

- **`lift_rank` exactness — REFUTED.** See ①②. Soft rank, not exact dimension.
- **Edge critical exponent — first attempt FAILED, second SUCCEEDED**
  (`theory/hardness_exponents.py`). The naïve subordination-iteration attempt
  broke (200k-cap artefact, `b≈0.9` discarded). Reframed onto two clean objects:
  - *Non-Hermitian, order-q EP* (companion `λ^q = s`): eigenvalue conditioning —
    an algorithm-independent lower bound — scales as `dist^{-(1-1/q)}`. Measured
    `b = 0.500 / 0.666 / 0.748 / 0.796` for `q = 2/3/4/5` (predicted `1-1/q`);
    splitting `s^{1/q}` confirms the order. **The hardness exponent IS quantized
    by the EP order.**
  - *Hermitian continuum edge*: actual Lanczos iterations to resolve an eigenvalue
    a gap `g` above a dense bulk scale as `g^{-0.43} ≈ g^{-1/2}` — the q=2 /
    square-root-edge class, by a different mechanism (Krylov, not conditioning).
  *Honest:* both exponents are CLASSICAL (Vishik–Lyusternik/Lidskii EP perturbation;
  Kaniel–Paige–Saad). What the experiment supports is the **unification** — one
  cost exponent set by the singularity type. Still open: the multi-parameter
  catastrophe stratification, and a rigorous lower bound.
- **Verified (unchanged, in `tests/` and `theory/`):** W=Φ bridge (`1e-13`),
  freeness defect (free `1e-3` vs non-free `28`, ratio `6143×`), R-transform
  additivity (`1e-14`), free CLT (`κ₄·K≈−1`), shock at `t_c=1`, Pastur `m₂`
  matches Monte-Carlo, the binary extractable/wall classifier.

---

## 3. The boldest open conjecture — a thermodynamics of computation

> **Conjecture.** The cost of extracting an answer near the non-removable singular
> set `Σ*` obeys **critical scaling with an exponent set by the singularity TYPE**:
>
> | singularity | predicted `b` | meaning |
> |-------------|---------------|---------|
> | isolated pole | `≈ 0` | deflatable — cheap regardless of distance |
> | √-edge (band edge) | `≈ 1/2` | critical slowing |
> | exceptional point, order q | `1 − 1/q` | algebraic |
> | continuum (structureless) | extensive | a genuine wall (Shor) |
>
> If these exponents **cluster into classes** (rather than smearing), there are
> *universality classes of computational hardness* — a physics of how hard a
> problem is, **measurable a priori** from the operator, before solving.

*Ours:* the cross-field statement + the universality-class framing. *Classical:*
critical slowing of fixed-point iteration near a branch point; per-field cost laws
(Krylov gap-dependence, EP perturbation theory). *Status:* the **EP-order column is
now measured cleanly** — `b_q = 1−1/q` for `q=2..5`, and the Hermitian edge gives
the q=2 exponent `½` as a real Lanczos cost (§2, `theory/hardness_exponents.py`).
The exponents themselves are classical; what is supported is the **quantization of
hardness by singularity type**. Still **unproven**: that this is a full Arnold
catastrophe stratification (the multi-parameter strata — cusp `⅔` from a genuine
3-fold coalescence, not just an algebraic order-3 EP), and a rigorous
algorithm-independent lower bound.

### What would turn this from rhetoric into a result

1. **Measure `b` cleanly for ≥4 singularity types** (pole, √-edge, EP of orders
   2–3, continuum) on controlled operators, with proper numerics — and show the
   exponents *cluster into classes*.
2. **Show the soft dial predicts real cost:** correlate `Φ₁` / `lift_rank` /
   `dist(z,Σ*)` against actual solver iterations (CG/Lanczos) across domains. If it
   does, the "complexity thermometer" is a measured fact, not a picture.

These two — *"tractable = a lift-rank"* and *"cost = criticality at Σ*"* — are the
only candidates here that could become a genuine paper, and only if the
universality experiment is carried out honestly.

---

*Everything above is synthesis of established results (Voiculescu, Speicher,
Biane; Kronecker/Prony; Tang; critical slowing). The contribution is the
cross-field framing — and the discipline of recording where it has been verified,
where it has been refuted, and where it remains an untested hope.*
