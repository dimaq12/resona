# Novelty & Attribution

**Author:** Dmitry Sierikov, 2026.  **License:** MIT (code).

This note states **honestly** what is original to `resona` and what is classical
prior art it stands on. Calibrated honesty is the point: a tool that overclaims is
a tool you cannot trust.

---

## What we do NOT claim (classical foundations, credited)

`resona` is built on established numerical linear algebra and free probability. We
claim none of the following — they are the field's, and we use them gratefully:

- **Stochastic Lanczos quadrature (SLQ)** for `Tr f(A)` / spectral densities —
  Golub & Meurant; Ubaru, Chen & Saad.
- **Lanczos / Krylov subspace methods**; **Hutchinson** trace estimation;
  **Kernel Polynomial Method (KPM)**.
- **Free probability** — Voiculescu (R-transform, free convolution), Speicher
  (free cumulants, non-crossing partitions); **subordination** (Biane).
- Standard results: condition number / Krylov deflation, the Marchenko–Pastur
  law, Tracy–Widom edge, Tang-style dequantization of low-rank problems.

If you need a specific sub-algorithm, it predates this package.

## What `resona` contributes (and reserves)

The contribution is **synthesis, framing, and one interface** — not new
sub-algorithms:

1. **A single primitive — `Spectral`, the "FFT of operators".** One object that
   unifies *probe → compose → read* for any operator given as a black-box matvec.
   The organizing claim: `Spectral.of(A)` is to operators what `fft` is to
   signals — the representation in which the hard operation (composition / spectral
   computation) is linear, computed matrix-free.

2. **Matrix-free operator composition as an algebra.** Exposing `A+B`, `A·B`
   (via the exact closure `(A+B)x = Ax + Bx`, and free additive/multiplicative
   convolution) as `+` and `@` on responses — composing spectra **without forming
   the composed operator and without diagonalizing it**.

3. **A built-in cost / feasibility law — the *Extraction Law*.** The cost to read
   a solution from the resolvent field scales as the inverse distance to the
   field's **non-removable** singular set (edges/shocks/continua, not isolated
   poles), with the singularity *type* setting the exponent; over a parameter
   family this cost is a field singular on the **discriminant**. With it: the
   **removable-vs-genuine test** (does a finite lift's effective rank saturate?)
   as a computable criterion for *cheap / liftable / genuine wall*, and the
   effective rank `Φ₁ = Tr(A)²/Tr(A²)` of the response as the **classical↔quantum /
   cheap↔hard dial**.

4. **The unifying synthesis.** Identifying the **defect** (numerical-analysis
   error) = the **free-convolution shock** = the **edge of chaos of the
   subordination fixed point** = the **spectral phase boundary** as one object,
   and using it to bridge: matrix-free spectral computation, free probability,
   computational capacity, and the classical/quantum boundary — through one
   measurable quantity (the response and its effective rank).

**Reserved:** the framing and API design of the single-primitive interface; the
formulation of the Extraction Law and its removable/genuine criterion; the
`Φ₁`-as-dial proposal; and the unifying synthesis above. These are research
contributions, offered openly under MIT, with attribution requested.

## The mathematical ideas we read as ours

Beyond the code, `resona` encodes a small set of research ideas. Each rests on
classical mathematics (credited above); **what is ours is the cross-field
synthesis and the specific conjectures** — stated here with their status, so the
claim is exact and bounded. (Full development lives in the companion research;
this is the intellectual core the library embodies.)

**1. The Extraction Law.** The cost to extract a solution from the resolvent
(Green's) field scales as `ε^{-a} · dist(z, Σ*)^{-b}`, where `Σ*` is the field's
**non-removable** singular set (edges, branch points, shocks, continua — *not*
isolated poles, which deflate), and the exponent `b` is set by the singularity
*type*. Over a parameter family the cost is a scalar field **singular on the
discriminant** — the phase diagram of computability.
*Ours:* the single statement holding the same shape across solves, spectra,
disorder, and the P↔BQP boundary. *Classical:* the per-field cost laws
(condition number / Krylov, critical slowing, BBP, sample-complexity ∝ rank).
*Status:* candidate framework; exponents partly verified numerically.

**2. The identity  defect = shock = edge = phase boundary.** The
numerical-analysis **defect** (`D_n = P_n − P_{2n}`, the frozen-prediction error,
the spectral curvature `κ_W`) is identified with the **free-convolution Burgers
shock**, the **edge of chaos of the subordination fixed point** (`|T'|→1`,
critical slowing), and the **spectral phase boundary** — one object across
numerical analysis, free probability, and dynamics.
*Ours:* the identification as one object. *Classical:* each component
(Voiculescu/Biane free convolution = complex Burgers; critical slowing; band
edges). *Status:* synthesis, numerically illustrated.

**3. Φ₁ as the dial of difficulty.** The effective rank of the response measure,
`Φ₁ = Tr(A)² / Tr(A²)`, is proposed as the measurable quantity that grades a
problem: **low Φ₁ ⇒ structured / cheap / dequantizable; high Φ₁ ⇒ genuine
frontier** (including the classical↔quantum boundary). With it, the
**removable-vs-genuine test**: a finite lift's effective rank *saturates*
(apparent wall, extractable) or *grows* (genuine wall, e.g. Shor's orbit).
*Ours:* Φ₁-as-boundary and the lift-saturation criterion. *Classical:*
participation ratio; Tang-style low-rank dequantization. *Status:* sharp,
falsifiable conjecture; demonstrated on examples, not a proven boundary theorem.

**4. The response measure as a conjugate pair, with an uncertainty.** An operator
is one measure `μ_B = Σ_i (v_iᵀ B v_i) δ(λ_i)` at two resolutions — its **density**
(`W = ∂λ/∂k`, eigenbasis-resolved) and its **moments** (`Tr(A^p B)`, matrix-free)
— related by `Σ_i λ_i^p W_i = Tr(A^p B)`, and **Fourier-conjugate** via
`φ(t)=Tr e^{-itA}`, giving `(λ-resolution) × (moment order) ≳ 1`.
*Ours:* the framing of W and Φ as one measure at two resolutions, bridged by
Lanczos quadrature, with a Heisenberg-type bound. *Classical:* the identity itself
(elementary); SLQ. *Status:* framing.

**5. "Any shock is a sum of linearities" = the R-transform = the Cole–Hopf of free
probability.** The R-transform linearizes the shock-forming free-convolution flow
exactly as Cole–Hopf / Carleman linearizes Burgers — verified: the flow is a
straight line in `R`; the shock lives only in the density.
*Ours:* the explicit bridge tying the defect-calculus Cole–Hopf/Carleman insight
to Voiculescu's R-transform. *Classical:* free convolution = Burgers; R-transform;
Cole–Hopf. *Status:* framing/bridge, numerically verified.

**6. The harvest principle.** The answer pre-exists in the resolvent field;
solving is **extraction, not creation**; an operator is a program, and a generic
medium already computes (the response / reservoir). The defect is the
side-channel to collect.
*Status:* organizing principle / worldview, not a theorem.

## Honest status

The Extraction Law and `Φ₁`-boundary are **candidate** statements (sharp,
falsifiable, partly verified numerically in the companion research), not finished
theorems. The package's *computations* are sound (verified against dense ground
truth in `tests/` and `examples/`); the *unifying claims* are research hypotheses,
labelled as such.
