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
   effective rank `Φ₁ = (Σw)²/Σw²` of the response as the **classical↔quantum /
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

## Honest status

The Extraction Law and `Φ₁`-boundary are **candidate** statements (sharp,
falsifiable, partly verified numerically in the companion research), not finished
theorems. The package's *computations* are sound (verified against dense ground
truth in `tests/` and `examples/`); the *unifying claims* are research hypotheses,
labelled as such.
