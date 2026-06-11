# Measuring difficulty

How hard is a problem — *before* you solve it?  resona reads difficulty as a
spectral / analytic property: a single dial (`Φ₁`), a removable-vs-genuine
classifier, and the Extraction Law.

## 1. The cost dial Φ₁ (one number)

```python
import numpy as np, resona
phi1 = resona.of(matvec, N).effective_rank()     # Φ₁ = Tr(A)²/Tr(A²)
```

Low `Φ₁` ⇒ structured / cheap / dequantizable (few modes carry everything); high
`Φ₁` ⇒ near the genuine frontier (full-rank, no cheap handle).  It is the dial that
separates classical-tractable from quantum-frontier (low-rank QML dequantizes;
`aˣ mod N` reads as full-rank — Shor's wall).

## 2. Removable vs genuine wall (from a signal)

```python
ok, ranks = resona.cost.is_extractable(signal)   # True ⇒ extractable, False ⇒ a wall
```

Lift a signal to its trajectory (Hankel) operator at a growing window and watch the
effective rank.  **Saturates** ⇒ a finite linear model exists ⇒ extractable (most
apparent walls are just bad coordinates).  **Keeps growing** ⇒ no finite chart ⇒ a
genuine wall (random, `aˣ mod N`).  Same dial as Koopman-linearizability,
dequantization, and minimal realization order.

## 3. The Extraction Law (the cost field)

```python
resona.cost.extraction_cost(eps, dist, a, b)     # Cost ~ ε^{-a} · dist(z,Σ*)^{-b}
a, b, c = resona.cost.fit_law(costs, eps, dist)   # fit the exponents from measurements
```

The cost of extracting an answer near the non-removable singular set `Σ*` (edges,
branch points, shocks) follows a power law; the exponent `b` is set by the
**singularity type**.  Measured: pole `b≈0` (deflatable), √-edge `b≈½`, exceptional
point of order q `b≈1−1/q`.  See [`theory/hardness_exponents.py`](../theory/hardness_exponents.py)
and [`FRONTIER.md`](../FRONTIER.md) §3.

## How it works

`Φ₁` is the participation ratio of the response — the effective number of modes.
The lift-rank test is `Φ₁` of the trajectory operator (Kronecker/Prony: a sum of
`r` modes has Hankel rank `r`).  The Extraction Law ties extraction cost to the
analytic geometry of the resolvent's singularities.

## Gotchas (honest)

- **`Φ₁` / `lift_rank` are SOFT ranks** (participation ratio) — monotone in the
  number of modes but **not** an exact counter. The binary extractable/wall split
  is sharp; the exact mode count needs a thresholded rank.
- **`Φ₁` is for PSD operators** (covariance/kernel/Hessian).
- The Extraction Law is a **research hypothesis** — exponents partly verified, the
  full catastrophe-stratification is open ([`FRONTIER.md`](../FRONTIER.md)).

## Worked examples

- **The Φ₁ dial, dequantization, the Shor wall** — [`examples/quantum/dequantize.py`](../examples/quantum/dequantize.py), [`examples/quantum/shor_wall.py`](../examples/quantum/shor_wall.py)
- **Extractable vs genuine wall + the cost-law fit** — [`examples/spectral_phenomena/extraction_law.py`](../examples/spectral_phenomena/extraction_law.py)
- **Hardness exponents `b_q = 1−1/q`** — [`theory/hardness_exponents.py`](../theory/hardness_exponents.py)
