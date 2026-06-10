# Quantum tasks — one dial, five problems

These five examples collect the quantum-flavoured problems from across the
program (`theory/`, `commonOperators/`, and the `frontier/` branch) and re-express
them on the **resona** primitives. They are bound by a single object: the
**effective rank Φ₁** (`resona.effective_rank` = `Tr(A)²/Tr(A²)`, the participation
ratio) — *how many modes a response really carries.*

> **The thesis.** "You need a quantum computer" is, in many famous cases, a claim
> about cost — an `O(2ⁿ)` or `O(D³)` wall. That wall is only real when the object
> is genuinely **high-rank**. When Φ₁ is **low**, the answer pre-exists in a few
> modes and is **harvestable classically** — the quantum advantage is *redundant*.
> resona both **measures** Φ₁ and **exploits** it. The honesty is that the same
> dial points at the wall we **cannot** pass.

| # | file | problem | resona / Φ₁ role | result |
|---|------|---------|------------------|--------|
| 1 | [`many_body_spectrum.py`](many_body_spectrum.py) | full spectrum of an n-qubit Hamiltonian (TFIM), no `eig` | `extreme()` + `moment(1,2)` → Beta closure | ~2% MAE, **56× at n=12**, O(D) vs O(D³) |
| 2 | [`phase_transition.py`](phase_transition.py) | locate the quantum phase transition, no diagonalization | response susceptibility Φ_η (CG + Hutchinson), E₀ from resona | onset at **h≈1.05** (exact h_c=1) |
| 3 | [`entanglement_transition.py`](entanglement_transition.py) | measurement-induced (volume↔area) entanglement transition | entanglement entropy = **effective rank over GF(2)** | p_c bracketed **≈0.16–0.22** |
| 4 | [`dequantize.py`](dequantize.py) | beat "exponential" low-rank quantum ML (Tang) | low Φ₁ ⇒ length-squared sampling harvests it | overlap **1.000** touching **0.006%** of data |
| 5 | [`shor_wall.py`](shor_wall.py) | the honest boundary: where we *don't* beat quantum | high Φ₁ of aˣ mod N ⇒ no handle | Φ₁ **~10×** the structured signal |

## The two halves

**Compute quantum spectra matrix-free (1–2).** Where textbooks diagonalize an
`O(8ⁿ)` Hamiltonian, resona rings it with a few probe vectors and reads the few
invariants that pin the answer down — the support and two moments give the whole
spectrum (a local Hamiltonian's level density is smooth, max-entropy ⇒ Beta); the
response susceptibility Φ_η turns on at the critical point. No matrix is ever
formed; cost is linear in dimension.

**Map the quantum/classical boundary with Φ₁ (3–5).** Entanglement entropy *is* an
effective rank (over GF(2)) — the measurement-induced transition is a rank-scaling
transition, extensive (volume, "free") vs sub-extensive (area, "structured").
Low-rank quantum ML is *dequantized* precisely because low Φ₁ is sampleable
(Tang). And Shor's wall is exactly where Φ₁ goes high: aˣ mod N has no low-rank
structure, so there is no classical harvest — and our own dial says so.

## Honesty

- These are **classical** algorithms (stochastic Lanczos quadrature, Hutchinson,
  CG, Gottesman–Knill stabilizer simulation, length-squared sampling) — all
  classical, credited in the repo's [`NOVELTY.md`](../../NOVELTY.md). The
  contribution is the **unifying lens**: that Φ₁ decides redundant-vs-genuine, and
  that resona measures it.
- We beat **exactly the low-Φ₁ class** (low-rank linear algebra, Clifford
  circuits, area-law states) — **not all quantum.** Shor, volume-law dynamics, and
  full-rank problems remain the genuine frontier. The boundary is the *feature*:
  a framework whose own dial pointed across the Shor wall would be one to distrust.
- The "speedups" shown are real but model-specific (TFIM); the Beta closure is a
  ~2% approximation of the spectrum, not exact — the value is the **O(D)-vs-O(D³)
  scaling**, demonstrated to n=12.

```bash
for f in many_body_spectrum phase_transition entanglement_transition dequantize shor_wall; do
    python3 examples/quantum/$f.py
done
```
