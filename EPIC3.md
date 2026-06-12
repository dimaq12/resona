# EPIC 3: the grand revise — "resona 2.0: nothing estimated that can be known"

This epic opens with ANSWERS, not features.  The owner asked three questions;
here is what the evidence (53 stands, two epics, one stress campaign, the
beauty pass) says — then the phases that act on it.

---

## ANSWER 1 — can the examples be more expressive and MORE ACCURATE?

**Yes, and the single biggest lever is embarrassing in the best way: the
gallery predates our own engines.**  Most stands were written before
deflate/KPM/certified/zoom/polish existed and still run plain `of()`.
A systematic RETROFIT pass should yield measured accuracy jumps:

| stand | today | lever | expected |
|---|---|---|---|
| killer_tasks GP log-det | 0.84% rel.err | deflate=K on the kernel's spikes + certified bracket | ~0.1–0.3%, with a [lo,hi] line |
| heisenberg_thermo | max err 19.4% (L=8 row) | deflate + more probes (now cheap) | < 8% |
| many_body_spectrum | ~2% MAE | KPM high-resolution density for the Beta closure inputs | ~1% |
| chern_from_noise | ±0.924 at L=12 (k=160 step-poly) | Chebyshev/Jackson projector (KPM-style) — cheaper per chain → L=16–20 affordable | ±0.97–0.99 |
| thermal_response calibration | 0.046 | probes ×4 (incremental stepping made them cheap) | ~0.02 |
| koopman VdP base mode | 7e-4 vs FFT | longer trajectory + rank sweep | ~1e-5 |
| zeta program | 3 heights, 0.91–0.97×ln(t/2π) | **Odlyzko zeros6 = the first 2,001,052 zeros** (we used 100k) | the Berry constant over a continuous height band; ±0.02-grade |
| spectra_to_machine ACT3 | 4e-16 of span | already at machine | leave (the floor is the floor) |

**Expressiveness:** two disciplines, not decoration: (a) every stand ends in
ONE punchline table (a few stands ramble); (b) PERSONA LANDING PAGES — five
half-page docs ("resona for quants / for physicists / for ML / for dynamics
/ for numerical analysts") that map each persona's first five tasks to five
calls.  The cookbook stays the reference; the landing pages are the doors.
NO plots — the text-metric discipline is the brand.

## ANSWER 2 — what should the API look like, given our use cases and users?

**The beauty pass's verdict stands: the SHAPE survived (8/10 blind
guessability).  2.0 is not a reorganization — it is an HONESTY COMPLETION.**
Who actually uses this library (read off our own gallery): quants/statisticians
(rie_clean, certified log-det), ML people (Hessians, trainability,
deflate), quantum/condensed matter (thermal, Chern, integrability),
dynamics/data people (koopman, cloud), numerical analysts (defect calculus,
track, certificates), mathematicians (free probability, zeta).  What they
ALL need is the same three things:

1. **EPISTEMIC PARITY** — the 2.0 north star: *every stochastic read offers
   `with_err`; every theorem-backed read offers `certified`; everything else
   is exact and says so.*  Today: trace/moment/effective_rank have bars;
   density/cumulants/extreme/kappa_w don't.  Close the surface.
2. **THE DIET** — apply the beauty-pass 2.0 bin in ONE break (small user
   base now = the cheapest moment in the library's life): split
   `trace`/`trace_certified`; remove `phi1`; rename `spectroscopy →
   defect_barycentres`; `synthesize` as the discoverable name over
   `from_measure`; lock `shock_time`'s knobs; consolidate apply's inline
   Lanczos (with a dedicated parity CI job).  One MIGRATION.md, one
   deprecation release (1.4) where old names still work and their docstrings
   point forward, then 2.0 removes.
3. **THE DOORS (interop, persona-driven)** — not new math, new sockets:
   accept `scipy.sparse.linalg.LinearOperator` natively everywhere a matvec
   is taken (one `_as_matvec` shim — quants and PDE people live there), and
   the genuinely new capability: **differentiable spectral reads** — for a
   parametric A(θ), d/dθ Tr f(A) = Tr(f′(A)·∂A/∂θ) estimated from the SAME
   probes; with the torch path already in the core this makes spectral
   regularizers trainable (penalize sharpness/condition during training).
   That is a new audience (ML) reached with ~one function (`grad_trace`),
   inside the boundary contract.

New-use-case candidates examined and NOT taken: streaming/online updates
(real demand unclear; revisit on request), sklearn wrappers (a thin veneer
that would rot), plotting helpers (against the brand).

## ANSWER 3 — how does everything unapplied integrate?

The full inventory, each item assigned exactly one fate:

**→ 2.0 core (the diet + parity):** trace split · phi1 removal ·
spectroscopy rename · synthesize alias · shock_time diet · apply-Lanczos
consolidation · density/cumulants/extreme/kappa_w error bars ·
r_inverse/s_inverse (the missing duals) · rie_clean domain note.

**→ new capability (phase NEW):** differentiable reads (grad_trace) ·
LinearOperator interop · generator_read_converged (the Richardson-verified
variant) — three sockets, all inside the +small cap.

**→ research (phase R, results land in theory/ + FRONTIER):**
zeros6 two-million-zero Berry scan · criticality-window scaling vs σ²
(parked in the stress campaign) · SFF retry via FILTERED typicality (the
honest path recorded when the SLQ-measure attempt failed) · the
W(A)-from-moments open question (when does σ² ∝ dist² hold — demoted gold,
still a real question) · orthogonal defect packets (99% empirical, shell
equivalence unproven — port only if the proof gap closes or is honestly
fenced).

**→ paper (phase P):** BDS/Method-8 — 35/35 + the new noise-robustness
section is a complete SIAM/JCP-grade manuscript; the library is its
artifact.  (The zeta Jacobi-rigidity note is the second candidate.)

**→ stays parked (with reasons):** Lindbladians (audience unclear until
asked), β-ensemble sampler (nice, not core), operator-valued free
probability (heavy; needs a driving use case), eta invariant (Chern stand
covers the topology story), defect cohomology (needs mathematics first),
⊠-deconvolution beyond MP (open math).

---

## Phases

- **Phase 0 — the retrofit audit (measure first).**  Re-run every stand
  with candidate engine upgrades IN A SCRATCH; record actual deltas in a
  table; only upgrades with measured improvement enter the gallery (the
  ratchet then ratifies them as the new baselines).  ~1 day.
- **Phase 1 — the retrofit + punchlines.**  Apply winning upgrades;
  punchline-table discipline; persona landing pages (5 × half-page).  ~1 day.
- **Phase 2 — 1.4 "the deprecation release".**  Old names intact, forward
  pointers in docstrings, MIGRATION.md published, new names live
  (trace_certified, defect_barycentres, synthesize, r_inverse/s_inverse),
  epistemic parity lands (bars everywhere).  Ship to PyPI.  ~1½ days.
- **Phase 3 — NEW sockets.**  grad_trace (differentiable reads, torch +
  numpy paths, verified against autograd ground truth), LinearOperator
  shim, generator_read_converged.  ~1 day.
- **Phase 4 — research week-end.**  zeros6 scan; criticality window
  scaling; SFF filtered-typicality retry (pre-registered: may fail again —
  then the parking note gets a second confirmation).  ~1 day.
- **Phase 5 — 2.0.**  Remove deprecated names; the parity CI job for the
  Lanczos consolidation; final beauty re-pass (same three personas, fresh);
  release with MIGRATION.md front and center.  GATE: the owner approves the
  break list explicitly before this phase runs.  ~1 day.
- **Phase P — the BDS manuscript** (parallel, background): draft from the
  stands; the owner decides on submission.

Contracts: all four inherited (elegance budget, ratchet, honesty ledger,
boundary) + one new: **the parity rule** — no stochastic read ships in 2.0
without its error bar; no theorem-backed read without its certificate path.

Estimated: ~6–7 focused days + the manuscript.  Every phase independently
shippable; Phase 5 (the break) is explicitly gated on the owner.
