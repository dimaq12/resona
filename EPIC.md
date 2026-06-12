# EPIC: resona 1.2 — "certified, fast, and two new worlds"

One epic, six phases.  Everything below obeys three non-negotiable contracts,
stated first because they matter more than any feature.

---

## Contract 1 — the elegance budget (API)

The library is one object, three verbs, a hub.  This epic adds **zero new
top-level verbs**.  Every feature enters in exactly one of three sanctioned
forms:

1. **A parameter on an existing verb** — `of(engine=, deflate=)`,
   `trace(certified=)`, `s.zoom(a, b)`.  Default values reproduce today's
   behaviour **bit-for-bit**.
2. **A sibling object where the mathematics is genuinely different** — exactly
   one: `resona.cloud(mv, N)` for non-Hermitian spectra (complex Ritz cloud).
   A `Spectral` pretending its nodes are real would be a lie; a sibling is
   honest.  The cloud is *small*: probe + read only, no compose (free
   probability for non-normal operators without extra structure does not
   exist, and we do not fake it).
3. **A bridge function or a leaf module** — `lift.koopman` (data → matvec,
   one function), `resona.thermal` (3 functions on top of `apply`).  Leaf
   modules never import the hub; the hub never imports them.

Anti-goals: no plugin systems, no config objects, no `options=dict(...)`, no
method with more than 5 parameters, no new dependency beyond numpy/scipy
(GPU enters via the array-API of the *user's own* arrays, not via a new dep).

## Contract 2 — the metric ratchet (no stand regresses)

The protocol that already caught three regressions this week, formalized:

- **Freeze `BASELINE2`** before any change: full gallery (47 stands) outputs +
  TIMES.tsv + pytest, via `.audit/run_gallery.py`.
- **Every phase closes** with: all tests green → full gallery → `compare.py`
  classification = only `IDENTICAL` / `TIMING-OR-ADDITIVE`.  Any quality
  metric that moves must move **in the better direction**, and the diff is
  quoted in the phase's commit message.
- **Bit-parity rule**: default-parameter paths produce bit-identical
  `Spectral` objects.  A change to default numerics is allowed only when it
  *provably improves* a verified metric (precedent: rayleigh_polish, 99%→100%).
- Every new feature ships **(a)** tests against dense/analytic ground truth,
  **(b)** a gallery stand that prints its own metrics, **(c)** cookbook +
  theorem-dictionary rows.  No exceptions — a feature without a verified
  stand does not merge.

## Contract 3 — the honesty ledger

Each feature's docstring states what it cannot do, in the first screen.
Pre-registered limits are listed per feature below; if implementation
discovers more, they are added, not hidden.

---

## Phase 0 — rails (½ day)

- Freeze `BASELINE2` (gallery + times at low machine load, 3-run median for
  the timing columns we compare informally).
- `CHANGELOG.md` (we released 1.0→1.1.1 without one; starts now, backfilled).
- CI gains a fast gallery smoke (5 cheapest stands) so the ratchet runs on
  every push, not only locally.

## Phase A — precision flagship (1½ days)

### A1. Certified bounds: `s.trace(f, certified=True) → (lo, hi)`

**Math.** Golub–Meurant: for f with sign-definite derivatives on the spectrum
interval (log, 1/x, exp — the money functions), Gauss quadrature of each
probe's tridiagonal is one-sided; Gauss–Radau with a prescribed endpoint is
one-sided the other way.  Per probe → rigorous bracket; over probes → a
bracket for the trace estimate itself.

**Design.** `of()` already computes (α, β) per probe — *keep them*
(`self._tridiags`, additive attribute, no behaviour change).  `certified=True`
builds the Radau-augmented tridiagonals (one extra row each — O(k²), free)
with endpoints from `extreme()` padded by the Ritz residual bound.

**Honest limit (pre-registered).** The bracket is conditional on the endpoint
bracket and on f's derivative signs; both are checked and the conditions are
part of the return contract.  This brackets the *quadrature* error, while
across probes the statistical scatter keeps its `with_err` treatment — the
docstring separates the two error sources explicitly.

**Stand.** `examples/certified_logdet.py`: GP log-det with **[lo, hi]**
printed next to the dense truth sitting inside the bracket, plus a row in
killer_tasks (additive).  *No other library hands users a certificate; this
is the epic's crown.*

### A2. Spectral zoom: `s.zoom(a, b, k=…) → Spectral`

**Math.** Chebyshev band-pass filter p(A) via `apply` machinery → probe the
filtered operator → Ritz values inside [a, b] at full resolution; polish via
`solve.rayleigh_polish` for machine precision in the window.

**Design.** Method on the hub (needs the carried matvec).  Returns a normal
`Spectral` whose measure is the window's — every read works on it.

**Honest limit.** Filter leakage near window edges (reported as the filter's
measured transition width); needs `s.matvec`.

**Stand.** extend `spectra_to_machine_precision.py` (interior eigenvalues
without dense seed — today's weakness, closed) — metrics may only improve.

## Phase B — speed engines (1½ days)

### B1. Deflated probing: `of(mv, N, deflate=K)`

**Math.** Hutch++ generalized to the measure level: capture the top-K
eigenpairs exactly (block Lanczos, polished), probe the deflated remainder
stochastically.  Trace variance drops from O(1/p) on the full operator to
O(1/p) on the *residual* — for spiked spectra (covariances, kernels, Hessians:
our core clientele) this is the published quadratic matvec saving.

**Design.** One parameter; `deflate=0` (default) is today's path, bit-frozen.
The returned object is still nodes+weights (K exact atoms with weight 1/N +
stochastic bulk) — every downstream read works unchanged.  Composes with
`with_err` and A1's certificates (certificates on the residual + exact part).

**Honest limit.** Pays K extra polished eigenpairs; for flat spectra the gain
is ~1 (documented, and `effective_rank` is the dial that predicts the gain —
the library advising on its own parameters).

### B2. KPM engine: `of(mv, N, engine="kpm")`

**Math.** Chebyshev moments + Jackson kernel; no reorthogonalization (the
O(N·k²) term vanishes), stable at k ~ thousands.  Same-object contract:
Chebyshev moments → Gauss nodes/weights via the existing
`_gauss_from_moments` — the caller cannot tell which engine ran except by
speed and resolution.

**Honest limit.** Needs a spectral interval estimate (one cheap Lanczos
pre-pass); Gibbs smoothing is the Jackson kernel's η — stated like `density`'s.

**Acceptance for B.** Bench table in COMPLEXITY.md (measured, both engines,
3 sizes); gallery ratchet clean; killer_tasks/JWST timings same or better.

## Phase C — invisible GPU (2 days, isolated worktree)

**Design.** `_lanczos`, `_lanczos_block`, `apply` dispatch on the array
namespace of the probe/matvec output (numpy → today's code path untouched;
torch/cupy → same algorithm on-device; the small k×k eigh stays on host).
Zero API: a user whose matvec eats torch tensors gets GPU silently.

**The ratchet here is absolute:** CPU path must be bit-identical (the
dispatch happens before any arithmetic).  CI has no GPU — tests use the
array-api-strict shim to pin the dispatch logic; a measured torch bench lands
in COMPLEXITY.md from the dev machine.

**Honest limit.** Float32-default backends are documented loudly (precision
ledger: fp32 SLQ ≈ 2–3 digits; pair with `solve`/A1 for more).

## Phase D — two new worlds (1½ days)

### D1. `resona.cloud(mv, N, k=, probes=)` — non-Hermitian probe

Arnoldi from several probes → complex Ritz cloud.  Reads: `.radius()`
(spectral radius, from below), `.abscissa()` (max Re — stability), `.nodes`,
and `pseudospectrum` integration (defect.py already owns the honest part).
**Pre-registered honesty:** for non-normal operators Ritz values can be far
from eigenvalues — *that is the pseudospectrum story*, and `cloud`'s
docstring opens with it; `.radius()/.abscissa()` are lower bounds.

Opens: Markov mixing (gap of P), stability margins, and D2.

### D2. Koopman bridge: `lift.koopman(snapshots) → (mv, rmv, N)`

Data matrix of a dynamical system → the action of the DMD/Koopman operator
(least-squares propagator via one thin SVD of the data; the operator is never
formed).  Eight lines.  Then the ENTIRE existing machine falls onto data:
`cloud` reads its spectrum, `cost.is_extractable`/`lift_rank` grade the
dynamics, `conserved_charge` hunts invariants.  *"Your time series is an
operator"* — the bridge that turns resona from an operator library into a
data library, at the cost of one function.

### D3. `resona.thermal` — leaf module on `apply` (3 functions)

`state(Hmv, beta, N)` (typicality vector e^{−βH/2}|r⟩), `expect(Hmv, Omv,
beta)`, `correlator(Hmv, Omv, beta, ts)` → ⟨O(t)O⟩_β and S(ω) matrix-free.
Condensed-matter bread; verified on small chains vs dense (existing
heisenberg stands gain an additive act).  **Honesty:** typicality error
~1/√(2^L · Z-participation), printed by the stand.

## Phase E — the jaw gallery (1½ days)

All examples-only — zero API:

- **`quantum/chern_from_noise.py`** — Haldane model; projector P = sign(H)
  via `apply`; local Chern marker from stochastic probes returns **an exact
  integer** (−1/0/+1 across the phase diagram).  Quantized output from noisy
  Lanczos — the single most jaw-dropping demo we can build on what exists.
- **`quantum/spectral_form_factor.py`** — |Tr Uᵗ|² via Hutchinson on
  repeated `apply(exp(−iHt))`: the ramp–plateau of quantum chaos at L≈20,
  where exact diagonalization is dead.  Ties to ⟨r⟩/integrability stands.
- **`science/koopman_dynamics.py`** — D2's showcase: Lorenz + a flow dataset;
  spectrum of the data's operator, `is_extractable` verdict, honest
  "turbulence is a wall, periodic orbit is a chart".
- **`spectral_phenomena/isospectral_drums.py`** — cospectral graphs:
  `density` identical, `local_spectrum` tells them apart — "can't hear the
  shape, CAN hear it from a point" (the basis-layer duality made audible).

## Phase F — release 1.2.0 (½ day)

Tour gains one stop (certified bounds + zoom = "when you need to be SURE");
dictionary rows for Radau/Hutch++/KPM/Koopman/Chern; COMPLEXITY benches;
CHANGELOG; PyPI release with the now-standard clean-venv verification, tag,
absolute-URL check.

---

## Order & dependencies

```
0 → A → B → D → E → F        (C runs in parallel from B onward, isolated worktree)
        A1 ← per-probe tridiags (additive to of())
        B1 ← uses solve.rayleigh_polish (exists), composes with A1
        E  ← needs D1 (SFF? no — apply only), D2 (koopman stand)
```

Estimated effort: ~8 focused days.  Each phase is independently shippable;
the ratchet runs at every phase boundary, so the epic can pause at any line
with the library better than before it started.

## Risk register (pre-mortem)

| risk | mitigation |
|---|---|
| Radau endpoint estimate invalidates the certificate | endpoints from `extreme()` + Ritz residual padding; certificate returns its *conditions*; tests include an adversarial near-singular case |
| KPM measure ≠ Lanczos measure downstream | same-object contract tested: moments 1–6 agree within stated tolerance on 5 operator types |
| `deflate` perturbs RNG stream of default path | separate code branch; default bit-parity test (exists) extended to `deflate=0` |
| GPU dispatch breaks CPU bit-parity | dispatch strictly before arithmetic; parity test in CI via array-api-strict |
| `cloud` over-trusted on non-normal ops | docstring opens with the Ritz≠eig warning; `.radius/.abscissa` named and documented as bounds; pseudospectrum cross-referenced |
| Koopman least-squares ill-posed (short data) | rank-truncated SVD with the cutoff REPORTED; `effective_rank` of the data Gram as the honest dial |
| scope creep (the 12th idea mid-epic) | it goes to this file's appendix, not the working tree |

## Appendix — parked (explicitly NOT in this epic)

fp32+defect-correction engine (research first → `theory/`), Lindbladians,
β-ensemble sampler, operator-valued free probability, prime-trace formula in
Jacobi coordinates (FRONTIER §5 open question), eta invariant / spectral flow.

**Parked DURING the epic (honesty log):**
- the spectral form factor |Tr Uᵗ|² via the SLQ measure FAILED its honesty
  probe (corr 0.45 vs dense at L=10): the quadrature measure does not carry
  eigenvalue PAIR correlations at fluctuation scale — SFF needs either exact
  spectra or filtered-typicality estimators (future work, not faked).
  Replaced in Phase E by `thermal_response.py` (S(ω) by typicality, which
  the machinery does support honestly).
