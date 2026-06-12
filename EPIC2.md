# EPIC 2: resona 1.3 — "the defect listens back" (+ the API beauty pass)

Ports the stress-survivors of REVISE_2026-06-12 (see FA/revise_stress/
STRESS_REPORT.md: 4 of 5 gold items passed; W(A)-from-moments was demoted by
its own table and is NOT here).  Inherits the three contracts of EPIC.md
(elegance budget · metric ratchet · honesty ledger) and adds a fourth, which
is the soul of this epic.

---

## Contract 4 — the boundary contract (anti-dump, anti-god-method)

The user's words are the spec: «не хочу либу-свалку, не хочу один метод
делает всё подряд».  Operationalized:

1. **One function = one mathematical observable.**  A function returns a
   measured quantity; it never also decides, warns, retries, or "helps".
   (Precedent kept: `condition()` returns a number, the docstring carries
   the epistemics.  Precedent rejected: a `pastur_grid` that prints
   warnings mid-iteration.)
2. **No silent domain extension.**  Every port ships with the EXACT domain
   the stress test verified, stated in the signature's vocabulary (e.g.
   BDS takes explicit `bands`; the Koopman read takes explicit
   `solver="be"` and refuses others rather than approximating).
3. **Family placement before code.**  Each function must name its module
   family and justify membership in ONE sentence before implementation;
   if the sentence needs "and", it's two functions or zero.
4. **Two-sketch rule.**  Every new public signature is designed TWICE
   (sketch A/B below), prototyped against the FA stress stands, and the
   loser is recorded in this file — so future-us knows the road not taken.
5. **The cap.**  This epic adds at most 5 public callables and 0 new
   modules.  If a design wants more, the design is wrong.

---

## What enters (the stress-test survivors)

| item | stress verdict | pre-registered limits (from STRESS_REPORT) |
|---|---|---|
| Koopman-from-defect | generalizes, O(n⁻²) exact on new families | BE-specific constant (CN deviates O(1)); float32 = noise floor |
| BDS / Method 8 | 35/35 + robust to 5% noise (M6 dies at 1e-5) | ±1-bin rounding (λ err ~2Δk/k); ~5× cost vs ratio; verified on band-decomposable discretizations only |
| wkernel.track + κ_W | C1–C9 + fresh-seed-stable | κ_W is an ACCURACY dial (C8 ρ=0.929), NOT a cost dial (blind ρ≈0.05) — cost is path length L |
| criticality observable | |T'|→1 exact at true edges | critical WINDOW shrinks with σ² (soft edge @ σ²=0.1: 0.30 at d=1e-3) — the observable is direct, the window is not universal |

NOT entering: W(A)-from-moments (demoted — family-dependent, sign flips);
its open question goes to FRONTIER, not to code.

---

## Phase 0 — docstring debt + the smallest observable (½ day)

- `free.py`: the free-CLT attractor paragraph (why free is everywhere;
  resona/theory/free_clt.py is the verified backing).
- `cost.is_extractable`: name the dichotomy — REMOVABLE (rank saturates,
  finite lift) vs GENUINE wall (rank grows); cross-ref the koopman stand's
  leniency note.
- `Spectral.__add__`: name the closure theorem ("the response algebra is
  closed under + — machine-precision closure; this is what compose rests
  on").
- **`subordination.contraction(spectral, xs, sigma2)`** → |T′(g*)| at the
  fixed point, vectorized.  A pure read (boundary contract §1: no warning,
  no flag — the user compares to 1 themselves; the docstring shows the
  two-line recipe and the window caveat).  Tests: two-atom edge →
  0.9998-ish; soft-edge window caveat reproduced.
  *Family sentence:* "subordination owns the Pastur fixed point; this is
  that fixed point's stability — same family, one observable."

## Phase A — the defect listens back (1½ days)

The defect module's docstring already says "error-as-information"; this
phase completes the sentence: the defect carries the GENERATOR and the
SPECTRUM.  Two functions, one family.

### A1. `defect.generator_read(P_n, P_2n, t, n, solver="be")`
Returns the leading-defect generator term (for BE: (t²/2n)·A²e^{−tA}u₀
scaled out → the A²-observable on the evolved state), i.e. what the solver
already computed and threw away.
- *Family sentence:* "defect.py owns D_n = P_n − P_2n; this is D_n's
  leading coefficient given the solver's expansion — same family."
- **Sketch A:** return the raw vector G·u₀-read + scalar n-order check.
  **Sketch B:** return a small result object with .vector/.order/.residual.
  Default bias: A (resona returns numbers and arrays, not objects —
  the Spectral/Cloud exceptions earn their classhood by carrying algebra).
- `solver` accepts "be" only at 1.3; anything else raises with the CN
  deviation number from the stress test in the message.
- Stand: post-process a legacy integrator (write the integrator as a plain
  20-line BE loop, treat it as a black box) → Koopman-generator spectrum vs
  truth; the float32 floor and the CN refusal demonstrated.

### A2. `defect.spectroscopy(P_n, P_2n, t, n, bands, symbol=None)`  [BDS]
Per-band barycentric read of the defect's power: returns (k̄_j, signal_j)
per band, and λ_j = symbol[round(k̄_j)] when the caller provides the
discretization's symbol lookup.
- *Family sentence:* "the defect's power spectrum is a measure; this is its
  per-band barycentre — a read on D_n, defect family."
- THE BOUNDARY (this is where the dump-risk lives): resona does NOT guess
  the basis.  `bands` is an explicit list of index masks; `symbol` is the
  caller's array.  No FFT inside, no grid assumptions — the caller who has
  a Fourier discretization passes `np.abs(fft(D))**2`-compatible masks in
  one line (shown in the docstring and the stand).  This keeps the function
  a 30-line estimator instead of a 300-line framework.
- **Sketch A:** operate on the defect field D (caller did the transform).
  **Sketch B:** operate on (P_n, P_2n) and transform inside.
  Default bias: A — stricter boundary, zero hidden basis choices.
  (Then the signature is `spectroscopy(D_hat_power, bands, symbol=None)` —
  even cleaner; decide at prototype against method8_35.)
- Adaptive-n selection (the original's best-of-n loop) stays in the STAND,
  not the library — it is protocol, not observable (boundary contract §1).
- Stand: the 35-equation suite rerun through the library function — must
  reproduce 35/35 and the noise table (M8 stable at 5e-2, M6 dies 1e-5).

## Phase B — the W story completed, narrowly (1 day)

### B1. `wkernel.track(eigvals0, eigvecs0, perturbations, path)`
Integrate dλ = W(k)·dk along the path with eigenvector continuation
(crossing-safe).  Returns λ(path) [+ final vectors].
- *Family sentence:* "wkernel owns ∂λ/∂k; track is its line integral."
- Pre-registered numbers to reproduce: crossing test 8.9e-15 (vs 3.3
  sorted); ODE ≥100× frozen-W on the non-commuting family.
- **Sketch A:** generator `track(...)` yields per-step; **Sketch B:**
  one-shot arrays.  Default bias: B (simplest thing that ships; per-step
  control is the caller's loop over sub-paths).

### B2. `wkernel.kappa_w(eigvals, eigvecs, perturbations)`
The curvature dial.  Docstring is one paragraph and half of it is the
boundary: **predicts frozen-W ACCURACY (C8: ρ=0.929); does NOT predict
global cost (blind-seed ρ≈0.05) — cost is path length.**  The narrow
framing IS the feature.

## Phase C — THE API BEAUTY PASS (the user's special order) (1 day)

After A and B land and the ratchet is green, stop building and LOOK.

- Three fresh-eyes reviews, run as personas over the FULL public surface
  (README dictionary + cookbook + `dir(resona)` + every signature):
  (1) the newcomer — "which name did I guess wrong, which function did I
  look for in the wrong module"; (2) the mathematician — "where does a name
  promise more/less than the theorem"; (3) the maintainer — "which two
  functions are one function, which one function is secretly two".
- Me: a written taxonomy audit — every public callable gets one line
  "module · observable · why here"; any line that reads awkwardly is a
  finding.
- Deliverable: `API_REVIEW.md` with verdicts in three bins:
  **polish** (rename/docstring/alias — applied immediately, back-compat),
  **propose-2.0** (real reorganization — written up, NOT applied; breaking
  changes need their own decision), **leave** (justified as-is).
- Pre-commitment: if the review finds nothing, the deliverable says
  "nothing found" — beauty passes that must find something find noise.

## Phase D — release 1.3.0 (½ day)

CHANGELOG, cookbook/dictionary/COMPLEXITY rows (each port: one row), tour
untouched unless Phase C says otherwise, build with the in-wheel
`__version__` check, PyPI, tag, clean-venv verification.

---

## Order & budget

0 → A → B → C → D.  Estimated ~4 focused days.  Public surface delta:
+5 callables (contraction, generator_read, spectroscopy, track, kappa_w),
+0 modules, +0 verbs.  Ratchet (BASELINE2 lineage) at every phase close.

## Risk register

| risk | mitigation |
|---|---|
| BDS generalization creep ("just add 2-D grids…") | bands/symbol are caller-side by design; 2-D lives in a stand, not the signature |
| generator_read silently used with non-BE data | solver param + raise with the measured CN deviation in the message |
| kappa_w read as a cost oracle | the docstring's first line is the negative result |
| track duplicating design() | track = forward line integral; design = inverse step; one cross-ref line in each docstring |
| beauty pass turning into a 2.0 rewrite | three-bin protocol: breaking ideas are PROPOSED, never applied in this epic |
| api cap pressure (a 6th function "really needed") | it goes to the appendix of this file with a sentence why it lost |
