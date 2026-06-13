---
name: library-change
description: "USE WHEN making ANY change to the resona library (or a similar metric-gated library) — a bugfix, optimization, refactor, or new feature, before touching library or example/stand code. Enforces the four-gate discipline so nothing ships unless it is quarantined + ratchet-clean, GUARANTEES an improvement to correctness/convergence/stability while regressing none, REDUCES API entropy without mixing concerns, and is released only after all of that is verified against ground truth. Invoke at the START of the change, not after."
---

# Library Change — the four-gate discipline

A change to the library is not done when it "runs" or "looks better". It is done
when it passes **all four gates**, each with measured numbers. (Hard-won lesson:
`phase_transition` shipped a stably-WRONG result for ages because the ratchet
checked output *stability*, not *correctness* — "looks saner" ≠ correct.)

Work the gates in order. Do not advance on a gate you have not actually measured.

## GATE 1 — QUARANTINE + RATCHET (isolation, no regression)
- Make the change in ISOLATION (a git worktree or branch), never edit-and-pray on `main`.
- It must pass BOTH arms of the ratchet:
  - **Diff arm** — `.audit/run_gallery.py <dir>` + `.audit/compare.py baseline <dir>`:
    no result line removed or changed except the ones you *intend* to change
    (timing lines are free). Any *intended* output change must be a re-baseline you
    can justify (see Gate 2), not a silent drift.
  - **Absolute arm** — `.audit/convergence_check.py`: zero NEW discarded
    solver-convergence flags, and every certificate PASSES.
- `python3 -m pytest -q` stays fully green.
- Re-running the touched stand twice must be **bit-identical** (seed any RNG;
  a non-deterministic stand cannot be verified).

## GATE 2 — GUARANTEED IMPROVEMENT (and zero regression)
The change MUST measurably improve **at least one** of these, and regress **none**:
- **correctness** — closer to DENSE / ANALYTIC ground truth. Verify against truth
  (free-fermion energy, `eigvalsh`, closed form, a from-scratch reimplementation),
  NOT against "it looks reasonable". Numbers before/after.
- **convergence** — fewer iterations / less wall-clock at the SAME accuracy
  (e.g. complex-shift κ instead of CG-on-squared κ²).
- **stability** — determinism, robustness on edge inputs, honest error bars
  (no stochastic estimate ever labelled "exact").
If you cannot point at a measured improvement in one of the three, it does not ship.

## GATE 3 — API ENTROPY DOWN, CONCERNS UNMIXED
- Any API change must LOWER entropy: fewer concepts, leaner surface, **one clear
  primitive per job**. Default to NO new top-level verbs.
- Never mix concerns — do not entangle unrelated responsibilities into one
  function/module just because it's convenient. Each thing does one thing.
- New surface is justified only by a removal, or a clear net simplification
  (the change makes the whole smaller/clearer, not just adds).

## GATE 4 — RELEASE (only after 1–3 are real)
- Bump version consistently (`pyproject.toml` + `__init__.__version__`).
- Update `CHANGELOG.md` — every number in it must be PRINTED by a test or stand,
  not asserted by hand; describe the measured before/after.
- Re-baseline any *intended* snapshot change (and add a certificate to the
  absolute arm so it can never silently regress again).
- `python3 -m build` + `python3 -m twine check dist/*`.
- Commit **under the alias the owner specifies** (NOT the default git identity),
  tag the version, `git push`, then `twine upload` (irreversible — confirm first).

> One line: quarantine it, prove it improves correctness/convergence/stability and
> breaks nothing, make the API smaller not bigger, then release with every claimed
> number reproduced. See also the `hypothesis-search` skill for the open-minded
> phase that PRECEDES a change.
