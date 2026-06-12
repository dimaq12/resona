# Migrating to resona 1.4 / 2.0

resona 1.4 was the **deprecation release**: every new name already worked,
every old name still worked.  **resona 2.0 (out now) removes the old names.**
Nothing about the math changes — same engines, same bit-frozen default paths,
same numbers (the 2.0 Lanczos consolidation inside `apply` is certified
bit-identical by the full example-gallery diff).

## Renames & splits

| you write today (≤1.3, still works in 1.4) | write instead (1.4+) | why |
|---|---|---|
| `s.trace(f, certified=True, support=…)` | `s.trace_certified(f, support=…)` | one name per return shape: `trace` always returns a number (or `(value, stderr)` with `with_err=True`); the certified BRACKET has its own name |
| `resona.defect.spectroscopy(power, bands)` | `resona.defect.defect_barycentres(power, bands)` | the function compresses each band to ONE coordinate (a barycentre); the old name promised a spectrum it does not deliver |
| `resona.from_measure(levels, w)` | `resona.synthesize(levels, w)` | the discoverable verb for "construct an operator with this spectrum"; `from_measure` remains as the precise synonym in 2.0 (it is not wrong, just hidden) |
| `resona.cost.phi1(s)` | `s.effective_rank()` | `phi1` was a thin alias; removed in 2.0 |

## New surface in 1.4 (no old equivalent)

- `resona.lift.r_inverse` / `resona.lift.s_inverse` — the missing duals of the
  R/S-transforms, for spectral design (monotone window required, bisected to
  machine tightness; they live beside `r_transform`/`s_transform`).
- **Error-bar parity**: `density(xs, with_err=True)`, `cumulants(n, with_err=True)`,
  `extreme(with_err=True)`, `wkernel.kappa_w(..., full=True)` — every
  stochastic read now offers its scatter, the same way `trace`/`moment`/
  `effective_rank` already did.

## Removed in 2.0 (done)

- `trace(certified=True)` — use `trace_certified(f, support=…)`; calling
  `trace` with the old kwarg now raises `TypeError`
- `defect.spectroscopy` — use `defect.defect_barycentres`
- `cost.phi1` — use `s.effective_rank()`
- `shock_time`'s four expert knobs (`n_t`, `n_x`, `thresh` are now internal
  constants 160/400/2%; the dials that remain: `t_max`, `eta`)

## Mechanical sweep

```bash
grep -rn "certified=True" your_code/ | grep trace      # → trace_certified
grep -rn "defect.spectroscopy\|cost.phi1" your_code/
```

If you do nothing: 1.4 keeps working and warns nowhere at import time —
deprecation lives in docstrings only (this library does not spam stderr).
Pin `resona<2` if you want the old names forever.
