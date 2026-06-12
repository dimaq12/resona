# resona for numerical analysts

You care about what's *provable*: certified enclosures, convergence orders,
where precision dies and why.  resona's epistemics are built for you — every
stochastic read offers an error bar, every theorem-backed read a certificate,
and the discretization error of a legacy solver is treated as a SIGNAL.

Your first five tasks:

| task | call |
|---|---|
| certified enclosure of vᵀf(A)v (Gauss–Radau, provable) | `resona.quadform(mv, "inv", v, certified=True, support=(a, None))` |
| separate k-truncation error from Monte-Carlo error in a trace | `s.trace_certified("log", support=…)` vs `s.trace("log", with_err=True)` |
| read the GENERATOR out of a solver's defect (D_n = P_n − P_2n) | `resona.defect.generator_read(P_n, P_2n, t, n)` — the Richardson gap as spectroscopy |
| follow eigenvalues through crossings along a parameter path | `resona.wkernel.track(A0, Bs, path)` (continuation: 8.9e-15 where sorting breaks by O(1)) |
| solve to full precision near a defective cluster (ε^{1/q} catastrophe) | `resona.solve.catastrophe_solve(coeffs_exact, target_digits=15)` — auto-budgets dps = q×target |

The defect-calculus stance: when a solver at resolution n disagrees with
itself at 2n, that disagreement has the exact form (t²/4n)·A²e^{−tA}u₀ —
an *observable* of the operator.  resona reads it instead of merely
extrapolating it away.

Worked examples: [`certified_logdet.py`](../../examples/certified_logdet.py),
[`defect_spectroscopy.py`](../../examples/defect_spectroscopy.py),
[`spectra_to_machine_precision.py`](../../examples/spectra_to_machine_precision.py)
(machine precision on ALL eigenvalues, matrix-free).  Guides:
[precision-and-defects](../precision-and-defects.md),
[inverse-problems](../inverse-problems.md).
