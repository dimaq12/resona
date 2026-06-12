# resona for dynamics & data people

You have trajectories — sensors, simulations, markets — and you want the
*operator* behind them: frequencies, damping, stability, predictability.

Your first five tasks:

| task | call |
|---|---|
| turn a time series into an operator (Koopman/DMD) | `mv, rmv, r = resona.lift.koopman(delay_stack(sig, d))` |
| its frequencies & damping (complex spectrum) | `resona.cloud(mv, r).nodes` — on the unit circle = conservative, inside = decaying |
| is the dynamics linearizable AT ALL? | `resona.cost.is_extractable(sig)` + the `lift_rank` growth curve |
| stability of a non-Hermitian system (transient growth) | `resona.cloud(mv, N).abscissa()` — the NUMERICAL abscissa, the honest transient read |
| evolve / filter with the operator you found | `resona.apply(mv, f, x)` |

Calibration you can quote: on a Van der Pol limit cycle the Koopman base
frequency lands within 3×10⁻⁴ Hz of the FFT peak — **finer than the FFT's own
1×10⁻³ bin**; on Lorenz the same dials honestly refuse (all modes decay, the
delay-embedding rank grows without closing — chaos has no finite linear
chart, and resona says so instead of hallucinating one).

Worked examples: [`science/koopman_dynamics.py`](../../examples/science/koopman_dynamics.py),
[`science/lorenz_control.py`](../../examples/science/lorenz_control.py),
[`signals.py`](../../examples/signals.py).  Guides:
[lifting-nonlinear](../lifting-nonlinear.md),
[measuring-difficulty](../measuring-difficulty.md).
