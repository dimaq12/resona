# resona for quants & statisticians

You have covariance/kernel matrices too large to factor, eigenvalues
contaminated by sampling noise, and risk numbers that need to be *defensible*.
resona reads spectral quantities from matvecs alone — and tells you how much
to trust each read.

Your first five tasks:

| task | call |
|---|---|
| log-det of a kernel/covariance, no Cholesky | `resona.of(mv, N, deflate=64).trace("log")` |
| …with a PROVABLE bracket (audit-grade) | `s.trace_certified("log", support=(jitter, None))` |
| clean a noisy sample covariance (RIE, the Ledoit–Péché estimator) | `resona.free.rie_clean(eigs, q=N/T)` |
| detect real factors vs noise (the BBP threshold) | `resona.of(mv, N).extreme()[1]` — outliers above the bulk edge are real; see [`spike_detection.py`](../../examples/spike_detection.py) |
| effective number of factors (participation) | `s.effective_rank(with_err=True)` |

The two keywords that pay your bills: `deflate=K` makes the top-K eigenpairs
(your factor spikes) **exact atoms** — variance on spiked traces drops orders
of magnitude; `trace_certified` returns a Gauss–Radau bracket the true value
provably lives in (truncation error, separated from sampling error).

Worked example: [`covariance_cleaning.py`](../../examples/covariance_cleaning.py)
(RIE beats both the raw sample and naive shrinkage, measured),
[`killer_tasks.py`](../../examples/killer_tasks.py) task 1 (GP log-det, 0.5%
matvec-only).  Guides: [reading-spectra](../reading-spectra.md),
[composing-operators](../composing-operators.md).
