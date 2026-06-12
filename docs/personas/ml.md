# resona for ML people

Your Hessian is d×d with d in the millions, but you can compute Hessian-vector
products (autograd gives them for free).  That is exactly resona's input.

Your first five tasks:

| task | call |
|---|---|
| sharpness λ_max(H) from HVPs | `resona.of(hvp, d).extreme()[1]` |
| total curvature Tr H, with an error bar | `s.moment(1, with_err=True)` |
| Hessian eigenvalue DENSITY (the loss landscape's shape) | `s.density(xs)` — negative mass = saddles, outliers = sharp directions |
| GP / Laplace marginal-likelihood log-det | `resona.of(Kmv, N, deflate=64).trace("log")` |
| effective dimensionality of a representation | `resona.of(cov_mv, d).effective_rank(with_err=True)` |

Notes that save you a day: `deflate=K` is Hutch++ — on spiked spectra
(kernels, covariances, NTKs are all spiked) it collapses the estimator
variance by orders of magnitude for K extra Lanczos vectors.  The default
read path runs on GPU transparently: hand resona a matvec that takes/returns
torch CUDA tensors and the harvest stays on device, bit-identical reads
(`deflate`/`engine="kpm"` are CPU-side today and say so if asked).

Worked examples: [`killer_tasks.py`](../../examples/killer_tasks.py) (tasks
1–2: GP log-det, Hessian sharpness),
[`image_anomaly.py`](../../examples/image_anomaly.py) (graph-Laplacian
filtering).  Guides: [reading-spectra](../reading-spectra.md),
[measuring-difficulty](../measuring-difficulty.md).
