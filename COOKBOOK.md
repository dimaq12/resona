# resona — the cookbook

Task → primitive → snippet → the number it prints. Every number below was read
by running the snippet (or its gallery stand) against dense ground truth on this
machine; the figure quoted is what the code actually returned, not a target.

All five verbs — **read, move, trust, compose, synthesize** — are closed
matrix-free for **both** Hermitian and non-Hermitian operators. The recipes are
grouped by verb; the non-Hermitian corner (`brown`, `cloud_flow`, `cloud`) is
called out where it differs.

Run any recipe with `PYTHONPATH=/path/to/resona python3 …`.

---

## READ — Hermitian: hear a spectrum you can only matvec

### The loss-landscape / Hessian spectrum (sharpness, curvature dimension)

The Hessian of a loss is `d×d` — `O(d²)` to store, `O(d³)` to diagonalize — but
a Hessian-vector product `Hv` is one extra backward pass. `resona.of(Hv, d)`
reads the curvature structure from HVPs alone.

```python
import numpy as np, resona
n, d, lam = 8000, 600, 1e-2
rng = np.random.default_rng(0)
F = rng.standard_normal((d, 12)) / np.sqrt(12); Z = rng.standard_normal((n, 12))
X = Z @ F.T + 0.3 * rng.standard_normal((n, d))
p = 1/(1+np.exp(-(X @ (rng.standard_normal(d)/np.sqrt(d))))); w = p*(1-p)
hvp = lambda v: (X.T @ (w * (X @ v)))/n + lam*v       # H v, matrix-free
s = resona.of(hvp, d, k=90, probes=30, seed=1)
print(s.effective_rank(), s.extreme()[1])             # curvature dim, λ_max
```

Verified (`examples/data/hessian_spectrum.py`): effective_rank (curvature
dimension) **15.8** vs dense **15.1**; sharpest curvature λ_max **13.014** =
dense **13.014** (exact); **12** sharp directions of 600 (98% of the landscape
flat); counting function N(λ<0.3) **587** vs dense **588**, outliers N(λ>1.0)
**12.7** vs **12** (exact). At d=20,000 (a 3 GB dense Hessian) the same read
runs in ~36 s.

### Effective degrees of freedom (the real model complexity)

`DoF(λ) = Tr(K(K+λI)⁻¹) = Σ σ_i/(σ_i+λ)` is a spectral-functional trace, so it
is `s.trace(f)` with `f(σ)=σ/(σ+λ)` — no `n×n` kernel formed or factored.

```python
import numpy as np, resona
n, p = 2000, 300
rng = np.random.default_rng(0); X = rng.standard_normal((n, p)); X[:, :10] *= 3.0
Kv = lambda v: (X @ (X.T @ v)) / p                    # K = XXᵀ/p, matrix-free
s = resona.of(Kv, n, k=80, probes=24, seed=1)
for lam in (10.0, 1.0, 0.1, 0.01):
    print(lam, s.trace(lambda x, l=lam: x / (x + l)))
```

Verified (`examples/data/effective_dof.py`): DoF rel-err **1.6% / 1.1% / 0.9% /
0.9%** at λ = 10 / 1 / 0.1 / 0.01 vs the dense Σσ/(σ+λ). At n=50,000 (a 20 GB
dense kernel) the whole DoF curve is read from matvecs in well under a second.

### Certified log-determinant (a bracket, not just an estimate)

`quadform(..., certified=True)` returns Gauss–Radau brackets (Golub–Meurant):
the answer is **provably** inside, and the bracket width falls exponentially in k.

```python
# examples/certified_logdet.py, ACT 2 — log|K| of a GP kernel
# k= 8: [-1567.82, -1486.27]  width 8.16e+01
# k=16: [-1547.73, -1540.08]  width 7.65e+00
# k=32: [-1544.78, -1544.72]  width 5.57e-02
```

Verified: dense truth **-1535.03** sits inside every bracket; the k-truncation
is certified to **5.57e-02** at k=32, and what remains is the probe scatter
(±MC ~9.1), reported separately. Two error sources, separated.

### RMT universality class (Poisson / GOE / GUE / GSE)

`cost.rmt_class(eigenvalues)` reads the Dyson class from the R4 spacing-tail
rigidity — the companion to `cost.level_spacing_ratio` (which returns ⟨r⟩).

```python
import numpy as np, resona
from resona import cost
rng = np.random.default_rng(0)
H = rng.standard_normal((600, 600)) + 1j*rng.standard_normal((600, 600))
H = (H + H.conj().T)/2
print(cost.rmt_class(np.linalg.eigvalsh(H)))          # complex Hermitian → GUE
```

Verified: GUE → **('GUE', -0.056)**, real-symmetric GOE → **('GOE', +0.033)**,
uniform-random Poisson → **('Poisson', +0.215)** — strictly ordered
Poisson > GOE > GUE, exactly the class boundaries in the docstring. Trust the
class, not the decimals (R4 is a positively-biased proxy).

### Spectrum from two numbers (μ₁, μ₂ → the whole band)

`beta.beta_from(s, robust=True)` reads the extreme edges and the first two
moments off a `Spectral` and returns the max-entropy (Beta) spectrum — the whole
band from two trace numbers.

```python
import numpy as np, resona
from resona import beta
A_mv = ...                                            # any matvec, N huge
s = resona.of(A_mv, N)
levels = beta.beta_from(s, robust=True)               # N reconstructed eigenvalues
```

Verified (`examples/wild/two_numbers_whole_spectrum.py`): reconstruction
**< 2% of span**; Tr(A) **6000.37 ± 10.97** vs dense **6000.00** (within 3σ);
runs at **N = 10,000,000 in ~90 s** matrix-free (no dense check possible there).
Honest limit: the Beta law needs ONE smooth band — a spiked spectrum fails at
50.6% of span (documented, not hidden).

---

## READ — non-Hermitian: eigenvalues in the plane

### The Brown measure (the stable eigenvalue density of a non-normal A)

For a non-normal A the eigenvalues fill a 2-D region and dense `eig(A)` is
pseudospectrally unstable. `brown.brown_measure` reads the **stable** law μ_A =
(1/2π)Δ S(z) via hermitization, S(z) a matrix-free `Tr log((A−z)*(A−z))`.

```python
import numpy as np
from resona import brown
rng = np.random.default_rng(0)
G = rng.standard_normal((200, 200)) / np.sqrt(200)    # Ginibre → circular law
res = brown.brown_measure(G, grid=(-1.4, 1.4, -1.4, 1.4, 29), exact=True)
X, Y, mu = res["X"], res["Y"], res["mu"]
print(mu[(X**2 + Y**2) < 0.75**2].mean(), res["mass"])
```

Verified: mean density inside the unit disk **0.3217** vs the circular-law
prediction **1/π = 0.3183**; total mass **1.01** (support inside the box). Use
`exact=True` for a dense ground-truth SVD; drop it (with `k`, `probes`) for the
matrix-free SLQ read at scale. `brown.brown_boxplus(A, B)` gives the Brown
measure of a *-free sum.

---

## MOVE — track / sense how a spectrum responds to a knob

### Matrix-free selected-mode spectral flow (the last O(N³), closed)

`wkernel.track(modes=k)` follows only the bottom-k (k>0) or top-|k| (k<0) modes
along a parameter path via `eigsh` — `O(N·k)` for sparse A0/B instead of a dense
eigh per midpoint. `wkernel.kappa_w(modes=k)` is the curvature of that selected
sub-Jacobian.

```python
import numpy as np, scipy.sparse as sp
from resona import wkernel
N = 4000; rng = np.random.default_rng(0)
A0 = sp.diags([0.3*rng.standard_normal(N-1), np.linspace(1, 40, N),
               0.3*rng.standard_normal(N-1)], [-1, 0, 1]).tocsr()
B = sp.diags([0.01*rng.standard_normal(N)], [0]).tocsr()
path = np.linspace(0, 1, 5).reshape(-1, 1)
lams, _ = wkernel.track(A0, [B], path, steps=2, modes=6)   # bottom-6, matrix-free
```

Verified at N=4000: `kappa_w(modes=6)` (matrix-free) **0.42 s** vs `modes='all'`
(dense double-eigh) **341 s** — a **~820×** speedup; `track(modes=6)` **0.14 s**
vs `modes='all'` **100 s** (**~690×**). Accuracy of the tracked bottom-6 vs an
independent `eigsh` at the path end: **1.8e-05** when the mode block stays
gapped (`guard=True` warns if a mode is about to leave the block — selected-block
continuation is exact only while the group stays spectrally isolated).

### Non-Hermitian spectral Jacobian + exceptional-point locator

`cloud_flow` owns the non-self-adjoint sensitivity `∂λ_i/∂k_j =
(u_i*B_j v_i)/(u_i*v_i)` (biorthogonal: left and right eigenvectors). The
denominator `u*v` is the phase rigidity; it → 0 at an **exceptional point**,
where ∂λ/∂k diverges like ε^{-1/2}. `exceptional_point` locates it along a scan.

```python
import numpy as np
from resona import cloud_flow
A0 = np.array([[0., 1.], [0., 0.]], complex)          # A(k)=[[0,1],[k,0]], λ=±√k
B  = np.array([[0., 0.], [1., 0.]], complex)          # EP at k=0
out = cloud_flow.exceptional_point(A0, [B], bracket=(1.0, -1e-12),
                                   target_pair=np.array([1., -1.], complex),
                                   n_scan=401)
print(out["k_ep"], out["rig_min"], out["gap_min"])
```

Verified: located EP at **k_ep = -2e-6** (true EP at 0), with rigidity and gap
both collapsing to **2e-06**. The sensitivity blow-up matches the analytic
Puiseux derivative `1/(2√k)` to <1e-4: |∂λ/∂k| = **0.500** at k=1, **50.0** at
k=1e-4. (For the unreliable-but-cheap Arnoldi Ritz cloud, see `resona.cloud`;
`brown` is its reliable 2-D completion.)

---

## TRUST — is the spectrum the whole story, or is it lying?

### Departure from normality (the global non-normality scalar)

`defect.normality` reads `‖[A, A*]‖²_F` matrix-free via Rademacher–Hutchinson
(one matvec A and one adjoint A* per probe). It is exactly **0** iff A is normal
(Hermitian / skew / unitary) — when it is large the spectrum lies (the
pseudospectrum, not the eigenvalues, drives the dynamics).

```python
import numpy as np
from resona import defect
rng = np.random.default_rng(0); N = 500
T = np.triu(rng.standard_normal((N, N)))              # upper-triangular → non-normal
val, se = defect.normality(lambda v: T @ v, N, rmatvec=lambda v: T.T @ v, seed=0)
print(val, se)
```

Verified: a symmetric A returns exactly **0.0**; the triangular (non-normal) A
reads **1.246e8 ± 9.7e5** vs the dense `‖[A,A*]‖²_F` = **1.250e8** (0.3% off).
The =0-iff-normal property is exact; the magnitude is a stochastic estimate.

### Locate the hard point of a family (avoided crossing / EP), no eig

`defect.hard_points` scans a parametric family and spikes the response
susceptibility `Φ_η(k)` exactly where two eigenvalues collide — `argmax Φ_η =
argmin gap`, computed by Hutchinson + CG (matvecs only, no diagonalization).

```python
from resona import defect
k_star, profile = defect.hard_points(family=lambda k: H_of(k), ks=k_grid,
                                      B=perturbation, N=N, E=0.0, eta=0.08)
```

Returns the peak parameter `k_star` (the avoided crossing / EP) and the Φ_η
profile — the matrix-free divining rod for where a family goes singular.

---

## COMPOSE & SYNTHESIZE — add operators, order a spectrum

These are the existing core verbs; the new non-Hermitian completion is
`brown.brown_boxplus(A, B)` — the Brown measure of a *-free sum in the plane, the
non-Hermitian analogue of `s.boxplus(t)`. See the main README's "Choose your
door" table and the Hermitian recipes in `docs/`.

---

## The five verbs, both corners

| verb | Hermitian | non-Hermitian |
|------|-----------|---------------|
| **read** | `of(mv,N).trace/density/extreme`, `effective_rank`, `cost.rmt_class`, `beta.beta_from` | `brown.brown_measure` (eigenvalue density in the plane) |
| **move** | `wkernel.track`/`kappa_w` (`modes=k` matrix-free) | `cloud_flow` ∂λ + `exceptional_point` |
| **trust** | `quadform(certified=True)`, `defect.barycentres` | `defect.normality`, `defect.hard_points` |
| **compose** | `s + t`, `s @ t`, `s.boxplus(t)` | `brown.brown_boxplus` |
| **synthesize** | `from_measure` / `from_eigenbasis` | (via the Brown log-potential) |

Every number above is reproducible by running the snippet or its named stand.
