# The tour — from zero to designing operators, in ten stops

Every stop: plain words first (🚶), then the one-line mathematical truth (🎓),
then runnable code. No stop needs the previous one's math — only its code.
All snippets run as-is after `pip install resona`.

```python
import numpy as np, resona
rng = np.random.default_rng(0)
```

---

## Stop 0 — the only thing resona ever asks of you: a matvec

🚶 An *operator* is anything that takes a vector and returns a vector of the
same size. You don't need its formula, its matrix, or its eigenvalues — only
the ability to *apply* it. That function `v ↦ Av` is called a **matvec**.
Everything resona learns — the eigenvalues and how strongly each one shows up —
is packed into one object called the *spectral measure*: think of it as the
operator's complete sound. Matvecs are everywhere:

```python
A = rng.standard_normal((2000, 2000)); A = A @ A.T / 2000
mv = lambda v: A @ v                  # a matrix you happen to have
# ...but equally: a graph (sum over neighbours), a PDE stencil (finite
# differences), an autograd Hessian-vector product, a quantum Hamiltonian
# acting on a state — anything that transforms vectors.
```

🎓 resona treats the operator as a black box and reconstructs its **spectral
measure** purely from the power sequence `{v, Av, A²v, …}` (Krylov data). The
matvec is the only oracle; total cost is counted in matvec calls.

---

## Stop 1 — listen to it

🚶 Strike a bell and the sound tells you its shape. `resona.of` "strikes" the
operator with a few random vectors and listens: which frequencies
(eigenvalues) ring, and how loudly (weights).

```python
s = resona.of(mv, 2000)               # ~400 matvecs, no matrix, no eig
print(s)                              # support, effective rank
lo, hi = s.extreme()                  # smallest & largest eigenvalue
rho = s.density(np.linspace(lo, hi, 300))   # the spectrum's shape
```

🎓 Stochastic Lanczos quadrature: each probe yields the Gauss quadrature of
the spectral measure seen from that vector — approximate eigenvalues (the Ritz
values) with weights; averaging
probes estimates the density of states. Extremes converge first (Kaniel–Paige).
Verified: 35-operator suite, `examples/spectra_to_machine_precision.py`.

---

## Stop 2 — ask it anything: trace(f)

🚶 Many "impossible" numbers are just sums over all eigenvalues. The
log-determinant — the workhorse of Gaussian-process learning and statistics —
is one line, and never builds the matrix:

```python
logdet = s.trace(np.log)              # log|A|, no Cholesky
exact  = np.linalg.slogdet(A)[1]
print(abs(logdet - exact) / abs(exact))    # a few % — from ~400 matvecs
```

🎓 `Tr f(A) = N·∫f dμ ≈ N·Σ wᵢ f(θᵢ)`. Any spectral functional: `Tr A⁻¹`,
partition functions `Tr e^{−βH}` (`examples/quantum/heisenberg_thermo.py`
does full thermodynamics at L=20 where dense eig needs 9 TB), Schatten norms,
counting functions. Stochastic accuracy ~2–4 digits; Stop 8 buys more.

---

## Stop 3 — how hard is my problem? (the dials)

🚶 Before paying for a computation, ask the operator how much structure it
has. `effective_rank` counts how many "modes" *really* participate: if a few
carry all the weight, it reads low — the problem is secretly small (cheap,
compressible, sometimes even "quantum" problems become classical — that is
dequantization). If every mode matters equally, it reads ~N: you are at the
genuine frontier, and no trick will save you.

```python
U = rng.standard_normal((2000, 8))
low  = resona.of(lambda v: U @ (U.T @ v), 2000)   # secretly rank-8
print(low.effective_rank())           # ≈ 8   → cheap
print(s.effective_rank())             # ≈ 1000 → genuinely full
print(s.condition())                  # κ = λmax/λmin (honest LOWER bound)
```

🎓 Φ₁ = (Σλ)²/Σλ² — the participation ratio, the dial of the Extraction Law
(`COMPLEXITY.md`). It decides dequantizability (`examples/quantum/dequantize.py`
vs `shor_wall.py`) and is the boundary where classical shortcuts die. Two more
dials: `cost.is_extractable` (lift-rank saturation) and `cost.level_spacing_ratio`
(⟨r⟩ = 0.386 integrable / 0.531 chaotic — `examples/quantum/integrability_detector.py`).

---

## Stop 4 — add operators without adding them

🚶 You measured two systems separately. What is the spectrum of their *sum*?
With matvecs, exactly: `s + t` re-probes `Ax + Bx`. Without any matvec —
knowing only the two spectra — there is still a universal rule, the same way
variances of independent noises add. (One catch: the rule assumes the two
operators are unrelated in a precise sense — *free*; `freeness_defect` below
tests exactly that.)

```python
t2 = resona.of(lambda v: A @ (A @ v) / 4, 2000)
both = s + t2                         # exact: probes (A + A²/4)x
m = s.boxplus(t2, order=4)            # measure-level: no joint matvec at all
```

🎓 `boxplus` is free additive convolution ⊞: free cumulants add,
`κₙ(A⊞B)=κₙ(A)+κₙ(B)` (Voiculescu/Speicher). Exact iff A, B are *free* —
test it: `free.freeness_defect` (≈0 free, O(1) correlated). The R-transform
`s.r(w)` linearizes ⊞ the way log linearizes convolution — "the Cole–Hopf of
free probability" (this library's framing; calibrated in `NOVELTY.md`).
`examples/spectral_phenomena/free_probability.py`.

---

## Stop 5 — use the operator like a function: f(A)·v

🚶 Solve a linear system, run heat through a network, propagate a quantum
state, low-pass-filter a signal that lives on a graph — all are "apply a
function of the operator to a vector", and all are the same call:

```python
b = rng.standard_normal(2000)
mv5 = lambda v: A @ v + v                          # A + I  (decently conditioned)
x  = resona.apply(mv5, lambda l: 1/l, b)           # solve (A+I)x = b
u  = resona.apply(mv, lambda l: np.exp(-0.1*l), b) # heat / diffusion
psi = resona.apply(mv, lambda l: np.exp(-1j*l), b.astype(complex),
                   hermitian=False)                # quantum e^{-iA}ψ
print(np.linalg.norm(A @ x + x - b) / np.linalg.norm(b))   # ~1e-7 from 48 matvecs
```

🎓 Krylov evaluation of matrix functions: Lanczos (symmetric) / Arnoldi
(general, complex f). One engine = solver + integrator + filter.
`examples/spectral_phenomena/universal_solver.py`, `affine_flow.py`,
the JWST denoiser in the README.

---

## Stop 6 — make the nonlinear linear (the lift)

🚶 Some "hopelessly nonlinear" problems are linear ones wearing a costume.
Find the right change of variables — the **lift** — and Stop 5 solves them.

```python
# logistic ODE  ẋ = x − x²  →  EXACTLY linear on z = (x, x², x³, …)
M = resona.lift.carleman_scalar([0.0, 1.0, -1.0], order=8)
x0, dt = 0.1, 0.05
z = np.array([x0 ** j for j in range(1, 9)])
z = resona.apply(lambda v: M @ v, lambda l: np.exp(dt * l), z, hermitian=False)
print(z[0].real)                      # x(0.05), machine-accurate per step
```

🎓 Carleman linearization; Cole–Hopf does Burgers (`examples/nonlinear_pde.py`,
residual 5e-9); over GF(p), `x^p ≡ x` makes ANY finite logic an exact linear
polynomial (`examples/logic/`, 0 errors on all pⁿ inputs). Existence of a lift
⇔ enough conserved charges — and `lift.conserved_charge` finds them blind.

---

## Stop 7 — write your own operator (synthesis)

🚶 So far we listened. Now compose: describe the spectrum you *want* — "two
bands with a gap, like an insulator" — and get a working operator that has it.

```python
xs = np.linspace(0, 5.5, 900)
band = lambda c: np.sqrt(np.clip(.25 - (xs-c)**2, 0, None))
rho = band(1.5) + band(3.5)                        # the ORDER: two bands, a gap
cdf = np.cumsum(rho); cdf /= cdf[-1]
levels = np.interp((np.arange(300)+.5)/300, cdf, xs)            # inverse CDF
al, be = resona.from_measure(levels, np.full(300, 1/300))       # REALIZE
mv_synth = lambda v: al*v + np.r_[be*v[1:], 0] + np.r_[0, be*v[:-1]]
print(resona.of(mv_synth, 300).extreme())                       # plays as ordered
```

🎓 The inverse spectral problem (Stieltjes): measure → Jacobi operator;
eigenvalues match the order to ~1e-14 (`examples/spectral_phenomena/operator_synthesis.py`,
which also closes the gap live with the free-heat knob `s.flow`/`s.shock_time`).
Honest limit: you get the tridiagonal *representative* — a measure does not fix
the eigenbasis (Horn: cross-moments are discarded information).

---

## Stop 8 — when numbers lie (precision & defects)

🚶 Two silent failure modes every numerics user eventually hits. (1) Near a
*cluster* of roots, double precision quietly keeps only a fraction of the
digits. (2) For a *non-normal* operator, the eigenvalues themselves mislead:
solvers see a whole "bloom" region, not points. resona detects both, sizes
them, and spends extra precision **only where the defect lives**:

```python
lam = resona.solve.rayleigh_polish(mv, s.extreme()[1], N=2000)  # 1 eig → ~1e-15
from fractions import Fraction as F
coeffs = resona.solve.exact_poly([1-F(1,10**5), 1, 1+F(1,10**5), 5])
roots, q, dps, naive = resona.solve.catastrophe_solve(coeffs)   # q=3 detected
J = np.diag([1., 1.], 1)               # the 3x3 Jordan block: eigenvalues {0,0,0}
r = resona.defect.pseudospectrum_radius(J, 1e-6)
print(q, dps, r)   # r ≈ ε^{1/3} = 0.01 — FOUR orders above ε: the spectrum lies
```

🎓 ε^{1/q} bloom of an order-q Jordan block (exact, tested q=2…5); the Arnold
A_{q−1} catastrophe costs 16/q digits and `dps = q·target` recovers them on
exact data — and *nothing* recovers them on rounded data (stated honestly).
GMRES follows the pseudospectrum, not the spectrum:
`examples/spectral_phenomena/nonnormal_convergence.py` — same spectrum, 14
iterations vs a stall. Guide: [precision-and-defects](precision-and-defects.md).

---

## Stop 9 — the walls (what resona will tell you it cannot do)

🚶 The best feature: before you burn compute, the dials say which of four
regimes you are in — and the fourth is a wall no trick removes.

| regime | dial reading | what to do |
|---|---|---|
| cheap | Φ₁ low / rank saturates | sample, compress, dequantize |
| liftable | charges found / ⟨r⟩ ≈ 0.39 | change variables, then linear tools |
| defect-bound | bloom ≫ ε / cluster q > 1 | pay precision ONLY on the q-core |
| genuine wall | Φ₁ ~ N, rank grows, no charges | no shortcut exists — and you know it *now* |

🎓 `examples/quantum/shor_wall.py` is the honest exhibit: `aˣ mod N` has no
lift-rank saturation — the dial reads "wall", which is *why* factoring needs
Shor. The library never claims to break it; it claims to *price* it.

---

## Where next

- **[Cookbook](README.md)** — "I want to…" recipes, one line each.
- **[Examples gallery](../examples/README.md)** — 44 verified scripts, from
  sorting to JWST images to quantum chains; each prints its own metrics.
- **[THEORY.md](../THEORY.md)** / **[FRONTIER.md](../FRONTIER.md)** — the
  unified picture, the open conjectures, and the recorded failures.
