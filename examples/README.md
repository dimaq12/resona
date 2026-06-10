# resona examples — the gallery

Every script is **self-contained** (numpy + scipy + resona only), runs in one
command, and prints its own metrics against ground truth. 🎯 = reaches **machine
precision** (≤1e-12) or **exact** results. Honest caveats are in each docstring.

```bash
python3 examples/<dir>/<name>.py
```

## Core (root)
| file | what | metric |
|------|------|--------|
| `killer_tasks.py` | 5 flagship matrix-free tasks (log-det, Hessian, A+B, trainability, Φ₁) | see table in README |
| `nonlinear_pde.py` | nonlinear Burgers via Cole–Hopf lift → `resona.apply` | residual 4.8e-9 |
| 🎯 `spectra_to_machine_precision.py` | 35 operators → spectra (Ritz seed → Rayleigh polish) | **1e-16, 99% machine-zero** |
| `spike_detection.py` | the BBP detection threshold — signal vs noise | λ=θ+1/θ above θ_c=1 |
| `anderson_localization.py` | metal→insulator transition, matrix-free LDOS | Λ: 0.97→0.15, 3.4s |
| `tracy_widom_edge.py` | universal fluctuation law of `extreme()` | std·N^⅔→1.27, exp −0.65 |
| `signals.py`, `image_anomaly.py` | spectral analysis of 1D signals / images | — |

## quantum/ — one dial (Φ₁), many problems · [details](quantum/README.md)
| file | what | metric |
|------|------|--------|
| `many_body_spectrum.py` | n-qubit TFIM spectrum from support+2 moments (Beta) | ~2% MAE, 56× at n=12 |
| `phase_transition.py` | locate the quantum phase transition, matrix-free | onset h≈1.05 (exact 1.0) |
| `entanglement_transition.py` | measurement-induced volume↔area transition | p_c≈0.16–0.22 |
| `dequantize.py` | beat low-rank quantum ML (Tang) by sampling Φ₁ | overlap 1.000 @ 0.006% data |
| `shor_wall.py` | the honest boundary — high Φ₁, no classical handle | Φ₁ ~10× structured |
| `laptop_quantum.py` | J1-J2 chain at L=16 (N=65536) where eigh is impossible | eigh 2.8e14 FLOP / 34 GB → resona 1.9s |
| `heisenberg_gap.py` | spectral gap from Z(β) curvature, no sector projection | gap scale ~1.1–1.5× (honest: scale only) |
| `hubbard_mott.py` | Hubbard Mott crossover via trace moments, no sign problem | DOS depletes 0.114→0.024 (79%) |
| `syk_chaos.py` | SYK free vs maximally-chaotic via self-averaging | CV gap +20% (correct direction) |
| `heisenberg_thermo.py` | full thermodynamics Z,F,E,S,Cv from the spectrum | E err 0.6% @ L=10; L=20 = 9 TB for eigh |

## graphs/
| file | what | metric |
|------|------|--------|
| 🎯 `spectral_sort.py` | sorting via the CDF-rank (response) operator | **0 mismatches** vs np.sort, 200K |
| `graph_structure.py` | bridges/articulation/k-core, O(1) after precompute | build 65ms @ V=5000; λ₂=0.876 |
| `inverse_graph_design.py` | design edge weights to hit a target spectrum (W kernel) | **60× fewer eigensolves** |
| `edge_weight_recovery.py` | recover hidden edge weights from the spectrum | **292× better accuracy** than FD |

## logic/ — the Carleman LIFT, made exact
| file | what | metric |
|------|------|--------|
| 🎯 `ternary_carleman.py` | GF(3) logic → linear polynomial operator (x³≡x) | **0 errors** on all 3ᴺ inputs |
| 🎯 `boolean_carleman.py` | precompute once → instant SAT / truth tables | **0 errors** vs brute force |
| `ternary_graph.py` | 3-valued-weight graph → Laplacian → spectral fingerprint | weight ratio 4.00× recovered |

## spectral_phenomena/
| file | what | metric |
|------|------|--------|
| 🎯 `defect_sort.py` | the defect IS the sorting operator | **0 mismatches** vs np.sort |
| 🎯 `universal_solver.py` | precompute the response field → instant solves (harvest) | 10,000 solves, **2.6e-15** residual |
| `arithmetic_manifold.py` | 12 bit-ops self-cluster by spectral fingerprint | silhouette 0.40 (partial, honest) |
| 🎯 `affine_flow.py` | exact exp(dt·J) flow for stiff/non-normal ODEs (`apply`) | accuracy 38–141× BE (timing caveat) |

## science/ — spectral lens on real physics
| file | what | metric |
|------|------|--------|
| `riemann_prime_wave.py` | Riemann-zeta zeros from prime number waves | 25/30 zeros, r=0.994 (von Mangoldt) |
| 🎯 `kepler_spectrum.py` | solar system as an operator; Kepler's 3rd law in the spectrum | exponent −1.49986 (−3/2) |
| `lorenz_control.py` | steer Lorenz chaos via W=∂λ/∂param | stabilized; dominant knob σ |
| 🎯 `maxwell_4_3.py` | the 4/3 EM-mass paradox as a spectral signature | ratio 1.333333 (caveat: scaling not rank-drop) |
| `dark_matter_rotation.py` | NGC 3198 rotation curve: Newton vs MOND vs halo | recovered a₀=1.15e-10 (Milgrom 1.2e-10) |

## wild/ — for fun (still honest)
| file | what | metric |
|------|------|--------|
| 🎯 `impossible_structures.py` | 1-look sorter + spectral hash + invisible store | EXACT sort, secret recovered 0 bit-err |
| `sacred_constants.py` | 100+ math constants as one operator (toy) | eff_rank 20.5/115 in log-space |

---

**Honesty.** The underlying algorithms (SLQ, Lanczos, Arnoldi, Hutchinson,
free probability, Carleman lift, length-squared sampling, Tarjan/k-core) are
**classical** and credited in [`NOVELTY.md`](../NOVELTY.md). Several demos are
physically *suggestive* rather than proofs (Maxwell, Riemann, dark matter,
sacred constants) — each says so in its docstring and reports the **real**
measured number, not a headline. Where a port's measured result fell short of the
original's claim (e.g. arithmetic-manifold clustering, affine-flow wall-clock),
the script states the honest number.
