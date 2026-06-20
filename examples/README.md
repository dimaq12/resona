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
| ⭐ `grand_tour.py` | **the whole theory in one chained pipeline** — all 8 modules, consistent | Δm 0.01, beta 0.6%, t_c≈1, design 3e-15 |
| `killer_tasks.py` | 5 flagship matrix-free tasks (log-det, Hessian, A+B, trainability, Φ₁) | see table in README |
| `nonlinear_pde.py` | nonlinear Burgers via Cole–Hopf lift → `resona.apply` | residual 4.8e-9 |
| 🎯 `spectra_to_machine_precision.py` | 35 operators → spectra (Ritz seed → Rayleigh polish) | **1e-16, 99% machine-zero** |
| `spike_detection.py` | the BBP detection threshold — signal vs noise | λ=θ+1/θ above θ_c=1 |
| `anderson_localization.py` | metal→insulator transition, matrix-free LDOS | Λ: 0.97→0.15, 3.4s |
| `tracy_widom_edge.py` | universal fluctuation law of `extreme()` | std·N^⅔→1.27, exp −0.65 |
| `inverse_spectral.py` | hear the shape of an operator — the inverse of `of` (`from_measure`) | recover 3e-7; eigenvalues-alone fail |
| `signals.py`, `image_anomaly.py` | spectral analysis of 1D signals / images | — |
| 🎯 `covariance_cleaning.py` | FREE DECONVOLUTION: un-add MP noise from a sample covariance (RIE) | 1.81x closer, 95% of oracle; risk self-deception 4.15x → 0.94x |
| 🎯 `defect_spectroscopy.py` | the solver's ERROR read back as physics (generator + band spectrum) | generator to 4.2e-5 (Richardson-checked); barycentre holds 5% noise, ratio dies at 1e-5 |
| 🎯 `certified_logdet.py` | Gauss–Radau brackets: the answer PROVABLY inside | GP variance certified, width →3.8e-4; truncation vs scatter separated |

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
| 🎯 `integrability_detector.py` | is it integrable? ⟨r⟩ + blind conserved-charge search | ⟨r⟩ 0.392/0.532 (refs .386/.531); 4 vs 1 charges |
| 🎯 `chern_from_noise.py` | a topological INTEGER from Krylov chains (Haldane, real space) | trivial: 0.0000 exact; topological ±0.92→±0.97 with L |
| `thermal_response.py` | S(ω) by typicality where dense dies (L=14, D=16384) | L=8 calibration printed; sum rule C(0)=⟨O²⟩ agrees |

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
| 🎯 `defect_sort.py` | the defect IS the sorting operator (+ its pseudospectrum) | **0 mismatches** vs np.sort; bloom ε^{1/q} exact |
| 🎯 `nonnormal_convergence.py` | GMRES follows the PSEUDOspectrum, not the spectrum | same spectrum: 14 iters vs stall; σ_min 1e-9 |
| 🎯 `operator_synthesis.py` | SYNTHESIZE operators: design a measure (bands, ⊞, flow) → realize a tridiagonal matvec | eig = order to **5.6e-15**; gap dip 0.054; two materials from one knob |
| 🎯 `isospectral_drums.py` | Kac executable: identical spectra, local_spectrum hears the shape | spectra equal to 1e-15; local profiles differ |
| 🎯 `universal_solver.py` | precompute the response field → instant solves (harvest) | 10,000 solves, **2.6e-15** residual |
| `arithmetic_manifold.py` | 12 bit-ops self-cluster by spectral fingerprint | silhouette 0.40 (partial, honest) |
| 🎯 `affine_flow.py` | exact exp(dt·J) flow for stiff/non-normal ODEs (`apply`) | accuracy 38–141× BE (timing caveat) |
| 🎯 `free_probability.py` | free cumulants, freeness criterion, R-transform additivity | R add. 9e-4; freeness 1e-4 vs 3.0 |
| `free_convolution_flow.py` | compose spectra w/o joint matvec; Pastur DOS; Burgers shock | Δm 0.01; m₂ vs MC; t_c≈1 |
| `extraction_law.py` | removable vs genuine walls by lift-rank saturation | periodic→extract, aˣ mod N→wall; law fit (1.5,0.8) |

## science/ — spectral lens on real physics
| file | what | metric |
|------|------|--------|
| `riemann_prime_wave.py` | Riemann-zeta zeros from prime number waves | 25/30 zeros, r=0.994 (von Mangoldt) |
| 🎯 `kepler_spectrum.py` | solar system as an operator; Kepler's 3rd law in the spectrum | exponent −1.49986 (−3/2) |
| `lorenz_control.py` | steer Lorenz chaos via W=∂λ/∂param | stabilized; dominant knob σ |
| 🎯 `maxwell_4_3.py` | the 4/3 EM-mass paradox as a spectral signature | ratio 1.333333 (caveat: scaling not rank-drop) |
| `dark_matter_rotation.py` | NGC 3198 rotation curve: Newton vs MOND vs halo | recovered a₀=1.15e-10 (Milgrom 1.2e-10) |
| 🎯 `hilbert_polya.py` | THE zeta-zero operator: built, verified, interrogated | eig=zeros to 2.8e-13; Weyl corr 0.9996; ⟨r⟩ 0.615 vs GUE ctrl 0.590; β-fluct 1.42x smoother than GUE |
| 🎯 `zeta_ascent.py` | Odlyzko's ascent: 100k zeros + #10^12 + #10^21 through two dials | ⟨r⟩=0.6000 at #10^12 (4 digits); β-rigidity excess decays 1.42x→1.06x |
| 🎯 `koopman_dynamics.py` | your time series IS an operator (Koopman bridge) | odd-harmonic ladder to 7e-4 vs FFT; chaos rank grows ∝ window |

## wild/ — for fun (still honest) · the "Professors' Wall" flagships
| file | what | metric |
|------|------|--------|
| 🎯 `free_probability_calculator.py` | predict the FULL spectrum of `A⊞B` / `A⊠B` from the two spectra alone, never forming the composite; the certificate knows its domain | ~1% vs dense; `freeness_defect` blows up off-domain |
| `sculpting_eigenvalues.py` | inverse spectral at scale — DESIGN `A(k)=A0+Σk_e B_e` so m chosen eigenvalues hit a target (matrix-free Hellmann–Feynman `W` + regularized `design`, sparse `eigsh`, no dense eigh) | m eigenvalues land on the target |
| `certifying_catastrophe.py` | the spectrum LIES for non-normal flow — predict transient growth + GMRES stall at the deepest ε-bloom, matrix-free, BEFORE any solve (`defect.normality` + `hard_points`) | locates the hard `z`; normal twin converges fast |
| `two_numbers_whole_spectrum.py` | support + 2 moments → the ENTIRE N-eigenvalue spectrum (Beta max-entropy) + a Gauss–Radau-certified log-det | <2% of span; certified interval contains the exact value |
| `kesten_mckay_at_scale.py` | the spectral density of a million-node d-regular graph, matrix-free, vs the analytic Kesten–McKay law | bulk <1%; edges smoothed (reported) |
| 🎯 `impossible_structures.py` | 1-look sorter + spectral hash + invisible store | EXACT sort, secret recovered 0 bit-err |
| `sacred_constants.py` | 100+ math constants as one operator (toy) | eff_rank 20.5/115 in log-space |

## data/ — the spectral lens on machine learning
| file | what | metric |
|------|------|--------|
| `hessian_spectrum.py` | the loss Hessian spectrum from HVPs alone — a bulk of flat directions plus a few sharp outliers, never forming the d×d Hessian | `effective_rank` + sharpest λ vs dense |
| `effective_dof.py` | the effective degrees of freedom `Tr(K(K+λI)⁻¹)` of a ridge/kernel fit, matrix-free over a λ-sweep — the AIC/GCV complexity curve | DoF(λ) vs the dense `Σσ/(σ+λ)` |

---

**Library, not boilerplate.** The examples call the theory modules
(`resona.wkernel/lift/beta/defect/free`) rather than re-deriving them — e.g.
`many_body_spectrum` uses `resona.beta`, the graph/Kepler demos use
`resona.wkernel`, the logic demos use `resona.lift.carleman_gf`,
`free_probability` uses `resona.free`. Where a demo's structure genuinely doesn't
fit a primitive (Lorenz's parameter-nonlinear Jacobian; Rayleigh-polish vs
Richardson), the script says so instead of forcing it.

**Honesty.** The underlying algorithms (SLQ, Lanczos, Arnoldi, Hutchinson,
free probability, Carleman lift, length-squared sampling, Tarjan/k-core) are
**classical** and credited in [`NOVELTY.md`](../NOVELTY.md). Several demos are
physically *suggestive* rather than proofs (Maxwell, Riemann, dark matter,
sacred constants) — each says so in its docstring and reports the **real**
measured number, not a headline. Where a port's measured result fell short of the
original's claim (e.g. arithmetic-manifold clustering, affine-flow wall-clock),
the script states the honest number.
