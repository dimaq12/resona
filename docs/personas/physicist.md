# resona for physicists (quantum & condensed matter)

Your Hamiltonian lives in a 2ⁿ-dimensional space and eigh died at n≈14.
Everything below is matrix-free — you supply `H @ v`, resona never forms or
diagonalizes anything.

Your first five tasks:

| task | call |
|---|---|
| full thermodynamics Z, E, S, Cv at any β | `resona.of(lambda v: -(H@v), D, deflate=24).trace(...)` — deflating −H makes the lowest levels EXACT |
| real-time dynamics e^{−iHt}ψ | `resona.apply(mv, lambda l: np.exp(-1j*t*l), psi, hermitian=False)` |
| finite-T correlator ⟨O(t)O⟩_β / spectral function S(ω) | `resona.thermal.correlator(Hmv, Omv, beta, ts, D)` |
| ground state + gap | `resona.of(mv, D).extreme()`, polish with `resona.solve.rayleigh_polish` |
| a topological invariant from real space | Chebyshev projector chains — [`chern_from_noise.py`](../../examples/quantum/chern_from_noise.py) (C = +0.985 at L=20, seconds) |

The trick worth stealing: thermal traces are dominated by the BOTTOM of the
spectrum, and `deflate` captures the TOP — so hand resona **−H**.  Its top-K
Ritz pairs are your K lowest levels, made exact atoms in the measure; the
random probes carry only the smooth bulk.  Measured on the Heisenberg chain:
max Z error 19.4% → 0.09% at L=8 for the same k.

Worked examples: [`heisenberg_thermo.py`](../../examples/quantum/heisenberg_thermo.py)
(L=20, D=10⁶, full thermodynamics in seconds),
[`thermal_response.py`](../../examples/quantum/thermal_response.py),
[`many_body_spectrum.py`](../../examples/quantum/many_body_spectrum.py)
(the whole spectrum from two moments).  Guides:
[solving-and-evolving](../solving-and-evolving.md),
[reading-spectra](../reading-spectra.md).
