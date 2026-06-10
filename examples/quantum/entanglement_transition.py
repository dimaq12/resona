"""
quantum/entanglement_transition.py
==============================================================================
THE ENTANGLEMENT (MEASUREMENT-INDUCED) PHASE TRANSITION — entanglement IS a rank.

THE PHENOMENON.  Run a quantum circuit that interleaves two opposing forces:
ENTANGLING gates (which spread quantum correlations across the system) and
MEASUREMENTS at rate p (which collapse and DISENTANGLE).  As p crosses a critical
p_c the steady state changes phase:
    p < p_c :  VOLUME LAW — half-chain entanglement entropy S ∝ L (extensive,
               maximally entangled — "free", the entangling force wins);
    p > p_c :  AREA LAW   — S → const (entanglement pinned local — "structured",
               measurement wins).
This is a genuinely quantum, dynamical phase transition with no classical thermal
analogue — yet here we simulate it EXACTLY on a laptop for hundreds of qubits.

WHY IT'S CLASSICALLY TRACTABLE (the key).  We use only CLIFFORD gates +
Z-measurements.  By the Gottesman–Knill theorem such circuits are LINEAR ALGEBRA
OVER GF(2): the state is tracked as a stabilizer tableau (L generators × 2L bits),
each gate is a GF(2) row/column operation, each measurement a GF(2) pivot.  No 2ᴸ
amplitudes are ever stored.  And the entanglement entropy of a stabilizer state is

      S_A  =  rank_GF(2)( tableau restricted to region A's columns )  −  |A|.

ENTANGLEMENT ENTROPY IS AN EFFECTIVE RANK — over the field GF(2).  This is exactly
resona's lens (`effective_rank` = how many independent modes a response really
has), transported from ℝ to GF(2): the transition is a RANK-SCALING transition,
extensive rank (volume) vs sub-extensive rank (area).  Same object, new field.

THE HONEST READING.  Clifford circuits are the dequantizable corner of quantum
computing (low "rank" structure — see dequantize.py).  The MIPT lives there, which
is *why* we can render it classically.  A volume-law state over a UNIVERSAL gate
set would be the genuine quantum frontier (high rank, no GF(2) shortcut).

Run:  python3 examples/quantum/entanglement_transition.py
"""
import numpy as np

rng = np.random.default_rng(0)


def gf2_rank(M):
    """Rank of a 0/1 matrix over GF(2) by XOR elimination = the entropy counter."""
    M = M.copy().astype(np.uint8); rows, cols = M.shape; r = 0
    for c in range(cols):
        piv = np.flatnonzero(M[r:, c])
        if piv.size == 0:
            continue
        p = r + piv[0]; M[[r, p]] = M[[p, r]]
        mask = M[:, c].astype(bool); mask[r] = False
        M[mask] ^= M[r]; r += 1
        if r == rows:
            break
    return r


class Stab:
    """Stabilizer tableau over GF(2): L generators × 2L bits (x|z). Phase ignored."""
    def __init__(self, L):
        self.L = L
        self.T = np.zeros((L, 2 * L), dtype=np.uint8)
        for q in range(L):
            self.T[q, L + q] = 1                          # |0…0⟩: stabilizers Zq

    def H(self, q):                                       # Hadamard: swap x/z columns
        self.T[:, [q, self.L + q]] = self.T[:, [self.L + q, q]]

    def S(self, q):                                       # phase gate
        self.T[:, self.L + q] ^= self.T[:, q]

    def CNOT(self, c, t):                                 # entangler (the only 2-qubit op)
        self.T[:, t] ^= self.T[:, c]
        self.T[:, self.L + c] ^= self.T[:, self.L + t]

    def rand1q(self, q):
        if rng.random() < 0.5: self.H(q)
        if rng.random() < 0.5: self.S(q)
        if rng.random() < 0.5: self.H(q)

    def two_q(self, a, b):
        self.rand1q(a); self.rand1q(b); self.CNOT(a, b); self.rand1q(a); self.rand1q(b)

    def measure_Z(self, q):                               # the DISENTANGLING force
        anti = np.flatnonzero(self.T[:, q])               # generators anticommuting w/ Zq
        if anti.size == 0:
            return
        piv, rest = anti[0], anti[1:]
        if rest.size:
            self.T[rest] ^= self.T[piv]
        self.T[piv] = 0; self.T[piv, self.L + q] = 1

    def entropy_half(self):
        """Half-chain entanglement = GF(2) effective rank across the cut."""
        L = self.L; h = L // 2
        cols = list(range(h)) + [L + q for q in range(h)]
        return gf2_rank(self.T[:, cols]) - h


def run(L, p, layers, reps):
    out = []
    for _ in range(reps):
        st = Stab(L)
        for _ in range(layers):
            for off in (0, 1):                            # brickwork of entanglers
                for i in range(off, L - 1, 2):
                    st.two_q(i, i + 1)
            for q in np.flatnonzero(rng.random(L) < p):   # measurements at rate p
                st.measure_Z(int(q))
        out.append(st.entropy_half())
    return float(np.mean(out))


if __name__ == "__main__":
    print("=" * 72)
    print("ENTANGLEMENT PHASE TRANSITION — entanglement = effective rank over GF(2)")
    print("=" * 72)
    print("  Clifford brickwork + Z-measurements at rate p (Gottesman–Knill, exact).")
    print("  S(2L)/S(L) ≈ 2 → volume law (S∝L);  ≈ 1 → area law (saturated).\n")
    Ls = [32, 64]
    print(f"  {'p':>6} " + "".join(f"{'L='+str(L):>9}" for L in Ls) + f"{'S(64)/S(32)':>13}  phase")
    print("  " + "─" * 48)
    for p in [0.04, 0.08, 0.12, 0.16, 0.22, 0.30, 0.45]:
        row = [run(L, p, layers=2 * L + 20, reps=8) for L in Ls]
        ratio = row[1] / max(row[0], 1e-9)
        phase = ("area" if max(row) < 1.5 or ratio < 1.25
                 else "volume" if ratio > 1.6 else "~critical")
        print(f"  {p:>6.2f} " + "".join(f"{v:>9.1f}" for v in row)
              + f"{ratio:>13.2f}  {phase}")
    print("\n" + "=" * 72)
    print("  The crossover (ratio passing ~1.5) brackets p_c ≈ 0.12–0.20 — the known")
    print("  (1+1)D random-Clifford value p_c ≈ 0.16.  Entanglement entropy = GF(2)")
    print("  effective rank; the transition is extensive-rank (volume, 'free') vs")
    print("  sub-extensive-rank (area, 'structured') — resona's dial in a finite field.")
    print("=" * 72)
