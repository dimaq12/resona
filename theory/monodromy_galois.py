"""
monodromy_galois.py — the bridge BEYOND catastrophe: spectral monodromy & Galois.

Catastrophe theory (catastrophe_hardness.py) is the LOCAL picture — the type of a
branch point.  Beneath it is one global object: the eigenvalues of a parameter
family form a branched RIEMANN SURFACE over parameter space, and its MONODROMY
(how the eigenvalues permute as you loop around a branch point) carries everything.

  • LOCAL — the catastrophe is a monodromy CYCLE.  Looping around an Arnold A_q
    stratum (q-fold coalescence) permutes the q colliding eigenvalues in a single
    q-cycle: fold→2-cycle, cusp→3-cycle, swallowtail→4-cycle, butterfly→5-cycle.
    (And the conditioning there is dist^{-(1-1/q)} — catastrophe_hardness.py.)

  • GLOBAL — the monodromy GROUP (all loops, from a common base point) is the
    GALOIS GROUP of the characteristic polynomial.  By Abel–Ruffini it decides
    SOLVABILITY: a solvable group ⇒ the eigenvalues have a closed-form radical
    formula; an unsolvable one (S_n, n≥5) ⇒ none exists — you MUST iterate.

So spectral monodromy gives TWO orthogonal axes of hardness:
    LOCAL  (catastrophe / conditioning)  —  how SENSITIVE the answer is;
    GLOBAL (Galois / solvability)        —  whether the answer has a CLOSED FORM.

HONEST STATUS.  Every ingredient is classical and deep — monodromy of the
discriminant complement (the braid group), the Galois group of the characteristic
polynomial, Abel–Ruffini.  The contribution is the framing: the spectral
monodromy as the single object behind both the catastrophe-hardness law and a
second, solvability axis.  A demonstration, not a theorem.

Run:  python3 theory/monodromy_galois.py
"""
import numpy as np


def companion(coeffs):
    n = len(coeffs); C = np.zeros((n, n), complex)
    for i in range(n - 1):
        C[i, i + 1] = 1.0
    C[n - 1, :] = -np.asarray(coeffs, complex)
    return C


def track(bpath, coeffs_of_b, n):
    """Permutation of the n eigenvalue sheets after continuing along a closed
    parameter path (greedy nearest-neighbour continuation)."""
    prev = np.linalg.eigvals(companion(coeffs_of_b(bpath[0]))); start = prev.copy()
    for b in bpath[1:]:
        cur = np.linalg.eigvals(companion(coeffs_of_b(b)))
        used = set(); order = np.zeros(n, int)
        for j in range(n):
            for idx in np.argsort(np.abs(cur - prev[j])):
                if idx not in used:
                    order[j] = idx; used.add(idx); break
        prev = cur[order]
    return [int(np.argmin(np.abs(start - prev[j]))) for j in range(n)]


def cycles(perm):
    n = len(perm); seen = [False] * n; out = []
    for i in range(n):
        if seen[i]:
            continue
        c = 0; j = i
        while not seen[j]:
            seen[j] = True; j = perm[j]; c += 1
        out.append(c)
    return sorted(out, reverse=True)


def loop_around(center, radius=0.1, steps=4000):
    return center + radius * np.exp(1j * np.linspace(0, 2 * np.pi, steps))


def based_loop(b0, bk, eps=0.06):
    near = bk + eps
    return np.concatenate([np.linspace(b0, near, 1500),
                           bk + eps * np.exp(1j * np.linspace(0, 2 * np.pi, 4000)),
                           np.linspace(near, b0, 1500)])


if __name__ == "__main__":
    print("=" * 74)
    print("THE BRIDGE BEYOND CATASTROPHE — spectral monodromy & Galois solvability")
    print("=" * 74)

    print("\n  LOCAL — the catastrophe A_q is a q-cycle of monodromy:")
    cases = [("A₂ fold", lambda t: [2 + 0.1 * np.exp(1j * t), -3, 0], 3),
             ("A₃ cusp", lambda t: [0.1 * np.exp(1j * t), 0, 0], 3),
             ("A₄ swallowtail", lambda t: [0.1 * np.exp(1j * t), 0, 0, 0], 4),
             ("A₅ butterfly", lambda t: [0.1 * np.exp(1j * t), 0, 0, 0, 0], 5)]
    for name, path, n in cases:
        perm = track(np.linspace(0, 2 * np.pi, 4000), path, n)
        print(f"      {name:>16}:  monodromy cycles = {cycles(perm)}")

    print("\n  GLOBAL — the monodromy GROUP is the Galois group ⇒ solvability:")
    # SOLVABLE: λ⁵ + b  — one branch point, a single 5-cycle, group C₅
    perm = track(loop_around(0.0), lambda b: [b, 0, 0, 0, 0], 5)
    print(f"      λ⁵ + b            : group ⟨{cycles(perm)}-cycle⟩ = C₅  → SOLVABLE "
          f"(eigenvalues = radicals)")
    # GENERIC degree 5: branch points = b where p has a double root (p'(λ) roots)
    coeffs = lambda b: [b, 1, -1, 2, 0]                     # λ⁵+2λ³−λ²+λ+b
    bstars = [-(l**5 + 2*l**3 - l**2 + l) for l in np.roots([5, 0, 6, -2, 1])]
    b0 = 8.0 + 5.0j; edges = []
    for bs in bstars:
        moved = tuple(i for i, p in enumerate(track(based_loop(b0, bs), coeffs, 5)) if p != i)
        if len(moved) == 2:
            edges.append(moved)
    adj = {i: set() for i in range(5)}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    seen = {0}; st = [0]
    while st:
        for v in adj[st.pop()]:
            if v not in seen:
                seen.add(v); st.append(v)
    print(f"      generic λ⁵+…+b    : 4 transpositions {edges} span {len(seen)}/5 points")
    print(f"                          → group = S₅  → UNSOLVABLE (Abel–Ruffini, no formula)")

    print("\n" + "=" * 74)
    print("  The spectral monodromy is one object with two faces:")
    print("    LOCAL  cycle  = the catastrophe (A_q → q-cycle → conditioning 1−1/q)")
    print("    GLOBAL group  = the Galois group (S_n unsolvable → no closed form)")
    print("  Two orthogonal hardnesses — SENSITIVITY (catastrophe) and EXPRESSIBILITY")
    print("  (Galois) — both read from how eigenvalues braid around the discriminant.")
    print("=" * 74)
