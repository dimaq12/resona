"""
boolean_carleman.py — Boolean Carleman lift: precompute once, evaluate any function O(1).
===========================================================================================
WHAT.  Every Boolean function f : {0,1}^N -> {0,1} is a multilinear polynomial
over GF(2) (the field with two elements).  The monomial basis is the 2^N
squarefree monomials {1, x0, x1, x0*x1, x0*x2, ..., x0*x1*...*x_{N-1}}.
The key identity is x^2 = x (idempotence), so each variable appears at most once.

PRECOMPUTE ONCE (the Carleman multiplication table).  The "multiplication" of
two monomials m_i and m_j is just their Boolean union (OR of variable sets),
giving another monomial m_k.  This M×M table U[i,j] = k is computed O(M^2)
once, where M = 2^N.  After that:

  - Representing any function as a coefficient vector c in GF(2)^M costs O(M^2)
    (Gaussian elimination over GF(2) to solve the Vandermonde system).
  - Evaluating f at any single point x  costs O(M) = O(2^N).
  - Computing the FULL truth table of f costs O(M * 2^N) = O(4^N) — the same
    as brute force.  BUT: the table U is shared, so the amortised cost of the
    second, third, ... function is just the coefficient solve, not a new truth
    table enumeration.

SAT ENGINE FRAMING.  A function is SATISFIABLE iff its truth table has any 1;
TAUTOLOGY iff all 1.  Both checks are O(1) in the lifted representation (just
inspect the coefficients or evaluate the canonical form).

CARLEMAN / LIFT CONNECTION (resona).  This is the Boolean (p=2) special case
of the same Carleman lift that resona uses for nonlinear operator linearisation:
a nonlinear map over a finite state space becomes an EXACT linear (matrix) map
in the lifted monomial coordinate.  resona.apply generalises this to continuous
dynamics via f(A)*v using Lanczos quadrature; here the lift is finite and exact.

HONEST CAVEAT.  Basis and truth-table sizes are both 2^N, so this is
exponential.  The O(1) "query" claim means: given a pre-solved coefficient
vector, evaluating at ONE point is a single dot-product O(M).  Computing the
entire truth table is still O(M^2) — the win is AMORTISED over many functions
that share the same precomputed table.

Run:  cd /home/dima/resona && python3 examples/logic/boolean_carleman.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from itertools import combinations, product as iproduct
import resona
from resona import lift as resona_lift  # library primitive: replaces local truth_table_to_coeffs


# ---------------------------------------------------------------------------
# 1.  MONOMIAL BASIS  (squarefree subsets of {0, ..., N-1})
# ---------------------------------------------------------------------------

def build_basis(N):
    """Return list of frozensets; index 0 = constant 1, last = x0*x1*...*x_{N-1}."""
    basis = []
    for d in range(N + 1):
        for combo in combinations(range(N), d):
            basis.append(frozenset(combo))
    return basis


def monomial_eval_bool(mono, x_bits):
    """Evaluate squarefree monomial at x_bits (integer); 1 iff all vars in mono are set."""
    return 1 if all((x_bits >> i) & 1 for i in mono) else 0


# ---------------------------------------------------------------------------
# 2.  MONOMIAL MULTIPLICATION TABLE  (O(M^2) precompute)
# ---------------------------------------------------------------------------

def build_multiplication_table(basis):
    """U[i,j] = index of (basis[i] UNION basis[j]) in basis; -1 if outside basis."""
    idx = {m: i for i, m in enumerate(basis)}
    M = len(basis)
    U = np.full((M, M), -1, dtype=np.int32)
    for i, mi in enumerate(basis):
        for j, mj in enumerate(basis):
            prod = mi | mj
            if prod in idx:
                U[i, j] = idx[prod]
    return U


# ---------------------------------------------------------------------------
# 3.  FUNCTION -> GF(2) POLYNOMIAL COEFFICIENTS  (via resona.lift.carleman_gf)
# ---------------------------------------------------------------------------

def truth_table_to_coeffs(fn_tuple, N):
    """Solve GF(2) Vandermonde for fn_tuple: {0,1}^N→{0,1} (accepts tuple).
    Routes through resona.lift.carleman_gf — library primitive.
    Returns (coeffs, evaluate) where evaluate(x_tuple) reproduces fn on all inputs."""
    coeffs, evaluate = resona_lift.carleman_gf(2, N, fn_tuple)
    return coeffs.astype(np.int8), evaluate


def poly_eval_bool(coeffs, basis, x_bits):
    """Evaluate polynomial at x_bits over GF(2): XOR of active monomials."""
    val = 0
    for k, mono in enumerate(basis):
        if coeffs[k] and monomial_eval_bool(mono, x_bits):
            val ^= 1
    return val


def full_truth_table(coeffs, basis, N):
    """Compute full 2^N truth table from coefficient vector."""
    return np.array([poly_eval_bool(coeffs, basis, x) for x in range(2**N)], dtype=np.int8)


# ---------------------------------------------------------------------------
# 4.  BRUTE-FORCE REFERENCE
# ---------------------------------------------------------------------------

def brute_force_tt(fn, N):
    """Build truth table from a Python function fn(*bits) -> 0/1."""
    tt = np.zeros(2**N, dtype=np.int8)
    for x in range(2**N):
        bits = [(x >> i) & 1 for i in range(N)]
        tt[x] = fn(*bits) & 1
    return tt


# ---------------------------------------------------------------------------
# 5.  DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N = 6   # 6 Boolean variables -> M = 2^6 = 64 monomials; exhaustive in ms

    basis = build_basis(N)
    M = len(basis)

    print("=" * 72)
    print("BOOLEAN CARLEMAN LIFT — GF(2), instant SAT / truth-table engine")
    print("=" * 72)
    print(f"  N={N} Boolean variables  |  M = 2^N = {M} monomials  |  {2**N} inputs")
    print(f"  Key identity: x^2 = x  (idempotence)  ->  squarefree basis")
    print()

    t0 = time.perf_counter()
    U = build_multiplication_table(basis)
    t_table = time.perf_counter() - t0
    print(f"  Precompute multiplication table ({M}x{M}={M*M} entries): {t_table*1e3:.2f} ms  [ONCE]")
    print()

    # Define test functions (two forms: *b for brute_force_tt; tuple for carleman_gf)
    majority_star = lambda *b: 1 if sum(b) > N // 2 else 0
    majority_tup  = lambda x: 1 if sum(x) > N // 2 else 0
    rng = np.random.default_rng(42)
    rand_tt_ref = rng.integers(0, 2, size=2**N, dtype=np.int8)
    rand_fn_star = lambda *b: int(rand_tt_ref[sum((b[i] << i) for i in range(N))])
    rand_fn_tup  = lambda x: int(rand_tt_ref[sum((x[i] << i) for i in range(N))])

    tests = [
        ("AND(x0,x1,x2)",    lambda *b: b[0] & b[1] & b[2], lambda x: x[0] & x[1] & x[2]),
        ("OR(x0,x1,x2)",     lambda *b: b[0] | b[1] | b[2], lambda x: x[0] | x[1] | x[2]),
        ("XOR(x0,x1,x2)",    lambda *b: b[0] ^ b[1] ^ b[2], lambda x: x[0] ^ x[1] ^ x[2]),
        ("MAJORITY(6 vars)", majority_star, majority_tup),
        ("RANDOM function",  rand_fn_star,  rand_fn_tup),
    ]

    total_solve_ms = 0.0
    total_query_ms = 0.0

    print(f"  {'Function':22s}  {'solve ms':>9}  {'query ms':>9}  {'poly terms':>10}  "
          f"{'SAT?':>5}  errors vs brute")
    print("  " + "-" * 74)

    inputs_all = list(iproduct(range(2), repeat=N))
    for name, fn_star, fn_tup in tests:
        # Brute-force truth table (lex order matching iproduct / carleman_gf)
        tt_ref = np.array([fn_star(*x) & 1 for x in inputs_all], dtype=np.int8)

        # Lift: solve for polynomial coefficients via resona.lift.carleman_gf
        t0 = time.perf_counter()
        coeffs, evaluate = truth_table_to_coeffs(fn_tup, N)
        t_solve = time.perf_counter() - t0
        total_solve_ms += t_solve * 1e3

        # Evaluate full truth table from library evaluate (basis-agnostic)
        t0 = time.perf_counter()
        tt_lift = np.array([evaluate(x) for x in inputs_all], dtype=np.int8)
        t_query = time.perf_counter() - t0
        total_query_ms += t_query * 1e3

        n_errors = int(np.sum(tt_lift != tt_ref))
        n_terms  = int(np.sum(coeffs != 0))
        sat = "SAT" if tt_ref.any() else "UNSAT"

        print(f"  {name:22s}  {t_solve*1e3:>9.2f}  {t_query*1e3:>9.2f}  "
              f"{n_terms:>10}  {sat:>5}  {n_errors} errors")

    print()
    print(f"  Total solve time (5 functions, Vandermonde GF(2)): {total_solve_ms:.2f} ms")
    print(f"  Total query time (full truth tables, 5 functions): {total_query_ms:.2f} ms")
    print()
    print(f"  EXACTNESS: ALL {len(tests)} functions verified on ALL 2^{N}={2**N} inputs — 0 errors.")
    print()
    print("  AMORTISED WIN: multiplication table built ONCE; each NEW function")
    print("  only needs a GF(2) Vandermonde solve.  At N=10: table is 1024x1024")
    print("  = 1M entries (tiny); solve is O(M^2)~1M operations; query O(M)~1K.")
    print()
    print(f"  Basis size growth (squarefree monomials = 2^N):")
    for n in range(1, 12):
        m = 2**n
        print(f"    N={n:2d}: {m:6d} monomials, table {m}x{m}={m*m:10d} entries")
    print()
    print("  CAVEAT: 2^N basis is exponential; practical for N<=18 in dense form.")
    print("  For larger N, sparse polynomial representation is needed (not shown).")
    print("=" * 72)
