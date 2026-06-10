"""
ternary_carleman.py — GF(3) Carleman lift: every ternary function is a linear operator.
=========================================================================================
WHAT.  Over the three-element field GF(3) = {0, 1, 2}, every function
f : {0,1,2}^N -> {0,1,2} is a multivariate polynomial modulo x^3 - x = 0
(Fermat's little theorem for p=3).  Lifting the N input variables into the
full monomial basis {x0^e0 * x1^e1 * ... | ei in {0,1,2}} — which has 3^N
elements — turns evaluation of f into a SINGLE dot-product with a coefficient
vector: f(x) = sum_k c_k * phi_k(x).

WHY IT IS POSSIBLE.  The Vandermonde matrix V over all 3^N inputs is square
and invertible over GF(3), so the Lagrange / polynomial interpolation problem
has a unique solution for every function.  Equivalently: the monomial functions
{phi_k} form an orthogonal basis of the function space GF(3)^{GF(3)^N}.  The
key identity is x^3 = x mod 3, which collapses all exponents to {0,1,2} and
keeps the basis finite.

CARLEMAN / LIFT CONNECTION.  This is the finite-field analogue of the Carleman
lift used throughout resona: a nonlinear map becomes EXACTLY linear in a lifted
(monomial) coordinate system.  In resona's continuous setting the lift is
approximate (truncated Taylor series); here it is EXACT because the field is
finite and the basis is complete.  The Carleman operator L(f) is the 3^N x 3^N
matrix that maps the lifted state phi(x(t)) to phi(x(t+1)) along the dynamics
defined by f — the same structure that resona.apply uses for nonlinear ODE
linearisation.

HONEST CAVEAT.  The basis size is 3^N, so this is exponential in N.  It is
useful for small N (N <= 6 comfortably) or as an exact algebraic reference.
For large N one would use a sparse polynomial representation or the SFT
(sparse Fourier / Walsh–Hadamard) approach.  The POINT of this demo is to
show the lift primitive in its exact, zero-error form.

Run:  cd /home/dima/resona && python3 examples/logic/ternary_carleman.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from itertools import product as iproduct


# ---------------------------------------------------------------------------
# 1.  MONOMIAL BASIS  (all exponent tuples in {0,1,2}^N)
# ---------------------------------------------------------------------------

def build_basis(N, p=3):
    """Return list of all exponent tuples; length = p^N."""
    return list(iproduct(range(p), repeat=N))


def monomial_eval(exp, x, p=3):
    """Evaluate monomial x0^e0 * x1^e1 * ... at point x in GF(p)^N."""
    val = 1
    for xi, ei in zip(x, exp):
        if ei > 0:
            val = (val * pow(int(xi), ei, p)) % p
    return val


# ---------------------------------------------------------------------------
# 2.  VANDERMONDE SYSTEM OVER GF(3)  ->  polynomial coefficients
# ---------------------------------------------------------------------------

def gf_gauss(A, b, p=3):
    """Gaussian elimination over GF(p); returns solution vector mod p."""
    n = len(b)
    A = A.copy() % p
    b = b.copy() % p
    # Modular inverse lookup for GF(p) — valid for any prime p.
    inv = [0] * p
    for x in range(1, p):
        for y in range(1, p):
            if (x * y) % p == 1:
                inv[x] = y
    for col in range(n):
        pivot = next((r for r in range(col, n) if A[r, col] != 0), -1)
        if pivot < 0:
            continue
        if pivot != col:
            A[[col, pivot]] = A[[pivot, col]]
            b[[col, pivot]] = b[[pivot, col]]
        f = inv[int(A[col, col])]
        A[col] = (A[col] * f) % p
        b[col] = (b[col] * f) % p
        for row in range(n):
            if row != col and A[row, col] != 0:
                factor = int(A[row, col])
                A[row] = (A[row] - factor * A[col]) % p
                b[row] = (b[row] - factor * b[col]) % p
    return b


def function_to_polynomial(truth_table, basis, p=3):
    """Solve V * coeffs = truth_table over GF(p) where V is the Vandermonde matrix."""
    M = len(basis)
    N = len(basis[0])
    inputs = list(iproduct(range(p), repeat=N))
    V = np.array([[monomial_eval(exp, x, p) for exp in basis]
                  for x in inputs], dtype=np.int64)
    return gf_gauss(V, np.array(truth_table, dtype=np.int64), p)


def poly_eval(coeffs, basis, x, p=3):
    """Evaluate polynomial sum_k c_k * phi_k(x) at x in GF(p)^N."""
    return int(sum(c * monomial_eval(e, x, p) for c, e in zip(coeffs, basis)) % p)


# ---------------------------------------------------------------------------
# 3.  DEMO
# ---------------------------------------------------------------------------

def verify_all(coeffs, basis, truth_table, inputs, p=3):
    """Return (n_errors, max_error) over all inputs."""
    errors = sum(1 for x, t in zip(inputs, truth_table)
                 if poly_eval(coeffs, basis, x, p) != t)
    return errors


if __name__ == "__main__":
    p = 3
    N = 2    # two ternary variables -> 3^2 = 9 monomials; verification exhaustive

    basis  = build_basis(N, p)
    M      = len(basis)
    inputs = list(iproduct(range(p), repeat=N))

    print("=" * 72)
    print("TERNARY CARLEMAN LIFT — GF(3), exact polynomial representation")
    print("=" * 72)
    print(f"  N={N} ternary variables  |  {M} monomials (3^N)  |  {len(inputs)} inputs (3^N)")
    print(f"  Key identity: x^3 = x mod 3  (Fermat's little theorem for p=3)")
    print(f"  Lift: f: {{0,1,2}}^N -> {{0,1,2}}  becomes  f(x) = c . phi(x)  EXACTLY")
    print()

    # Five ternary functions to demonstrate
    funcs = {
        "min(a,b)            ": lambda a, b: min(a, b),
        "max(a,b)            ": lambda a, b: max(a, b),
        "NOT a  (2-a mod 3)  ": lambda a, b: (2 - a) % 3,
        "half-sum  (a+b)*2%3 ": lambda a, b: ((a + b) * 2) % 3,
        "a*b mod 3 (GF3 mult)": lambda a, b: (a * b) % 3,
    }

    t_pre = 0.0
    t_query_total = 0.0
    print(f"  {'Function':40s}  {'poly terms':>10}  {'errors':>7}  verified")
    print("  " + "-" * 68)
    for name, fn in funcs.items():
        truth = np.array([fn(a, b) for a, b in inputs], dtype=np.int64)

        t0 = time.perf_counter()
        coeffs = function_to_polynomial(truth, basis, p)
        t_pre += time.perf_counter() - t0

        t0 = time.perf_counter()
        errs = verify_all(coeffs, basis, truth, inputs, p)
        t_query_total += time.perf_counter() - t0

        n_terms = int(np.sum(coeffs != 0))
        ok = "YES — 0 errors" if errs == 0 else f"NO  — {errs} errors"
        print(f"  {name:40s}  {n_terms:>10}  {errs:>7}  {ok}")

    print()
    print(f"  Verification: ALL {len(funcs)} functions checked on ALL 3^{N}={3**N} inputs.")
    print(f"  Lift cost (Vandermonde solve, GF(3)):  {t_pre*1e3:.2f} ms total for {len(funcs)} functions")
    print(f"  Query cost (dot-product evaluation):   {t_query_total*1e6/len(funcs):.1f} us per full truth-table scan")
    print()
    print("  EXACTNESS: Carleman lift over GF(3) is EXACT — not approximate.")
    print("  In resona's continuous setting the same lift is truncated (approx).")
    print("  Here, finite field + complete basis = zero truncation error.")
    print()

    # Show basis growth: GF(2) vs GF(3)
    print("  Basis size scaling (Boolean vs Ternary):")
    print(f"  {'N':>4}  {'GF(2) 2^N':>10}  {'GF(3) 3^N':>10}  {'ratio':>7}")
    for n in range(1, 8):
        b2, b3 = 2**n, 3**n
        print(f"  {n:>4}  {b2:>10}  {b3:>10}  {b3/b2:>7.1f}x")
    print()
    print("  CAVEAT: 3^N basis grows fast — practical for N<=6 without sparsity.")
    print("=" * 72)
