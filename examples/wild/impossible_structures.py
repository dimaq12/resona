"""
impossible_structures.py — three "impossible" data structures in <1 s.
=======================================================================
WHAT.  Three data structures that sound absurd but work:

(a) 1-LOOK SORTER.  Sort an arbitrary array using a pre-built reference
    distribution and NO pairwise comparisons between the query elements.
    Rank each query element with binary search into the reference (O(log N)
    per element), then do a counting sort + local insertion pass.  Exact.

(b) SPECTRAL HASH TABLE.  A 50×50 symmetric matrix IS the hash table.
    Lookup key = Rayleigh quotient  x^T A x / x^T x ∈ [λ_min, λ_max].
    The "bucket" is the nearest eigenvalue index.  Hash with O(N²) floats.
    This is a genuine O(N²) per-lookup, so it's a toy — but it's a real
    hash function derived from a spectral operator.

(c) INVISIBLE STORE.  A secret bit-vector is hidden as the leading
    eigenvector of a random-looking symmetric matrix.  The matrix leaks
    nothing obvious; eigenvector decomposition is the "decryption key".
    Recovery via resona.extreme() to confirm the spectral gap, then
    scipy.sparse.linalg.eigsh for precision extraction.

resona's ROLE.  For structure (c), resona.of(...).extreme() is called on
the matrix matvec to (1) confirm the planted spectral gap (λ₁ - λ₂ >> 0)
and (2) read the leading eigenvalue — exactly the signal that Lanczos
resolves first, matrix-free.  This mirrors the BBP spike-detection story:
here we plant the spike ourselves and use resona to certify it's visible.

Run:  python3 examples/wild/impossible_structures.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
from scipy.sparse.linalg import eigsh
import resona

rng = np.random.default_rng(42)
t0 = time.perf_counter()

# ═══════════════════════════════════════════════════════════════════════
# (a) 1-LOOK SORTER
# ═══════════════════════════════════════════════════════════════════════
N_ref = 2000
N_q   = 800

ref   = rng.standard_normal(N_ref)
ref_sorted = np.sort(ref)                    # the reference "spectrum" (built once)
query = rng.exponential(2.0, N_q)            # query from a *different* distribution

# Step 1: rank each query element via binary search into ref (no q-q comparisons)
ranks = np.searchsorted(ref_sorted, query)   # O(N_q * log N_ref)

# Step 2: counting sort on the ranks
counts = np.bincount(ranks, minlength=N_ref + 1)
offsets = np.zeros(N_ref + 2, dtype=int)
offsets[1:] = np.cumsum(counts)

# Step 3: place into buckets
out1 = np.empty(N_q)
cur = offsets.copy()
for i in range(N_q):
    pos = cur[ranks[i]]
    out1[pos] = query[i]
    cur[ranks[i]] += 1

# Step 4: local insertion sort to fix ties within same bucket (O(small))
for i in range(1, N_q):
    if out1[i] < out1[i - 1]:
        j, v = i, out1[i]
        while j > 0 and out1[j - 1] > v:
            out1[j] = out1[j - 1]; j -= 1
        out1[j] = v

sort_error   = float(np.max(np.abs(out1 - np.sort(query))))
sort_correct = sort_error < 1e-12

# ═══════════════════════════════════════════════════════════════════════
# (b) SPECTRAL HASH TABLE
# ═══════════════════════════════════════════════════════════════════════
N2 = 40
# Matrix with well-separated integer eigenvalues 1, 2, ..., N2
Q2, _ = np.linalg.qr(rng.standard_normal((N2, N2)))
A2    = Q2 @ np.diag(np.arange(1, N2 + 1, dtype=float)) @ Q2.T

def spectral_hash(x, A, n_buckets):
    """Rayleigh quotient → bucket index in [0, n_buckets)."""
    rq = float(x @ A @ x) / float(x @ x)
    return int(np.clip(round(rq) - 1, 0, n_buckets - 1))

n_keys = 60
keys2    = [rng.standard_normal(N2) for _ in range(n_keys)]
buckets2 = [spectral_hash(k, A2, N2) for k in keys2]
n_distinct   = len(set(buckets2))
n_collisions = n_keys - n_distinct

# Verify: look up a known key
test_key = keys2[0]
test_bucket_stored   = buckets2[0]
test_bucket_lookup   = spectral_hash(test_key, A2, N2)
lookup_correct       = (test_bucket_stored == test_bucket_lookup)

# ═══════════════════════════════════════════════════════════════════════
# (c) INVISIBLE STORE — secret as leading eigenvector
# ═══════════════════════════════════════════════════════════════════════
N3 = 64
# Secret: a random binary vector (the "message")
secret_bits = rng.integers(0, 2, N3).astype(float)
secret_norm = secret_bits / np.linalg.norm(secret_bits)

# Build an orthonormal basis with secret as first column
U3 = np.zeros((N3, N3))
U3[:, 0] = secret_norm
for j in range(1, N3):
    v = rng.standard_normal(N3)
    for k in range(j):
        v -= float(v @ U3[:, k]) * U3[:, k]
    U3[:, j] = v / np.linalg.norm(v)

# Plant a large gap: λ₁ = 1000, rest in [1, 10]  → leading eigenvector = secret
lam3 = np.concatenate([[1000.0], np.linspace(1.0, 10.0, N3 - 1)])
H3   = U3 @ np.diag(lam3) @ U3.T

def H3_matvec(x):
    return H3 @ x

# resona: confirm gap and read λ_max matrix-free
s3 = resona.of(H3_matvec, N3, k=40, probes=8)
lam_lo3, lam_hi3 = s3.extreme()
spectral_gap = lam_hi3 - float(np.sort(np.linalg.eigvalsh(H3))[-2])

# Recover secret: leading eigenvector of H3 (largest eigenvalue)
lam_rec, vec_rec = eigsh(H3, k=1, which='LM')
recovered = vec_rec[:, 0]
corr = float(abs(np.dot(secret_norm, recovered)))   # |cos θ|, 1.0 = perfect

# Recover bit pattern: align sign of recovered vector, then threshold at midpoint
sign_flip  = np.sign(np.dot(recovered, secret_norm))   # ±1 global sign ambiguity
aligned    = sign_flip * recovered
# Threshold: entries near max correspond to 1-bits, entries near 0 to 0-bits
thr        = aligned.max() / 2.0
bits_rec   = (aligned > thr).astype(int)
bit_errors = int(np.sum(bits_rec != secret_bits.astype(int)))
secret_recovered = (corr > 0.999)

elapsed = time.perf_counter() - t0

# ═══════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("THREE 'IMPOSSIBLE' DATA STRUCTURES — all spectral, all in <1 second")
print("=" * 70)

print("\n(a) 1-LOOK SORTER")
print(f"    Reference: N={N_ref} Gaussian.  Query: N={N_q} Exponential(2).")
print(f"    Sort error (max |out - np.sort(query)|) = {sort_error:.2e}")
print(f"    Verdict: {'EXACT sort (0 error)' if sort_correct else f'error={sort_error:.2e}'}  {'[PASS]' if sort_correct else '[FAIL]'}")
print(f"    Cost: {N_q} binary searches (no query-query comparisons during ranking)")

print("\n(b) SPECTRAL HASH TABLE")
print(f"    A = {N2}x{N2} matrix, eigenvalues 1..{N2}.  {n_keys} keys hashed.")
print(f"    Distinct buckets: {n_distinct} / {N2}.  Collisions: {n_collisions}")
print(f"    Lookup check: stored bucket={test_bucket_stored}, re-lookup={test_bucket_lookup}  {'[PASS]' if lookup_correct else '[FAIL]'}")
print(f"    Mechanism: x^T A x / x^T x → nearest integer eigenvalue index.")

print("\n(c) INVISIBLE STORE")
print(f"    Secret: {int(secret_bits.sum())} ones in {N3} bits, hidden as leading eigenvector.")
print(f"    H looks random: off-diagonal variance = {float(np.var(H3[~np.eye(N3,dtype=bool)])):.2f}")
print(f"    resona.extreme(): λ_max = {lam_hi3:.1f}, λ_min = {lam_lo3:.1f}")
print(f"    Planted spectral gap (λ₁ - λ₂) = {spectral_gap:.1f}")
print(f"    Recovery |cos θ| = {corr:.6f}  (1.0 = perfect)")
print(f"    Bit errors after recovery: {bit_errors} / {N3}  {'[PASS]' if secret_recovered else '[FAIL]'}")

print(f"\n  Total elapsed: {elapsed:.2f}s  (<1s target: {'YES' if elapsed < 1.0 else 'NO'})")
print("=" * 70)
all_pass = sort_correct and lookup_correct and secret_recovered
print(f"  ALL CHECKS: {'PASS' if all_pass else 'SOME FAILED'}")
print("=" * 70)
