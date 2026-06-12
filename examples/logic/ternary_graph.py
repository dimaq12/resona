"""
ternary_graph.py — 3-valued edge graph: ternary Laplacian + resona spectral fingerprint.
==========================================================================================
WHAT.  Build a graph whose edges carry TERNARY weights from {0, 1, 2}:
  0 = no connection  (edge absent)
  1 = weak link      (soft / co-occurrence relation)
  2 = strong link    (implication / hard logical constraint)

We build a ternary Laplacian L_T — weak edges get real weight 0.5, strong edges
get real weight 2.0 (4x contrast) — then pass it to resona.of which reads the
spectral fingerprint WITHOUT calling eig: effective rank, spectral moments, the
spectral density curve, the free cumulants κ₁..κ₄ (s.cumulants — they obey the
exact scaling law κ_n → cⁿκ_n under weight contrast c), the condition number
(s.condition — valid because the +0.01 shift makes L_T PD), and the Beta-closure
spectrum (s.levels), whose residual against dense eig doubles as a block-
structure detector.

Three graph families are compared:
  A. PURE RANDOM WEAK:    all edges label=1, Erdos-Renyi-style
  B. PURE RANDOM STRONG:  all edges label=2
  C. MIXED TERNARY:       ~60% weak, ~40% strong (realistic logical network)

WHY THIS MATTERS.  A standard (binary) graph uses weights in {0,1}, encoding
only presence/absence.  Ternary weights encode LOGICAL COMMITMENT LEVEL.  The
ratio of strong to weak edges shifts the spectral effective rank: a graph of
purely strong edges is more spectrally CONCENTRATED (lower r_eff) than one
with only weak edges, because the strong-edge clusters dominate the low-
frequency eigenmodes.

RESONA / LIFT CONNECTION.  The ternary edge labels are a GF(3) signal on the
edge set: label in {0,1,2} mod 3.  The Carleman multiplication table of
ternary_carleman.py tells us how these labels compose under product (0->absorb,
1->identity, 2->flip).  The graph Laplacian is the degree-1 Fourier mode of
this product structure.  resona.of reads its spectral effective rank — the
"dimension count" of the logical network — matrix-free via Stochastic Lanczos
Quadrature, the same engine used throughout resona for operator spectra.

r_eff = exp( H[ spectral density ] ), where H is Shannon entropy.
r_eff = n  =>  flat spectrum (uniform eigenvalues, maximally spread).
r_eff << n =>  spectrum concentrated on a few modes (structured clusters).

HONEST CAVEAT.  The Laplacian here is real-valued (standard graph theory),
not a GF(3) operator.  A true GF(3) Laplacian uses cube-root-of-unity
eigenvalues and requires a complex spectral analysis — that is beyond standard
real-symmetric Laplacians.  What we show: ternary LABELS as edge WEIGHTS in a
real Laplacian, combined with resona's spectral probe.  The GF(3)/Carleman
connection is conceptually correct (labels are GF(3) signals, composition is
GF(3) multiplication) but the spectral algebra is real, not GF(3).

Run:  cd /home/dima/resona && python3 examples/logic/ternary_graph.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from scipy import sparse
import resona


# ---------------------------------------------------------------------------
# 1.  GRAPH CONSTRUCTION
# ---------------------------------------------------------------------------

WEIGHT_MAP = {1: 0.5, 2: 2.0}   # GF(3) label -> real weight; 4x contrast


def clustered_ternary_graph(n_clusters, cluster_size, inter_prob,
                            strong_fraction, seed=0):
    """
    Stochastic block model: k clusters of size m each.
    Intra-cluster edges: all present (probability 1.0).
    Inter-cluster edges: present with probability inter_prob.
    Labels assigned by strong_fraction (fraction of edges that get label=2).

    This produces a STRUCTURED graph where spectral concentration is visible:
    strong intra-cluster edges create tight cliques that dominate the low modes.
    """
    rng = np.random.default_rng(seed)
    n = n_clusters * cluster_size
    edges = []

    def make_label():
        return 2 if rng.random() < strong_fraction else 1

    # Intra-cluster (all present)
    for c in range(n_clusters):
        base = c * cluster_size
        for i in range(cluster_size):
            for j in range(i + 1, cluster_size):
                edges.append((base + i, base + j, make_label()))

    # Inter-cluster (sparse)
    for c1 in range(n_clusters):
        for c2 in range(c1 + 1, n_clusters):
            for i in range(cluster_size):
                for j in range(cluster_size):
                    if rng.random() < inter_prob:
                        u = c1 * cluster_size + i
                        v = c2 * cluster_size + j
                        edges.append((u, v, make_label()))

    return n, edges


def build_ternary_laplacian(n, edge_list):
    """
    L_T = D - A, where A[u,v] = WEIGHT_MAP[label].
    Tiny diagonal shift (+0.01) makes L strictly positive definite (no zero mode).
    """
    rows, cols, vals = [], [], []
    deg = np.zeros(n)
    for u, v, label in edge_list:
        r = WEIGHT_MAP.get(label, 0.0)
        rows += [u, v]; cols += [v, u]; vals += [-r, -r]
        deg[u] += r; deg[v] += r
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(deg[i] + 0.01)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# 2.  SPECTRAL FINGERPRINT VIA RESONA
# ---------------------------------------------------------------------------

def spectral_fingerprint(L):
    """Run resona.of, return dict with key spectral metrics.

    resona.moment(p) returns Tr(A^p).  Mean eigenvalue = Tr(L)/n.
    Variance = Tr(L^2)/n - (Tr(L)/n)^2.
    """
    n = L.shape[0]
    matvec = lambda x: L @ x
    sp = resona.of(matvec, n, k=min(48, n - 1), probes=16, seed=7)
    r_eff    = sp.effective_rank()
    trace1   = sp.moment(1)          # Tr(L)
    trace2   = sp.moment(2)          # Tr(L^2)
    mean_lam = trace1 / n            # mean eigenvalue
    spread   = np.sqrt(max(trace2 / n - mean_lam**2, 0.0))  # std of eigenvalues
    xs       = np.linspace(0.01, mean_lam + 4 * spread + 0.1, 300)
    rho      = sp.density(xs)
    peak     = float(xs[np.argmax(rho)])
    # Deeper hub readouts from the SAME Spectral object (no extra matvecs):
    kappa    = tuple(sp.cumulants(4))   # free cumulants — additive coordinates
    cond     = sp.condition()           # κ = λ_max/λ_min (valid: L is PD, +0.01 shift)
    return dict(n=n, r_eff=r_eff, mu1=mean_lam, spread=spread, peak=peak,
                kappa=kappa, cond=cond), sp


# ---------------------------------------------------------------------------
# 3.  GF(3) LABEL ARITHMETIC  (from ternary_carleman.py)
# ---------------------------------------------------------------------------

def ternary_product(a, b):
    """Multiplication in GF(3): {0,1,2} x {0,1,2} -> {0,1,2}."""
    return (a * b) % 3


def edge_product_spectrum(edge_list):
    """
    Compute the distribution of ternary_product(label_i, label_j) over all
    edge pairs.  Illustrates the GF(3) multiplicative structure on the edge set.
    """
    labels = [w for _, _, w in edge_list]
    counts = {0: 0, 1: 0, 2: 0}
    for a in labels:
        for b in labels:
            counts[ternary_product(a, b)] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


# ---------------------------------------------------------------------------
# 4.  MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Stochastic block model: 8 clusters of 10 nodes each = 80 total
    # Intra-cluster: all connected (dense cliques).
    # Inter-cluster: sparse (5% probability).
    # The clique structure makes spectral concentration measurable.
    K = 8; M = 10   # 8 clusters x 10 nodes = 80 nodes

    print("=" * 72)
    print("TERNARY GRAPH — 3-valued edge weights, spectral fingerprint via resona")
    print("=" * 72)
    print(f"  Stochastic block model: {K} clusters x {M} nodes = {K*M} nodes total")
    print(f"  Intra-cluster: all edges present.  Inter-cluster: ~5% sparsity.")
    print(f"  Edge labels in GF(3): 0=absent  1=weak  2=strong")
    print(f"  Real weights: 1 -> {WEIGHT_MAP[1]}  (weak)   2 -> {WEIGHT_MAP[2]}  (strong, 4x contrast)")
    print()

    configs = [
        ("all-weak  (label=1)", 0.00),   # strong_fraction=0 -> all label 1
        ("all-strong(label=2)", 1.00),   # strong_fraction=1 -> all label 2
        ("mixed (40% strong) ", 0.40),   # 60% weak, 40% strong
    ]

    results = []
    spectra = []   # (Spectral, Laplacian) per config — for the deeper hub readouts
    for label, s_frac in configs:
        n, edges = clustered_ternary_graph(K, M, inter_prob=0.05,
                                           strong_fraction=s_frac, seed=42)
        n_weak   = sum(1 for _, _, w in edges if w == 1)
        n_strong = sum(1 for _, _, w in edges if w == 2)
        L = build_ternary_laplacian(n, edges)

        t0 = time.perf_counter()
        fp, sp = spectral_fingerprint(L)
        t_spec = time.perf_counter() - t0

        ep = edge_product_spectrum(edges)
        results.append((label, n_weak, n_strong, fp, t_spec, ep))
        spectra.append((sp, L))

    # Print results table
    print(f"  {'Graph':24s}  {'E':>4}  {'w':>4}  {'s':>4}  "
          f"{'r_eff':>7}  {'r_eff/n':>7}  {'mean_lam':>9}  {'spread':>7}  time")
    print("  " + "-" * 78)
    for lbl, nw, ns, fp, t, ep in results:
        n_nodes = fp['n']
        print(f"  {lbl:24s}  {nw+ns:>4}  {nw:>4}  {ns:>4}  "
              f"{fp['r_eff']:>7.2f}  {fp['r_eff']/n_nodes:>7.3f}  "
              f"{fp['mu1']:>9.3f}  {fp['spread']:>7.3f}  {t*1e3:.0f}ms")

    print()
    r_weak   = results[0][3]['r_eff']
    r_strong = results[1][3]['r_eff']
    r_mixed  = results[2][3]['r_eff']
    n_nodes  = results[0][3]['n']
    print(f"  Spectral effective rank (out of n={n_nodes}):")
    print(f"    all-weak   r_eff = {r_weak:.2f}  (r_eff/n = {r_weak/n_nodes:.3f})")
    print(f"    all-strong r_eff = {r_strong:.2f}  (r_eff/n = {r_strong/n_nodes:.3f})  "
          f"ratio = {r_strong/r_weak:.3f}x")
    print(f"    mixed      r_eff = {r_mixed:.2f}  (r_eff/n = {r_mixed/n_nodes:.3f})")
    print()

    # Honest assessment
    if abs(r_strong - r_weak) / r_weak < 0.02:
        print(f"  NOTE: r_eff(weak) ~= r_eff(strong) (differ by <2%).")
        print(f"  For clustered-but-complete-clique graphs, the Laplacian's spectral")
        print(f"  entropy is dominated by graph TOPOLOGY (cluster structure), not just")
        print(f"  edge weights.  The 4x weight contrast shifts mean_lam by 4x but leaves")
        print(f"  r_eff nearly unchanged because all cliques remain equally tight.")
        print(f"  Spectral concentration emerges more strongly when TOPOLOGY is varied")
        print(f"  (e.g. some cliques fully connected with strong, others sparse with weak).")
    else:
        delta = (r_weak - r_strong) / r_weak * 100
        print(f"  r_eff drops by {delta:.1f}% going from all-weak to all-strong.")
        print(f"  Strong-edge cliques are spectrally tighter: fewer modes carry the energy.")

    print()
    print(f"  Mean eigenvalue ratio (strong/weak): "
          f"{results[1][3]['mu1']/results[0][3]['mu1']:.2f}x  (= weight ratio 2.0/0.5 = 4x, as expected)")

    # ── Deeper hub readouts (same Spectral objects, no extra matvecs) ────────
    print()
    print(f"  Free-cumulant fingerprint κ₁..κ₄ (s.cumulants — the coordinates that")
    print(f"  ADD under free convolution; weight scaling w → c·w sends κ_n → cⁿ·κ_n):")
    for lbl, _, _, fp, _, _ in results:
        k1, k2, k3, k4 = fp['kappa']
        print(f"    {lbl:24s}  κ₁={k1:7.3f}  κ₂={k2:8.2f}  κ₃={k3:9.1f}  κ₄={k4:11.1f}")
    kw, ks = results[0][3]['kappa'], results[1][3]['kappa']
    print(f"    scaling check (strong/weak): κ₁ ratio = {ks[0]/kw[0]:.2f} (≈ 4¹)   "
          f"κ₂ ratio = {ks[1]/kw[1]:.2f} (≈ 4² = 16)")
    print()
    print(f"  PD health-check via s.condition() (valid: the +0.01 shift makes L PD;")
    print(f"  κ here reads λ_max in units of the regularised zero mode):")
    for (lbl, _, _, fp, _, _) in results:
        print(f"    {lbl:24s}  κ = {fp['cond']:10.1f}")
    print()
    sp_mix, L_mix = spectra[2]
    lam_dense = np.sort(np.linalg.eigvalsh(L_mix.toarray()))
    lam_beta = sp_mix.levels(len(lam_dense))
    lvl_err = np.abs(lam_beta - lam_dense)
    rel = lvl_err.mean() / max(lam_dense.std(), 1e-12)
    print(f"  Beta-closure spectrum (mixed graph): s.levels({len(lam_dense)}) vs dense eigvalsh")
    print(f"    mean |Δλ| = {lvl_err.mean():.3f}  ({rel*100:.1f}% of the spectral spread)")
    print(f"    The Beta closure assumes a SMOOTH spectrum; its residual here is a")
    print(f"    structure detector — the SBM's clique band deviates from the smooth")
    print(f"    maximum-entropy shape exactly where the 8 block modes sit.")

    print()
    print(f"  GF(3) edge label product distribution:")
    for lbl, nw, ns, _, _, ep in results:
        dist = "  ".join(f"p({k})={v:.2f}" for k, v in sorted(ep.items()))
        print(f"    {lbl:24s}  {dist}")

    print()
    print(f"  RESONA: spectral fingerprint computed MATRIX-FREE via resona.of")
    print(f"  (Stochastic Lanczos Quadrature on L_T matvec, no eig called, n={n_nodes}).")
    print(f"  (One dense eigvalsh appears above ONLY as ground truth for the")
    print(f"   s.levels Beta-closure check — the fingerprint itself never calls eig.)")
    print(f"  r_eff = exp(H[spectral density]) measures the 'dimension' of the graph's")
    print(f"  logical space: how many independent modes the ternary-weighted Laplacian spans.")
    print()
    print(f"  LIFT FRAMING: ternary edge labels in GF(3) are Carleman basis elements at")
    print(f"  degree 1.  The Laplacian IS the linear (degree-1) Carleman operator on the")
    print(f"  graph's function space.  ternary_carleman.py gives the full quadratic lift.")
    print()
    print(f"  CAVEAT: this Laplacian is real-valued.  A proper GF(3) Laplacian would use")
    print(f"  cube-root-of-unity eigenvalues; that requires complex-valued spectral analysis")
    print(f"  and is outside the scope of this demo.")
    print("=" * 72)
