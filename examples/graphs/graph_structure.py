"""
graph_structure.py — precompute graph structure once, query O(1).
=============================================================================
WHAT.  Build a random graph of a few thousand nodes, run three O(V+E)
passes — BFS connectivity, iterative Tarjan DFS for bridges/articulation
points, Batagelj-Zaversnik k-core decomposition — then answer structural
queries in O(1) (array/dict lookup).

WHY.  Real-time graph analytics (network monitoring, social graphs, routing)
needs instant answers to "is (u,v) a bridge?", "is v an articulation point?",
"what k-core is v in?".  Paying O(V+E) per query is unacceptable; paying it
*once* at build time makes every subsequent query a single dict/array access.

resona's ROLE.  After the structural build we form the graph Laplacian (as a
sparse matvec) and call resona.of to get the spectral effective_rank and an
estimate of the algebraic connectivity (λ₂, the Fiedler value).  These give
a *spectral complexity readout*: effective_rank measures how many independent
modes the graph has (high ⇒ complex / multi-scale structure), and λ₂ measures
how well-connected the graph is (small λ₂ ⇒ nearly disconnected).  Together
they complement the combinatorial decomposition with a continuous complexity
signature — bridging graph algorithms to the resona lens.

Run:  python3 examples/graphs/graph_structure.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import resona

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# GraphOperator — Tarjan + BFS + k-core, all in one build pass
# ---------------------------------------------------------------------------

class GraphOperator:
    """Precompute ALL structural properties once (O(V+E)); every query is O(1)."""

    def __init__(self, n_vertices: int, edges: list):
        t0 = time.perf_counter()
        self.n = n_vertices
        self.adj = [[] for _ in range(n_vertices)]
        for u, v in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

        self._bfs_connectivity()
        self._tarjan_bridges_articulations()
        self._kcore()

        self.build_time = time.perf_counter() - t0

    # --- BFS connectivity ------------------------------------------------
    def _bfs_connectivity(self):
        self.comp = [-1] * self.n
        self.depth = [-1] * self.n
        comp_sizes = []
        cid = 0
        for root in range(self.n):
            if self.comp[root] != -1:
                continue
            q = [root]; head = 0
            self.comp[root] = cid; self.depth[root] = 0; sz = 0
            while head < len(q):
                u = q[head]; head += 1; sz += 1
                for w in self.adj[u]:
                    if self.comp[w] == -1:
                        self.comp[w] = cid
                        self.depth[w] = self.depth[u] + 1
                        q.append(w)
            comp_sizes.append(sz); cid += 1
        self.n_comps = cid
        self._comp_sizes = comp_sizes

    def component_id(self, v: int) -> int:   return self.comp[v]
    def component_size(self, v: int) -> int: return self._comp_sizes[self.comp[v]]
    def connected(self, u: int, v: int) -> bool: return self.comp[u] == self.comp[v]

    # --- Tarjan iterative DFS for bridges + articulation points ----------
    def _tarjan_bridges_articulations(self):
        tin = [-1] * self.n
        low  = [-1] * self.n
        self.is_articulation = [False] * self.n
        self.bridges = set()
        timer = [0]

        for start in range(self.n):
            if tin[start] != -1:
                continue
            stack = [(start, -1, 0)]   # (vertex, parent, adj_index)
            parent_of = {start: -1}
            while stack:
                u, par, idx = stack[-1]
                if tin[u] == -1:
                    tin[u] = low[u] = timer[0]; timer[0] += 1
                found_child = False
                while idx < len(self.adj[u]):
                    w = self.adj[u][idx]; idx += 1
                    stack[-1] = (u, par, idx)
                    if w == par:
                        continue
                    if tin[w] != -1:
                        low[u] = min(low[u], tin[w])
                    else:
                        parent_of[w] = u
                        stack.append((w, u, 0))
                        found_child = True
                        break
                if not found_child:
                    stack.pop()
                    if par != -1:
                        low[par] = min(low[par], low[u])
                        if low[u] > tin[par]:
                            e = (min(par, u), max(par, u))
                            self.bridges.add(e)
                        if low[u] >= tin[par]:
                            self.is_articulation[par] = True

            # Root: articulation iff it has >1 DFS tree children
            root_children = sum(1 for w in self.adj[start] if parent_of.get(w) == start)
            if root_children > 1:
                self.is_articulation[start] = True

    def is_bridge_edge(self, u: int, v: int) -> bool:
        return (min(u, v), max(u, v)) in self.bridges

    def is_articulation_vertex(self, v: int) -> bool:
        return self.is_articulation[v]

    # --- Batagelj-Zaversnik k-core decomposition -------------------------
    def _kcore(self):
        deg = [len(self.adj[v]) for v in range(self.n)]
        max_d = max(deg) if self.n else 0
        buckets = [[] for _ in range(max_d + 1)]
        for v in range(self.n):
            buckets[deg[v]].append(v)
        pos = {v: i for d in range(max_d + 1) for i, v in enumerate(buckets[d])}
        self.coreness = [0] * self.n
        removed = [False] * self.n
        for d in range(max_d + 1):
            for v in list(buckets[d]):
                if removed[v]:
                    continue
                self.coreness[v] = d; removed[v] = True
                for w in self.adj[v]:
                    if removed[w]:
                        continue
                    dw = deg[w]; pw = pos[w]
                    last = buckets[dw][-1]
                    buckets[dw][pw] = last; pos[last] = pw
                    buckets[dw].pop()
                    deg[w] -= 1
                    buckets[deg[w]].append(w); pos[w] = len(buckets[deg[w]]) - 1

    def get_coreness(self, v: int) -> int:
        return self.coreness[v]


# ---------------------------------------------------------------------------
# Build graph & run structure analysis
# ---------------------------------------------------------------------------

V, E_target = 5_000, 25_000
print("=" * 72)
print(f"GRAPH STRUCTURE — precompute once, O(1) queries  (V={V}, E≈{E_target})")
print("=" * 72)

# Random Erdos-Renyi-ish graph
src = rng.integers(0, V, E_target)
dst = rng.integers(0, V, E_target)
mask = src != dst
edges = list(zip(src[mask].tolist(), dst[mask].tolist()))
E = len(edges)

print(f"\n  Graph: {V} vertices, {E} edges (random)")

op = GraphOperator(V, edges)
print(f"\n  Build time          : {op.build_time*1000:.1f} ms")
print(f"  Components          : {op.n_comps}")
print(f"  Bridges             : {len(op.bridges)}")
print(f"  Articulation points : {sum(op.is_articulation)}")
print(f"  Max coreness        : {max(op.coreness)}")

# ---------------------------------------------------------------------------
# O(1) query timing
# ---------------------------------------------------------------------------

N_QUERIES = 10_000
test_u = rng.integers(0, V, N_QUERIES).tolist()
test_v = rng.integers(0, V, N_QUERIES).tolist()

t0 = time.perf_counter()
for u, v in zip(test_u, test_v):
    op.connected(u, v)
conn_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
for u, v in zip(test_u, test_v):
    op.is_bridge_edge(u, v)
bridge_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
for v in test_u:
    op.is_articulation_vertex(v)
artic_ms = (time.perf_counter() - t0) * 1000

print(f"\n  Query times ({N_QUERIES:,} queries each):")
print(f"    connected()            : {conn_ms:.2f} ms  ({conn_ms/N_QUERIES*1e3:.0f} µs/query)")
print(f"    is_bridge_edge()       : {bridge_ms:.2f} ms  ({bridge_ms/N_QUERIES*1e3:.0f} µs/query)")
print(f"    is_articulation_vertex : {artic_ms:.3f} ms  ({artic_ms/N_QUERIES*1e3:.1f} µs/query)")

# ---------------------------------------------------------------------------
# resona spectral readout of the graph Laplacian
# ---------------------------------------------------------------------------

print(f"\n  resona spectral complexity readout (graph Laplacian):")

# Build sparse Laplacian
rows, cols, data_vals = [], [], []
deg_arr = np.zeros(V)
for u, v in edges:
    rows += [u, v]; cols += [v, u]; data_vals += [-1.0, -1.0]
    deg_arr[u] += 1; deg_arr[v] += 1

L_offdiag = sparse.csr_matrix((data_vals, (rows, cols)), shape=(V, V))
L_diag = sparse.diags(deg_arr)
L = L_diag + L_offdiag   # graph Laplacian

matvec = lambda v: L @ v
s = resona.of(matvec, V, k=64, probes=16)

eff_rank = s.effective_rank()
lam_min, lam_max = s.extreme()

# Algebraic connectivity via eigsh (few exact eigenpairs, matrix-free Lanczos)
try:
    vals_small = eigsh(L, k=6, which="SM", tol=1e-6, return_eigenvectors=False)
    vals_small.sort()
    lam2 = float(vals_small[1]) if len(vals_small) > 1 else float("nan")
    alg_conn_str = f"{lam2:.4f}"
except Exception as e:
    alg_conn_str = f"n/a ({e})"

print(f"    Laplacian support   : [{lam_min:.3f}, {lam_max:.3f}]")
print(f"    Effective rank      : {eff_rank:.1f}  (out of {V}; ratio={eff_rank/V:.3f})")
print(f"    Algebraic conn. λ₂  : {alg_conn_str}  (eigsh; >0 ⟹ connected)")
print(f"    (λ₂>0 confirms the giant component is connected;")
print(f"     effective_rank/{V} ≈ complexity of the spectral structure)")

print()
print("=" * 72)
print("  Build O(V+E), queries O(1). resona maps the Laplacian to a spectral")
print("  complexity score — complementing combinatorial analysis with the")
print("  continuous eigenvalue distribution without a full diagonalisation.")
print("=" * 72)
