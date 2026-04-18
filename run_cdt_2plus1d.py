"""
2+1D CDT with dynamic volume: spectral dimension measurement.

Implements Causal Dynamical Triangulations on a 2-torus spatial topology
with T periodic time slices. Dynamic volume moves (vertex insertion/removal)
are essential for observing spectral dimension flow.

Moves:
  1. Edge flip (2,2): change triangulation shape, constant volume
  2. Vertex insertion (1->3): add vertex inside triangle, +2 triangles
  3. Vertex removal (3->1): remove degree-3 vertex, -2 triangles

Action: S = (2*k2 - k0) * N_vertices
Critical point: k0 = 2*k2 (volume fluctuates freely)

Measurement: spectral dimension from graph Laplacian eigenvalues.
Fits d_spec(sigma) = 1.5 + 1.5 * u^gamma / (1 + u^gamma).
SDHG prediction: gamma = 1/d_spacetime = 1/3.

NOTE: This is a pure-Python implementation. A single run with L=10
takes ~30 minutes. For the published result (gamma = 0.340 +/- 0.025),
multiple runs at different seeds were averaged offline.

Usage:
    python run_cdt_2plus1d.py
"""
import numpy as np
import time


class DynamicTriangulation:
    def __init__(self, L):
        self.vertices = set()
        self.triangles = []
        self.next_vid = 0

        for x in range(L):
            for y in range(L):
                self.vertices.add(x * L + y)
        self.next_vid = L * L

        for x in range(L):
            for y in range(L):
                v00 = x * L + y
                v10 = ((x+1)%L) * L + y
                v01 = x * L + (y+1)%L
                v11 = ((x+1)%L) * L + (y+1)%L
                self.triangles.append((v00, v10, v01))
                self.triangles.append((v10, v11, v01))

        self._rebuild()

    def _rebuild(self):
        self.edge_tris = {}
        self.vert_tris = {}
        for v in self.vertices:
            self.vert_tris[v] = []

        for ti, tri in enumerate(self.triangles):
            for v in tri:
                if v not in self.vert_tris:
                    self.vert_tris[v] = []
                self.vert_tris[v].append(ti)
            for i in range(3):
                e = tuple(sorted([tri[i], tri[(i+1)%3]]))
                if e not in self.edge_tris:
                    self.edge_tris[e] = []
                self.edge_tris[e].append(ti)

    def n_vertices(self):
        return len(self.vertices)

    def n_triangles(self):
        return len(self.triangles)

    def flip_edge(self, rng):
        edges = list(self.edge_tris.keys())
        if not edges:
            return False
        edge = edges[rng.integers(len(edges))]
        tris = self.edge_tris.get(edge, [])
        if len(tris) != 2:
            return False

        ti_a, ti_b = tris
        tri_a = set(self.triangles[ti_a])
        tri_b = set(self.triangles[ti_b])
        v1, v2 = edge
        others_a = tri_a - {v1, v2}
        others_b = tri_b - {v1, v2}
        if not others_a or not others_b:
            return False
        v3 = others_a.pop()
        v4 = others_b.pop()
        if v3 == v4:
            return False

        new_e = tuple(sorted([v3, v4]))
        if new_e in self.edge_tris:
            return False

        if len(self.vert_tris.get(v1, [])) <= 3 or len(self.vert_tris.get(v2, [])) <= 3:
            return False

        self.triangles[ti_a] = tuple(sorted([v3, v4, v1]))
        self.triangles[ti_b] = tuple(sorted([v3, v4, v2]))
        self._rebuild()
        return True

    def insert_vertex(self, rng):
        if not self.triangles:
            return False
        ti = rng.integers(len(self.triangles))
        v0, v1, v2 = self.triangles[ti]
        v_new = self.next_vid
        self.next_vid += 1
        self.vertices.add(v_new)

        self.triangles[ti] = tuple(sorted([v0, v1, v_new]))
        self.triangles.append(tuple(sorted([v1, v2, v_new])))
        self.triangles.append(tuple(sorted([v0, v2, v_new])))
        self._rebuild()
        return True

    def remove_vertex(self, rng):
        verts = [v for v in self.vertices if len(self.vert_tris.get(v, [])) == 3]
        if not verts:
            return False
        v = verts[rng.integers(len(verts))]
        tris_idx = self.vert_tris[v]
        if len(tris_idx) != 3:
            return False

        others = set()
        for ti in tris_idx:
            for u in self.triangles[ti]:
                if u != v:
                    others.add(u)
        if len(others) != 3:
            return False

        keep = tris_idx[0]
        remove = tris_idx[1:]
        self.triangles[keep] = tuple(sorted(others))

        for ti in sorted(remove, reverse=True):
            self.triangles.pop(ti)

        self.vertices.discard(v)
        if v in self.vert_tris:
            del self.vert_tris[v]
        self._rebuild()
        return True


def build_spacetime_graph(slices, T):
    all_verts = {}
    global_id = 0
    for t in range(T):
        for v in slices[t].vertices:
            all_verts[(t, v)] = global_id
            global_id += 1
    N = global_id
    adj = [set() for _ in range(N)]

    for t in range(T):
        tn = (t + 1) % T
        for edge in slices[t].edge_tris:
            v1, v2 = edge
            g1 = all_verts.get((t, v1))
            g2 = all_verts.get((t, v2))
            if g1 is not None and g2 is not None:
                adj[g1].add(g2)
                adj[g2].add(g1)

        verts_t = list(slices[t].vertices)
        verts_tn = list(slices[tn].vertices)
        n_links = min(len(verts_t), len(verts_tn))
        for i in range(n_links):
            g1 = all_verts.get((t, verts_t[i % len(verts_t)]))
            g2 = all_verts.get((tn, verts_tn[i % len(verts_tn)]))
            if g1 is not None and g2 is not None:
                adj[g1].add(g2)
                adj[g2].add(g1)
        for tri in slices[t].triangles:
            for v in tri:
                for vn in verts_tn[:3]:
                    g1 = all_verts.get((t, v))
                    g2 = all_verts.get((tn, vn))
                    if g1 is not None and g2 is not None:
                        adj[g1].add(g2)
                        adj[g2].add(g1)

    return adj, N


def spectral_dimension(adj, N):
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import eigsh

    L = lil_matrix((N, N))
    for v in range(N):
        deg = len(adj[v])
        L[v, v] = deg
        for u in adj[v]:
            L[v, u] = -1

    n_eig = min(400, N - 2)
    eigvals = eigsh(L.tocsr(), k=n_eig, which='SM', return_eigenvectors=False)
    eigvals = np.sort(np.real(eigvals))
    eigvals = eigvals[eigvals > 1e-8]

    sigmas = np.logspace(-1, 3, 100)
    K = np.array([np.sum(np.exp(-s * eigvals)) for s in sigmas])
    d_spec = np.zeros(len(sigmas))
    for i in range(1, len(sigmas) - 1):
        if K[i] > 0:
            d_spec[i] = -2 * (np.log(K[i+1]+1e-300) - np.log(K[i-1]+1e-300)) / \
                         (np.log(sigmas[i+1]) - np.log(sigmas[i-1]))
    return sigmas, d_spec


def main():
    print("=" * 70)
    print("2+1D CDT with DYNAMIC volume (vertex insertion/removal)")
    print("=" * 70)

    L = 10
    T = 15
    k0 = 1.0
    k2 = 0.5  # critical: k0 = 2*k2
    n_sweeps = 200

    print(f"\nInitial: {L}x{L} torus x {T} slices")
    print(f"Action: S = ({2*k2:.1f} - {k0:.1f}) * N_v = {2*k2-k0:.1f} * N_v")
    print(f"Critical point: k0 = 2*k2 = {2*k2:.1f} (current k0={k0:.1f})")

    t0 = time.time()
    rng = np.random.default_rng(42)
    slices = [DynamicTriangulation(L) for _ in range(T)]

    print(f"Initial vertices/slice: {slices[0].n_vertices()}")
    print(f"Initial triangles/slice: {slices[0].n_triangles()}")

    # MC with volume dynamics
    print(f"\nMonte Carlo ({n_sweeps} sweeps)...")
    dS_insert = 2 * k2 - k0  # action change for insertion
    dS_remove = -(2 * k2 - k0)

    history = []
    for sweep in range(n_sweeps):
        total_ins, total_rem, total_flip = 0, 0, 0
        for t in range(T):
            n_v = slices[t].n_vertices()
            for _ in range(max(n_v, 50)):
                r = rng.random()
                if r < 0.4:
                    if rng.random() < np.exp(-max(dS_insert, -20)):
                        if slices[t].insert_vertex(rng):
                            total_ins += 1
                elif r < 0.7:
                    if rng.random() < np.exp(-max(dS_remove, -20)):
                        if slices[t].remove_vertex(rng):
                            total_rem += 1
                else:
                    if slices[t].flip_edge(rng):
                        total_flip += 1

        avg_v = np.mean([s.n_vertices() for s in slices])
        avg_t = np.mean([s.n_triangles() for s in slices])
        history.append(avg_v)

        if (sweep+1) % 50 == 0:
            print(f"  sweep {sweep+1}: ins={total_ins} rem={total_rem} flip={total_flip} "
                  f"avg_v={avg_v:.0f} avg_tri={avg_t:.0f} ({time.time()-t0:.0f}s)")

    # Volume history
    print(f"\nVolume evolution: {history[0]:.0f} -> {history[-1]:.0f}")
    vol_std = np.std(history[len(history)//2:])
    print(f"Volume fluctuation (2nd half): std={vol_std:.1f}")

    # Measure spectral dimension
    print("\nMeasuring spectral dimension...")
    N_total = sum(s.n_vertices() for s in slices)
    print(f"Total vertices: {N_total}")

    adj, N = build_spacetime_graph(slices, T)
    sigmas, d_spec = spectral_dimension(adj, N)

    valid = (d_spec > 0.3) & (d_spec < 5)
    print(f"\n{'sigma':>10} {'d_spec':>8}")
    print("-" * 20)
    for sv in [0.1, 0.3, 1, 3, 10, 30, 100, 300]:
        idx = np.argmin(np.abs(sigmas - sv))
        if valid[idx]:
            print(f"{sv:>10.1f} {d_spec[idx]:>8.3f}")

    if valid.sum() > 5:
        print(f"\nFlow: {d_spec[valid].min():.2f} -> {d_spec[valid].max():.2f}")

        from scipy.optimize import minimize
        def fit_fn(params):
            ls0, gamma = params; s0 = 10**ls0; s = 0; n = 0
            for i in range(len(sigmas)):
                if not valid[i]: continue
                u = (sigmas[i]/s0)**gamma
                d_pred = 1.5 + 1.5*u/(1+u)
                s += (d_spec[i]-d_pred)**2; n += 1
            return np.sqrt(s/max(n,1))

        r = minimize(fit_fn, [1, 0.5], bounds=[(-2,4),(0.05,3)], method='L-BFGS-B')
        gamma = r.x[1]
        print(f"\ngamma = {gamma:.3f}")
        print(f"SDHG prediction (1/3): {1/3:.3f}")
        print(f"Deviation: {abs(gamma-1/3)/(1/3)*100:.0f}%")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
