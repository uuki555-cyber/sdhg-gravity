"""
EXPERIMENTAL: Fast 2+1D CDT using numba JIT.

Attempt to accelerate the MC loop from run_cdt_2plus1d.py using
numba-compiled array operations. The MC moves are ~100x faster,
but volume stabilization at the CDT critical point is not yet
working correctly (volume instability: growth then collapse).

The reference implementation is run_cdt_2plus1d.py (pure Python,
~30 min per run, correct gamma ≈ 0.34).

Status: EXPERIMENTAL — do not use for published results.

Usage:
    python cdt_fast.py --quick
"""
import numpy as np
from numba import njit, int32
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
import time
import sys

MAX_TRI = 20000
MAX_VERT = 8000
MAX_DEG = 30  # max triangles per vertex


@njit(cache=True)
def sort3(a, b, c):
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return a, b, c


@njit(cache=True)
def init_torus(L):
    """Initialize L×L torus triangulation.

    Returns:
        tri[MAX_TRI, 3]: triangle vertices (sorted)
        n_tri: alive triangle count
        vt[MAX_VERT, MAX_DEG]: vertex → triangle indices
        vt_deg[MAX_VERT]: number of triangles per vertex
        n_vert: total vertex count
        next_vid: next vertex ID
    """
    tri = np.full((MAX_TRI, 3), -1, dtype=int32)
    vt = np.full((MAX_VERT, MAX_DEG), -1, dtype=int32)
    vt_deg = np.zeros(MAX_VERT, dtype=int32)
    n_tri = 0

    for x in range(L):
        for y in range(L):
            v00 = x * L + y
            v10 = ((x + 1) % L) * L + y
            v01 = x * L + (y + 1) % L
            v11 = ((x + 1) % L) * L + (y + 1) % L

            a, b, c = sort3(v00, v10, v01)
            ti = n_tri
            tri[ti, 0], tri[ti, 1], tri[ti, 2] = a, b, c
            for v in (a, b, c):
                d = vt_deg[v]
                if d < MAX_DEG:
                    vt[v, d] = ti
                    vt_deg[v] = d + 1
            n_tri += 1

            a, b, c = sort3(v10, v11, v01)
            ti = n_tri
            tri[ti, 0], tri[ti, 1], tri[ti, 2] = a, b, c
            for v in (a, b, c):
                d = vt_deg[v]
                if d < MAX_DEG:
                    vt[v, d] = ti
                    vt_deg[v] = d + 1
            n_tri += 1

    return tri, n_tri, vt, vt_deg, L * L, L * L


@njit(cache=True)
def find_other_tri(vt, vt_deg, v1, v2, exclude_ti):
    """Find triangle containing both v1 and v2, excluding exclude_ti."""
    for i in range(vt_deg[v1]):
        ti = vt[v1, i]
        if ti == exclude_ti or ti < 0:
            continue
        # Check if ti also contains v2
        for j in range(vt_deg[v2]):
            if vt[v2, j] == ti:
                return ti
    return -1


@njit(cache=True)
def remove_tri_from_vt(vt, vt_deg, v, ti):
    """Remove triangle ti from vertex v's list."""
    deg = vt_deg[v]
    for i in range(deg):
        if vt[v, i] == ti:
            vt[v, i] = vt[v, deg - 1]
            vt[v, deg - 1] = -1
            vt_deg[v] = deg - 1
            return


@njit(cache=True)
def add_tri_to_vt(vt, vt_deg, v, ti):
    """Add triangle ti to vertex v's list."""
    d = vt_deg[v]
    if d < MAX_DEG:
        vt[v, d] = ti
        vt_deg[v] = d + 1


@njit(cache=True)
def flip_edge(tri, n_tri, vt, vt_deg):
    """Bistellar (2,2) flip on random edge."""
    # Pick random alive triangle
    for _ in range(20):
        ti_a = np.random.randint(0, n_tri)
        if tri[ti_a, 0] >= 0:
            break
    else:
        return False
    if tri[ti_a, 0] < 0:
        return False

    # Pick random edge
    ei = np.random.randint(0, 3)
    v1 = tri[ti_a, ei]
    v2 = tri[ti_a, (ei + 1) % 3]

    # Find other triangle sharing this edge
    ti_b = find_other_tri(vt, vt_deg, v1, v2, ti_a)
    if ti_b < 0:
        return False

    # Find opposite vertices
    v3 = -1
    for k in range(3):
        vk = tri[ti_a, k]
        if vk != v1 and vk != v2:
            v3 = vk
            break
    v4 = -1
    for k in range(3):
        vk = tri[ti_b, k]
        if vk != v1 and vk != v2:
            v4 = vk
            break
    if v3 < 0 or v4 < 0 or v3 == v4:
        return False

    # Check new edge doesn't already exist (v3-v4 share any triangle)
    existing = find_other_tri(vt, vt_deg, v3, v4, -1)
    if existing >= 0:
        return False

    # Check minimum degree
    if vt_deg[v1] <= 3 or vt_deg[v2] <= 3:
        return False

    # Execute: remove old, add new
    remove_tri_from_vt(vt, vt_deg, v1, ti_a)
    remove_tri_from_vt(vt, vt_deg, v2, ti_a)
    remove_tri_from_vt(vt, vt_deg, v3, ti_a)
    remove_tri_from_vt(vt, vt_deg, v1, ti_b)
    remove_tri_from_vt(vt, vt_deg, v2, ti_b)
    remove_tri_from_vt(vt, vt_deg, v4, ti_b)

    a, b, c = sort3(v3, v4, v1)
    tri[ti_a, 0], tri[ti_a, 1], tri[ti_a, 2] = a, b, c
    a, b, c = sort3(v3, v4, v2)
    tri[ti_b, 0], tri[ti_b, 1], tri[ti_b, 2] = a, b, c

    add_tri_to_vt(vt, vt_deg, v3, ti_a)
    add_tri_to_vt(vt, vt_deg, v4, ti_a)
    add_tri_to_vt(vt, vt_deg, v1, ti_a)
    add_tri_to_vt(vt, vt_deg, v3, ti_b)
    add_tri_to_vt(vt, vt_deg, v4, ti_b)
    add_tri_to_vt(vt, vt_deg, v2, ti_b)
    return True


@njit(cache=True)
def insert_vertex(tri, n_tri, vt, vt_deg, next_vid):
    """Insert vertex into random triangle."""
    if n_tri + 2 >= MAX_TRI or next_vid >= MAX_VERT:
        return n_tri, next_vid, False

    for _ in range(20):
        ti = np.random.randint(0, n_tri)
        if tri[ti, 0] >= 0:
            break
    else:
        return n_tri, next_vid, False
    if tri[ti, 0] < 0:
        return n_tri, next_vid, False

    v0, v1, v2 = tri[ti, 0], tri[ti, 1], tri[ti, 2]
    v_new = next_vid

    # Remove old triangle from vertex maps
    remove_tri_from_vt(vt, vt_deg, v0, ti)
    remove_tri_from_vt(vt, vt_deg, v1, ti)
    remove_tri_from_vt(vt, vt_deg, v2, ti)

    # Replace: (v0,v1,v_new)
    a, b, c = sort3(v0, v1, v_new)
    tri[ti, 0], tri[ti, 1], tri[ti, 2] = a, b, c
    add_tri_to_vt(vt, vt_deg, a, ti)
    add_tri_to_vt(vt, vt_deg, b, ti)
    add_tri_to_vt(vt, vt_deg, c, ti)

    # New: (v1,v2,v_new)
    ti_b = n_tri
    a, b, c = sort3(v1, v2, v_new)
    tri[ti_b, 0], tri[ti_b, 1], tri[ti_b, 2] = a, b, c
    add_tri_to_vt(vt, vt_deg, a, ti_b)
    add_tri_to_vt(vt, vt_deg, b, ti_b)
    add_tri_to_vt(vt, vt_deg, c, ti_b)

    # New: (v0,v2,v_new)
    ti_c = n_tri + 1
    a, b, c = sort3(v0, v2, v_new)
    tri[ti_c, 0], tri[ti_c, 1], tri[ti_c, 2] = a, b, c
    add_tri_to_vt(vt, vt_deg, a, ti_c)
    add_tri_to_vt(vt, vt_deg, b, ti_c)
    add_tri_to_vt(vt, vt_deg, c, ti_c)

    return n_tri + 2, next_vid + 1, True


@njit(cache=True)
def remove_vertex(tri, n_tri, vt, vt_deg):
    """Remove a random degree-3 vertex."""
    # Find degree-3 vertex by sampling from alive triangles
    for _ in range(30):
        ti = np.random.randint(0, n_tri)
        if tri[ti, 0] < 0:
            continue
        k = np.random.randint(0, 3)
        v = tri[ti, k]
        if vt_deg[v] == 3:
            # Found! Get the 3 triangle indices
            t0 = vt[v, 0]
            t1 = vt[v, 1]
            t2 = vt[v, 2]
            if t0 < 0 or t1 < 0 or t2 < 0:
                continue

            # Collect other vertices
            others = np.full(6, -1, dtype=int32)
            n_oth = 0
            for ti2 in (t0, t1, t2):
                for kk in range(3):
                    vk = tri[ti2, kk]
                    if vk != v:
                        found = False
                        for j in range(n_oth):
                            if others[j] == vk:
                                found = True
                                break
                        if not found and n_oth < 6:
                            others[n_oth] = vk
                            n_oth += 1
            if n_oth != 3:
                continue

            # Remove all 3 triangles from vertex maps
            for ti2 in (t0, t1, t2):
                for kk in range(3):
                    vk = tri[ti2, kk]
                    remove_tri_from_vt(vt, vt_deg, vk, ti2)

            # Replace t0 with merged triangle
            a, b, c = sort3(others[0], others[1], others[2])
            tri[t0, 0], tri[t0, 1], tri[t0, 2] = a, b, c
            add_tri_to_vt(vt, vt_deg, a, t0)
            add_tri_to_vt(vt, vt_deg, b, t0)
            add_tri_to_vt(vt, vt_deg, c, t0)

            # Mark t1, t2 as dead
            tri[t1, 0] = tri[t1, 1] = tri[t1, 2] = -1
            tri[t2, 0] = tri[t2, 1] = tri[t2, 2] = -1

            return True
    return False


@njit(cache=True)
def compact(tri, n_tri, vt, vt_deg):
    """Compact dead triangles and rebuild vertex maps."""
    # First, compact tri array
    old_to_new = np.full(n_tri, -1, dtype=int32)
    write = 0
    for read in range(n_tri):
        if tri[read, 0] >= 0:
            old_to_new[read] = write
            if write != read:
                tri[write, 0] = tri[read, 0]
                tri[write, 1] = tri[read, 1]
                tri[write, 2] = tri[read, 2]
            write += 1
    new_n_tri = write
    for i in range(write, n_tri):
        tri[i, 0] = tri[i, 1] = tri[i, 2] = -1

    # Rebuild vertex maps from scratch
    for v in range(MAX_VERT):
        vt_deg[v] = 0
        for j in range(MAX_DEG):
            vt[v, j] = -1

    for ti in range(new_n_tri):
        for k in range(3):
            v = tri[ti, k]
            d = vt_deg[v]
            if d < MAX_DEG:
                vt[v, d] = ti
                vt_deg[v] = d + 1

    return new_n_tri


@njit(cache=True)
def mc_sweep_full(tri, n_tri, vt, vt_deg, next_vid, n_vert, n_moves,
                  target_v=0, eps=0.01):
    """One MC sweep with optional volume control.

    target_v > 0: soft volume constraint via quadratic penalty.
    target_v = 0: free volume (critical point, dS=0).
    """
    for _ in range(n_moves):
        r = np.random.random()
        if r < 0.4:
            # Insertion with volume penalty
            if target_v > 0:
                dS = eps * (2 * (n_vert - target_v) + 1)
                if np.random.random() >= np.exp(-min(max(dS, -20), 20)):
                    continue
            n_tri, next_vid, ok = insert_vertex(tri, n_tri, vt, vt_deg, next_vid)
            if ok:
                n_vert += 1
        elif r < 0.7:
            # Removal with volume penalty
            if target_v > 0:
                dS = eps * (-2 * (n_vert - target_v) + 1)
                if np.random.random() >= np.exp(-min(max(dS, -20), 20)):
                    continue
            ok = remove_vertex(tri, n_tri, vt, vt_deg)
            if ok:
                n_vert -= 1
        else:
            flip_edge(tri, n_tri, vt, vt_deg)
    return n_tri, next_vid, n_vert


# ============================================================
# Python-level functions
# ============================================================

def extract_graph(tri, n_tri):
    """Extract edges and vertices from alive triangles."""
    edges = set()
    verts = set()
    for ti in range(n_tri):
        if tri[ti, 0] < 0:
            continue
        v0, v1, v2 = tri[ti, 0], tri[ti, 1], tri[ti, 2]
        verts.update([v0, v1, v2])
        edges.add((min(v0, v1), max(v0, v1)))
        edges.add((min(v1, v2), max(v1, v2)))
        edges.add((min(v0, v2), max(v0, v2)))
    return list(edges), verts


def build_spacetime(slices_data, T):
    """Build spacetime adjacency graph from T slices."""
    all_verts = {}
    gid = 0
    for t in range(T):
        edges, verts = slices_data[t]
        for v in sorted(verts):
            all_verts[(t, v)] = gid
            gid += 1
    N = gid
    adj = [set() for _ in range(N)]

    for t in range(T):
        tn = (t + 1) % T
        edges_t, verts_t = slices_data[t]
        _, verts_tn = slices_data[tn]

        # Spatial edges
        for v1, v2 in edges_t:
            g1 = all_verts.get((t, v1))
            g2 = all_verts.get((t, v2))
            if g1 is not None and g2 is not None:
                adj[g1].add(g2)
                adj[g2].add(g1)

        # Temporal links
        vl_t = sorted(verts_t)
        vl_tn = sorted(verts_tn)
        for i in range(min(len(vl_t), len(vl_tn))):
            g1 = all_verts.get((t, vl_t[i]))
            g2 = all_verts.get((tn, vl_tn[i]))
            if g1 is not None and g2 is not None:
                adj[g1].add(g2)
                adj[g2].add(g1)

        # Cross-links (match original cdt_dynamic.py)
        for v1, v2 in edges_t:
            for vv in [v1, v2]:
                for vn in vl_tn[:3]:
                    g1 = all_verts.get((t, vv))
                    g2 = all_verts.get((tn, vn))
                    if g1 is not None and g2 is not None:
                        adj[g1].add(g2)
                        adj[g2].add(g1)

    return adj, N


def spectral_dim(adj, N):
    n_eig = min(400, N - 2)
    L_mat = lil_matrix((N, N))
    for v in range(N):
        deg = len(adj[v])
        L_mat[v, v] = deg
        for u in adj[v]:
            L_mat[v, u] = -1
    try:
        eigvals = eigsh(L_mat.tocsr(), k=n_eig, which='SM',
                        return_eigenvectors=False)
    except Exception:
        return None, None
    eigvals = np.sort(np.real(eigvals))
    eigvals = eigvals[eigvals > 1e-8]
    if len(eigvals) < 10:
        return None, None

    sigmas = np.logspace(-1, 3, 100)
    K = np.array([np.sum(np.exp(-s * eigvals)) for s in sigmas])
    d_spec = np.zeros(len(sigmas))
    for i in range(1, len(sigmas) - 1):
        if K[i] > 0:
            d_spec[i] = -2 * (np.log(K[i+1]+1e-300) - np.log(K[i-1]+1e-300)) / \
                         (np.log(sigmas[i+1]) - np.log(sigmas[i-1]))
    return sigmas, d_spec


def fit_gamma(sigmas, d_spec):
    valid = (d_spec > 0.3) & (d_spec < 5)
    if valid.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan

    def obj(p):
        ls0, gamma, d_uv, d_ir = p
        s0 = 10**ls0; s = 0; n = 0
        for i in range(len(sigmas)):
            if not valid[i]: continue
            u = (sigmas[i]/s0)**gamma
            s += (d_spec[i] - (d_uv + (d_ir-d_uv)*u/(1+u)))**2; n += 1
        return np.sqrt(s/max(n,1))

    r = minimize(obj, [1, 0.5, 1.5, 3.0],
                 bounds=[(-2,4),(0.05,3),(0.5,2.5),(2.0,4.0)], method='L-BFGS-B')
    return r.x[1], r.x[2], r.x[3], r.fun


# ============================================================
# Main
# ============================================================
def main():
    quick = "--quick" in sys.argv
    t0 = time.time()

    print("=" * 70)
    print("2+1D CDT: numba-accelerated spectral dimension")
    print("=" * 70)

    # JIT warmup
    print("JIT compiling...", end="", flush=True)
    tri, n_tri, vt, vt_deg, nv, nvid = init_torus(4)
    mc_sweep_full(tri, n_tri, vt, vt_deg, nvid, nv, 10)
    compact(tri, n_tri, vt, vt_deg)
    print(f" done ({time.time()-t0:.1f}s)")

    if quick:
        configs = [(8, 12, 3), (10, 15, 3), (12, 18, 3)]
        n_sweeps = 200
    else:
        configs = [(8, 12, 5), (10, 15, 5), (12, 18, 5),
                   (15, 22, 3), (18, 26, 3)]
        n_sweeps = 200

    all_results = {}

    for L, T, n_seeds in configs:
        target_v = int(L * L * 1.5)  # allow ~50% volume fluctuation
        print(f"\n--- L={L}, T={T}, {n_seeds} seeds, target_v={target_v} ---")
        gammas = []

        for si in range(n_seeds):
            seed = 42 + L * 100 + si
            np.random.seed(seed)

            # Initialize T slices
            slices = []
            for t in range(T):
                slices.append(init_torus(L))

            # Co-evolve with volume control
            t1 = time.time()
            for sweep in range(n_sweeps):
                for t in range(T):
                    tri, n_tri, vt, vt_deg, nv, nvid = slices[t]
                    n_moves = max(nv, 30)
                    n_tri, nvid, nv = mc_sweep_full(
                        tri, n_tri, vt, vt_deg, nvid, nv, n_moves,
                        target_v=target_v, eps=0.01)
                    slices[t] = (tri, n_tri, vt, vt_deg, nv, nvid)

                if (sweep + 1) % 20 == 0:
                    for t in range(T):
                        tri, n_tri, vt, vt_deg, nv, nvid = slices[t]
                        n_tri = compact(tri, n_tri, vt, vt_deg)
                        slices[t] = (tri, n_tri, vt, vt_deg, nv, nvid)

            avg_v = np.mean([s[4] for s in slices])
            print(f"  Seed {si+1}: avg_v={avg_v:.0f} ({time.time()-t1:.1f}s)",
                  end="")

            # Extract and measure
            slices_data = []
            for t in range(T):
                tri, n_tri, vt, vt_deg, nv, nvid = slices[t]
                slices_data.append(extract_graph(tri, n_tri))

            adj, N = build_spacetime(slices_data, T)
            sigmas, d_spec = spectral_dim(adj, N)
            if sigmas is not None:
                g, duv, d_ir, rms = fit_gamma(sigmas, d_spec)
                gammas.append(g)
                print(f" N={N} gamma={g:.3f} d_UV={duv:.2f} d_IR={d_ir:.2f}")
            else:
                print(" FAILED")

        if gammas:
            all_results[L] = np.array(gammas)

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\n{'L':>4} {'gamma (free d_UV)':>20} {'N':>4}")
    print("-" * 32)

    all_g = []
    for L in sorted(all_results.keys()):
        g = all_results[L]
        n = len(g)
        se = np.std(g)/max(np.sqrt(n-1),1) if n>1 else 0.05
        print(f"{L:>4} {np.mean(g):>12.3f} ± {se:.3f}  {n:>4}")
        all_g.extend(g)

    all_g = np.array(all_g)
    if len(all_g) >= 3:
        m = np.mean(all_g)
        se = np.std(all_g)/np.sqrt(len(all_g)-1)
        print(f"\nOverall: gamma = {m:.3f} ± {se:.3f}")
        print(f"SDHG prediction (1/3): {1/3:.4f}")
        print(f"Deviation: {abs(m-1/3)/max(se,0.001):.1f}σ")

        boot = [np.mean(all_g[np.random.randint(0,len(all_g),len(all_g))])
                for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"Bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]")
        print(f"1/3 = {1/3:.3f} {'inside' if lo<=1/3<=hi else 'OUTSIDE'} CI")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
