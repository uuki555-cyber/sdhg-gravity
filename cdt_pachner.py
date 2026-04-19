"""
Proper 2+1D CDT with spacetime Pachner moves.

Implements (2,3)/(3,2) moves on a tetrahedralized spacetime manifold.
Measures spectral dimension via random walk on dual graph.

Usage:
    python cdt_pachner.py
"""
import random
import numpy as np
from scipy.optimize import minimize
import time
import sys


class CDTComplex:
    """Simplicial complex with incremental face tracking."""

    def __init__(self, tets_list):
        self.tets = {}       # tid -> sorted 4-tuple
        self.next_id = 0
        self.face_to_tets = {}  # face (3-tuple) -> set of tids
        self.vert_to_tets = {}  # vertex -> set of tids

        for t in tets_list:
            self._add_tet(t)

    def n_tets(self):
        return len(self.tets)

    def _add_tet(self, verts):
        tid = self.next_id
        self.next_id += 1
        tet = tuple(sorted(verts))
        self.tets[tid] = tet

        for fi in range(4):
            face = tuple(tet[j] for j in range(4) if j != fi)
            self.face_to_tets.setdefault(face, set()).add(tid)

        for v in tet:
            self.vert_to_tets.setdefault(v, set()).add(tid)

        return tid

    def _remove_tet(self, tid):
        tet = self.tets[tid]
        for fi in range(4):
            face = tuple(tet[j] for j in range(4) if j != fi)
            s = self.face_to_tets.get(face)
            if s:
                s.discard(tid)
                if not s:
                    del self.face_to_tets[face]

        for v in tet:
            s = self.vert_to_tets.get(v)
            if s:
                s.discard(tid)
                if not s:
                    del self.vert_to_tets[v]

        del self.tets[tid]

    def get_neighbor(self, tid, fi):
        tet = self.tets[tid]
        face = tuple(tet[j] for j in range(4) if j != fi)
        others = self.face_to_tets.get(face, set()) - {tid}
        return others.pop() if others else None

    def edge_exists(self, d, e):
        """Check if edge (d,e) exists in the complex."""
        td = self.vert_to_tets.get(d, set())
        te = self.vert_to_tets.get(e, set())
        return bool(td & te)

    def tets_with_edge(self, d, e):
        """Find all tets containing edge (d,e)."""
        td = self.vert_to_tets.get(d, set())
        te = self.vert_to_tets.get(e, set())
        return td & te

    def move_23(self):
        """(2,3) Pachner move: 2 tets sharing a face -> 3 tets sharing an edge."""
        tids = list(self.tets.keys())
        tid1 = random.choice(tids)
        fi = random.randint(0, 3)
        tid2 = self.get_neighbor(tid1, fi)
        if tid2 is None:
            return False

        tet1 = set(self.tets[tid1])
        tet2 = set(self.tets[tid2])
        shared = tet1 & tet2
        if len(shared) != 3:
            return False

        d = (tet1 - shared).pop()
        e = (tet2 - shared).pop()

        # Check: edge (d,e) must not already exist
        if self.edge_exists(d, e):
            return False

        a, b, c = sorted(shared)

        # Execute
        self._remove_tet(tid1)
        self._remove_tet(tid2)
        self._add_tet([a, b, d, e])
        self._add_tet([b, c, d, e])
        self._add_tet([a, c, d, e])
        return True

    def move_32(self):
        """(3,2) Pachner move: 3 tets sharing an edge -> 2 tets sharing a face."""
        tids = list(self.tets.keys())
        tid1 = random.choice(tids)
        tet1 = self.tets[tid1]

        # Pick random edge from this tet
        edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        ei = random.randint(0, 5)
        d, e = tet1[edges[ei][0]], tet1[edges[ei][1]]

        # Find all tets sharing edge (d,e)
        shared_tids = self.tets_with_edge(d, e)
        if len(shared_tids) != 3:
            return False

        # Collect other vertices (should be exactly 3 forming a triangle)
        others = set()
        for tid in shared_tids:
            others |= set(self.tets[tid]) - {d, e}
        if len(others) != 3:
            return False

        a, b, c = sorted(others)

        # Check: face (a,b,c) must not already exist
        face_abc = (a, b, c)
        existing = self.face_to_tets.get(face_abc, set())
        # It should have 0 tets (the face doesn't exist yet)
        if existing - shared_tids:
            return False

        # Execute
        for tid in list(shared_tids):
            self._remove_tet(tid)
        self._add_tet([a, b, c, d])
        self._add_tet([a, b, c, e])
        return True

    def verify(self):
        """Check manifold consistency."""
        unmatched = 0
        over = 0
        for face, ts in self.face_to_tets.items():
            if len(ts) == 1:
                unmatched += 1
            elif len(ts) > 2:
                over += 1
        matched = sum(1 for ts in self.face_to_tets.values() if len(ts) == 2)
        return matched, unmatched, over

    def random_walk_dspec(self, n_walks=200000, sigma_max=400):
        """Spectral dimension via random walk on dual graph."""
        tids = list(self.tets.keys())
        n = len(tids)
        tid_to_idx = {t: i for i, t in enumerate(tids)}

        # Build adjacency as arrays for speed
        adj = [[] for _ in range(n)]
        for i, tid in enumerate(tids):
            tet = self.tets[tid]
            for fi in range(4):
                nb = self.get_neighbor(tid, fi)
                if nb is not None and nb in tid_to_idx:
                    adj[i].append(tid_to_idx[nb])

        P = np.zeros(sigma_max + 1)
        for _ in range(n_walks):
            start = random.randint(0, n - 1)
            if not adj[start]:
                continue
            pos = start
            for sig in range(1, sigma_max + 1):
                if not adj[pos]:
                    break
                pos = random.choice(adj[pos])
                if pos == start:
                    P[sig] += 1
        P /= n_walks
        return P


def build_initial_cdt(L, T):
    """Build initial CDT spacetime: L×L torus × T time slices."""
    n_sv = L * L
    tets = []

    # Use same base triangulation for all slices
    base_tris = []
    for x in range(L):
        for y in range(L):
            v00 = x*L+y
            v10 = ((x+1)%L)*L+y
            v01 = x*L+(y+1)%L
            v11 = ((x+1)%L)*L+(y+1)%L
            base_tris.append(tuple(sorted([v00, v10, v01])))
            base_tris.append(tuple(sorted([v10, v11, v01])))

    # Thermalize base triangulation
    # (simple edge flips)
    vt = {}
    for v in range(n_sv):
        vt[v] = []
    for ti, tri in enumerate(base_tris):
        for v in tri:
            vt[v].append(ti)

    acc = 0
    for _ in range(5000):
        ti = random.randint(0, len(base_tris)-1)
        ei = random.randint(0, 2)
        v1, v2 = base_tris[ti][ei], base_tris[ti][(ei+1)%3]
        mn = min(v1, v2)
        ti2 = None
        for t in vt.get(mn, []):
            if t != ti and max(v1,v2) in base_tris[t]:
                ti2 = t
                break
        if ti2 is None:
            continue
        v3 = [v for v in base_tris[ti] if v != v1 and v != v2][0]
        v4 = [v for v in base_tris[ti2] if v != v1 and v != v2][0]
        if v3 == v4:
            continue
        mn34 = min(v3, v4)
        if any(max(v3,v4) in base_tris[t] for t in vt.get(mn34, [])):
            continue
        if len(vt[v1]) <= 3 or len(vt[v2]) <= 3:
            continue
        base_tris[ti] = tuple(sorted([v3, v4, v1]))
        base_tris[ti2] = tuple(sorted([v3, v4, v2]))
        vt = {}
        for v in range(n_sv):
            vt[v] = []
        for i, tri in enumerate(base_tris):
            for v in tri:
                vt[v].append(i)
        acc += 1

    print(f"  Spatial flips: {acc}/5000")

    # Build tetrahedra: 3 per triangle per slab
    for t in range(T):
        tn = (t + 1) % T
        for tri in base_tris:
            a, b, c = tri  # sorted
            at, bt, ct = t*n_sv+a, t*n_sv+b, t*n_sv+c
            atn, btn, ctn = tn*n_sv+a, tn*n_sv+b, tn*n_sv+c
            tets.append([at, bt, ct, atn])
            tets.append([bt, ct, atn, btn])
            tets.append([ct, atn, btn, ctn])

    return CDTComplex(tets)


def extract_dspec(P, sigma_max):
    """Extract d_spec(sigma) from return probability P."""
    sigs, ds = [], []
    for s in range(8, sigma_max - 8, 2):
        w = max(3, s // 5)
        lo = [P[j] for j in range(max(1, s-w), s+1) if P[j] > 0]
        hi = [P[j] for j in range(s, min(sigma_max, s+w)+1) if P[j] > 0]
        if lo and hi:
            Plo, Phi = np.mean(lo), np.mean(hi)
            if Plo > 0 and Phi > 0:
                d = -2 * (np.log(Phi) - np.log(Plo)) / \
                    (np.log(s + w/2) - np.log(max(1, s - w/2)))
                if 0.5 < d < 8:
                    sigs.append(s)
                    ds.append(d)
    return np.array(sigs, dtype=float), np.array(ds)


def main():
    t0 = time.time()
    L, T = 8, 12
    n_pachner = 5000
    n_walks = 200000
    sigma_max = 400

    print("=" * 60)
    print("2+1D CDT with spacetime Pachner moves")
    print("=" * 60)

    print(f"\nBuilding initial CDT (L={L}, T={T})...")
    cdt = build_initial_cdt(L, T)
    m, u, o = cdt.verify()
    print(f"  Tets: {cdt.n_tets()}, matched: {m}, unmatched: {u}, over: {o}")

    # Measure BEFORE Pachner moves
    print(f"\nMeasuring d_spec (no Pachner moves)...")
    P0 = cdt.random_walk_dspec(n_walks, sigma_max)
    sigs0, ds0 = extract_dspec(P0, sigma_max)
    print(f"  d_spec range: {ds0.min():.2f} - {ds0.max():.2f}")

    # Apply Pachner moves
    print(f"\nApplying {n_pachner} Pachner moves...")
    acc_23, acc_32, att = 0, 0, 0
    for i in range(n_pachner):
        att += 1
        if random.random() < 0.5:
            if cdt.move_23():
                acc_23 += 1
        else:
            if cdt.move_32():
                acc_32 += 1
        if (i+1) % 1000 == 0:
            m, u, o = cdt.verify()
            print(f"  {i+1}: tets={cdt.n_tets()} matched={m} "
                  f"unmatched={u} 23={acc_23} 32={acc_32} "
                  f"({time.time()-t0:.0f}s)")

    m, u, o = cdt.verify()
    print(f"\nAfter Pachner: tets={cdt.n_tets()}, "
          f"matched={m}, unmatched={u}, over={o}")
    print(f"Accepted: (2,3)={acc_23}, (3,2)={acc_32}")

    # Measure AFTER Pachner moves
    print(f"\nMeasuring d_spec (after Pachner)...")
    P1 = cdt.random_walk_dspec(n_walks, sigma_max)
    sigs1, ds1 = extract_dspec(P1, sigma_max)

    print(f"\n{'sigma':>6} {'before':>8} {'after':>8}")
    print("-" * 24)
    for s in [10, 20, 40, 80, 150, 300]:
        d0 = ds0[np.argmin(np.abs(sigs0 - s))] if len(sigs0) > 0 else 0
        d1 = ds1[np.argmin(np.abs(sigs1 - s))] if len(sigs1) > 0 else 0
        print(f"{s:>6} {d0:>8.3f} {d1:>8.3f}")

    # Fit SDHG formula to AFTER data
    if len(ds1) > 10:
        def obj(p):
            s0 = 10**p[0]
            u = (sigs1/s0)**p[1]
            return np.sqrt(np.mean((ds1 - (p[2]+(p[3]-p[2])*u/(1+u)))**2))
        r = minimize(obj, [1.5, 0.5, 2.0, 3.0],
                     bounds=[(0,3),(0.05,3),(0.5,4),(2,4)], method='L-BFGS-B')
        print(f"\nFit: gamma={r.x[1]:.3f} d_UV={r.x[2]:.2f} "
              f"d_IR={r.x[3]:.2f} RMS={r.fun:.3f}")
        print(f"SDHG prediction: gamma = 1/3 = {1/3:.4f}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
