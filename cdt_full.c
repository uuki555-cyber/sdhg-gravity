/*
 * Full 2+1D CDT: spatial MC + proper tetrahedra + dual graph random walk.
 *
 * 1. Thermalize T spatial slices via edge flips (4,4 move)
 * 2. Build spacetime tetrahedra from thermalized slices
 * 3. Build dual graph (shared faces)
 * 4. Random walk on dual graph -> spectral dimension
 *
 * Compile: cl /O2 /Fe:cdt_full.exe cdt_full.c
 * Usage:  cdt_full L T n_therm n_walks sigma_max seed
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * Spatial triangulation (2D torus)
 * ============================================================ */
#define MAX_STRI 2000   /* max spatial triangles per slice */
#define MAX_SEDGE 3000  /* max spatial edges per slice */
#define MAX_SVERT 500   /* max spatial vertices per slice */
#define MAX_SDEG 20     /* max triangles per vertex */

typedef struct {
    int tri[MAX_STRI][3];   /* spatial triangles (sorted) */
    int n_tri;
    /* vertex -> triangle map */
    int vt[MAX_SVERT][MAX_SDEG];
    int vt_deg[MAX_SVERT];
    int n_vert;
    int L;
} SpatialSlice;

static void sort3(int *a, int *b, int *c) {
    int t;
    if (*a > *b) { t=*a; *a=*b; *b=t; }
    if (*b > *c) { t=*b; *b=*c; *c=t; }
    if (*a > *b) { t=*a; *a=*b; *b=t; }
}

static void rebuild_slice(SpatialSlice *s) {
    int i, k, v;
    for (v = 0; v < MAX_SVERT; v++) s->vt_deg[v] = 0;
    for (i = 0; i < s->n_tri; i++) {
        for (k = 0; k < 3; k++) {
            v = s->tri[i][k];
            if (v >= 0 && v < MAX_SVERT && s->vt_deg[v] < MAX_SDEG)
                s->vt[v][s->vt_deg[v]++] = i;
        }
    }
}

static void init_slice(SpatialSlice *s, int L) {
    int x, y, v00, v10, v01, v11, a, b, c;
    memset(s, 0, sizeof(SpatialSlice));
    s->L = L;
    s->n_vert = L * L;
    s->n_tri = 0;
    for (x = 0; x < L; x++) {
        for (y = 0; y < L; y++) {
            v00 = x*L+y; v10 = ((x+1)%L)*L+y;
            v01 = x*L+(y+1)%L; v11 = ((x+1)%L)*L+(y+1)%L;
            a=v00; b=v10; c=v01; sort3(&a,&b,&c);
            s->tri[s->n_tri][0]=a; s->tri[s->n_tri][1]=b; s->tri[s->n_tri][2]=c;
            s->n_tri++;
            a=v10; b=v11; c=v01; sort3(&a,&b,&c);
            s->tri[s->n_tri][0]=a; s->tri[s->n_tri][1]=b; s->tri[s->n_tri][2]=c;
            s->n_tri++;
        }
    }
    rebuild_slice(s);
}

/* Find other triangle sharing edge (va,vb), excluding ti_excl */
static int find_adj_tri(SpatialSlice *s, int va, int vb, int ti_excl) {
    int i, ti, k, has_b;
    int vmin = va < vb ? va : vb;
    for (i = 0; i < s->vt_deg[vmin]; i++) {
        ti = s->vt[vmin][i];
        if (ti == ti_excl) continue;
        has_b = 0;
        for (k = 0; k < 3; k++)
            if (s->tri[ti][k] == vb) { has_b = 1; break; }
        if (has_b) return ti;
    }
    return -1;
}

/* Edge flip on spatial slice */
static int flip_spatial_edge(SpatialSlice *s) {
    int ti_a, ti_b, ei, v1, v2, v3, v4, k, a, b, c;

    /* Pick random triangle and edge */
    ti_a = rand() % s->n_tri;
    ei = rand() % 3;
    v1 = s->tri[ti_a][ei];
    v2 = s->tri[ti_a][(ei+1)%3];

    ti_b = find_adj_tri(s, v1, v2, ti_a);
    if (ti_b < 0) return 0;

    /* Find opposite vertices */
    v3 = -1;
    for (k = 0; k < 3; k++)
        if (s->tri[ti_a][k] != v1 && s->tri[ti_a][k] != v2)
            { v3 = s->tri[ti_a][k]; break; }
    v4 = -1;
    for (k = 0; k < 3; k++)
        if (s->tri[ti_b][k] != v1 && s->tri[ti_b][k] != v2)
            { v4 = s->tri[ti_b][k]; break; }
    if (v3 < 0 || v4 < 0 || v3 == v4) return 0;

    /* Check new edge doesn't exist */
    if (find_adj_tri(s, v3, v4, -1) >= 0) return 0;

    /* Check min degree */
    if (s->vt_deg[v1] <= 3 || s->vt_deg[v2] <= 3) return 0;

    /* Execute */
    a=v3; b=v4; c=v1; sort3(&a,&b,&c);
    s->tri[ti_a][0]=a; s->tri[ti_a][1]=b; s->tri[ti_a][2]=c;
    a=v3; b=v4; c=v2; sort3(&a,&b,&c);
    s->tri[ti_b][0]=a; s->tri[ti_b][1]=b; s->tri[ti_b][2]=c;

    rebuild_slice(s);
    return 1;
}

/* ============================================================
 * Spacetime tetrahedra + dual graph
 * ============================================================ */
#define MAX_TETS 200000

typedef struct {
    int v[4];
    int nb[4];
    int n_nb;
} Tet;

static Tet tets[MAX_TETS];
static int n_tets = 0;

static void sort4(int a[4]) {
    int i, j, tmp;
    for (i = 0; i < 3; i++)
        for (j = i+1; j < 4; j++)
            if (a[i] > a[j]) { tmp=a[i]; a[i]=a[j]; a[j]=tmp; }
}

static void build_tetrahedra(SpatialSlice *slices, int T) {
    int t, L, n_sv, ti;
    L = slices[0].L;
    n_sv = L * L;
    n_tets = 0;

    for (t = 0; t < T; t++) {
        int tn = (t + 1) % T;
        for (ti = 0; ti < slices[0].n_tri; ti++) {
            int a = slices[0].tri[ti][0];
            int b = slices[0].tri[ti][1];
            int c = slices[0].tri[ti][2];
            int at = t*n_sv+a, bt = t*n_sv+b, ct = t*n_sv+c;
            int atn = tn*n_sv+a, btn = tn*n_sv+b, ctn = tn*n_sv+c;

            /* 3-tet prism: T1=(a,b,c,a'), T2=(b,c,a',b'), T3=(c,a',b',c') */
            if (n_tets+2 < MAX_TETS) {
                tets[n_tets].v[0]=at; tets[n_tets].v[1]=bt;
                tets[n_tets].v[2]=ct; tets[n_tets].v[3]=atn;
                sort4(tets[n_tets].v); tets[n_tets].n_nb=0; n_tets++;

                tets[n_tets].v[0]=bt; tets[n_tets].v[1]=ct;
                tets[n_tets].v[2]=atn; tets[n_tets].v[3]=btn;
                sort4(tets[n_tets].v); tets[n_tets].n_nb=0; n_tets++;

                tets[n_tets].v[0]=ct; tets[n_tets].v[1]=atn;
                tets[n_tets].v[2]=btn; tets[n_tets].v[3]=ctn;
                sort4(tets[n_tets].v); tets[n_tets].n_nb=0; n_tets++;
            }
        }
    }
}

/* Face comparison for qsort */
typedef struct { int v[3]; int tet_idx; } FaceEntry;

static int face_cmp(const void *a, const void *b) {
    const FaceEntry *fa = (const FaceEntry *)a, *fb = (const FaceEntry *)b;
    if (fa->v[0] != fb->v[0]) return fa->v[0] - fb->v[0];
    if (fa->v[1] != fb->v[1]) return fa->v[1] - fb->v[1];
    return fa->v[2] - fb->v[2];
}

static void build_dual_graph(void) {
    /* Sort-based face matching (no hash collisions) */
    int n_faces = n_tets * 4;
    FaceEntry *all_faces = (FaceEntry *)malloc(n_faces * sizeof(FaceEntry));
    int ti, fi, i;

    for (ti = 0; ti < n_tets; ti++) tets[ti].n_nb = 0;

    /* Extract all faces */
    for (ti = 0; ti < n_tets; ti++) {
        int fv[4][3] = {
            {tets[ti].v[0], tets[ti].v[1], tets[ti].v[2]},
            {tets[ti].v[0], tets[ti].v[1], tets[ti].v[3]},
            {tets[ti].v[0], tets[ti].v[2], tets[ti].v[3]},
            {tets[ti].v[1], tets[ti].v[2], tets[ti].v[3]},
        };
        for (fi = 0; fi < 4; fi++) {
            int idx = ti * 4 + fi;
            all_faces[idx].v[0] = fv[fi][0];
            all_faces[idx].v[1] = fv[fi][1];
            all_faces[idx].v[2] = fv[fi][2];
            all_faces[idx].tet_idx = ti;
        }
    }

    /* Sort all faces */
    qsort(all_faces, n_faces, sizeof(FaceEntry), face_cmp);

    /* Match consecutive identical faces */
    for (i = 0; i < n_faces - 1; i++) {
        if (all_faces[i].v[0] == all_faces[i+1].v[0] &&
            all_faces[i].v[1] == all_faces[i+1].v[1] &&
            all_faces[i].v[2] == all_faces[i+1].v[2]) {
            int a = all_faces[i].tet_idx;
            int b = all_faces[i+1].tet_idx;
            if (a != b) {
                if (tets[a].n_nb < 4) tets[a].nb[tets[a].n_nb++] = b;
                if (tets[b].n_nb < 4) tets[b].nb[tets[b].n_nb++] = a;
            }
            i++; /* skip the matched pair */
        }
    }

    free(all_faces);
}

/* ============================================================
 * Random walk on dual graph
 * ============================================================ */
static void measure_spectral_dim(int n_walks, int sigma_max) {
    double *P = (double *)calloc(sigma_max+1, sizeof(double));
    int i, sigma;

    for (i = 0; i < n_walks; i++) {
        int start = rand() % n_tets;
        if (tets[start].n_nb == 0) continue;
        int pos = start;
        for (sigma = 1; sigma <= sigma_max; sigma++) {
            if (tets[pos].n_nb == 0) break;
            pos = tets[pos].nb[rand() % tets[pos].n_nb];
            if (pos == start) P[sigma] += 1.0;
        }
    }
    for (i = 0; i <= sigma_max; i++) P[i] /= n_walks;

    /* Output d_spec */
    printf("sigma d_spec P_return\n");
    int w = 5;
    for (i = w+2; i <= sigma_max-w-2; i += 2) {
        double Plo=0, Phi=0; int nlo=0, nhi=0, j;
        for (j=i-w; j<=i; j++) if(P[j]>0){Plo+=P[j];nlo++;}
        for (j=i; j<=i+w; j++) if(P[j]>0){Phi+=P[j];nhi++;}
        if (nlo>0 && nhi>0) {
            Plo/=nlo; Phi/=nhi;
            if (Plo>1e-15 && Phi>1e-15) {
                double d = -2.0*(log(Phi)-log(Plo))/(log((double)(i+w/2))-log((double)(i-w/2)));
                if (d>0 && d<10) printf("%d %.4f %.6e\n", i, d, P[i]);
            }
        }
    }
    free(P);
}

/* ============================================================
 * Main
 * ============================================================ */
int main(int argc, char **argv) {
    int L=10, T=15, n_therm=5000, n_walks=200000, sigma_max=500, seed=42;
    int t, sweep, acc;

    if (argc >= 2) L = atoi(argv[1]);
    if (argc >= 3) T = atoi(argv[2]);
    if (argc >= 4) n_therm = atoi(argv[3]);
    if (argc >= 5) n_walks = atoi(argv[4]);
    if (argc >= 6) sigma_max = atoi(argv[5]);
    if (argc >= 7) seed = atoi(argv[6]);

    srand(seed);

    fprintf(stderr, "CDT: L=%d T=%d therm=%d walks=%d seed=%d\n",
            L, T, n_therm, n_walks, seed);

    /* Initialize T spatial slices */
    SpatialSlice *slices = (SpatialSlice *)calloc(T, sizeof(SpatialSlice));
    for (t = 0; t < T; t++)
        init_slice(&slices[t], L);

    fprintf(stderr, "Initial: %d tri/slice, %d vert/slice\n",
            slices[0].n_tri, slices[0].n_vert);

    /* Thermalize: flip edges on slice 0, copy to all slices.
       This ensures all slices share the same triangulation,
       which is required for consistent prism face matching.
       (CDT (4,4) move = flip applied uniformly to all slices.) */
    fprintf(stderr, "Thermalizing (%d flips on shared triangulation)...", n_therm);
    fflush(stderr);
    acc = 0;
    for (sweep = 0; sweep < n_therm; sweep++)
        acc += flip_spatial_edge(&slices[0]);
    /* Copy thermalized triangulation to all other slices */
    for (t = 1; t < T; t++)
        memcpy(&slices[t], &slices[0], sizeof(SpatialSlice));
    fprintf(stderr, " done. Accepted: %d/%d (%.0f%%)\n",
            acc, n_therm, 100.0*acc/n_therm);

    /* Build spacetime tetrahedra */
    fprintf(stderr, "Building tetrahedra...");
    fflush(stderr);
    build_tetrahedra(slices, T);
    fprintf(stderr, " %d tets\n", n_tets);

    /* Build dual graph */
    fprintf(stderr, "Building dual graph...");
    fflush(stderr);
    build_dual_graph();
    {
        int total_nb = 0, i;
        for (i = 0; i < n_tets; i++) total_nb += tets[i].n_nb;
        fprintf(stderr, " avg_nb=%.2f\n", (double)total_nb/n_tets);
    }

    /* Measure spectral dimension */
    fprintf(stderr, "Random walk...\n");
    measure_spectral_dim(n_walks, sigma_max);

    free(slices);
    fprintf(stderr, "Done.\n");
    return 0;
}
