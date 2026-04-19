/*
 * Proper 2+1D CDT: spacetime simplicial complex with dual-graph random walk.
 *
 * Structure: Each time slab is filled with tetrahedra.
 * For each spatial triangle (v0,v1,v2) at time t, the prism to t+1
 * is divided into 3 tetrahedra:
 *   (3,1): (v0_t, v1_t, v2_t, v0_{t+1})
 *   (2,2): (v1_t, v2_t, v0_{t+1}, v1_{t+1})
 *   (1,3): (v2_t, v0_{t+1}, v1_{t+1}, v2_{t+1})
 *
 * Spectral dimension: random walk on dual graph (tet→tet via shared face).
 *
 * Compile: cl /O2 cdt_proper.c   or   gcc -O2 -o cdt_proper cdt_proper.c -lm
 * Usage:  cdt_proper L T n_walks sigma_max
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * Data structures
 * ============================================================ */
#define MAX_TETS 200000
#define MAX_NB   4

typedef struct {
    int v[4];         /* 4 vertex indices */
    int nb[4];        /* 4 neighbor tet indices (-1 = none) */
    int n_nb;         /* actual neighbor count */
} Tet;

static Tet tets[MAX_TETS];
static int n_tets = 0;

/* RNG */
static unsigned int rng_state = 42;
static unsigned int xorshift32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static int rand_int(int n) { return (int)(xorshift32() % (unsigned)n); }
static double rand_double(void) {
    return (xorshift32() & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}

/* ============================================================
 * Build initial CDT spacetime
 * ============================================================ */

/* Vertex index: slice t, spatial vertex v -> t * n_sv + v */
static int vid(int t, int v, int T, int n_sv) {
    return (t % T) * n_sv + v;
}

/* Sort 4 integers */
static void sort4(int a[4]) {
    int i, j, tmp;
    for (i = 0; i < 3; i++)
        for (j = i+1; j < 4; j++)
            if (a[i] > a[j]) { tmp = a[i]; a[i] = a[j]; a[j] = tmp; }
}

/* Check if tets a and b share exactly 3 vertices (a face) */
static int share_face(int a, int b) {
    int shared = 0, i, j;
    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++)
            if (tets[a].v[i] == tets[b].v[j]) { shared++; break; }
    return shared == 3;
}

static void build_spacetime(int L, int T) {
    int n_sv = L * L;         /* spatial vertices per slice */
    int n_st = 2 * L * L;     /* spatial triangles per slice */
    int x, y, t, ti;

    n_tets = 0;

    /* For each time slab and each spatial triangle, create 3 tetrahedra */
    /* Spatial triangles: for square (x,y), lower=(v00,v10,v01), upper=(v10,v11,v01) */

    for (t = 0; t < T; t++) {
        int tn = (t + 1) % T;

        for (x = 0; x < L; x++) {
            for (y = 0; y < L; y++) {
                int v00 = x * L + y;
                int v10 = ((x+1)%L) * L + y;
                int v01 = x * L + (y+1)%L;
                int v11 = ((x+1)%L) * L + (y+1)%L;

                /* Lower triangle: (v00, v10, v01) */
                {
                    int w0 = vid(t, v00, T, n_sv);
                    int w1 = vid(t, v10, T, n_sv);
                    int w2 = vid(t, v01, T, n_sv);
                    int w0n = vid(tn, v00, T, n_sv);
                    int w1n = vid(tn, v10, T, n_sv);
                    int w2n = vid(tn, v01, T, n_sv);

                    /* (3,1): w0,w1,w2, w0n */
                    if (n_tets < MAX_TETS) {
                        tets[n_tets].v[0] = w0; tets[n_tets].v[1] = w1;
                        tets[n_tets].v[2] = w2; tets[n_tets].v[3] = w0n;
                        sort4(tets[n_tets].v);
                        tets[n_tets].n_nb = 0;
                        n_tets++;
                    }
                    /* (2,2): w1,w2, w0n,w1n */
                    if (n_tets < MAX_TETS) {
                        tets[n_tets].v[0] = w1; tets[n_tets].v[1] = w2;
                        tets[n_tets].v[2] = w0n; tets[n_tets].v[3] = w1n;
                        sort4(tets[n_tets].v);
                        tets[n_tets].n_nb = 0;
                        n_tets++;
                    }
                    /* (1,3): w2, w0n,w1n,w2n */
                    if (n_tets < MAX_TETS) {
                        tets[n_tets].v[0] = w2; tets[n_tets].v[1] = w0n;
                        tets[n_tets].v[2] = w1n; tets[n_tets].v[3] = w2n;
                        sort4(tets[n_tets].v);
                        tets[n_tets].n_nb = 0;
                        n_tets++;
                    }
                }

                /* Upper triangle: (v10, v11, v01) */
                {
                    int w0 = vid(t, v10, T, n_sv);
                    int w1 = vid(t, v11, T, n_sv);
                    int w2 = vid(t, v01, T, n_sv);
                    int w0n = vid(tn, v10, T, n_sv);
                    int w1n = vid(tn, v11, T, n_sv);
                    int w2n = vid(tn, v01, T, n_sv);

                    if (n_tets < MAX_TETS) {
                        tets[n_tets].v[0] = w0; tets[n_tets].v[1] = w1;
                        tets[n_tets].v[2] = w2; tets[n_tets].v[3] = w0n;
                        sort4(tets[n_tets].v);
                        tets[n_tets].n_nb = 0;
                        n_tets++;
                    }
                    if (n_tets < MAX_TETS) {
                        tets[n_tets].v[0] = w1; tets[n_tets].v[1] = w2;
                        tets[n_tets].v[2] = w0n; tets[n_tets].v[3] = w1n;
                        sort4(tets[n_tets].v);
                        tets[n_tets].n_nb = 0;
                        n_tets++;
                    }
                    if (n_tets < MAX_TETS) {
                        tets[n_tets].v[0] = w2; tets[n_tets].v[1] = w0n;
                        tets[n_tets].v[2] = w1n; tets[n_tets].v[3] = w2n;
                        sort4(tets[n_tets].v);
                        tets[n_tets].n_nb = 0;
                        n_tets++;
                    }
                }
            }
        }
    }

    fprintf(stderr, "Built %d tetrahedra (%d spatial tri x %d slabs x 3)\n",
            n_tets, 2*L*L, T);

    /* Build neighbor lists (shared faces) */
    /* For efficiency: hash faces and find matches */
    fprintf(stderr, "Building dual graph...");
    fflush(stderr);

    /* Face hash: encode 3 sorted vertex indices as key */
    /* Simple approach: for each tet, compute 4 faces, store in hash table */
    typedef struct { int v[3]; int tet_idx; } FaceEntry;

    int hash_size = n_tets * 8;  /* generous */
    FaceEntry *hash_table = (FaceEntry *)calloc(hash_size, sizeof(FaceEntry));
    int *hash_used = (int *)calloc(hash_size, sizeof(int));

    int total_nb = 0;

    for (ti = 0; ti < n_tets; ti++) {
        /* 4 faces of tetrahedron ti */
        int faces[4][3] = {
            {tets[ti].v[0], tets[ti].v[1], tets[ti].v[2]},
            {tets[ti].v[0], tets[ti].v[1], tets[ti].v[3]},
            {tets[ti].v[0], tets[ti].v[2], tets[ti].v[3]},
            {tets[ti].v[1], tets[ti].v[2], tets[ti].v[3]},
        };
        int fi;
        for (fi = 0; fi < 4; fi++) {
            /* Hash the face */
            unsigned int h = (unsigned)(faces[fi][0] * 73856093 ^
                                        faces[fi][1] * 19349663 ^
                                        faces[fi][2] * 83492791);
            h = h % (unsigned)hash_size;

            /* Linear probe */
            int found = 0;
            int probe;
            for (probe = 0; probe < 100; probe++) {
                int idx = (h + probe) % hash_size;
                if (!hash_used[idx]) {
                    /* Empty slot: store this face */
                    hash_used[idx] = 1;
                    hash_table[idx].v[0] = faces[fi][0];
                    hash_table[idx].v[1] = faces[fi][1];
                    hash_table[idx].v[2] = faces[fi][2];
                    hash_table[idx].tet_idx = ti;
                    break;
                }
                /* Check if same face */
                if (hash_table[idx].v[0] == faces[fi][0] &&
                    hash_table[idx].v[1] == faces[fi][1] &&
                    hash_table[idx].v[2] == faces[fi][2]) {
                    /* Found matching face! Connect the two tets */
                    int other = hash_table[idx].tet_idx;
                    if (other != ti) {
                        if (tets[ti].n_nb < MAX_NB) {
                            tets[ti].nb[tets[ti].n_nb++] = other;
                            total_nb++;
                        }
                        if (tets[other].n_nb < MAX_NB) {
                            tets[other].nb[tets[other].n_nb++] = ti;
                            total_nb++;
                        }
                    }
                    found = 1;
                    break;
                }
            }
        }
    }

    free(hash_table);
    free(hash_used);

    fprintf(stderr, " done. Total neighbor links: %d, avg: %.1f\n",
            total_nb, (double)total_nb / n_tets);
}

/* ============================================================
 * Random walk on dual graph
 * ============================================================ */
static void measure_spectral_dimension(int n_walks, int sigma_max) {
    double *P = (double *)calloc(sigma_max + 1, sizeof(double));
    int i, sigma;

    fprintf(stderr, "Random walk: %d walks, sigma_max=%d on %d tets\n",
            n_walks, sigma_max, n_tets);

    for (i = 0; i < n_walks; i++) {
        int start = rand_int(n_tets);
        if (tets[start].n_nb == 0) continue;
        int pos = start;

        for (sigma = 1; sigma <= sigma_max; sigma++) {
            if (tets[pos].n_nb == 0) break;
            pos = tets[pos].nb[rand_int(tets[pos].n_nb)];
            if (pos == start) P[sigma] += 1.0;
        }
    }

    for (i = 0; i <= sigma_max; i++) P[i] /= n_walks;

    /* Output d_spec */
    printf("sigma d_spec P_return\n");
    {
        int w = 5;
        for (i = w + 2; i <= sigma_max - w - 2; i += 2) {
            double P_lo = 0, P_hi = 0;
            int lo_n = 0, hi_n = 0, j;
            for (j = i - w; j <= i; j++) {
                if (P[j] > 0) { P_lo += P[j]; lo_n++; }
            }
            for (j = i; j <= i + w; j++) {
                if (P[j] > 0) { P_hi += P[j]; hi_n++; }
            }
            if (lo_n > 0 && hi_n > 0) {
                P_lo /= lo_n; P_hi /= hi_n;
                if (P_lo > 1e-15 && P_hi > 1e-15) {
                    double d = -2.0 * (log(P_hi) - log(P_lo)) /
                                      (log((double)(i + w/2)) - log((double)(i - w/2)));
                    if (d > 0 && d < 10)
                        printf("%d %.4f %.6e\n", i, d, P[i]);
                }
            }
        }
    }

    free(P);
}

/* ============================================================
 * Main
 * ============================================================ */
int main(int argc, char **argv) {
    int L = 10, T = 15, n_walks = 200000, sigma_max = 500;

    if (argc >= 3) { L = atoi(argv[1]); T = atoi(argv[2]); }
    if (argc >= 4) n_walks = atoi(argv[3]);
    if (argc >= 5) sigma_max = atoi(argv[4]);

    fprintf(stderr, "Proper 2+1D CDT: L=%d T=%d\n", L, T);

    build_spacetime(L, T);
    measure_spectral_dimension(n_walks, sigma_max);

    fprintf(stderr, "Done.\n");
    return 0;
}
