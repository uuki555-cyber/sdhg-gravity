/*
 * 2+1D CDT simulation in C — fast spectral dimension measurement.
 *
 * Same algorithm as run_cdt_2plus1d.py (Python reference implementation)
 * but ~1000x faster.
 *
 * Outputs spacetime graph as edge list to stdout.
 * Python wrapper (cdt_fast_c.py) reads this and computes spectral dimension.
 *
 * Compile: cl /O2 /Fe:cdt_sim.exe cdt_sim.c   (MSVC)
 *     or:  gcc -O2 -o cdt_sim cdt_sim.c -lm    (GCC/MinGW)
 *
 * Usage:  cdt_sim L T n_sweeps seed target_v
 *         cdt_sim 10 15 200 42 150
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TRI   50000
#define MAX_VERT  20000
#define MAX_DEG   40

/* ============================================================
 * Triangulation data structure
 * ============================================================ */
typedef struct {
    int tri[MAX_TRI][3];    /* triangle vertices (sorted), -1 = dead */
    int n_tri;              /* total slots used */
    int n_alive;            /* alive triangles */

    int vt[MAX_VERT][MAX_DEG];  /* vertex -> triangle indices */
    int vt_deg[MAX_VERT];       /* degree (# triangles) per vertex */

    int n_vert;             /* alive vertex count */
    int next_vid;           /* next vertex ID to assign */
    int L;                  /* original grid size */
} Triangulation;

/* ============================================================
 * Utility
 * ============================================================ */
static unsigned int rng_state;

static unsigned int xorshift32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static int rand_int(int n) {
    return (int)(xorshift32() % (unsigned int)n);
}

static double rand_double(void) {
    return (xorshift32() & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}

static void sort3(int *a, int *b, int *c) {
    int t;
    if (*a > *b) { t = *a; *a = *b; *b = t; }
    if (*b > *c) { t = *b; *b = *c; *c = t; }
    if (*a > *b) { t = *a; *a = *b; *b = t; }
}

/* ============================================================
 * Triangulation operations
 * ============================================================ */
static void rebuild(Triangulation *T) {
    int i, k, ti, v;
    /* Clear vertex maps */
    for (i = 0; i < T->next_vid && i < MAX_VERT; i++)
        T->vt_deg[i] = 0;

    T->n_alive = 0;
    for (ti = 0; ti < T->n_tri; ti++) {
        if (T->tri[ti][0] < 0) continue;
        T->n_alive++;
        for (k = 0; k < 3; k++) {
            v = T->tri[ti][k];
            if (v >= 0 && v < MAX_VERT && T->vt_deg[v] < MAX_DEG) {
                T->vt[v][T->vt_deg[v]] = ti;
                T->vt_deg[v]++;
            }
        }
    }
}

static void compact(Triangulation *T) {
    int read, write = 0;
    for (read = 0; read < T->n_tri; read++) {
        if (T->tri[read][0] >= 0) {
            if (write != read) {
                T->tri[write][0] = T->tri[read][0];
                T->tri[write][1] = T->tri[read][1];
                T->tri[write][2] = T->tri[read][2];
            }
            write++;
        }
    }
    for (read = write; read < T->n_tri; read++)
        T->tri[read][0] = T->tri[read][1] = T->tri[read][2] = -1;
    T->n_tri = write;
    rebuild(T);
}

static void init_torus(Triangulation *T, int L) {
    int x, y, v00, v10, v01, v11, a, b, c;
    memset(T, 0, sizeof(Triangulation));
    T->L = L;
    T->n_vert = L * L;
    T->next_vid = L * L;
    T->n_tri = 0;

    /* Set all triangles to dead initially */
    for (x = 0; x < MAX_TRI; x++)
        T->tri[x][0] = T->tri[x][1] = T->tri[x][2] = -1;

    for (x = 0; x < L; x++) {
        for (y = 0; y < L; y++) {
            v00 = x * L + y;
            v10 = ((x + 1) % L) * L + y;
            v01 = x * L + (y + 1) % L;
            v11 = ((x + 1) % L) * L + (y + 1) % L;

            a = v00; b = v10; c = v01;
            sort3(&a, &b, &c);
            T->tri[T->n_tri][0] = a;
            T->tri[T->n_tri][1] = b;
            T->tri[T->n_tri][2] = c;
            T->n_tri++;

            a = v10; b = v11; c = v01;
            sort3(&a, &b, &c);
            T->tri[T->n_tri][0] = a;
            T->tri[T->n_tri][1] = b;
            T->tri[T->n_tri][2] = c;
            T->n_tri++;
        }
    }
    rebuild(T);
}

/* Find the OTHER triangle sharing edge (v1,v2), excluding exclude_ti */
static int find_other_tri(Triangulation *T, int v1, int v2, int exclude_ti) {
    int i, ti, k;
    int va = (v1 < v2) ? v1 : v2;
    int vb = (v1 < v2) ? v2 : v1;

    for (i = 0; i < T->vt_deg[va]; i++) {
        ti = T->vt[va][i];
        if (ti == exclude_ti || T->tri[ti][0] < 0) continue;
        for (k = 0; k < 3; k++) {
            if (T->tri[ti][k] == vb) return ti;
        }
    }
    return -1;
}

static int flip_edge(Triangulation *T) {
    int ti_a, ti_b, ei, v1, v2, v3, v4, k, a, b, c;

    /* Pick random alive triangle */
    int attempts = 0;
    do {
        ti_a = rand_int(T->n_tri);
        if (++attempts > 30) return 0;
    } while (T->tri[ti_a][0] < 0);

    /* Pick random edge */
    ei = rand_int(3);
    v1 = T->tri[ti_a][ei];
    v2 = T->tri[ti_a][(ei + 1) % 3];

    /* Find other triangle */
    ti_b = find_other_tri(T, v1, v2, ti_a);
    if (ti_b < 0) return 0;

    /* Find opposite vertices */
    v3 = -1;
    for (k = 0; k < 3; k++)
        if (T->tri[ti_a][k] != v1 && T->tri[ti_a][k] != v2)
            { v3 = T->tri[ti_a][k]; break; }
    v4 = -1;
    for (k = 0; k < 3; k++)
        if (T->tri[ti_b][k] != v1 && T->tri[ti_b][k] != v2)
            { v4 = T->tri[ti_b][k]; break; }

    if (v3 < 0 || v4 < 0 || v3 == v4) return 0;

    /* Check new edge doesn't exist */
    if (find_other_tri(T, v3, v4, -1) >= 0) return 0;

    /* Check minimum degree */
    if (T->vt_deg[v1] <= 3 || T->vt_deg[v2] <= 3) return 0;

    /* Execute flip */
    a = v3; b = v4; c = v1; sort3(&a, &b, &c);
    T->tri[ti_a][0] = a; T->tri[ti_a][1] = b; T->tri[ti_a][2] = c;
    a = v3; b = v4; c = v2; sort3(&a, &b, &c);
    T->tri[ti_b][0] = a; T->tri[ti_b][1] = b; T->tri[ti_b][2] = c;

    rebuild(T);
    return 1;
}

static int insert_vertex(Triangulation *T) {
    int ti, v0, v1, v2, v_new, a, b, c;

    if (T->n_tri + 2 >= MAX_TRI || T->next_vid >= MAX_VERT) return 0;

    /* Pick random alive triangle */
    int attempts = 0;
    do {
        ti = rand_int(T->n_tri);
        if (++attempts > 30) return 0;
    } while (T->tri[ti][0] < 0);

    v0 = T->tri[ti][0]; v1 = T->tri[ti][1]; v2 = T->tri[ti][2];
    v_new = T->next_vid++;
    T->n_vert++;

    /* Replace original */
    a = v0; b = v1; c = v_new; sort3(&a, &b, &c);
    T->tri[ti][0] = a; T->tri[ti][1] = b; T->tri[ti][2] = c;

    /* Add two new */
    a = v1; b = v2; c = v_new; sort3(&a, &b, &c);
    T->tri[T->n_tri][0] = a; T->tri[T->n_tri][1] = b; T->tri[T->n_tri][2] = c;
    T->n_tri++;

    a = v0; b = v2; c = v_new; sort3(&a, &b, &c);
    T->tri[T->n_tri][0] = a; T->tri[T->n_tri][1] = b; T->tri[T->n_tri][2] = c;
    T->n_tri++;

    rebuild(T);
    return 1;
}

static int remove_vertex(Triangulation *T) {
    int ti, k, v, i;
    int t_idx[3], n_t = 0;
    int others[6], n_oth = 0;
    int a, b, c, j, found;

    /* Find degree-3 vertex by sampling triangles */
    int attempts = 0;
    do {
        ti = rand_int(T->n_tri);
        if (T->tri[ti][0] < 0) { attempts++; continue; }
        k = rand_int(3);
        v = T->tri[ti][k];
        if (T->vt_deg[v] == 3) goto found_v;
        attempts++;
    } while (attempts < 50);
    return 0;

found_v:
    /* Collect 3 triangles */
    n_t = 0;
    for (i = 0; i < T->vt_deg[v] && n_t < 3; i++)
        t_idx[n_t++] = T->vt[v][i];
    if (n_t != 3) return 0;

    /* Collect other vertices */
    n_oth = 0;
    for (i = 0; i < 3; i++) {
        for (k = 0; k < 3; k++) {
            int vk = T->tri[t_idx[i]][k];
            if (vk == v) continue;
            found = 0;
            for (j = 0; j < n_oth; j++)
                if (others[j] == vk) { found = 1; break; }
            if (!found && n_oth < 6)
                others[n_oth++] = vk;
        }
    }
    if (n_oth != 3) return 0;

    /* Replace first triangle with merged */
    a = others[0]; b = others[1]; c = others[2];
    sort3(&a, &b, &c);
    T->tri[t_idx[0]][0] = a;
    T->tri[t_idx[0]][1] = b;
    T->tri[t_idx[0]][2] = c;

    /* Mark others dead */
    T->tri[t_idx[1]][0] = T->tri[t_idx[1]][1] = T->tri[t_idx[1]][2] = -1;
    T->tri[t_idx[2]][0] = T->tri[t_idx[2]][1] = T->tri[t_idx[2]][2] = -1;

    T->n_vert--;
    rebuild(T);
    return 1;
}

/* ============================================================
 * MC sweep with volume control
 * ============================================================ */
static void mc_sweep(Triangulation *T, int n_moves, int target_v, double eps) {
    int i;
    double r, dS;

    for (i = 0; i < n_moves; i++) {
        r = rand_double();
        if (r < 0.4) {
            /* Insert with volume penalty */
            if (target_v > 0) {
                dS = eps * (2 * (T->n_vert - target_v) + 1);
                if (rand_double() >= exp(-dS)) continue;
            }
            insert_vertex(T);
        } else if (r < 0.7) {
            /* Remove with volume penalty */
            if (target_v > 0) {
                dS = eps * (-2 * (T->n_vert - target_v) + 1);
                if (rand_double() >= exp(-dS)) continue;
            }
            remove_vertex(T);
        } else {
            flip_edge(T);
        }
    }
}

/* ============================================================
 * Main: simulate and output spacetime graph
 * ============================================================ */
int main(int argc, char **argv) {
    int L, T_slices, n_sweeps, seed, target_v;
    int t, sweep, ti, k, i, j;
    Triangulation *slices;
    double eps = 0.01;

    if (argc < 6) {
        fprintf(stderr, "Usage: cdt_sim L T n_sweeps seed target_v\n");
        fprintf(stderr, "  Example: cdt_sim 10 15 200 42 150\n");
        return 1;
    }

    L = atoi(argv[1]);
    T_slices = atoi(argv[2]);
    n_sweeps = atoi(argv[3]);
    seed = atoi(argv[4]);
    target_v = atoi(argv[5]);

    rng_state = (unsigned int)seed;
    if (rng_state == 0) rng_state = 1;

    /* Allocate slices */
    slices = (Triangulation *)calloc(T_slices, sizeof(Triangulation));
    if (!slices) { fprintf(stderr, "Memory allocation failed\n"); return 1; }

    /* Initialize */
    for (t = 0; t < T_slices; t++)
        init_torus(&slices[t], L);

    fprintf(stderr, "CDT: L=%d T=%d sweeps=%d seed=%d target_v=%d\n",
            L, T_slices, n_sweeps, seed, target_v);

    /* Thermalize */
    for (sweep = 0; sweep < n_sweeps; sweep++) {
        for (t = 0; t < T_slices; t++) {
            int n_moves = slices[t].n_vert > 30 ? slices[t].n_vert : 30;
            mc_sweep(&slices[t], n_moves, target_v, eps);
        }
        /* Compact every 20 sweeps */
        if ((sweep + 1) % 20 == 0) {
            int avg_v = 0;
            for (t = 0; t < T_slices; t++) {
                compact(&slices[t]);
                avg_v += slices[t].n_vert;
            }
            if ((sweep + 1) % 50 == 0)
                fprintf(stderr, "  sweep %d: avg_v=%d\n",
                        sweep + 1, avg_v / T_slices);
        }
    }

    /* Output spacetime graph as edge list */
    /* First: assign global vertex IDs */
    int *gid_offset = (int *)calloc(T_slices + 1, sizeof(int));
    int N = 0;
    /* Collect alive vertices per slice */
    int **alive_v = (int **)calloc(T_slices, sizeof(int *));
    int *n_alive_v = (int *)calloc(T_slices, sizeof(int));

    for (t = 0; t < T_slices; t++) {
        alive_v[t] = (int *)calloc(MAX_VERT, sizeof(int));
        char *seen = (char *)calloc(MAX_VERT, sizeof(char));
        n_alive_v[t] = 0;

        for (ti = 0; ti < slices[t].n_tri; ti++) {
            if (slices[t].tri[ti][0] < 0) continue;
            for (k = 0; k < 3; k++) {
                int v = slices[t].tri[ti][k];
                if (v >= 0 && v < MAX_VERT && !seen[v]) {
                    seen[v] = 1;
                    alive_v[t][n_alive_v[t]++] = v;
                }
            }
        }
        /* Sort for consistent temporal links */
        for (i = 0; i < n_alive_v[t] - 1; i++)
            for (j = i + 1; j < n_alive_v[t]; j++)
                if (alive_v[t][i] > alive_v[t][j]) {
                    int tmp = alive_v[t][i];
                    alive_v[t][i] = alive_v[t][j];
                    alive_v[t][j] = tmp;
                }

        gid_offset[t] = N;
        N += n_alive_v[t];
        free(seen);
    }
    gid_offset[T_slices] = N;

    fprintf(stderr, "Graph: N=%d vertices\n", N);

    /* Output: N, then edges */
    printf("%d\n", N);

    for (t = 0; t < T_slices; t++) {
        int tn = (t + 1) % T_slices;

        /* Spatial edges from triangles */
        for (ti = 0; ti < slices[t].n_tri; ti++) {
            if (slices[t].tri[ti][0] < 0) continue;
            for (k = 0; k < 3; k++) {
                int va = slices[t].tri[ti][k];
                int vb = slices[t].tri[ti][(k + 1) % 3];
                int ga = -1, gb = -1;
                /* Binary search for va in alive_v[t] */
                { int lo=0, hi=n_alive_v[t]-1;
                  while(lo<=hi) { int mid=(lo+hi)/2;
                    if(alive_v[t][mid]==va){ga=gid_offset[t]+mid;break;}
                    else if(alive_v[t][mid]<va)lo=mid+1; else hi=mid-1; }}
                /* Binary search for vb */
                { int lo=0, hi=n_alive_v[t]-1;
                  while(lo<=hi) { int mid=(lo+hi)/2;
                    if(alive_v[t][mid]==vb){gb=gid_offset[t]+mid;break;}
                    else if(alive_v[t][mid]<vb)lo=mid+1; else hi=mid-1; }}
                if (ga >= 0 && gb >= 0 && ga < gb)
                    printf("%d %d\n", ga, gb);
            }
        }

        /* Temporal links: 1-to-1 by sorted order */
        {   int n_link = n_alive_v[t] < n_alive_v[tn] ?
                         n_alive_v[t] : n_alive_v[tn];
            for (i = 0; i < n_link; i++)
                printf("%d %d\n", gid_offset[t] + i, gid_offset[tn] + i);
        }

        /* CDT (2,2) diagonal: for each spatial edge (a,b),
           cross-connect a_t<->b_{t+1} and b_t<->a_{t+1} */
        for (ti = 0; ti < slices[t].n_tri; ti++) {
            if (slices[t].tri[ti][0] < 0) continue;
            for (k = 0; k < 3; k++) {
                int va = slices[t].tri[ti][k];
                int vb = slices[t].tri[ti][(k+1)%3];
                if (va > vb) continue;
                int ga_t=-1, gb_t=-1, ga_tn=-1, gb_tn=-1;
                {int lo=0,hi=n_alive_v[t]-1; while(lo<=hi){int mid=(lo+hi)/2;
                 if(alive_v[t][mid]==va){ga_t=gid_offset[t]+mid;break;}
                 else if(alive_v[t][mid]<va)lo=mid+1;else hi=mid-1;}}
                {int lo=0,hi=n_alive_v[t]-1; while(lo<=hi){int mid=(lo+hi)/2;
                 if(alive_v[t][mid]==vb){gb_t=gid_offset[t]+mid;break;}
                 else if(alive_v[t][mid]<vb)lo=mid+1;else hi=mid-1;}}
                {int lo=0,hi=n_alive_v[tn]-1; while(lo<=hi){int mid=(lo+hi)/2;
                 if(alive_v[tn][mid]==va){ga_tn=gid_offset[tn]+mid;break;}
                 else if(alive_v[tn][mid]<va)lo=mid+1;else hi=mid-1;}}
                {int lo=0,hi=n_alive_v[tn]-1; while(lo<=hi){int mid=(lo+hi)/2;
                 if(alive_v[tn][mid]==vb){gb_tn=gid_offset[tn]+mid;break;}
                 else if(alive_v[tn][mid]<vb)lo=mid+1;else hi=mid-1;}}
                if(ga_t>=0 && gb_tn>=0) printf("%d %d\n", ga_t, gb_tn);
                if(gb_t>=0 && ga_tn>=0) printf("%d %d\n", gb_t, ga_tn);
            }
        }
    }

    /* Cleanup */
    for (t = 0; t < T_slices; t++) free(alive_v[t]);
    free(alive_v);
    free(n_alive_v);
    free(gid_offset);
    free(slices);

    fprintf(stderr, "Done.\n");
    return 0;
}
