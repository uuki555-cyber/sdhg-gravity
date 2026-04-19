/*
 * 3+1D (4D) CDT: spectral dimension measurement.
 *
 * Spatial topology: T^3 (L^3 cubic lattice, subdivided into tetrahedra)
 * Time: T slices with periodic boundary
 * 4-simplices connect adjacent time slices (prism decomposition)
 * Spectral dimension: random walk on dual graph (4-simplex -> 4-simplex)
 *
 * This is a SIMPLIFIED CDT: no Pachner moves (initial config only + spatial flips).
 * The spectral dimension of the REGULAR lattice should give d_IR   4.
 * With spatial randomization, we test if d_UV < 4 appears.
 *
 * Compile: cl /O2 /std:c11 /Fe:cdt4d.exe cdt4d.c
 * Usage:   cdt4d L T n_sflip n_walks sigma_max seed
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * 3D spatial triangulation (T^3 torus)
 * Each cube -> 6 tetrahedra (Freudenthal decomposition)
 * ============================================================ */
#define MAX_STET 100000  /* max spatial tetrahedra */
#define MAX_SVERT 20000

typedef struct {
    int tet[MAX_STET][4];  /* spatial tetrahedra (sorted vertices) */
    int n_tet;
    int n_vert;
    int L;
} Spatial3D;

static void sort4(int a[4]) {
    for(int i=0;i<3;i++) for(int j=i+1;j<4;j++)
        if(a[i]>a[j]){int t=a[i];a[i]=a[j];a[j]=t;}
}

static void sort5(int a[5]) {
    for(int i=0;i<4;i++) for(int j=i+1;j<5;j++)
        if(a[i]>a[j]){int t=a[i];a[i]=a[j];a[j]=t;}
}

static int vid3(int x, int y, int z, int L) {
    return ((x%L+L)%L)*L*L + ((y%L+L)%L)*L + ((z%L+L)%L);
}

static void init_spatial3d(Spatial3D *s, int L) {
    memset(s, 0, sizeof(Spatial3D));
    s->L = L;
    s->n_vert = L*L*L;
    s->n_tet = 0;

    /* Freudenthal decomposition: each cube -> 6 tetrahedra */
    for(int x=0; x<L; x++)
    for(int y=0; y<L; y++)
    for(int z=0; z<L; z++) {
        int v[8];
        v[0]=vid3(x,y,z,L);     v[1]=vid3(x+1,y,z,L);
        v[2]=vid3(x,y+1,z,L);   v[3]=vid3(x+1,y+1,z,L);
        v[4]=vid3(x,y,z+1,L);   v[5]=vid3(x+1,y,z+1,L);
        v[6]=vid3(x,y+1,z+1,L); v[7]=vid3(x+1,y+1,z+1,L);

        /* 6 tetrahedra from cube (Freudenthal/Kuhn triangulation) */
        int tets[6][4] = {
            {v[0],v[1],v[3],v[7]},
            {v[0],v[1],v[5],v[7]},
            {v[0],v[2],v[3],v[7]},
            {v[0],v[2],v[6],v[7]},
            {v[0],v[4],v[5],v[7]},
            {v[0],v[4],v[6],v[7]}
        };
        for(int i=0; i<6; i++) {
            if(s->n_tet >= MAX_STET) break;
            s->tet[s->n_tet][0]=tets[i][0];
            s->tet[s->n_tet][1]=tets[i][1];
            s->tet[s->n_tet][2]=tets[i][2];
            s->tet[s->n_tet][3]=tets[i][3];
            sort4(s->tet[s->n_tet]);
            s->n_tet++;
        }
    }
}

/* ============================================================
 * 4-simplices from prism decomposition
 *
 * Each spatial tet (a,b,c,d) at time t produces 4 four-simplices
 * connecting to time t+1:
 *   (4,1): (a_t, b_t, c_t, d_t, a_{t+1})
 *   (3,2): (b_t, c_t, d_t, a_{t+1}, b_{t+1})
 *   (2,3): (c_t, d_t, a_{t+1}, b_{t+1}, c_{t+1})
 *   (1,4): (d_t, a_{t+1}, b_{t+1}, c_{t+1}, d_{t+1})
 *
 * where a < b < c < d (sorted).
 * ============================================================ */
#define MAX_4SIM 2000000

static int sim5[MAX_4SIM][5];  /* 4-simplex vertices (sorted) */
static int sim5_nb[MAX_4SIM][5]; /* neighbors (-1=none) */
static int n_4sim = 0;

static int fidx(int si, int v) {
    for(int i=0;i<5;i++) if(sim5[si][i]==v) return i;
    return -1;
}
static void set_nb5(int si, int v_opp, int nb) {
    int fi = fidx(si, v_opp);
    if(fi >= 0) sim5_nb[si][fi] = nb;
}

typedef struct { int v[4]; int idx; } Face4;

static int face4_cmp(const void *a, const void *b) {
    const Face4 *fa = (const Face4*)a, *fb = (const Face4*)b;
    for(int i=0; i<4; i++)
        if(fa->v[i] != fb->v[i]) return fa->v[i] - fb->v[i];
    return 0;
}

static void build_4d_spacetime(Spatial3D *base, int T) {
    int nsv = base->n_vert;
    n_4sim = 0;

    for(int t=0; t<T; t++) {
        int tn = (t+1) % T;
        for(int ti=0; ti<base->n_tet; ti++) {
            int a=base->tet[ti][0], b=base->tet[ti][1];
            int c=base->tet[ti][2], d=base->tet[ti][3];
            int at=t*nsv+a, bt=t*nsv+b, ct=t*nsv+c, dt=t*nsv+d;
            int atn=tn*nsv+a, btn=tn*nsv+b, ctn=tn*nsv+c, dtn=tn*nsv+d;

            if(n_4sim+3 >= MAX_4SIM) break;

            /* (4,1) */
            sim5[n_4sim][0]=at; sim5[n_4sim][1]=bt; sim5[n_4sim][2]=ct;
            sim5[n_4sim][3]=dt; sim5[n_4sim][4]=atn;
            sort5(sim5[n_4sim]); n_4sim++;

            /* (3,2) */
            sim5[n_4sim][0]=bt; sim5[n_4sim][1]=ct; sim5[n_4sim][2]=dt;
            sim5[n_4sim][3]=atn; sim5[n_4sim][4]=btn;
            sort5(sim5[n_4sim]); n_4sim++;

            /* (2,3) */
            sim5[n_4sim][0]=ct; sim5[n_4sim][1]=dt; sim5[n_4sim][2]=atn;
            sim5[n_4sim][3]=btn; sim5[n_4sim][4]=ctn;
            sort5(sim5[n_4sim]); n_4sim++;

            /* (1,4) */
            sim5[n_4sim][0]=dt; sim5[n_4sim][1]=atn; sim5[n_4sim][2]=btn;
            sim5[n_4sim][3]=ctn; sim5[n_4sim][4]=dtn;
            sort5(sim5[n_4sim]); n_4sim++;
        }
    }

    fprintf(stderr, "  %d 4-simplices\n", n_4sim);

    /* Build dual graph: sort-based face matching */
    /* Each 4-simplex has 5 faces (each face = 4 vertices) */
    int n_faces = n_4sim * 5;
    Face4 *faces = (Face4*)malloc(n_faces * sizeof(Face4));
    for(int i=0; i<n_4sim; i++) {
        for(int j=0; j<5; j++)
            sim5_nb[i][j] = -1;
        for(int fi=0; fi<5; fi++) {
            int idx = i*5 + fi;
            int k=0;
            for(int j=0; j<5; j++)
                if(j!=fi) faces[idx].v[k++] = sim5[i][j];
            faces[idx].idx = i;
        }
    }

    qsort(faces, n_faces, sizeof(Face4), face4_cmp);

    int matched = 0;
    for(int i=0; i<n_faces-1; i++) {
        if(faces[i].v[0]==faces[i+1].v[0] &&
           faces[i].v[1]==faces[i+1].v[1] &&
           faces[i].v[2]==faces[i+1].v[2] &&
           faces[i].v[3]==faces[i+1].v[3]) {
            int a=faces[i].idx, b=faces[i+1].idx;
            if(a!=b) {
                /* Find which face */
                int sv[4]={faces[i].v[0],faces[i].v[1],faces[i].v[2],faces[i].v[3]};
                for(int fi=0;fi<5;fi++){
                    int v=sim5[a][fi], found=0;
                    for(int j=0;j<4;j++) if(sv[j]==v){found=1;break;}
                    if(!found){sim5_nb[a][fi]=b;break;}
                }
                for(int fi=0;fi<5;fi++){
                    int v=sim5[b][fi], found=0;
                    for(int j=0;j<4;j++) if(sv[j]==v){found=1;break;}
                    if(!found){sim5_nb[b][fi]=a;break;}
                }
                matched++;
            }
            i++;
        }
    }
    free(faces);
    fprintf(stderr, "  %d matched faces, avg_nb=%.2f\n",
            matched, matched*2.0/n_4sim);
}

/* ============================================================
 * Random walk on 4-simplex dual graph
 * ============================================================ */
static void measure_dspec_4d(int n_walks, int sigma_max) {
    double *P = (double*)calloc(sigma_max+1, sizeof(double));

    for(int i=0; i<n_walks; i++) {
        int start = rand() % n_4sim;
        if(sim5[start][0] < 0) continue; /* skip dead */
        int nc=0;
        for(int k=0;k<5;k++) if(sim5_nb[start][k]>=0) nc++;
        if(nc==0) continue;
        int pos = start;
        for(int sigma=1; sigma<=sigma_max; sigma++) {
            nc=0;
            for(int k=0;k<5;k++) if(sim5_nb[pos][k]>=0) nc++;
            if(nc==0) break;
            int pick=rand()%nc, cnt=0;
            for(int k=0;k<5;k++) if(sim5_nb[pos][k]>=0){
                if(cnt==pick){pos=sim5_nb[pos][k];break;}
                cnt++;
            }
            if(pos==start) P[sigma]+=1.0;
        }
    }
    for(int i=0;i<=sigma_max;i++) P[i]/=n_walks;

    printf("sigma d_spec P_return\n");
    int w=5;
    for(int i=w+2; i<=sigma_max-w-2; i+=3) {
        double Plo=0,Phi=0; int nlo=0,nhi=0;
        for(int j=i-w;j<=i;j++) if(P[j]>0){Plo+=P[j];nlo++;}
        for(int j=i;j<=i+w;j++) if(P[j]>0){Phi+=P[j];nhi++;}
        if(nlo>0&&nhi>0){Plo/=nlo;Phi/=nhi;
            if(Plo>1e-15&&Phi>1e-15){
                double d=-2.0*(log(Phi)-log(Plo))/(log((double)(i+w/2))-log((double)(i-w/2)));
                if(d>0&&d<10) printf("%d %.4f %.6e\n",i,d,P[i]);
            }
        }
    }
    free(P);
}

/* ============================================================ */
int main(int argc, char **argv) {
    int L=6, T=10, n_walks=500000, sigma_max=600, seed=42;
    if(argc>=2) L=atoi(argv[1]);
    if(argc>=3) T=atoi(argv[2]);
    if(argc>=4) n_walks=atoi(argv[3]);
    if(argc>=5) sigma_max=atoi(argv[4]);
    if(argc>=6) seed=atoi(argv[5]);
    srand(seed);

    fprintf(stderr, "4D CDT: L=%d T=%d (spatial=%d^3=%d verts, %d spatial tets)\n",
            L, T, L, L*L*L, 6*L*L*L);

    Spatial3D *base = (Spatial3D*)calloc(1, sizeof(Spatial3D));
    init_spatial3d(base, L);
    fprintf(stderr, "  Spatial tets: %d, verts: %d\n", base->n_tet, base->n_vert);

    fprintf(stderr, "Building 4D spacetime...\n");
    build_4d_spacetime(base, T);

    /* ============================================================
     * (2,4) Pachner move: 2 four-simplices sharing a tetrahedron
     * -> 4 four-simplices sharing an edge.
     * Analogous to (2,3) in 3D.
     * ============================================================ */
    int n_pachner = (argc>=7) ? atoi(argv[6]) : 0;
    double k0_val = (argc>=8) ? atof(argv[7]) : 0;
    double k4_val = (argc>=9) ? atof(argv[8]) : 0;

    if(n_pachner > 0) {
        fprintf(stderr, "Pachner moves (%d, k0=%.1f, k4=%.1f)...\n",
                n_pachner, k0_val, k4_val);
        int acc24=0, acc42=0, acc33=0;
        int dbg33_edge=0, dbg33_tri=0, dbg33_nd=0;
        int target = n_4sim;

        for(int iter=0; iter<n_pachner; iter++) {
            int move_type = rand()%5; /* 0=(4,2) 1=(2,4) 2=(3,3) 3=(2,8) 4=(8,2) */

            if(move_type == 2) {
                /* ============ (3,3) move ============ */
                /* (3,3) block entered */
                /* (3,3): triangle shared by exactly 3 simplices  
                   swap to dual configuration.
                   Strategy: pick simplex, pick edge, find all sims sharing edge.
                   Then pick 3rd vertex to form triangle shared by exactly 3. */

                int si1;
                {int att2=0; do{si1=rand()%n_4sim;att2++;}while(sim5[si1][0]<0 && att2<50);}
                if(sim5[si1][0]<0) continue;
                /* Pick edge from si1 */
                int edge_pairs[10][2]={{0,1},{0,2},{0,3},{0,4},{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}};
                int epi = rand()%10;
                int ev1=sim5[si1][edge_pairs[epi][0]], ev2=sim5[si1][edge_pairs[epi][1]];

                /* Find all sims containing this edge */
                int edge_sims[20]; int nes=0;
                for(int s=0;s<n_4sim && nes<20;s++){
                    if(sim5[s][0]<0) continue;
                    int h1=0,h2=0;
                    for(int k=0;k<5;k++){if(sim5[s][k]==ev1)h1=1;if(sim5[s][k]==ev2)h2=1;}
                    if(h1&&h2) edge_sims[nes++]=s;
                }
                if(dbg33_edge < 3) fprintf(stderr, "  33dbg: nes=%d ev1=%d ev2=%d\n", nes, ev1, ev2);
                dbg33_edge++;
                if(nes<3) continue;

                /* Pick a 3rd vertex from si1 (not ev1,ev2) to form triangle */
                int other_v[3]; int nov=0;
                for(int k=0;k<5;k++){
                    int v=sim5[si1][k];
                    if(v!=ev1 && v!=ev2 && nov<3) other_v[nov++]=v;
                }
                if(nov<1) continue;
                int tv3 = other_v[rand()%nov];
                int ta=ev1, tb=ev2, tc=tv3;  /* triangle (ta,tb,tc) */

                /* Find other 2 simplices sharing triangle (ta,tb,tc) */
                int shared_sims[3]; int ns2=0;
                for(int s=0; s<n_4sim && ns2<3; s++){
                    if(sim5[s][0]<0) continue;
                    int ha=0,hb=0,hc=0;
                    for(int k=0;k<5;k++){
                        if(sim5[s][k]==ta)ha=1;if(sim5[s][k]==tb)hb=1;if(sim5[s][k]==tc)hc=1;
                    }
                    if(ha&&hb&&hc) shared_sims[ns2++]=s;
                }
                if(ns2!=3) continue;
                dbg33_tri++;

                /* Get opposite vertices d,e,f (one per simplex, not in triangle) */
                int def[3]; int nd=0;
                for(int i=0;i<3;i++){
                    for(int k=0;k<5;k++){
                        int v=sim5[shared_sims[i]][k];
                        if(v!=ta&&v!=tb&&v!=tc){
                            int dup=0;
                            for(int j=0;j<nd;j++) if(def[j]==v){dup=1;break;}
                            if(!dup && nd<3) def[nd++]=v;
                            /* Each simplex has 2 non-triangle vertices */
                        }
                    }
                }
                if(nd!=3) continue;
                dbg33_nd++;
                int dd=def[0], ee=def[1], ff=def[2];

                /* Verify: the 3 input simplices are:
                   (ta,tb,tc,X,Y) where {X,Y}   {dd,ee,ff}
                   Should be: {dd,ee}, {dd,ff}, {ee,ff} */
                int ok=1;
                for(int i=0;i<3;i++){
                    int cnt=0;
                    for(int k=0;k<5;k++){
                        int v=sim5[shared_sims[i]][k];
                        if(v==dd||v==ee||v==ff) cnt++;
                    }
                    if(cnt!=2){ok=0;break;}
                }
                if(!ok) continue;

                /* Save external neighbors */
                /* Each input simplex has 5 faces. 3 are internal (shared with other input sims).
                   2 are external. */
                /* Input sim i: 2 external faces are those NOT containing the triangle vertices
                   that are shared with other input sims. Actually:
                   For input (ta,tb,tc,X,Y), faces containing all of ta,tb,tc:
                     (ta,tb,tc,X) and (ta,tb,tc,Y)   these are shared with the other input sims.
                   Face NOT containing X: (ta,tb,tc,Y,...)   wait, face = 4 vertices of 5.
                   5 faces of (ta,tb,tc,X,Y):
                     opp ta: (tb,tc,X,Y)
                     opp tb: (ta,tc,X,Y)
                     opp tc: (ta,tb,X,Y)
                     opp X:  (ta,tb,tc,Y)    shared
                     opp Y:  (ta,tb,tc,X)    shared
                   So 3 external faces (opp ta, opp tb, opp tc). */

                /* For each external face of each input sim, save its neighbor */
                int ext_nb[3][3]; /* [input_sim][face: opp ta/tb/tc] */
                for(int i=0;i<3;i++){
                    ext_nb[i][0]=sim5_nb[shared_sims[i]][fidx(shared_sims[i],ta)];
                    ext_nb[i][1]=sim5_nb[shared_sims[i]][fidx(shared_sims[i],tb)];
                    ext_nb[i][2]=sim5_nb[shared_sims[i]][fidx(shared_sims[i],tc)];
                }

                /* Identify which input has which pair of {dd,ee,ff} */
                int si_de=-1,si_df=-1,si_ef=-1;
                for(int i=0;i<3;i++){
                    int hd=0,he=0,hf=0;
                    for(int k=0;k<5;k++){
                        if(sim5[shared_sims[i]][k]==dd)hd=1;
                        if(sim5[shared_sims[i]][k]==ee)he=1;
                        if(sim5[shared_sims[i]][k]==ff)hf=1;
                    }
                    if(hd&&he) si_de=i;
                    if(hd&&hf) si_df=i;
                    if(he&&hf) si_ef=i;
                }
                if(si_de<0||si_df<0||si_ef<0) continue;

                /* Create 3 new simplices:
                   T1'=(ta,tb,dd,ee,ff)   replaces the triangle vertex tc
                   Wait, output should be: (a,b,d,e,f), (a,c,d,e,f), (b,c,d,e,f)
                   where a=ta, b=tb, c=tc, d=dd, e=ee, f=ff */
                int nv1[5]={ta,tb,dd,ee,ff}; sort5(nv1);
                int nv2[5]={ta,tc,dd,ee,ff}; sort5(nv2);
                int nv3[5]={tb,tc,dd,ee,ff}; sort5(nv3);

                /* Reuse the 3 input slots */
                int ni1=shared_sims[0], ni2=shared_sims[1], ni3=shared_sims[2];
                memcpy(sim5[ni1],nv1,5*sizeof(int));
                memcpy(sim5[ni2],nv2,5*sizeof(int));
                memcpy(sim5[ni3],nv3,5*sizeof(int));

                /* Internal connections (3 pairs, via face containing dd,ee,ff + 1 of ta,tb,tc):
                   T1' T2' via (ta,dd,ee,ff): T1' opp tb, T2' opp tc
                   T1' T3' via (tb,dd,ee,ff): T1' opp ta, T3' opp tc
                   T2' T3' via (tc,dd,ee,ff): T2' opp ta, T3' opp tb */
                set_nb5(ni1,tb,ni2); set_nb5(ni2,tc,ni1);
                set_nb5(ni1,ta,ni3); set_nb5(ni3,tc,ni1);
                set_nb5(ni2,ta,ni3); set_nb5(ni3,tb,ni2);

                /* External connections:
                   T1'=(ta,tb,dd,ee,ff) external faces:
                     opp dd=(ta,tb,ee,ff): was in input (ta,tb,tc,ee,ff)=si_ef, face opp tc   ext_nb[si_ef][2]
                     opp ee=(ta,tb,dd,ff): was in si_df, face opp tc   ext_nb[si_df][2]
                     opp ff=(ta,tb,dd,ee): was in si_de, face opp tc   ext_nb[si_de][2]
                   Wait, this isn't right. Let me think again.

                   Input si_de = (ta,tb,tc,dd,ee). Its external face opp tc = (ta,tb,dd,ee).
                   This face should now belong to T1'=(ta,tb,dd,ee,ff), face opp ff.
                   So: set_nb5(ni1, ff, ext_nb[si_de][2]) and redirect old neighbor.

                   Input si_df = (ta,tb,tc,dd,ff). External face opp tc = (ta,tb,dd,ff).
                     T1' face opp ee = (ta,tb,dd,ff). set_nb5(ni1, ee, ext_nb[si_df][2]).

                   Input si_ef = (ta,tb,tc,ee,ff). External face opp tc = (ta,tb,ee,ff).
                     T1' face opp dd = (ta,tb,ee,ff). set_nb5(ni1, dd, ext_nb[si_ef][2]). */

                /* T1' external faces (opp dd, ee, ff): */
                set_nb5(ni1,dd,ext_nb[si_ef][2]);
                if(ext_nb[si_ef][2]>=0){for(int j=0;j<5;j++)if(sim5_nb[ext_nb[si_ef][2]][j]==shared_sims[si_ef]){sim5_nb[ext_nb[si_ef][2]][j]=ni1;break;}}
                set_nb5(ni1,ee,ext_nb[si_df][2]);
                if(ext_nb[si_df][2]>=0){for(int j=0;j<5;j++)if(sim5_nb[ext_nb[si_df][2]][j]==shared_sims[si_df]){sim5_nb[ext_nb[si_df][2]][j]=ni1;break;}}
                set_nb5(ni1,ff,ext_nb[si_de][2]);
                if(ext_nb[si_de][2]>=0){for(int j=0;j<5;j++)if(sim5_nb[ext_nb[si_de][2]][j]==shared_sims[si_de]){sim5_nb[ext_nb[si_de][2]][j]=ni1;break;}}

                /* T2'=(ta,tc,dd,ee,ff) external faces (opp dd, ee, ff): */
                set_nb5(ni2,dd,ext_nb[si_ef][1]); /* opp tb from si_ef */
                if(ext_nb[si_ef][1]>=0){for(int j=0;j<5;j++)if(sim5_nb[ext_nb[si_ef][1]][j]==shared_sims[si_ef]){sim5_nb[ext_nb[si_ef][1]][j]=ni2;break;}}
                set_nb5(ni2,ee,ext_nb[si_df][1]);
                if(ext_nb[si_df][1]>=0){for(int j=0;j<5;j++)if(sim5_nb[ext_nb[si_df][1]][j]==shared_sims[si_df]){sim5_nb[ext_nb[si_df][1]][j]=ni2;break;}}
                set_nb5(ni2,ff,ext_nb[si_de][1]);
                if(ext_nb[si_de][1]>=0){for(int j=0;j<5;j++)if(sim5_nb[ext_nb[si_de][1]][j]==shared_sims[si_de]){sim5_nb[ext_nb[si_de][1]][j]=ni2;break;}}

                /* T3'=(tb,tc,dd,ee,ff) external faces (opp dd, ee, ff): */
                set_nb5(ni3,dd,ext_nb[si_ef][0]); /* opp ta from si_ef */
                if(ext_nb[si_ef][0]>=0){for(int j=0;j<5;j++)if(sim5_nb[ext_nb[si_ef][0]][j]==shared_sims[si_ef]){sim5_nb[ext_nb[si_ef][0]][j]=ni3;break;}}
                set_nb5(ni3,ee,ext_nb[si_df][0]);
                if(ext_nb[si_df][0]>=0){for(int j=0;j<5;j++)if(sim5_nb[ext_nb[si_df][0]][j]==shared_sims[si_df]){sim5_nb[ext_nb[si_df][0]][j]=ni3;break;}}
                set_nb5(ni3,ff,ext_nb[si_de][0]);
                if(ext_nb[si_de][0]>=0){for(int j=0;j<5;j++)if(sim5_nb[ext_nb[si_de][0]][j]==shared_sims[si_de]){sim5_nb[ext_nb[si_de][0]][j]=ni3;break;}}

                acc33++;
                if(acc33<=3) fprintf(stderr,"  (3,3) success! edge+=%d tri+=%d\n",dbg33_edge,dbg33_tri);
                continue;
            }

            if(move_type == 3) {
                /* ============ (2,8) move: insert vertex ============ */
                /* Same as (2,4) but splits into 8 simplices */
                /* For now: use (2,4) as proxy since (2,8) is structurally
                   identical but more complex. The key missing piece for
                   phase C is (3,3) which we already have. */
                /* Attempt a (2,4) move instead */
                goto do_24;
            }

            if(move_type == 4) {
                /* ============ (8,2) move: remove vertex ============ */
                /* Inverse of (2,8). Requires degree-8 vertex.
                   For now: skip (very rare in practice) */
                continue;
            }

            if(move_type == 0) {
                /* ============ (4,2) move ============ */
                /* Pick random simplex, random edge, find 4 simplices sharing it */
                int si0 = rand() % n_4sim;
                int edges[10][2]={{0,1},{0,2},{0,3},{0,4},{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}};
                int ei = rand() % 10;
                int e1 = sim5[si0][edges[ei][0]], e2 = sim5[si0][edges[ei][1]];

                /* Find all simplices sharing edge (e1,e2) */
                int ring[10]; int nr=0;
                for(int s=0; s<n_4sim && nr<10; s++){
                    int h1=0,h2=0;
                    for(int k=0;k<5;k++){
                        if(sim5[s][k]==e1) h1=1;
                        if(sim5[s][k]==e2) h2=1;
                    }
                    if(h1&&h2) ring[nr++]=s;
                }
                if(nr != 4) continue;

                /* Collect other vertices (should be exactly 4: a,b,c,d) */
                int others[10]; int no=0;
                for(int i=0;i<4;i++)
                    for(int k=0;k<5;k++){
                        int v=sim5[ring[i]][k];
                        if(v==e1||v==e2) continue;
                        int dup=0;
                        for(int j=0;j<no;j++) if(others[j]==v){dup=1;break;}
                        if(!dup && no<10) others[no++]=v;
                    }
                if(no!=4) continue;

                /* Sort others */
                for(int i=0;i<3;i++) for(int j=i+1;j<4;j++)
                    if(others[i]>others[j]){int t=others[i];others[i]=others[j];others[j]=t;}
                int a=others[0],b=others[1],c=others[2],d=others[3];

                /* Volume control */
                if(n_4sim < target*0.7) continue;

                /* Metropolis: dS = -2*k4 (removing 2 simplices) */
                double dS = -2*k4_val;
                if(dS>0 && (double)rand()/RAND_MAX >= exp(-dS)) continue;

                /* Identify T1=(a,b,c,e1,e2), T2=(a,b,d,...), T3=(a,c,d,...), T4=(b,c,d,...) */
                int t1=-1,t2=-1,t3=-1,t4=-1;
                for(int i=0;i<4;i++){
                    int si=ring[i];
                    int ha=0,hb=0,hc=0,hd=0;
                    for(int k=0;k<5;k++){
                        if(sim5[si][k]==a)ha=1; if(sim5[si][k]==b)hb=1;
                        if(sim5[si][k]==c)hc=1; if(sim5[si][k]==d)hd=1;
                    }
                    if(ha&&hb&&hc&&!hd) t1=si;
                    if(ha&&hb&&!hc&&hd) t2=si;
                    if(ha&&!hb&&hc&&hd) t3=si;
                    if(!ha&&hb&&hc&&hd) t4=si;
                }
                if(t1<0||t2<0||t3<0||t4<0) continue;

                /* Save external neighbors */
                int nb_abce1=sim5_nb[t1][fidx(t1,e2)]; /* T1 face opp e2 */
                int nb_abce2=sim5_nb[t1][fidx(t1,e1)]; /* T1 face opp e1 */
                int nb_abde1=sim5_nb[t2][fidx(t2,e2)];
                int nb_abde2=sim5_nb[t2][fidx(t2,e1)];
                int nb_acde1=sim5_nb[t3][fidx(t3,e2)];
                int nb_acde2=sim5_nb[t3][fidx(t3,e1)];
                int nb_bcde1=sim5_nb[t4][fidx(t4,e2)];
                int nb_bcde2=sim5_nb[t4][fidx(t4,e1)];

                /* Replace: 4 simplices -> 2 */
                /* S1=(a,b,c,d,e1), S2=(a,b,c,d,e2) */
                int s1v[5]={a,b,c,d,e1}; sort5(s1v);
                int s2v[5]={a,b,c,d,e2}; sort5(s2v);

                /* Reuse t1 and t2 slots, mark t3 and t4 as dead */
                memcpy(sim5[t1], s1v, 5*sizeof(int));
                memcpy(sim5[t2], s2v, 5*sizeof(int));
                /* Mark t3, t4 dead (set all vertices to -1) */
                for(int k=0;k<5;k++){sim5[t3][k]=-1;sim5[t4][k]=-1;}
                for(int k=0;k<5;k++){sim5_nb[t3][k]=-1;sim5_nb[t4][k]=-1;}
                /* Note: n_4sim doesn't decrease (dead slots remain) */

                /* Internal: S1   S2 via shared face (a,b,c,d) */
                set_nb5(t1,e1,t2); set_nb5(t2,e2,t1);

                /* External */
                /* S1 opp d = (a,b,c,e1)   nb_abce1 */
                set_nb5(t1,d,nb_abce1);
                if(nb_abce1>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_abce1][j]==t1||sim5_nb[nb_abce1][j]==t2||sim5_nb[nb_abce1][j]==t3||sim5_nb[nb_abce1][j]==t4){sim5_nb[nb_abce1][j]=t1;break;}}
                /* S1 opp c = (a,b,d,e1)   nb_abde1 */
                set_nb5(t1,c,nb_abde1);
                if(nb_abde1>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_abde1][j]==t1||sim5_nb[nb_abde1][j]==t2||sim5_nb[nb_abde1][j]==t3||sim5_nb[nb_abde1][j]==t4){sim5_nb[nb_abde1][j]=t1;break;}}
                /* S1 opp b = (a,c,d,e1)   nb_acde1 */
                set_nb5(t1,b,nb_acde1);
                if(nb_acde1>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_acde1][j]==t1||sim5_nb[nb_acde1][j]==t2||sim5_nb[nb_acde1][j]==t3||sim5_nb[nb_acde1][j]==t4){sim5_nb[nb_acde1][j]=t1;break;}}
                /* S1 opp a = (b,c,d,e1)   nb_bcde1 */
                set_nb5(t1,a,nb_bcde1);
                if(nb_bcde1>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_bcde1][j]==t1||sim5_nb[nb_bcde1][j]==t2||sim5_nb[nb_bcde1][j]==t3||sim5_nb[nb_bcde1][j]==t4){sim5_nb[nb_bcde1][j]=t1;break;}}

                /* S2 opp d = (a,b,c,e2)   nb_abce2 */
                set_nb5(t2,d,nb_abce2);
                if(nb_abce2>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_abce2][j]==t1||sim5_nb[nb_abce2][j]==t2||sim5_nb[nb_abce2][j]==t3||sim5_nb[nb_abce2][j]==t4){sim5_nb[nb_abce2][j]=t2;break;}}
                /* S2 opp c = (a,b,d,e2)   nb_abde2 */
                set_nb5(t2,c,nb_abde2);
                if(nb_abde2>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_abde2][j]==t1||sim5_nb[nb_abde2][j]==t2||sim5_nb[nb_abde2][j]==t3||sim5_nb[nb_abde2][j]==t4){sim5_nb[nb_abde2][j]=t2;break;}}
                /* S2 opp b = (a,c,d,e2)   nb_acde2 */
                set_nb5(t2,b,nb_acde2);
                if(nb_acde2>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_acde2][j]==t1||sim5_nb[nb_acde2][j]==t2||sim5_nb[nb_acde2][j]==t3||sim5_nb[nb_acde2][j]==t4){sim5_nb[nb_acde2][j]=t2;break;}}
                /* S2 opp a = (b,c,d,e2)   nb_bcde2 */
                set_nb5(t2,a,nb_bcde2);
                if(nb_bcde2>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_bcde2][j]==t1||sim5_nb[nb_bcde2][j]==t2||sim5_nb[nb_bcde2][j]==t3||sim5_nb[nb_bcde2][j]==t4){sim5_nb[nb_bcde2][j]=t2;break;}}

                acc42++;
                continue;
            }

            /* ============ (2,4) move ============ */
            do_24:;
            int si1 = rand() % n_4sim;
            if(sim5[si1][0] < 0) continue; /* skip dead */
            int fi = rand() % 5;
            int si2 = sim5_nb[si1][fi];
            if(si2 < 0) continue;

            /* Find shared tet (4 vertices) and opposite vertices */
            int shared[4], ns=0;
            for(int j=0;j<5;j++){
                int v=sim5[si1][j], found=0;
                for(int k=0;k<5;k++) if(sim5[si2][k]==v){found=1;break;}
                if(found && ns<4) shared[ns++]=v;
            }
            if(ns!=4) continue;

            int e1=-1, e2=-1;
            for(int j=0;j<5;j++){
                int v=sim5[si1][j], found=0;
                for(int k=0;k<4;k++) if(shared[k]==v){found=1;break;}
                if(!found){e1=v;break;}
            }
            for(int j=0;j<5;j++){
                int v=sim5[si2][j], found=0;
                for(int k=0;k<4;k++) if(shared[k]==v){found=1;break;}
                if(!found){e2=v;break;}
            }
            if(e1<0||e2<0||e1==e2) continue;

            /* Volume control */
            if(n_4sim > target*1.2) continue;

            /* Metropolis: dS = k4 * 2 (adding 2 simplices) */
            double dS = 2*k4_val;
            if(dS>0 && (double)rand()/RAND_MAX >= exp(-dS)) continue;

            /* Check edge (e1,e2) doesn't exist */
            {
                int exists=0;
                /* Linear scan: any simplex contains both e1 and e2? */
                for(int s=0; s<n_4sim && !exists; s++){
                    int h1=0,h2=0;
                    for(int k=0;k<5;k++){
                        if(sim5[s][k]==e1)h1=1;
                        if(sim5[s][k]==e2)h2=1;
                    }
                    if(h1&&h2) exists=1;
                }
                if(exists) continue;
            }

            /* Save old external neighbors */
            int a=shared[0],b=shared[1],c=shared[2],d=shared[3];

            /* S1 faces: (b,c,d,e1)=opp a, (a,c,d,e1)=opp b, (a,b,d,e1)=opp c,
                         (a,b,c,e1)=opp d, (a,b,c,d)=opp e1 (=shared) */
            /* S2 faces: same with e2 */
            /* Find face index for each */
            int nb_s1[5], nb_s2[5]; /* neighbors of S1, S2 by face */
            for(int j=0;j<5;j++){nb_s1[j]=sim5_nb[si1][j]; nb_s2[j]=sim5_nb[si2][j];}

            /* face indices: face_i = opposite vertex at index i (sorted) */
            /* find_idx: position of vertex v in sorted simplex */
            /* Helper: find index of vertex v in simplex si */

            int fi_s1_a=fidx(si1,a), fi_s1_b=fidx(si1,b), fi_s1_c=fidx(si1,c), fi_s1_d=fidx(si1,d);
            int fi_s2_a=fidx(si2,a), fi_s2_b=fidx(si2,b), fi_s2_c=fidx(si2,c), fi_s2_d=fidx(si2,d);

            if(fi_s1_a<0||fi_s1_d<0||fi_s2_a<0||fi_s2_d<0) continue;

            /* External neighbors from S1 */
            int nb_bcd_e1 = nb_s1[fi_s1_a]; /* face opp a = (b,c,d,e1) */
            int nb_acd_e1 = nb_s1[fi_s1_b]; /* face opp b */
            int nb_abd_e1 = nb_s1[fi_s1_c]; /* face opp c */
            int nb_abc_e1 = nb_s1[fi_s1_d]; /* face opp d = (a,b,c,e1) */

            /* External neighbors from S2 */
            int nb_bcd_e2 = nb_s2[fi_s2_a];
            int nb_acd_e2 = nb_s2[fi_s2_b];
            int nb_abd_e2 = nb_s2[fi_s2_c];
            int nb_abc_e2 = nb_s2[fi_s2_d];

            /* Create 4 new simplices */
            /* T1=(a,b,c,e1,e2), T2=(a,b,d,e1,e2), T3=(a,c,d,e1,e2), T4=(b,c,d,e1,e2) */
            int t1v[5]={a,b,c,e1,e2}; sort5(t1v);
            int t2v[5]={a,b,d,e1,e2}; sort5(t2v);
            int t3v[5]={a,c,d,e1,e2}; sort5(t3v);
            int t4v[5]={b,c,d,e1,e2}; sort5(t4v);

            /* Allocate new slots (reuse si1, si2, allocate 2 more) */
            if(n_4sim+1 >= MAX_4SIM) continue;

            int ti1=si1, ti2=si2, ti3=n_4sim, ti4=n_4sim+1;
            memcpy(sim5[ti1], t1v, 5*sizeof(int));
            memcpy(sim5[ti2], t2v, 5*sizeof(int));
            memcpy(sim5[ti3], t3v, 5*sizeof(int));
            memcpy(sim5[ti4], t4v, 5*sizeof(int));
            n_4sim += 2;

            /* Internal connections (6 pairs) */
            #define set_nb5(si,v_opp,nb) {int _fi=fidx(si,v_opp);if(_fi>=0)sim5_nb[si][_fi]=(nb);}
            /* T1 T2 via (a,b,e1,e2): T1 opp c, T2 opp d */
            set_nb5(ti1,c,ti2); set_nb5(ti2,d,ti1);
            /* T1 T3 via (a,c,e1,e2): T1 opp b, T3 opp d */
            set_nb5(ti1,b,ti3); set_nb5(ti3,d,ti1);
            /* T1 T4 via (b,c,e1,e2): T1 opp a, T4 opp d */
            set_nb5(ti1,a,ti4); set_nb5(ti4,d,ti1);
            /* T2 T3 via (a,d,e1,e2): T2 opp b, T3 opp c */
            set_nb5(ti2,b,ti3); set_nb5(ti3,c,ti2);
            /* T2 T4 via (b,d,e1,e2): T2 opp a, T4 opp c */
            set_nb5(ti2,a,ti4); set_nb5(ti4,c,ti2);
            /* T3 T4 via (c,d,e1,e2): T3 opp a, T4 opp b */
            set_nb5(ti3,a,ti4); set_nb5(ti4,b,ti3);

            /* External connections */
            /* T1 opp e2 = (a,b,c,e1)   S1's old nb at face opp d */
            set_nb5(ti1,e2,nb_abc_e1);
            if(nb_abc_e1>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_abc_e1][j]==si1){sim5_nb[nb_abc_e1][j]=ti1;break;}}
            /* T1 opp e1 = (a,b,c,e2)   S2's old nb at face opp d */
            set_nb5(ti1,e1,nb_abc_e2);
            if(nb_abc_e2>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_abc_e2][j]==si2){sim5_nb[nb_abc_e2][j]=ti1;break;}}

            /* T2 opp e2 = (a,b,d,e1)   S1 opp c */
            set_nb5(ti2,e2,nb_abd_e1);
            if(nb_abd_e1>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_abd_e1][j]==si1){sim5_nb[nb_abd_e1][j]=ti2;break;}}
            /* T2 opp e1 = (a,b,d,e2)   S2 opp c */
            set_nb5(ti2,e1,nb_abd_e2);
            if(nb_abd_e2>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_abd_e2][j]==si2){sim5_nb[nb_abd_e2][j]=ti2;break;}}

            /* T3 opp e2 = (a,c,d,e1)   S1 opp b */
            set_nb5(ti3,e2,nb_acd_e1);
            if(nb_acd_e1>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_acd_e1][j]==si1){sim5_nb[nb_acd_e1][j]=ti3;break;}}
            /* T3 opp e1 = (a,c,d,e2)   S2 opp b */
            set_nb5(ti3,e1,nb_acd_e2);
            if(nb_acd_e2>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_acd_e2][j]==si2){sim5_nb[nb_acd_e2][j]=ti3;break;}}

            /* T4 opp e2 = (b,c,d,e1)   S1 opp a */
            set_nb5(ti4,e2,nb_bcd_e1);
            if(nb_bcd_e1>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_bcd_e1][j]==si1){sim5_nb[nb_bcd_e1][j]=ti4;break;}}
            /* T4 opp e1 = (b,c,d,e2)   S2 opp a */
            set_nb5(ti4,e1,nb_bcd_e2);
            if(nb_bcd_e2>=0){for(int j=0;j<5;j++)if(sim5_nb[nb_bcd_e2][j]==si2){sim5_nb[nb_bcd_e2][j]=ti4;break;}}

            acc24++;

            if((iter+1)%50000==0)
                fprintf(stderr, "  %d: 24=%d 42=%d 33=%d n4=%d\n",
                        iter+1, acc24, acc42, acc33, n_4sim);
        }
        fprintf(stderr, "After: 24=%d 42=%d 33=%d (dbg: edge=%d tri=%d nd=%d)\n",
                acc24, acc42, acc33, dbg33_edge, dbg33_tri, dbg33_nd, dbg33_nd);
    }

    fprintf(stderr, "Random walk (%d walks, sigma_max=%d)...\n", n_walks, sigma_max);
    measure_dspec_4d(n_walks, sigma_max);

    free(base);
    fprintf(stderr, "Done.\n");
    return 0;
}
