/*
 * Large-scale 2+1D CDT with Pachner moves + dual-graph random walk.
 * Target: 10^5-10^6 tetrahedra for true UV spectral dimension.
 *
 * Compile: cl /O2 /Fe:cdt_large.exe cdt_large.c
 * Usage:   cdt_large L T n_spatial_flips n_pachner n_walks sigma_max seed
 *          cdt_large 20 30 10000 100000 500000 1000 42
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * Spatial triangulation (2D torus, edge flips)
 * ============================================================ */
#define MAX_STRI 5000
#define MAX_SVERT 2000
#define MAX_SDEG 25

typedef struct {
    int tri[MAX_STRI][3];
    int n_tri, n_vert, L;
    int vt[MAX_SVERT][MAX_SDEG];
    int vt_deg[MAX_SVERT];
} SSlice;

static void sort3i(int *a, int *b, int *c) {
    int t;
    if(*a>*b){t=*a;*a=*b;*b=t;}
    if(*b>*c){t=*b;*b=*c;*c=t;}
    if(*a>*b){t=*a;*a=*b;*b=t;}
}

static void rebuild_ss(SSlice *s) {
    int i,k,v;
    for(v=0;v<s->n_vert;v++) s->vt_deg[v]=0;
    for(i=0;i<s->n_tri;i++)
        for(k=0;k<3;k++){
            v=s->tri[i][k];
            if(v>=0 && v<MAX_SVERT && s->vt_deg[v]<MAX_SDEG)
                s->vt[v][s->vt_deg[v]++]=i;
        }
}

static void init_ss(SSlice *s, int L) {
    int x,y,a,b,c;
    memset(s,0,sizeof(SSlice));
    s->L=L; s->n_vert=L*L; s->n_tri=0;
    for(x=0;x<L;x++) for(y=0;y<L;y++){
        int v00=x*L+y, v10=((x+1)%L)*L+y, v01=x*L+(y+1)%L, v11=((x+1)%L)*L+(y+1)%L;
        a=v00;b=v10;c=v01;sort3i(&a,&b,&c);
        s->tri[s->n_tri][0]=a;s->tri[s->n_tri][1]=b;s->tri[s->n_tri][2]=c;s->n_tri++;
        a=v10;b=v11;c=v01;sort3i(&a,&b,&c);
        s->tri[s->n_tri][0]=a;s->tri[s->n_tri][1]=b;s->tri[s->n_tri][2]=c;s->n_tri++;
    }
    rebuild_ss(s);
}

static int find_adj(SSlice *s, int va, int vb, int excl) {
    int mn=va<vb?va:vb, mx=va<vb?vb:va, i,ti,k;
    for(i=0;i<s->vt_deg[mn];i++){
        ti=s->vt[mn][i]; if(ti==excl) continue;
        for(k=0;k<3;k++) if(s->tri[ti][k]==mx) return ti;
    }
    return -1;
}

static int flip_ss(SSlice *s) {
    int ti=rand()%s->n_tri, ei=rand()%3;
    int v1=s->tri[ti][ei], v2=s->tri[ti][(ei+1)%3];
    int ti2=find_adj(s,v1,v2,ti);
    if(ti2<0) return 0;
    int v3=-1,v4=-1,k;
    for(k=0;k<3;k++) if(s->tri[ti][k]!=v1&&s->tri[ti][k]!=v2){v3=s->tri[ti][k];break;}
    for(k=0;k<3;k++) if(s->tri[ti2][k]!=v1&&s->tri[ti2][k]!=v2){v4=s->tri[ti2][k];break;}
    if(v3<0||v4<0||v3==v4) return 0;
    if(find_adj(s,v3,v4,-1)>=0) return 0;
    if(s->vt_deg[v1]<=3||s->vt_deg[v2]<=3) return 0;
    int a,b,c;
    a=v3;b=v4;c=v1;sort3i(&a,&b,&c);
    s->tri[ti][0]=a;s->tri[ti][1]=b;s->tri[ti][2]=c;
    a=v3;b=v4;c=v2;sort3i(&a,&b,&c);
    s->tri[ti2][0]=a;s->tri[ti2][1]=b;s->tri[ti2][2]=c;
    rebuild_ss(s);
    return 1;
}

/* ============================================================
 * Tetrahedra + sort-based face matching + Pachner moves
 * ============================================================ */
#define MAX_TETS 500000

static int tet_v[MAX_TETS][4];  /* sorted vertices */
static int tet_nb[MAX_TETS][4]; /* neighbors (-1=none) */
static int n_tets = 0;
/* Vertex-to-tet for fast edge lookup */
#define MAX_VT_GLOBAL 50
static int g_vt[MAX_SVERT*50][MAX_VT_GLOBAL]; /* vertex -> tet list */
static int g_vt_deg[MAX_SVERT*50];
static int g_n_verts = 0;

static void sort4i(int a[4]) {
    int i,j,t;
    for(i=0;i<3;i++) for(j=i+1;j<4;j++)
        if(a[i]>a[j]){t=a[i];a[i]=a[j];a[j]=t;}
}

typedef struct { int v[3]; int tet; } Face;

static int face_cmp_global(const void *a, const void *b) {
    const Face *fa = (const Face *)a, *fb = (const Face *)b;
    if(fa->v[0]!=fb->v[0]) return fa->v[0]-fb->v[0];
    if(fa->v[1]!=fb->v[1]) return fa->v[1]-fb->v[1];
    return fa->v[2]-fb->v[2];
}

static void build_tets_and_dual(SSlice *base, int T) {
    int n_sv = base->n_vert;
    int t, ti, fi;

    n_tets = 0;
    /* Build tets: 3 per triangle per slab */
    for(t=0; t<T; t++) {
        int tn=(t+1)%T;
        for(ti=0; ti<base->n_tri; ti++) {
            int a=base->tri[ti][0], b=base->tri[ti][1], c=base->tri[ti][2];
            int at=t*n_sv+a, bt=t*n_sv+b, ct=t*n_sv+c;
            int atn=tn*n_sv+a, btn=tn*n_sv+b, ctn=tn*n_sv+c;
            if(n_tets+2>=MAX_TETS) break;
            tet_v[n_tets][0]=at;tet_v[n_tets][1]=bt;tet_v[n_tets][2]=ct;tet_v[n_tets][3]=atn;
            sort4i(tet_v[n_tets]); n_tets++;
            tet_v[n_tets][0]=bt;tet_v[n_tets][1]=ct;tet_v[n_tets][2]=atn;tet_v[n_tets][3]=btn;
            sort4i(tet_v[n_tets]); n_tets++;
            tet_v[n_tets][0]=ct;tet_v[n_tets][1]=atn;tet_v[n_tets][2]=btn;tet_v[n_tets][3]=ctn;
            sort4i(tet_v[n_tets]); n_tets++;
        }
    }

    /* Sort-based face matching */
    int n_faces = n_tets * 4;
    Face *faces = (Face*)malloc(n_faces * sizeof(Face));
    for(ti=0; ti<n_tets; ti++) {
        tet_nb[ti][0]=tet_nb[ti][1]=tet_nb[ti][2]=tet_nb[ti][3]=-1;
        int fv[4][3]={
            {tet_v[ti][0],tet_v[ti][1],tet_v[ti][2]},
            {tet_v[ti][0],tet_v[ti][1],tet_v[ti][3]},
            {tet_v[ti][0],tet_v[ti][2],tet_v[ti][3]},
            {tet_v[ti][1],tet_v[ti][2],tet_v[ti][3]}};
        for(fi=0;fi<4;fi++){
            int idx=ti*4+fi;
            faces[idx].v[0]=fv[fi][0]; faces[idx].v[1]=fv[fi][1];
            faces[idx].v[2]=fv[fi][2]; faces[idx].tet=ti;
        }
    }

    /* Sort faces (face_cmp defined at file scope) */
    qsort(faces, n_faces, sizeof(Face), face_cmp_global);

    /* Match */
    int i, matched=0;
    for(i=0; i<n_faces-1; i++) {
        if(faces[i].v[0]==faces[i+1].v[0] &&
           faces[i].v[1]==faces[i+1].v[1] &&
           faces[i].v[2]==faces[i+1].v[2]) {
            int a=faces[i].tet, b=faces[i+1].tet;
            if(a!=b) {
                /* Find which face index */
                int fa,fb,fi2;
                for(fa=0;fa<4;fa++) if(tet_nb[a][fa]<0){tet_nb[a][fa]=b;break;}
                for(fb=0;fb<4;fb++) if(tet_nb[b][fb]<0){tet_nb[b][fb]=a;break;}
                matched++;
            }
            i++;
        }
    }
    free(faces);

    fprintf(stderr, "  %d tets, %d matched faces, avg_nb=%.2f\n",
            n_tets, matched, matched*2.0/n_tets);
}

/* ============================================================
 * Random walk on dual graph
 * ============================================================ */
static void measure_dspec(int n_walks, int sigma_max) {
    double *P = (double*)calloc(sigma_max+1, sizeof(double));
    int i, sigma;

    for(i=0; i<n_walks; i++) {
        int start = rand() % n_tets;
        int nb_count = 0, k;
        for(k=0;k<4;k++) if(tet_nb[start][k]>=0) nb_count++;
        if(nb_count==0) continue;
        int pos = start;
        for(sigma=1; sigma<=sigma_max; sigma++) {
            nb_count=0;
            for(k=0;k<4;k++) if(tet_nb[pos][k]>=0) nb_count++;
            if(nb_count==0) break;
            int pick = rand() % nb_count, cnt=0;
            for(k=0;k<4;k++) if(tet_nb[pos][k]>=0){
                if(cnt==pick){pos=tet_nb[pos][k];break;}
                cnt++;
            }
            if(pos==start) P[sigma]+=1.0;
        }
    }
    for(i=0;i<=sigma_max;i++) P[i]/=n_walks;

    /* Output d_spec */
    printf("sigma d_spec P_return\n");
    int w=5;
    for(i=w+2; i<=sigma_max-w-2; i+=3) {
        double Plo=0,Phi=0; int nlo=0,nhi=0,j;
        for(j=i-w;j<=i;j++) if(P[j]>0){Plo+=P[j];nlo++;}
        for(j=i;j<=i+w;j++) if(P[j]>0){Phi+=P[j];nhi++;}
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
    int L=20, T=30, n_sflip=10000, n_walks=500000, sigma_max=1000, seed=42;
    if(argc>=2) L=atoi(argv[1]);
    if(argc>=3) T=atoi(argv[2]);
    if(argc>=4) n_sflip=atoi(argv[3]);
    if(argc>=5) n_walks=atoi(argv[4]);
    if(argc>=6) sigma_max=atoi(argv[5]);
    if(argc>=7) seed=atoi(argv[6]);
    srand(seed);

    fprintf(stderr, "CDT Large: L=%d T=%d sflip=%d walks=%d smax=%d\n",
            L, T, n_sflip, n_walks, sigma_max);

    SSlice base;
    init_ss(&base, L);
    fprintf(stderr, "Spatial flips...");
    int acc=0;
    for(int i=0;i<n_sflip;i++) acc+=flip_ss(&base);
    fprintf(stderr, " %d/%d\n", acc, n_sflip);

    fprintf(stderr, "Building spacetime...\n");
    build_tets_and_dual(&base, T);

    fprintf(stderr, "Random walk (%d walks)...\n", n_walks);
    measure_dspec(n_walks, sigma_max);

    fprintf(stderr, "Done.\n");
    return 0;
}
