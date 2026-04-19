/*
 * Production 2+1D CDT: C implementation with Pachner moves.
 *
 * Features:
 * - Proper tetrahedralization (prism decomposition, avg_nb=4.0)
 * - (2,3)/(3,2) Pachner moves with incremental neighbor updates
 * - Regge action: S = -k0*N0 + k3*N3, Metropolis acceptance
 * - Dual-graph random walk for spectral dimension
 *
 * Compile: cl /O2 /std:c11 /Fe:cdt_main.exe cdt_main.c
 * Usage:   cdt_main L T n_therm n_pachner n_walks sigma_max k0 k3 seed
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TETS  10000000
#define MAX_VERTS 2000000
#define MAX_VT    60

/* Tet storage */
static int tv[MAX_TETS][4];      /* sorted vertices */
static int tn[MAX_TETS][4];      /* neighbor tet for each face (-1=none) */
static char ta[MAX_TETS];        /* alive flag */
static int n_tets_total = 0;     /* total slots used */
static int n_tets_alive = 0;

/* Vertex-to-tet map */
static int vt[MAX_VERTS][MAX_VT];
static int vt_deg[MAX_VERTS];
static int n_verts = 0;

/* Free list for tet reuse */
static int free_list[MAX_TETS];
static int n_free = 0;

/* ============================================================ */
static void sort4(int a[4]) {
    int i,j,t;
    for(i=0;i<3;i++) for(j=i+1;j<4;j++)
        if(a[i]>a[j]){t=a[i];a[i]=a[j];a[j]=t;}
}

static int find_v_idx(int ti, int v) {
    for(int i=0;i<4;i++) if(tv[ti][i]==v) return i;
    return -1;
}

/* Add tet, return index */
static int add_tet(int v0, int v1, int v2, int v3) {
    int ti;
    if(n_free > 0) {
        ti = free_list[--n_free];
    } else {
        ti = n_tets_total++;
        if(ti >= MAX_TETS) { fprintf(stderr,"MAX_TETS!\n"); exit(1); }
    }
    tv[ti][0]=v0; tv[ti][1]=v1; tv[ti][2]=v2; tv[ti][3]=v3;
    sort4(tv[ti]);
    tn[ti][0]=tn[ti][1]=tn[ti][2]=tn[ti][3]=-1;
    ta[ti]=1;
    n_tets_alive++;
    /* Update vertex-to-tet */
    for(int i=0;i<4;i++){
        int v=tv[ti][i];
        if(v<MAX_VERTS && vt_deg[v]<MAX_VT)
            vt[v][vt_deg[v]++] = ti;
    }
    return ti;
}

static void remove_tet(int ti) {
    ta[ti]=0;
    n_tets_alive--;
    /* Remove from vertex-to-tet */
    for(int i=0;i<4;i++){
        int v=tv[ti][i];
        for(int j=0;j<vt_deg[v];j++){
            if(vt[v][j]==ti){
                vt[v][j]=vt[v][vt_deg[v]-1];
                vt_deg[v]--;
                break;
            }
        }
    }
    free_list[n_free++] = ti;
}

/* Set neighbor: ti's face opposite v_opp points to nb */
static void set_nb(int ti, int v_opp, int nb) {
    int fi = find_v_idx(ti, v_opp);
    if(fi >= 0) tn[ti][fi] = nb;
}

/* Get neighbor at face opposite v_opp */
static int get_nb(int ti, int v_opp) {
    int fi = find_v_idx(ti, v_opp);
    return fi >= 0 ? tn[ti][fi] : -1;
}

/* Update old_nb's pointer: if it pointed to old_ti, make it point to new_ti */
static void redirect_nb(int old_nb, int old_ti, int new_ti) {
    if(old_nb < 0) return;
    for(int i=0;i<4;i++){
        if(tn[old_nb][i]==old_ti){
            tn[old_nb][i]=new_ti;
            return;
        }
    }
}

/* Check if edge (d,e) exists */
static int edge_exists(int d, int e) {
    for(int i=0;i<vt_deg[d];i++){
        int ti=vt[d][i];
        if(!ta[ti]) continue;
        for(int j=0;j<4;j++) if(tv[ti][j]==e) return 1;
    }
    return 0;
}

/* Find tets containing both d and e */
static int find_edge_tets(int d, int e, int *result, int max_n) {
    int count=0;
    for(int i=0;i<vt_deg[d];i++){
        int ti=vt[d][i];
        if(!ta[ti]) continue;
        for(int j=0;j<4;j++) if(tv[ti][j]==e){
            if(count<max_n) result[count]=ti;
            count++;
            break;
        }
    }
    return count;
}

/* ============================================================
 * (2,3) Pachner move
 * ============================================================ */
static int move_23(void) {
    /* Pick random alive tet and random face */
    int ti1, attempts=0;
    do { ti1=rand()%n_tets_total; } while(!ta[ti1] && ++attempts<100);
    if(!ta[ti1]) return 0;

    int fi = rand()%4;
    int ti2 = tn[ti1][fi];
    if(ti2<0 || !ta[ti2]) return 0;

    /* Shared face = {tv[ti1] except tv[ti1][fi]} */
    int d = tv[ti1][fi];  /* vertex of ti1 NOT in shared face */

    /* Find e = vertex of ti2 NOT in shared face */
    int e = -1;
    for(int i=0;i<4;i++){
        int v=tv[ti2][i], found=0;
        for(int j=0;j<4;j++) if(j!=fi && tv[ti1][j]==v){found=1;break;}
        if(!found){e=v;break;}
    }
    if(e<0 || d==e) return 0;

    /* Check edge (d,e) doesn't exist */
    if(edge_exists(d,e)) return 0;

    /* Shared face vertices */
    int shared[3], si=0;
    for(int i=0;i<4;i++) if(i!=fi) shared[si++]=tv[ti1][i];
    int a=shared[0], b=shared[1], c=shared[2];

    /* Save old neighbors before removing */
    /* ti1's non-shared faces: opposite a, b, c (faces containing d) */
    int nb_abd = get_nb(ti1, c);  /* face {a,b,d} = opposite c */
    int nb_acd = get_nb(ti1, b);  /* face {a,c,d} = opposite b */
    int nb_bcd = get_nb(ti1, a);  /* face {b,c,d} = opposite a */

    /* ti2's non-shared faces: opposite a, b, c (faces containing e) */
    int nb_abe = get_nb(ti2, c);  /* face {a,b,e} */
    int nb_ace = get_nb(ti2, b);  /* face {a,c,e} */
    int nb_bce = get_nb(ti2, a);  /* face {b,c,e} */

    /* Remove old tets */
    remove_tet(ti1);
    remove_tet(ti2);

    /* Add 3 new tets */
    int t3 = add_tet(a,b,d,e);  /* has faces: (a,b,d),(a,b,e),(a,d,e),(b,d,e) */
    int t4 = add_tet(b,c,d,e);  /* has faces: (b,c,d),(b,c,e),(b,d,e),(c,d,e) */
    int t5 = add_tet(a,c,d,e);  /* has faces: (a,c,d),(a,c,e),(a,d,e),(c,d,e) */

    /* Internal connections */
    set_nb(t3, a, t4); /* t3's face (b,d,e) -> t4 (opposite a in t3) */
    /* Wait, (b,d,e) is opposite a only if a is the min. Need to find which face. */
    /* t3 = sorted(a,b,d,e). Face opposite a = {b,d,e}. Face opposite b = {a,d,e}. etc */
    /* t3 face {b,d,e} = opp a -> connects to t4 (which has face {b,d,e}) */
    set_nb(t3, a, t4);  /* t3 face opp a = {b,d,e} <-> t4 */
    set_nb(t4, c, t3);  /* t4 face opp c = {b,d,e} <-> t3 */

    set_nb(t4, b, t5);  /* t4 face opp b = {c,d,e} <-> t5 */
    set_nb(t5, a, t4);  /* t5 face opp a = {c,d,e} <-> t4 */

    set_nb(t3, b, t5);  /* t3 face opp b = {a,d,e} <-> t5 */
    set_nb(t5, c, t3);  /* t5 face opp c = {a,d,e} <-> t3 */

    /* External connections */
    set_nb(t3, e, nb_abd);  redirect_nb(nb_abd, ti1, t3);  /* {a,b,d} */
    set_nb(t3, d, nb_abe);  redirect_nb(nb_abe, ti2, t3);  /* {a,b,e} */
    set_nb(t4, e, nb_bcd);  redirect_nb(nb_bcd, ti1, t4);  /* {b,c,d} */
    set_nb(t4, d, nb_bce);  redirect_nb(nb_bce, ti2, t4);  /* {b,c,e} */
    set_nb(t5, e, nb_acd);  redirect_nb(nb_acd, ti1, t5);  /* {a,c,d} */
    set_nb(t5, d, nb_ace);  redirect_nb(nb_ace, ti2, t5);  /* {a,c,e} */

    return 1;
}

/* ============================================================
 * (3,2) Pachner move
 * ============================================================ */
static int move_32(void) {
    /* Pick random alive tet, random edge */
    int ti1, attempts=0;
    do { ti1=rand()%n_tets_total; } while(!ta[ti1] && ++attempts<100);
    if(!ta[ti1]) return 0;

    int edges[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    int ei = rand()%6;
    int d=tv[ti1][edges[ei][0]], e=tv[ti1][edges[ei][1]];

    /* Find all tets with edge (d,e) */
    int edge_tets[10], n_et;
    n_et = find_edge_tets(d, e, edge_tets, 10);
    if(n_et != 3) return 0;

    /* Collect other vertices (should be exactly 3) */
    int others[10]; int n_oth=0;
    for(int i=0;i<3;i++){
        int ti=edge_tets[i];
        for(int j=0;j<4;j++){
            int v=tv[ti][j];
            if(v==d||v==e) continue;
            int dup=0;
            for(int k=0;k<n_oth;k++) if(others[k]==v){dup=1;break;}
            if(!dup) others[n_oth++]=v;
        }
    }
    if(n_oth!=3) return 0;

    int a=others[0], b=others[1], c=others[2];
    /* Sort a<b<c */
    if(a>b){int t=a;a=b;b=t;}
    if(b>c){int t=b;b=c;c=t;}
    if(a>b){int t=a;a=b;b=t;}

    /* Identify which tet is which:
       T3=(a,b,d,e), T4=(b,c,d,e), T5=(a,c,d,e)
       We need to find them in edge_tets */
    int t3=-1,t4=-1,t5=-1;
    for(int i=0;i<3;i++){
        int ti=edge_tets[i];
        int has_a=0,has_b=0,has_c=0;
        for(int j=0;j<4;j++){
            if(tv[ti][j]==a) has_a=1;
            if(tv[ti][j]==b) has_b=1;
            if(tv[ti][j]==c) has_c=1;
        }
        if(has_a&&has_b&&!has_c) t3=ti;
        if(has_b&&has_c&&!has_a) t4=ti;
        if(has_a&&has_c&&!has_b) t5=ti;
    }
    if(t3<0||t4<0||t5<0) return 0;

    /* Save external neighbors */
    int nb_abd = get_nb(t3, e);
    int nb_abe = get_nb(t3, d);
    int nb_bcd = get_nb(t4, e);
    int nb_bce = get_nb(t4, d);
    int nb_acd = get_nb(t5, e);
    int nb_ace = get_nb(t5, d);

    /* Remove 3 old tets */
    remove_tet(t3);
    remove_tet(t4);
    remove_tet(t5);

    /* Add 2 new tets */
    int ti1_new = add_tet(a,b,c,d);
    int ti2_new = add_tet(a,b,c,e);

    /* Internal: shared face (a,b,c) */
    set_nb(ti1_new, d, ti2_new);  /* ti1 face opp d = {a,b,c} */
    set_nb(ti2_new, e, ti1_new);  /* ti2 face opp e = {a,b,c} */

    /* External */
    set_nb(ti1_new, c, nb_abd); redirect_nb(nb_abd, t3, ti1_new);
    set_nb(ti1_new, b, nb_acd); redirect_nb(nb_acd, t5, ti1_new);
    set_nb(ti1_new, a, nb_bcd); redirect_nb(nb_bcd, t4, ti1_new);

    set_nb(ti2_new, c, nb_abe); redirect_nb(nb_abe, t3, ti2_new);
    set_nb(ti2_new, b, nb_ace); redirect_nb(nb_ace, t5, ti2_new);
    set_nb(ti2_new, a, nb_bce); redirect_nb(nb_bce, t4, ti2_new);

    return 1;
}

/* ============================================================
 * Spatial triangulation + initial tets (same as cdt_large.c)
 * ============================================================ */
#define MAX_STRI 25000
#define MAX_SDEG 30
typedef struct {
    int tri[MAX_STRI][3]; int n_tri, n_vert, L;
    int svt[12000][MAX_SDEG]; int svt_deg[12000];
} SSlice;

static void sort3(int*a,int*b,int*c){int t;
    if(*a>*b){t=*a;*a=*b;*b=t;}if(*b>*c){t=*b;*b=*c;*c=t;}if(*a>*b){t=*a;*a=*b;*b=t;}}

static void rebuild_ss(SSlice*s){int i,k,v;
    for(v=0;v<s->n_vert;v++)s->svt_deg[v]=0;
    for(i=0;i<s->n_tri;i++)for(k=0;k<3;k++){v=s->tri[i][k];
        if(v<MAX_VERTS&&s->svt_deg[v]<MAX_SDEG)s->svt[v][s->svt_deg[v]++]=i;}}

static void init_ss(SSlice*s,int L){int x,y,a,b,c;
    memset(s,0,sizeof(SSlice));s->L=L;s->n_vert=L*L;s->n_tri=0;
    for(x=0;x<L;x++)for(y=0;y<L;y++){
        int v00=x*L+y,v10=((x+1)%L)*L+y,v01=x*L+(y+1)%L,v11=((x+1)%L)*L+(y+1)%L;
        a=v00;b=v10;c=v01;sort3(&a,&b,&c);
        s->tri[s->n_tri][0]=a;s->tri[s->n_tri][1]=b;s->tri[s->n_tri][2]=c;s->n_tri++;
        a=v10;b=v11;c=v01;sort3(&a,&b,&c);
        s->tri[s->n_tri][0]=a;s->tri[s->n_tri][1]=b;s->tri[s->n_tri][2]=c;s->n_tri++;}
    rebuild_ss(s);}

static int find_adj_ss(SSlice*s,int va,int vb,int excl){
    int mn=va<vb?va:vb,i,ti,k;
    for(i=0;i<s->svt_deg[mn];i++){ti=s->svt[mn][i];if(ti==excl)continue;
        for(k=0;k<3;k++)if(s->tri[ti][k]==(va<vb?vb:va))return ti;}return -1;}

static int flip_ss(SSlice*s){
    int ti=rand()%s->n_tri,ei=rand()%3;
    int v1=s->tri[ti][ei],v2=s->tri[ti][(ei+1)%3];
    int ti2=find_adj_ss(s,v1,v2,ti);if(ti2<0)return 0;
    int v3=-1,v4=-1,k;
    for(k=0;k<3;k++)if(s->tri[ti][k]!=v1&&s->tri[ti][k]!=v2){v3=s->tri[ti][k];break;}
    for(k=0;k<3;k++)if(s->tri[ti2][k]!=v1&&s->tri[ti2][k]!=v2){v4=s->tri[ti2][k];break;}
    if(v3<0||v4<0||v3==v4)return 0;
    if(find_adj_ss(s,v3,v4,-1)>=0)return 0;
    if(s->svt_deg[v1]<=3||s->svt_deg[v2]<=3)return 0;
    int a,b,c;
    a=v3;b=v4;c=v1;sort3(&a,&b,&c);s->tri[ti][0]=a;s->tri[ti][1]=b;s->tri[ti][2]=c;
    a=v3;b=v4;c=v2;sort3(&a,&b,&c);s->tri[ti2][0]=a;s->tri[ti2][1]=b;s->tri[ti2][2]=c;
    rebuild_ss(s);return 1;}

/* Build initial tets + full face matching (sort-based) */
typedef struct { int v[3]; int tet; } FaceE;
static int face_cmp(const void*a,const void*b){
    const FaceE*fa=a,*fb=b;
    if(fa->v[0]!=fb->v[0])return fa->v[0]-fb->v[0];
    if(fa->v[1]!=fb->v[1])return fa->v[1]-fb->v[1];
    return fa->v[2]-fb->v[2];}

static void build_initial(SSlice*base, int T){
    int nsv=base->n_vert, t, ti, fi;
    n_tets_total=0; n_tets_alive=0; n_free=0;
    memset(vt_deg,0,sizeof(vt_deg));
    n_verts = nsv * T;

    for(t=0;t<T;t++){int tn=(t+1)%T;
        for(ti=0;ti<base->n_tri;ti++){
            int a=base->tri[ti][0],b=base->tri[ti][1],c=base->tri[ti][2];
            int at=t*nsv+a,bt=t*nsv+b,ct=t*nsv+c;
            int atn=tn*nsv+a,btn=tn*nsv+b,ctn=tn*nsv+c;
            add_tet(at,bt,ct,atn);
            add_tet(bt,ct,atn,btn);
            add_tet(ct,atn,btn,ctn);
        }
    }

    /* Sort-based face matching */
    int nf=n_tets_alive*4;
    FaceE*faces=(FaceE*)malloc(nf*sizeof(FaceE));
    int idx=0;
    for(ti=0;ti<n_tets_total;ti++){if(!ta[ti])continue;
        int fv[4][3]={
            {tv[ti][1],tv[ti][2],tv[ti][3]},
            {tv[ti][0],tv[ti][2],tv[ti][3]},
            {tv[ti][0],tv[ti][1],tv[ti][3]},
            {tv[ti][0],tv[ti][1],tv[ti][2]}};
        for(fi=0;fi<4;fi++){faces[idx].v[0]=fv[fi][0];
            faces[idx].v[1]=fv[fi][1];faces[idx].v[2]=fv[fi][2];
            faces[idx].tet=ti;idx++;}}

    qsort(faces,idx,sizeof(FaceE),face_cmp);
    int matched=0;
    for(int i=0;i<idx-1;i++){
        if(faces[i].v[0]==faces[i+1].v[0]&&
           faces[i].v[1]==faces[i+1].v[1]&&
           faces[i].v[2]==faces[i+1].v[2]){
            int a=faces[i].tet, b=faces[i+1].tet;
            if(a!=b){
                /* Find face indices */
                for(fi=0;fi<4;fi++){
                    int ok=1;
                    for(int j=0;j<3;j++){
                        int fv=tv[a][(fi==0?1:fi==1?0:fi==2?0:1)]; /* wrong */
                        /* Just find which face */
                    }
                }
                /* Simpler: for tet a, face opposite vertex NOT in the shared face */
                int shared_v[3]={faces[i].v[0],faces[i].v[1],faces[i].v[2]};
                for(fi=0;fi<4;fi++){
                    int v=tv[a][fi], found=0;
                    for(int j=0;j<3;j++)if(shared_v[j]==v){found=1;break;}
                    if(!found){tn[a][fi]=b;break;}
                }
                for(fi=0;fi<4;fi++){
                    int v=tv[b][fi], found=0;
                    for(int j=0;j<3;j++)if(shared_v[j]==v){found=1;break;}
                    if(!found){tn[b][fi]=a;break;}
                }
                matched++;
            }
            i++;
        }
    }
    free(faces);
    fprintf(stderr,"  %d tets, %d matched, avg_nb=%.2f\n",
            n_tets_alive,matched,matched*2.0/n_tets_alive);
}

/* ============================================================
 * Random walk
 * ============================================================ */
static void measure_dspec(int nw, int smax){
    double *P=(double*)calloc(smax+1,sizeof(double));
    for(int i=0;i<nw;i++){
        int start; do{start=rand()%n_tets_total;}while(!ta[start]);
        int pos=start, nc;
        for(int sig=1;sig<=smax;sig++){
            nc=0; for(int k=0;k<4;k++)if(tn[pos][k]>=0)nc++;
            if(nc==0)break;
            int pick=rand()%nc,cnt=0;
            for(int k=0;k<4;k++)if(tn[pos][k]>=0){
                if(cnt==pick){pos=tn[pos][k];break;}cnt++;}
            if(pos==start) P[sig]+=1.0;
        }
    }
    for(int i=0;i<=smax;i++)P[i]/=nw;
    printf("sigma d_spec P_return\n");
    int w=5;
    for(int i=w+2;i<=smax-w-2;i+=3){
        double Plo=0,Phi=0;int nlo=0,nhi=0;
        for(int j=i-w;j<=i;j++)if(P[j]>0){Plo+=P[j];nlo++;}
        for(int j=i;j<=i+w;j++)if(P[j]>0){Phi+=P[j];nhi++;}
        if(nlo>0&&nhi>0){Plo/=nlo;Phi/=nhi;
            if(Plo>1e-15&&Phi>1e-15){
                double d=-2.0*(log(Phi)-log(Plo))/(log((double)(i+w/2))-log((double)(i-w/2)));
                if(d>0&&d<10)printf("%d %.4f %.6e\n",i,d,P[i]);
            }
        }
    }
    free(P);
}

/* ============================================================ */
int main(int argc,char**argv){
    int L=15,T=22,n_sflip=5000,n_pachner=50000,nw=500000,smax=600,seed=42;
    double k0=0,k3=0;
    if(argc>=2)L=atoi(argv[1]);
    if(argc>=3)T=atoi(argv[2]);
    if(argc>=4)n_sflip=atoi(argv[3]);
    if(argc>=5)n_pachner=atoi(argv[4]);
    if(argc>=6)nw=atoi(argv[5]);
    if(argc>=7)smax=atoi(argv[6]);
    if(argc>=8)k0=atof(argv[7]);
    if(argc>=9)k3=atof(argv[8]);
    if(argc>=10)seed=atoi(argv[9]);
    srand(seed);

    fprintf(stderr,"CDT: L=%d T=%d sflip=%d pachner=%d walks=%d k0=%.2f k3=%.2f\n",
            L,T,n_sflip,n_pachner,nw,k0,k3);

    SSlice *basep = (SSlice*)calloc(1,sizeof(SSlice));
    if(!basep){fprintf(stderr,"SSlice alloc failed\n");return 1;}
    #define base (*basep)
    init_ss(&base,L);
    fprintf(stderr,"Spatial flips...");
    int acc=0; for(int i=0;i<n_sflip;i++)acc+=flip_ss(&base);
    fprintf(stderr," %d/%d\n",acc,n_sflip);

    fprintf(stderr,"Building initial spacetime...\n");
    build_initial(&base,T);

    /* Pachner moves with Regge action */
    fprintf(stderr,"Pachner moves (%d)...\n",n_pachner);
    int a23=0,a32=0;
    int target=n_tets_alive;
    for(int i=0;i<n_pachner;i++){
        int nt=n_tets_alive;
        /* Volume control + Regge action */
        double dS;
        if(rand()%2==0){
            /* (2,3): N3 += 1 */
            dS = k3;
            if(nt>target*1.15) continue; /* hard cap */
            if(dS<=0 || (double)rand()/RAND_MAX < exp(-dS))
                a23 += move_23();
        } else {
            /* (3,2): N3 -= 1 */
            dS = -k3;
            if(nt<target*0.85) continue;
            if(dS<=0 || (double)rand()/RAND_MAX < exp(-dS))
                a32 += move_32();
        }
        if((i+1)%10000==0)
            fprintf(stderr,"  %d: tets=%d 23=%d 32=%d\n",i+1,n_tets_alive,a23,a32);
    }
    fprintf(stderr,"After Pachner: tets=%d, 23=%d, 32=%d\n",n_tets_alive,a23,a32);

    fprintf(stderr,"Random walk (%d)...\n",nw);
    measure_dspec(nw,smax);

    fprintf(stderr,"Done.\n");
    return 0;
}
