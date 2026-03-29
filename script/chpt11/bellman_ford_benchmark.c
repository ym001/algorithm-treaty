/*
 * ================================================================
 *  SSSP BENCHMARK -- 6 ALGORITHMES, GRAPHES SYNTHETIQUES ET REELS
 *
 *  Algorithmes :
 *    1. BF Classique    -- Bellman (1958) / Ford (1956)
 *    2. SPFA            -- Moore (1959) / Duan & Wu (1994)
 *    3. BF Randomise    -- Bannister & Eppstein, ANALCO 2012
 *    4. BF Potentiel    -- Cantone & Maugeri, ICTCS 2019
 *    5. BF-ADAPT v1     -- Proposition originale (seuil noeuds)
 *    6. BF-ADAPT v2     -- Proposition originale (seuil arcs, adaptatif)
 *
 *  Graphes reels supportes :
 *    ROUTIERS : roadNet-CA/PA/TX.txt (SNAP Stanford)
 *               Format : u[TAB]v, lignes '#' ignorees (non pondere)
 *               URL    : snap.stanford.edu/data/roadNet-CA.txt.gz
 *
 *    FINANCIERS: soc-sign-bitcoinalpha.csv / soc-sign-bitcoin-otc.csv
 *               Format : source,target,rating,timestamp (CSV, pas d'entete)
 *               Poids  : rating in [-10,+10] (negatif = mefiance)
 *               URL    : snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz
 *               Ref    : Kumar et al., IEEE ICDM 2016
 *
 *  Usage :
 *    ./bf_bench                           (synthetiques seulement)
 *    ./bf_bench roadNet-PA.txt            (+ un routier)
 *    ./bf_bench roadNet-CA.txt roadNet-PA.txt roadNet-TX.txt \
 *               soc-sign-bitcoinalpha.csv soc-sign-bitcoin-otc.csv
 *
 *  Compilation : gcc -O2 -o bf_bench bellman_ford_benchmark.c -lm
 * ================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>
#include <math.h>

/* ── Limites ───────────────────────────────────────────────── */
#define INF    (INT_MAX / 2)
#define MAX_V  500000    /* noeuds max par graphe               */
#define MAX_E  650000    /* arcs max par graphe                 */
/* Note memoire : 6 tableaux dist d[MAX_V] = 6 * 500k * 4 = 12 MB   */
/* HexEntry pool = 500k * 56 bytes = 28 MB                           */
/* EdgeGraph.e = MAX_E * 12 = 7.8 MB                                 */
/* AdjGraph.head = MAX_V * 8 = 4 MB                                  */
/* Pour les graphes synthetiques (n <= 50000), la memoire est legere. */
/* Pour les roadNets (n ~ 500k), les algos allouent ~30 MB chacun.   */
/* 6 algos * 30 MB = 180 MB total -> OK sur machine moderne.         */

/* ── Structures ────────────────────────────────────────────── */
typedef struct { int u, v, w; } Edge;

typedef struct {
    int  n, m;
    Edge e[MAX_E];
} EdgeGraph;

typedef struct AdjNode { int to, w; struct AdjNode *next; } AdjNode;
typedef struct {
    int      n;
    AdjNode *head[MAX_V];
} AdjGraph;

typedef struct {
    int n, m;
    int offset[MAX_V + 1];
    int adj_v[MAX_E];
    int adj_w[MAX_E];
} CSRGraph;

typedef struct {
    long long relaxations, comparisons;
    int       iters;
    double    time_ms;
    bool      neg_cycle;
} Result;

/* ── Utilitaires ────────────────────────────────────────────── */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void dist_init(int *d, int n, int src) {
    for (int i = 0; i < n; i++) d[i] = INF;
    d[src] = 0;
}

static void add_edge(EdgeGraph *eg, AdjGraph *ag, int u, int v, int w) {
    if (u < 0 || v < 0 || u >= MAX_V || v >= MAX_V) return;
    if (eg->m >= MAX_E - 1) return;
    eg->e[eg->m++] = (Edge){u, v, w};
    if (eg->n <= u) eg->n = u + 1;
    if (eg->n <= v) eg->n = v + 1;
    AdjNode *nd = malloc(sizeof *nd);
    nd->to = v; nd->w = w; nd->next = ag->head[u]; ag->head[u] = nd;
    if (ag->n <= u) ag->n = u + 1;
    if (ag->n <= v) ag->n = v + 1;
}

static void adjgraph_free(AdjGraph *ag) {
    for (int i = 0; i < ag->n; i++) {
        AdjNode *c = ag->head[i];
        while (c) { AdjNode *t = c->next; free(c); c = t; }
        ag->head[i] = NULL;
    }
    ag->n = 0;
}

static void graph_reset(EdgeGraph *eg, AdjGraph *ag) {
    eg->n = 0; eg->m = 0;
    adjgraph_free(ag);
}

static void build_csr(const EdgeGraph *eg, CSRGraph *csr) {
    csr->n = eg->n; csr->m = eg->m;
    memset(csr->offset, 0, (eg->n + 1) * sizeof(int));
    for (int i = 0; i < eg->m; i++) csr->offset[eg->e[i].u + 1]++;
    for (int u = 0; u < eg->n; u++) csr->offset[u+1] += csr->offset[u];
    int *pos = calloc(eg->n, sizeof(int));
    for (int i = 0; i < eg->m; i++) {
        int u = eg->e[i].u, k = csr->offset[u] + pos[u]++;
        csr->adj_v[k] = eg->e[i].v;
        csr->adj_w[k] = eg->e[i].w;
    }
    free(pos);
}

/* ================================================================
 *  ALGORITHME 1 -- BF CLASSIQUE
 * ================================================================ */
Result bf_classic(const EdgeGraph *g, int src, int *dist) {
    Result r = {0};
    dist_init(dist, g->n, src);
    /* Cap: min(n-1, 2000) couvre la chaine (1499 passes) et borne
     * les grands graphes a cycles negatifs (2000 * m max)             */
    int max_pass = (g->n - 1 < 2000) ? g->n - 1 : 2000;
    double t0 = now_ms();
    for (int pass = 0; pass < max_pass; pass++) {
        r.iters++; bool upd = false;
        for (int i = 0; i < g->m; i++) {
            int u=g->e[i].u, v=g->e[i].v, w=g->e[i].w; r.comparisons++;
            if (dist[u]!=INF && dist[u]+w < dist[v]) {
                dist[v]=dist[u]+w; r.relaxations++; upd=true;
            }
        }
        if (!upd) break;
    }
    for (int i=0; i<g->m; i++) {
        int u=g->e[i].u, v=g->e[i].v, w=g->e[i].w;
        if (dist[u]!=INF && dist[u]+w<dist[v]) { r.neg_cycle=true; break; }
    }
    r.time_ms = now_ms() - t0;
    return r;
}

/* ================================================================
 *  ALGORITHME 2 -- SPFA
 *  CORRECTION : queue circulaire bornee O(n).
 *  L'implementation naive malloc(n*n) est incorrect pour grands graphes.
 *  Avec le garde inq[], au plus n noeuds simultanement en file.
 * ================================================================ */
Result spfa(const AdjGraph *ag, int src, int *dist) {
    Result r = {0};
    int n = ag->n;
    dist_init(dist, n, src);
    bool *inq = calloc(n, sizeof(bool));
    int  *cnt = calloc(n, sizeof(int));
    int  *Q   = malloc((n + 2) * sizeof(int));
    int   cap = n + 2, hd = 0, tl = 0;
    Q[tl++ % cap] = src; inq[src] = true; cnt[src] = 1;
    /* Cap total: evite O(n*m) sur graphes a cycles negatifs            */
    long long total_iters = 0;
    const long long ITER_CAP = (long long)n * 40;
    double t0 = now_ms();
    while (hd != tl) {
        if (++total_iters > ITER_CAP) { r.neg_cycle = true; goto spfa_end; }
        int u = Q[hd++ % cap]; inq[u] = false; r.iters++;
        for (AdjNode *nd = ag->head[u]; nd; nd = nd->next) {
            int v=nd->to, w=nd->w; r.comparisons++;
            if (dist[u]!=INF && dist[u]+w < dist[v]) {
                dist[v]=dist[u]+w; r.relaxations++;
                if (!inq[v]) {
                    inq[v]=true; Q[tl++ % cap]=v;
                    if (++cnt[v]>n) { r.neg_cycle=true; goto spfa_end; }
                }
            }
        }
    }
spfa_end:
    r.time_ms = now_ms() - t0;
    free(inq); free(cnt); free(Q);
    return r;
}

/* ================================================================
 *  ALGORITHME 3 -- BF RANDOMISE  (Bannister & Eppstein 2012)
 * ================================================================ */
static void shuffle(Edge *e, int m, unsigned *seed) {
    for (int i=m-1; i>0; i--) {
        *seed = *seed*1664525u+1013904223u;
        int j=(int)((*seed>>16)%(unsigned)(i+1));
        Edge t=e[i]; e[i]=e[j]; e[j]=t;
    }
}

Result bf_randomized(EdgeGraph *g, int src, int *dist) {
    Result r = {0};
    dist_init(dist, g->n, src);
    unsigned seed = 0xdeadbeef;
    int max_pass_r = (g->n - 1 < 2000) ? g->n - 1 : 2000;
    double t0 = now_ms();
    for (int pass = 0; pass < max_pass_r; pass++) {
        r.iters++; shuffle(g->e, g->m, &seed); bool upd = false;
        for (int i=0; i<g->m; i++) {
            int u=g->e[i].u, v=g->e[i].v, w=g->e[i].w; r.comparisons++;
            if (dist[u]!=INF && dist[u]+w<dist[v]) {
                dist[v]=dist[u]+w; r.relaxations++; upd=true;
            }
        }
        if (!upd) break;
    }
    for (int i=0; i<g->m; i++) {
        int u=g->e[i].u, v=g->e[i].v, w=g->e[i].w;
        if (dist[u]!=INF && dist[u]+w<dist[v]) { r.neg_cycle=true; break; }
    }
    r.time_ms = now_ms() - t0;
    return r;
}

/* ================================================================
 *  ALGORITHME 4 -- BF POTENTIEL  (Cantone & Maugeri 2019)
 * ================================================================ */
typedef struct { long long key; int node; } HEntry;
typedef struct { HEntry *data; int size, cap; } Heap;
static Heap *hnew(int c){Heap*h=malloc(sizeof*h);h->data=malloc(c*sizeof(HEntry));h->size=0;h->cap=c;return h;}
static void hfree(Heap*h){free(h->data);free(h);}
static void hpush(Heap*h,int node,long long key){
    if(h->size==h->cap){h->cap*=2;h->data=realloc(h->data,h->cap*sizeof(HEntry));}
    int i=h->size++;h->data[i]=(HEntry){key,node};
    while(i>0){int p=(i-1)/2;if(h->data[p].key<=h->data[i].key)break;
        HEntry t=h->data[p];h->data[p]=h->data[i];h->data[i]=t;i=p;}
}
static HEntry hpop(Heap*h){
    HEntry top=h->data[0];h->data[0]=h->data[--h->size];int i=0;
    while(1){int l=2*i+1,r=2*i+2,m=i;
        if(l<h->size&&h->data[l].key<h->data[m].key)m=l;
        if(r<h->size&&h->data[r].key<h->data[m].key)m=r;
        if(m==i)break;HEntry t=h->data[m];h->data[m]=h->data[i];h->data[i]=t;i=m;}
    return top;
}

Result bf_potential(const AdjGraph *ag, int src, int *dist) {
    Result r = {0}; int n=ag->n;
    dist_init(dist, n, src);
    long long *d_scan=malloc(n*sizeof*d_scan);
    int *in_heap=calloc(n,sizeof*in_heap), *scnt=calloc(n,sizeof*scnt);
    for(int i=0;i<n;i++) d_scan[i]=(long long)INF;
    Heap *H=hnew(n+16);
    hpush(H,src,(long long)dist[src]-d_scan[src]); in_heap[src]=1;
    double t0=now_ms();
    /* Limite la taille du heap pour eviter OOM sur cycles negatifs    */
    const int HEAP_MAX = n * 20 + 100000;
    while(H->size>0){
        if(H->size > HEAP_MAX) { r.neg_cycle=true; break; } /* guard OOM */
        HEntry he=hpop(H); int u=he.node; long long k=he.key;
        if(k!=((long long)dist[u]-d_scan[u])) continue;
        in_heap[u]=0; d_scan[u]=(long long)dist[u]; r.iters++;
        if(++scnt[u]>n){r.neg_cycle=true;break;}
        for(AdjNode*nd=ag->head[u];nd;nd=nd->next){
            int v=nd->to,w=nd->w; r.comparisons++;
            if(dist[u]!=INF&&dist[u]+w<dist[v]){
                dist[v]=dist[u]+w; r.relaxations++;
                hpush(H,v,(long long)dist[v]-d_scan[v]); in_heap[v]=1;
            }
        }
    }
    r.time_ms=now_ms()-t0;
    hfree(H); free(d_scan); free(in_heap); free(scnt);
    return r;
}

/* ================================================================
 *  ALGORITHME 5a -- BF-ADAPT v1  [PROPOSITION ORIGINALE]
 *  Seuil : rho(t) = |A(t)| / n  (densite de NOEUDS actifs)
 *  SCAN si rho > 1/4, SPFA sinon.
 * ================================================================ */
Result bf_adapt_v1(const CSRGraph *csr, int src, int *dist) {
    Result r = {0}; int n=csr->n;
    dist_init(dist, n, src);
    const int THRESH=(n/4<1)?1:n/4;
    bool *active=calloc(n,sizeof(bool)),*active_new=calloc(n,sizeof(bool));
    bool *inq=calloc(n,sizeof(bool)); int *cnt=calloc(n,sizeof(int));
    int qcap=n>64?n:64, *Q=malloc(qcap*sizeof(int)), hd=0, tl=0;
    active[src]=true; int n_active=1, scan_passes=0;
    const int MAX_SCAN = (n < 2000) ? n : 2000;
    double t0=now_ms();
    while(n_active>0){
        if(n_active>THRESH){
            r.iters++;
            if(++scan_passes>MAX_SCAN){r.neg_cycle=true;break;}
            memset(active_new,0,n*sizeof(bool));
            int n_new=0; bool neg=false;
            for(int u=0;u<n;u++){
                for(int k=csr->offset[u];k<csr->offset[u+1];k++){
                    int v=csr->adj_v[k],w=csr->adj_w[k]; r.comparisons++;
                    if(dist[u]!=INF&&dist[u]+w<dist[v]){
                        dist[v]=dist[u]+w; r.relaxations++;
                        if(!active_new[v]){active_new[v]=true;n_new++;}
                        if(++cnt[v]>n){r.neg_cycle=true;neg=true;break;}
                    }
                }
                if(neg)break;
            }
            if(neg)break;
            bool *tmp=active;active=active_new;active_new=tmp;
            n_active=n_new;
        } else {
            hd=tl=0;
            for(int v=0;v<n;v++){
                if(!active[v])continue;
                if(tl>=qcap){qcap*=2;Q=realloc(Q,qcap*sizeof(int));}
                Q[tl++]=v;inq[v]=true;active[v]=false;
            }
            n_active=0; bool switched=false;
            while(hd<tl&&!switched){
                int u=Q[hd++];inq[u]=false;r.iters++;
                for(int k=csr->offset[u];k<csr->offset[u+1];k++){
                    int v=csr->adj_v[k],w=csr->adj_w[k];r.comparisons++;
                    if(dist[u]!=INF&&dist[u]+w<dist[v]){
                        dist[v]=dist[u]+w;r.relaxations++;
                        if(!inq[v]){
                            if(tl>=qcap){qcap*=2;Q=realloc(Q,qcap*sizeof(int));}
                            Q[tl++]=v;inq[v]=true;
                            if(++cnt[v]>n){r.neg_cycle=true;goto v1_end;}
                            if((tl-hd)>THRESH*3){
                                for(int i=hd;i<tl;i++){
                                    int x=Q[i];
                                    if(!active[x]){active[x]=true;n_active++;}
                                    inq[x]=false;
                                }
                                hd=tl=0;switched=true;break;
                            }
                        }
                    }
                }
            }
        }
    }
v1_end:
    r.time_ms=now_ms()-t0;
    free(active);free(active_new);free(inq);free(cnt);free(Q);
    return r;
}

/* ================================================================
 *  ALGORITHME 5b -- BF-ADAPT v2  [PROPOSITION -- REVISION]
 *
 *  THEORIE : Critere de commutation optimal (Theoreme 5.3)
 *
 *  Densite d'arcs actifs : epsilon(t) = Sigma_{u in A(t)} deg(u) / m
 *  != rho(t)*d_moy sur graphes heterogenes (ex. hubs financiers).
 *
 *  SCAN optimal si  epsilon(t) >= 1/eta
 *  SPFA optimal si  epsilon(t) <  1/eta
 *  eta = c_q/c_s = ratio de cout SPFA/SCAN, estime EN LIGNE.
 *
 *  Avantage sur v1 : detecte quand des hubs a haut degre sont actifs
 *  (epsilon >> rho) et evite SPFA inefficace dans ce cas.
 * ================================================================ */
Result bf_adapt_v2(const CSRGraph *csr, int src, int *dist) {
    Result r = {0}; int n=csr->n, m=csr->m;
    dist_init(dist, n, src);
    double eta=2.0, t_s=0, a_s=0, t_q=0, a_q=0;
    int thresh=m>0?(int)(m/eta):1; if(thresh<1)thresh=1;
    bool *active=calloc(n,sizeof(bool)),*active_new=calloc(n,sizeof(bool));
    bool *inq=calloc(n,sizeof(bool)); int *cnt=calloc(n,sizeof(int));
    int qcap=n>64?n:64, *Q=malloc(qcap*sizeof(int)), hd=0, tl=0;
    active[src]=true;
    int n_active=1, n_ae=csr->offset[src+1]-csr->offset[src], sp=0;
    const int MAX_SP = (n < 2000) ? n : 2000;
    double t0=now_ms();
    while(n_active>0){
        thresh=(int)(m/eta); if(thresh<1)thresh=1;
        if(n_ae>=thresh){
            r.iters++;
            if(++sp>MAX_SP){r.neg_cycle=true;break;}
            double ts=now_ms();
            memset(active_new,0,n*sizeof(bool));
            int n_new=0,n_new_ae=0; bool neg=false;
            for(int u=0;u<n;u++){
                for(int k=csr->offset[u];k<csr->offset[u+1];k++){
                    int v=csr->adj_v[k],w=csr->adj_w[k];r.comparisons++;
                    if(dist[u]!=INF&&dist[u]+w<dist[v]){
                        dist[v]=dist[u]+w;r.relaxations++;
                        if(!active_new[v]){
                            active_new[v]=true;n_new++;
                            n_new_ae+=csr->offset[v+1]-csr->offset[v];
                        }
                        if(++cnt[v]>n){r.neg_cycle=true;neg=true;break;}
                    }
                }
                if(neg)break;
            }
            if(neg)break;
            t_s+=now_ms()-ts; a_s+=(double)m;
            bool *tmp=active;active=active_new;active_new=tmp;
            n_active=n_new; n_ae=n_new_ae;
        } else {
            hd=tl=0;
            for(int v=0;v<n;v++){
                if(!active[v])continue;
                if(tl>=qcap){qcap*=2;Q=realloc(Q,qcap*sizeof(int));}
                Q[tl++]=v;inq[v]=true;active[v]=false;
            }
            n_active=0; n_ae=0; bool switched=false;
            double ts=now_ms(); long arcs_q=0;
            while(hd<tl&&!switched){
                int u=Q[hd++];inq[u]=false;r.iters++;
                arcs_q+=csr->offset[u+1]-csr->offset[u];
                for(int k=csr->offset[u];k<csr->offset[u+1];k++){
                    int v=csr->adj_v[k],w=csr->adj_w[k];r.comparisons++;
                    if(dist[u]!=INF&&dist[u]+w<dist[v]){
                        dist[v]=dist[u]+w;r.relaxations++;
                        if(!inq[v]){
                            if(tl>=qcap){qcap*=2;Q=realloc(Q,qcap*sizeof(int));}
                            Q[tl++]=v;inq[v]=true;
                            n_ae+=csr->offset[v+1]-csr->offset[v]; n_active++;
                            if(++cnt[v]>n){r.neg_cycle=true;goto v2_end;}
                            if(n_ae>=thresh){
                                for(int i=hd;i<tl;i++){
                                    int x=Q[i]; if(!active[x])active[x]=true;
                                    inq[x]=false;
                                }
                                hd=tl=0; switched=true; break;
                            }
                        }
                    }
                }
            }
            t_q+=now_ms()-ts; a_q+=(double)arcs_q;
            if(!switched){n_active=0;n_ae=0;}
        }
        if(a_s>1e5&&a_q>1e4){
            double cs=t_s/a_s, cq=t_q/a_q;
            if(cs>1e-15&&cq>0){
                eta=cq/cs;
                if(eta<1.0)eta=1.0;
                if(eta>8.0)eta=8.0;
            }
        }
    }
v2_end:
    r.time_ms=now_ms()-t0;
    free(active);free(active_new);free(inq);free(cnt);free(Q);
    return r;
}

/* ================================================================
 *  CHARGEMENT DE GRAPHES REELS
 *
 *  load_snap_road :
 *    roadNet-CA/PA/TX.txt -- reseau routier SNAP Stanford
 *    Format : "# commentaire" puis "u\tv" (tab-separe, 0-indexed)
 *    Poids assigns = 1 (graphe non pondere)
 *    Arcs BIDIRECTIONNELS (route = double sens)
 *    Statistiques reelles (roadNet-PA) :
 *      1,088,092 noeuds  |  1,541,898 arcs  |  degre moy 2.83
 *      Diametre ~786     |  clustering 0.046
 *
 *  load_snap_bitcoin :
 *    soc-sign-bitcoinalpha.csv  -- Bitcoin Alpha (SNAP)
 *    soc-sign-bitcoin-otc.csv   -- Bitcoin OTC   (SNAP)
 *    Format : "source,target,rating,timestamp" (CSV sans entete)
 *    Poids = rating in [-10, +10]
 *    Semantique : -10=mefiance totale, +10=confiance totale
 *    Statistiques Bitcoin Alpha :
 *      3,783 noeuds  |  24,186 arcs  |  ~22% d'arcs negatifs
 *    Statistiques Bitcoin OTC :
 *      5,881 noeuds  |  35,592 arcs  |  ~20% d'arcs negatifs
 *    Interet algorithmique :
 *      Graphe signe dirige reel avec poids negatifs.
 *      SSSP = chemin de mefiance minimale depuis un trader.
 *      Cycles negatifs possibles (mefiances mutuelles en cycle).
 * ================================================================ */

static int load_snap_road(const char *fname, EdgeGraph *eg, AdjGraph *ag) {
    FILE *f = fopen(fname, "r");
    if (!f) { printf("  [fichier non trouve: %s]\n", fname); return -1; }
    graph_reset(eg, ag);
    char line[256];
    int max_node = -1, loaded = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        int u, v;
        if (sscanf(line, "%d %d", &u, &v) != 2) continue;
        if (u < 0 || v < 0 || u >= MAX_V || v >= MAX_V) continue;
        if (eg->m + 2 > MAX_E - 2) break;
        add_edge(eg, ag, u, v, 1);
        add_edge(eg, ag, v, u, 1);   /* bidirectionnel */
        if (u > max_node) max_node = u;
        if (v > max_node) max_node = v;
        loaded += 2;
    }
    fclose(f);
    if (max_node >= 0) { eg->n = max_node + 1; ag->n = max_node + 1; }
    return loaded;
}

static int load_snap_bitcoin(const char *fname, EdgeGraph *eg, AdjGraph *ag) {
    FILE *f = fopen(fname, "r");
    if (!f) { printf("  [fichier non trouve: %s]\n", fname); return -1; }
    graph_reset(eg, ag);
    /* Remapping des IDs : Bitcoin OTC a des IDs non contigus jusqu'a ~7600 */
    static int id_map[MAX_V];
    memset(id_map, -1, sizeof(id_map));
    int n_nodes = 0;
    char line[256];
    int loaded = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        int u, v, w; long long ts;
        if (sscanf(line, "%d,%d,%d,%lld", &u, &v, &w, &ts) != 4) continue;
        if (u <= 0 || v <= 0 || u >= MAX_V || v >= MAX_V) continue;
        if (id_map[u] < 0) id_map[u] = n_nodes++;
        if (id_map[v] < 0) id_map[v] = n_nodes++;
        int nu = id_map[u], nv = id_map[v];
        if (eg->m >= MAX_E - 1) break;
        add_edge(eg, ag, nu, nv, w);
        loaded++;
    }
    fclose(f);
    return loaded;
}

/* ================================================================
 *  load_erc20_stablecoins
 *
 *  Dataset : ERC20 Stablecoins (SNAP Stanford, NeurIPS 2022)
 *  Ref     : Shamsi et al., Chartalist, NeurIPS 2022
 *  URL     : snap.stanford.edu/data/ERC20-stablecoins.html
 *  Fichier : token_transfers_V3.0.0.csv (le plus complet)
 *            ou V1/V2 pour des versions plus petites
 *  Periode : Avril 2022 - Novembre 2022 (crash Terra Luna)
 *  Tokens  : USDT, USDC, DAI, UST, PAX, WLUNA
 *
 *  Format CSV (sans entete) :
 *    from_address , to_address , token_address , value , block_number
 *    Les adresses sont des strings hex Ethereum : "0x1234...abcd"
 *
 *  Mapping des adresses hex -> entiers consecutifs
 *  par table de hachage legere (hash % HSIZE avec chaining).
 *
 *  Poids SSSP = -round(log10(value_normalized))
 *    value_normalized = value / 1e18  (wei -> ETH/token)
 *    Poids < 0 si transfert > 1 token  (profitable = chemin court)
 *    Poids > 0 si transfert < 1 token  (micropaiement = cout eleve)
 *    Interpretation : plus court chemin = flux de valeur maximal.
 *
 *  Cycles negatifs possibles si boucles de gros transferts.
 *  Interet : detecter les baleine et circuits d'arbitrage.
 *
 *  Note : 70M+ lignes -> on charge MAX_E arcs (limite structurelle).
 *  Pour V1 (~2M lignes) : chargement integral possible.
 * ================================================================ */

/* Table de hachage pour adresses Ethereum (hex strings) */
#define HEX_HSIZE  (1 << 17)   /* 131072 buckets                   */
#define HEX_HMASK  (HEX_HSIZE - 1)

typedef struct HexEntry {
    char              addr[43];  /* "0x" + 40 hex + '\0'            */
    int               id;
    struct HexEntry  *next;
} HexEntry;

static HexEntry  *hex_table[HEX_HSIZE];
static HexEntry   hex_pool[MAX_V];
static int        hex_pool_used = 0;
static int        hex_n_nodes   = 0;

static void hex_table_reset(void) {
    memset(hex_table, 0, sizeof(hex_table));
    hex_pool_used = 0;
    hex_n_nodes   = 0;
}

/* FNV-1a hash sur les 42 chars de l'adresse */
static unsigned hex_hash(const char *addr) {
    unsigned h = 2166136261u;
    for (int i = 0; i < 42 && addr[i]; i++) {
        h ^= (unsigned char)addr[i];
        h *= 16777619u;
    }
    return h & HEX_HMASK;
}

/* Retourne l'ID entier de l'adresse (cree si absent) */
static int hex_get_or_create(const char *addr) {
    unsigned slot = hex_hash(addr);
    for (HexEntry *e = hex_table[slot]; e; e = e->next)
        if (memcmp(e->addr, addr, 42) == 0) return e->id;
    if (hex_pool_used >= MAX_V) return -1;
    HexEntry *ne   = &hex_pool[hex_pool_used++];
    memcpy(ne->addr, addr, 42); ne->addr[42] = '\0';
    ne->id         = hex_n_nodes++;
    ne->next       = hex_table[slot];
    hex_table[slot]= ne;
    return ne->id;
}

#define ERC20_TOP_N  5000   /* taille du sous-graphe extrait */

/* ── Tokenisation d'une ligne ERC20 ─────────────────────────────
 * Retourne 1 si ok (from42, to42, val_str remplis), 0 sinon.     */
static int erc20_parse_line(char *line, char sep,
                             char from42[43], char to42[43],
                             char val_str[32]) {
    int ll = (int)strlen(line);
    while (ll>0 && (line[ll-1]=='\n'||line[ll-1]=='\r')) line[--ll]='\0';
    if (ll < 20 || strncmp(line,"block",5)==0) return 0;

    char *tok[10]={NULL}; int ntok=0; char *p=line;
    if (sep==' ') {
        while (ntok<10&&*p) {
            while(*p==' ')p++;
            if(!*p)break;
            tok[ntok++]=p;
            while(*p&&*p!=' ')p++;
            if(*p){*p='\0';p++;}
        }
    } else {
        while (ntok<10) {
            tok[ntok++]=p;
            char *q=strchr(p,sep);
            if(!q)break;
            *q='\0';p=q+1;
        }
    }
    if (ntok<7) return 0;

    /* Trouver les 2 premieres adresses 0x (from, to) */
    char *from=NULL, *to=NULL; int ac=0;
    for (int i=0;i<ntok;i++) {
        char *s=tok[i]; if(!s)continue;
        while(*s=='"')s++;
        int sl=(int)strlen(s);
        while(sl>0&&s[sl-1]=='"'){s[sl-1]='\0';sl--;}
        if (s[0]=='0'&&(s[1]=='x'||s[1]=='X')&&sl>=10) {
            ac++;
            if (ac==1) from=s;
            else if (ac==2) { to=s; break; }
        }
    }
    if (!from||!to) return 0;

    /* value = derniere colonne */
    char *vs=tok[ntok-1];
    while(*vs=='"')vs++;
    int vl=(int)strlen(vs);
    while(vl>0&&vs[vl-1]=='"'){vs[vl-1]='\0';vl--;}

    strncpy(from42,from,42); from42[42]='\0';
    strncpy(to42,  to,  42); to42[42]='\0';
    strncpy(val_str,vs,31);  val_str[31]='\0';
    return 1;
}

/* ── Detecter le separateur sur la 1ere ligne de donnees ──────── */
static char erc20_detect_sep(FILE *f) {
    char line[2048]; char sep=',';
    while (fgets(line, sizeof(line), f)) {
        int ll=(int)strlen(line);
        while(ll>0&&(line[ll-1]=='\n'||line[ll-1]=='\r'))line[--ll]='\0';
        if (ll==0||strncmp(line,"block",5)==0) continue;
        int nc=0,nt=0,ns=0;
        for(int i=0;i<ll;i++){
            if(line[i]==',')nc++;
            else if(line[i]=='\t')nt++;
            else if(line[i]==' ')ns++;
        }
        sep=(nt>=5)?'\t':(nc>=5)?',':(ns>=5)?' ':',';
        printf("     '%s' sep='%c' ex:%.60s\n",
               (sep=='\t')?"TAB":(sep==',')?"CSV":"SPC", sep, line);
        break;
    }
    return sep;
}

/* ── Passe 1 sur un fichier : incrementer les compteurs ────────── */
static void erc20_pass1(FILE *f, char sep) {
    char line[2048], from42[43], to42[43], val_str[32];
    while (fgets(line, sizeof(line), f)) {
        if (!erc20_parse_line(line, sep, from42, to42, val_str)) continue;

        unsigned sf=hex_hash(from42);
        HexEntry *ef=NULL;
        for(ef=hex_table[sf];ef;ef=ef->next)
            if(memcmp(ef->addr,from42,42)==0)break;
        if(!ef && hex_pool_used<MAX_V){
            ef=&hex_pool[hex_pool_used++];
            memcpy(ef->addr,from42,42);ef->addr[42]='\0';
            ef->id=0;ef->next=hex_table[sf];hex_table[sf]=ef;
        }
        if(ef) ef->id++;

        unsigned st=hex_hash(to42);
        HexEntry *et=NULL;
        for(et=hex_table[st];et;et=et->next)
            if(memcmp(et->addr,to42,42)==0)break;
        if(!et && hex_pool_used<MAX_V){
            et=&hex_pool[hex_pool_used++];
            memcpy(et->addr,to42,42);et->addr[42]='\0';
            et->id=0;et->next=hex_table[st];hex_table[st]=et;
        }
        if(et) et->id++;
    }
}

/* ── Passe 2 sur un fichier : charger les arcs hub-hub ─────────── */
static int erc20_pass2(FILE *f, char sep, EdgeGraph *eg, AdjGraph *ag) {
    char line[2048], from42[43], to42[43], val_str[32];
    int loaded=0;
    while (fgets(line, sizeof(line), f)) {
        if (!erc20_parse_line(line, sep, from42, to42, val_str)) continue;
        if (eg->m>=MAX_E-1) break;

        unsigned sf=hex_hash(from42);
        int nu=-1;
        for(HexEntry *e=hex_table[sf];e;e=e->next)
            if(memcmp(e->addr,from42,42)==0){nu=e->id;break;}

        unsigned st=hex_hash(to42);
        int nv=-1;
        for(HexEntry *e=hex_table[st];e;e=e->next)
            if(memcmp(e->addr,to42,42)==0){nv=e->id;break;}

        if(nu<0||nv<0||nu==nv) continue;

        double val=atof(val_str);
        int w=(val<=0.0)?20:-(int)round(log10(val));
        if(w<-20)w=-20; if(w>20)w=20;

        add_edge(eg, ag, nu, nv, w);
        loaded++;
    }
    return loaded;
}

/* ── Fonction principale : charge N fichiers ERC20 ensemble ─────── */
static int load_erc20_multi(const char **fnames, int nfiles,
                             EdgeGraph *eg, AdjGraph *ag) {
    /* Ouvrir tous les fichiers */
    FILE **fps = malloc(nfiles * sizeof(FILE*));
    char  *seps= malloc(nfiles * sizeof(char));
    int    n_open = 0;

    for(int i=0;i<nfiles;i++){
        fps[i] = fopen(fnames[i], "r");
        if(!fps[i]){
            printf("  [absent: %s]\n", fnames[i]);
            seps[i]=',';
        } else {
            printf("\n  -> %s : ", fnames[i]);
            fflush(stdout);
            seps[i] = erc20_detect_sep(fps[i]);
            n_open++;
        }
    }
    if(n_open==0){ free(fps); free(seps); return -1; }

    graph_reset(eg, ag);

    /* ── PASSE 1 : compter sur tous les fichiers ouverts ─────────── */
    printf("     Passe 1/2: comptage transactions par adresse...\n");
    fflush(stdout);

    for(int i=0;i<HEX_HSIZE;i++) hex_table[i]=NULL;
    hex_pool_used=0; hex_n_nodes=0;

    long long total=0;
    for(int i=0;i<nfiles;i++){
        if(!fps[i]) continue;
        rewind(fps[i]);
        long long before = hex_pool_used;
        erc20_pass1(fps[i], seps[i]);
        long long after = hex_pool_used;
        printf("     %s : %lld adresses nouvelles (cumul=%d)\n",
               fnames[i], after-before, hex_pool_used);
        total += after-before;
    }
    int n_unique = hex_pool_used;
    printf("     Total : %d adresses uniques\n", n_unique);

    /* ── Selectionner le top-ERC20_TOP_N ──────────────────────────── */
    int top_n = (n_unique < ERC20_TOP_N) ? n_unique : ERC20_TOP_N;

    int *counts = malloc(n_unique * sizeof(int));
    for(int i=0;i<n_unique;i++) counts[i]=hex_pool[i].id;

    int cmp_desc(const void *a, const void *b){
        return *(const int*)b - *(const int*)a;
    }
    qsort(counts, n_unique, sizeof(int), cmp_desc);
    int threshold = counts[top_n-1];
    free(counts);

    /* Reassigner les IDs finaux */
    int n_sel=0;
    for(int i=0;i<n_unique;i++){
        if(hex_pool[i].id>=threshold && n_sel<ERC20_TOP_N)
            hex_pool[i].id=n_sel++;
        else
            hex_pool[i].id=-1;
    }
    printf("     Top-%d selectionnes (seuil: >= %d transactions)\n",
           n_sel, threshold);

    /* ── PASSE 2 : arcs entre hubs sur tous les fichiers ──────────── */
    printf("     Passe 2/2: chargement des arcs hub-hub...\n");
    eg->n=n_sel; ag->n=n_sel;
    int total_arcs=0;

    for(int i=0;i<nfiles;i++){
        if(!fps[i]) continue;
        rewind(fps[i]);
        int a = erc20_pass2(fps[i], seps[i], eg, ag);
        printf("     %s : %d arcs hub-hub\n", fnames[i], a);
        total_arcs += a;
        fclose(fps[i]);
    }
    free(fps); free(seps);
    return total_arcs;
}
static int load_ethereum_exchanges(const char *fname,
                                   EdgeGraph *eg, AdjGraph *ag) {
    FILE *f = fopen(fname, "r");
    if (!f) { printf("  [fichier non trouve: %s]\n", fname); return -1; }
    graph_reset(eg, ag);
    hex_table_reset();

    char line[1024];
    int  loaded = 0;
    char sep = ',';
    bool header_seen = false;

    while (fgets(line, sizeof(line), f)) {
        size_t ll = strlen(line);
        while (ll > 0 && (line[ll-1]=='\n'||line[ll-1]=='\r')) line[--ll]='\0';
        if (ll == 0 || line[0]=='#') continue;

        if (!header_seen) {
            header_seen = true;
            int ntabs=0, ncommas=0;
            for (size_t i=0; i<ll; i++) {
                if (line[i]=='\t') ntabs++;
                if (line[i]==',')  ncommas++;
            }
            sep = (ntabs > ncommas) ? '\t' : ',';
            if (line[0]!='0') continue;
        }

        /* Tokenisation manuelle */
        char *fields[8] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
        int   nf = 0;
        char *p = line;
        while (nf < 8) {
            fields[nf++] = p;
            char *q = strchr(p, sep);
            if (!q) break;
            *q = '\0'; p = q + 1;
        }
        if (nf < 3) continue;

        /* Retirer guillemets */
        for (int i=0; i<nf; i++) {
            char *s = fields[i]; size_t sl = strlen(s);
            if (sl >= 2 && s[0]=='"' && s[sl-1]=='"') { s[sl-1]='\0'; fields[i]++; }
        }

        char *from    = fields[0];
        char *to      = fields[1];
        char *val_str = (nf >= 3) ? fields[2] : "0";

        if (strlen(from)<4 || strlen(to)<4) continue;
        /* Adresses Ethereum hex OU IDs entiers */
        int nu = hex_get_or_create(from);
        int nv = hex_get_or_create(to);
        if (nu < 0 || nv < 0 || nu == nv) continue;
        if (eg->m >= MAX_E - 1) break;

        /* Poids identique a ERC20 */
        int w;
        if (val_str[0]=='0' && (val_str[1]=='x'||val_str[1]=='X')) {
            size_t hexlen = strlen(val_str) - 2;
            int dec_equiv = (int)(hexlen * 10 / 12);
            w = (dec_equiv > 18) ? -(dec_equiv - 18) : (int)(18 - dec_equiv);
        } else {
            size_t vlen = strlen(val_str);
            if (vlen==0||val_str[0]=='0') w=10;
            else if (vlen>18) w=-(int)(vlen-18);
            else              w= (int)(18-vlen);
        }
        if (w < -20) w=-20; if (w > 20) w=20;

        add_edge(eg, ag, nu, nv, w);
        loaded++;
    }
    fclose(f);
    return loaded;
}

/* ================================================================
 *  GENERATEURS SYNTHETIQUES
 * ================================================================ */
static void gen_random(EdgeGraph *eg, AdjGraph *ag,
                       int n, int m, int wmin, int wmax, unsigned seed) {
    graph_reset(eg,ag); eg->n=n; ag->n=n; srand(seed); int tries=0;
    while(eg->m<m && tries<m*20){tries++;
        int u=rand()%n,v=rand()%n; if(u==v)continue;
        add_edge(eg,ag,u,v,wmin+rand()%(wmax-wmin+1));}
}
static void gen_grid(EdgeGraph *eg, AdjGraph *ag, int side) {
    int n=side*side; graph_reset(eg,ag); eg->n=n; ag->n=n; srand(7);
    int dx[]={0,1,0,-1},dy[]={1,0,-1,0};
    for(int r=0;r<side;r++) for(int c=0;c<side;c++){
        int u=r*side+c;
        for(int d=0;d<4;d++){int nr=r+dx[d],nc=c+dy[d];
            if(nr<0||nr>=side||nc<0||nc>=side)continue;
            add_edge(eg,ag,u,nr*side+nc,1+rand()%20);}
    }
}
static void gen_dense(EdgeGraph *eg, AdjGraph *ag, int n) {
    graph_reset(eg,ag); eg->n=n; ag->n=n; srand(99);
    for(int u=0;u<n&&eg->m<MAX_E-n;u++) for(int v=0;v<n;v++){
        if(u==v||eg->m>=MAX_E-1)continue; add_edge(eg,ag,u,v,1+rand()%100);}
}
static void gen_neg_weights(EdgeGraph *eg, AdjGraph *ag, int n, int m) {
    graph_reset(eg,ag); eg->n=n; ag->n=n; srand(123);
    int *h=malloc(n*sizeof*h);
    for(int i=0;i<n;i++) h[i]=rand()%51;
    for(int i=1;i<n;i++){int u=rand()%i,w=(1+rand()%5)+h[u]-h[i];add_edge(eg,ag,u,i,w);}
    int tries=0;
    while(eg->m<m && tries<m*10){tries++;
        int u=rand()%n,v=rand()%n; if(u==v)continue;
        add_edge(eg,ag,u,v,(1+rand()%5)+h[u]-h[v]);}
    free(h);
}
static void gen_road_synth(EdgeGraph *eg, AdjGraph *ag, int n) {
    graph_reset(eg,ag); eg->n=n; ag->n=n; srand(17);
    int *px=malloc(n*sizeof*px),*py=malloc(n*sizeof*py);
    for(int i=0;i<n;i++){px[i]=rand()%1000;py[i]=rand()%1000;}
    for(int u=0;u<n&&eg->m+4<MAX_E;u++){
        int best[4]={-1,-1,-1,-1}; long long bd[4]={LLONG_MAX,LLONG_MAX,LLONG_MAX,LLONG_MAX};
        for(int v=0;v<n;v++){if(v==u)continue;
            long long dx=px[u]-px[v],dy=py[u]-py[v],d2=dx*dx+dy*dy;
            for(int k=0;k<4;k++){if(d2<bd[k]){for(int j=3;j>k;j--){best[j]=best[j-1];bd[j]=bd[j-1];}
                best[k]=v;bd[k]=d2;break;}}}
        for(int k=0;k<4;k++){if(best[k]<0)continue;
            add_edge(eg,ag,u,best[k],(int)(sqrt((double)bd[k])/10.0)+1);}}
    free(px);free(py);
}
static void gen_chain(EdgeGraph *eg, AdjGraph *ag, int n) {
    graph_reset(eg,ag); eg->n=n; ag->n=n;
    for(int i=n-2;i>=0;i--) add_edge(eg,ag,i,i+1,1);
}

/* ================================================================
 *  GENERATEUR FINANCIER 1 -- RESEAU FOREX MULTI-DEVISES
 *
 *  Modele : marche de change FX inter-bancaire.
 *  Ref    : Bertimas & Lo (1998), Journal of Financial Markets.
 *           Avellaneda & Stoikov (2008), Quant Finance.
 *
 *  Structure :
 *    Noeuds = (devise i, slot temporel t)
 *             i in [0, n_ccy-1], t in [0, n_slots-1]
 *    Noeud u = t * n_ccy + i
 *
 *    Arc temporel   : (i,t) -> (i,t+1), poids 0
 *                     (conserver la devise d'un slot au suivant)
 *
 *    Arc de change  : (i,t) -> (j,t+1), poids w_ij
 *                     (convertir devise i en devise j au slot t)
 *    w_ij = round(-1000 * log(rate_ij))
 *    Poids negatif  <=> rate > 1.0 (echange profitable)
 *
 *  Taux calibres sur les taux BCE 2024 (8 devises majeures) :
 *    USD=0, EUR=1, GBP=2, JPY=3, CHF=4, CAD=5, AUD=6, NZD=7
 *    puis n_ccy-8 devises emergentes autour des majors.
 *
 *  Absence de cycle negatif garantie :
 *    Construction par potentiels de Johnson.
 *    Poids reduit w_r(i,j) = w_ij + h[i] - h[j] >= 0
 *    => pas d'opportunite d'arbitrage sans fin.
 *    Bruit intraday positif seul (0 a +30 pts de base).
 *
 *  Proprietes resultantes :
 *    ~25-35% d'arcs negatifs (conversions profitables)
 *    Degre moyen des hubs majors : n_slots * n_ccy
 *    Structure hub-and-spoke : 8 majors tres connectes
 *    Diametre : ~n_slots (progression temporelle)
 *
 *  Interet SSSP : trouver le chemin de conversion optimal
 *  depuis USD au slot 0 vers toutes les devises a tous les slots.
 * ================================================================ */
static void gen_forex(EdgeGraph *eg, AdjGraph *ag,
                      int n_ccy, int n_slots, unsigned seed) {
    int n = n_ccy * n_slots;
    if (n > MAX_V) { n_slots = MAX_V / n_ccy; n = n_ccy * n_slots; }
    graph_reset(eg, ag); eg->n = n; ag->n = n;
    srand(seed);

    /* Potentiels de Johnson : h[i] = log-prix relatif a USD * 1000    */
    /* Valeurs calibrees sur taux BCE 2024 (arrondis a l'entier)        */
    static const int h_major[8] = { 0, 82, -234, -500, 103, -309, 417, 483 };
    /* Matrice de base entre majors : w_base[i][j] = h[j] - h[i]       */
    /* (no-arbitrage exact, puis bruit positif ajoute)                  */
    int *h = malloc(n_ccy * sizeof(int));
    int N_MAJ = (n_ccy >= 8) ? 8 : n_ccy;
    for (int i = 0; i < N_MAJ; i++) h[i] = h_major[i];
    /* Devises emergentes : potentiel aleatoire calibre entre -600/+600  */
    for (int i = N_MAJ; i < n_ccy; i++) {
        int parent = rand() % N_MAJ;
        /* Potentiel proche d'un major avec bruit regionnal             */
        h[i] = h[parent] + (rand() % 201) - 100;
    }

    /* Arcs temporels : (i,t) -> (i,t+1), poids 0                      */
    for (int t = 0; t < n_slots-1; t++)
        for (int i = 0; i < n_ccy && eg->m < MAX_E-1; i++)
            add_edge(eg, ag, t*n_ccy+i, (t+1)*n_ccy+i, 0);

    /* Arcs de change intra-slot ----------------------------------------*/
    for (int t = 0; t < n_slots-1 && eg->m < MAX_E - n_ccy*n_ccy; t++) {
        /* Majors <-> majors : marche interbancaire complet              */
        for (int i = 0; i < N_MAJ; i++)
        for (int j = 0; j < N_MAJ; j++) {
            if (i == j) continue;
            /* Poids = h[j]-h[i] (no-arbitrage) + bruit positif 0..30  */
            int w = (h[j] - h[i]) + (rand() % 31);
            if (eg->m < MAX_E-1)
                add_edge(eg, ag, t*n_ccy+i, t*n_ccy+j, w);
        }
        /* Emergentes : connectees a 2-3 majors (banque correspondante)  */
        for (int i = N_MAJ; i < n_ccy; i++) {
            int n_banks = 2 + rand() % 2;  /* 2 ou 3 banques corres.   */
            for (int b = 0; b < n_banks; b++) {
                int j = rand() % N_MAJ;
                /* (i->j) */
                int wij = (h[j]-h[i]) + (rand()%31);
                /* (j->i) */
                int wji = (h[i]-h[j]) + (rand()%31);
                if (eg->m < MAX_E-1) add_edge(eg,ag, t*n_ccy+i, t*n_ccy+j, wij);
                if (eg->m < MAX_E-1) add_edge(eg,ag, t*n_ccy+j, t*n_ccy+i, wji);
            }
        }
    }
    free(h);
}

/* ================================================================
 *  GENERATEUR FINANCIER 2 -- RESEAU DE CONFIANCE CRYPTO (TRUST)
 *
 *  Modele : reseau de reputation/mefiance entre traders crypto.
 *  Inspire de : Bitcoin Alpha / OTC (SNAP, Kumar et al. 2016)
 *               soc-sign-bitcoinalpha / soc-sign-bitcoin-otc
 *
 *  Structure power-law (loi de puissance) :
 *    Degre d'un noeud suit P(k) ~ k^(-gamma), gamma~2.3
 *    (conforme aux reseaux Bitcoin reels mesures par Kumar)
 *    Quelques hubs tres connectes (grands traders/exchanges)
 *    Majorite de petits traders (degre 1-5)
 *
 *  Poids in [-10, +10] :
 *    Positif = confiance (+1 a +10, avec clustering par communaute)
 *    Negatif = mefiance (-1 a -10, traders frauduleux/defaillants)
 *    ~25% d'arcs negatifs (conforme aux stats SNAP)
 *
 *  Cycles negatifs POSSIBLES :
 *    Si trois traders se melient mutuellement -> cycle negatif
 *    Detecte et signale par tous les algos
 *    Cas algorithmique difficile pour BF classique (n passes)
 *    BF-ADAPT v2 detecte le cycle TRES rapidement via le seuil arcs
 *
 *  Generation :
 *    1. Attribution des degres par loi de puissance (BA model)
 *    2. Connexions preferentielles (attachment proportionnel au degre)
 *    3. Poids : 75% positif aleatoire, 25% negatif aleatoire
 *    4. Quelques "fraudes en cercle" (triangles negatifs) injectes
 * ================================================================ */
static void gen_crypto_trust(EdgeGraph *eg, AdjGraph *ag,
                             int n, int m_target, unsigned seed) {
    graph_reset(eg, ag); eg->n = n; ag->n = n;
    srand(seed);

    /* Tableau de degres cumules pour l'attachement preferentiel        */
    int *deg = calloc(n, sizeof(int));

    /* Phase 1 : graphe de base (quelques hubs initiaux)                */
    int n_hubs = (n / 50 < 5) ? 5 : n / 50;  /* ~2% de hubs           */
    for (int i = 0; i < n_hubs && i < n; i++) {
        deg[i] = 20 + rand() % 80;  /* hubs : degre initial 20-100     */
    }
    for (int i = n_hubs; i < n; i++) {
        deg[i] = 1 + rand() % 4;    /* petits traders : degre 1-4      */
    }

    /* Phase 2 : generation des arcs                                    */
    int total_deg = 0;
    for (int i = 0; i < n; i++) total_deg += deg[i];

    int loaded = 0;
    int tries  = 0;
    while (loaded < m_target && tries < m_target * 10) {
        tries++;
        if (eg->m >= MAX_E - 1) break;

        /* Choisir u proportionnellement a son degre (attachement pref.) */
        int target_u = rand() % total_deg;
        int u = 0, cum = 0;
        while (u < n-1 && cum + deg[u] <= target_u) { cum += deg[u]; u++; }

        /* Choisir v aleatoirement parmi les autres                     */
        int v = rand() % n;
        if (v == u) { v = (v + 1) % n; }

        /* Poids : 75% positif, 25% negatif                            */
        int w;
        if (rand() % 100 < 25)
            w = -(1 + rand() % 10);   /* mefiance : -1 a -10           */
        else
            w = 1 + rand() % 10;      /* confiance : +1 a +10          */

        add_edge(eg, ag, u, v, w);
        loaded++;
    }

    /* Phase 3 : injecter quelques triangles negatifs (fraudes circulaires)
     * Cree des cycles negatifs realistes (3 traders se melient mutuellement)
     * Proportion : ~1% des noeuds impliques                            */
    int n_fraud_triangles = n / 100;
    for (int t = 0; t < n_fraud_triangles && eg->m + 3 < MAX_E; t++) {
        int a = rand() % n;
        int b = rand() % n; if (b == a) b = (b+1)%n;
        int c = rand() % n; if (c == a || c == b) c = (c+2)%n;
        int fw = -(3 + rand() % 5);  /* mefiance forte : -3 a -7       */
        add_edge(eg, ag, a, b, fw);
        add_edge(eg, ag, b, c, fw);
        add_edge(eg, ag, c, a, fw);
    }
    free(deg);
}


static int  d1[MAX_V], d2[MAX_V], d3[MAX_V], d4[MAX_V], d5[MAX_V], d6[MAX_V];
static CSRGraph csr_g;

static bool check6(int n) {
    for(int i=0;i<n;i++)
        if(d1[i]!=d2[i]||d1[i]!=d3[i]||d1[i]!=d4[i]||
           d1[i]!=d5[i]||d1[i]!=d6[i]) return false;
    return true;
}

static void run(const char *name, EdgeGraph *eg, AdjGraph *ag, int src) {
    build_csr(eg, &csr_g);
    int n = eg->n, m = eg->m;

    Result r1 = bf_classic   (eg,     src, d1);
    Result r2 = spfa         (ag,     src, d2);
    Result r3 = bf_randomized(eg,     src, d3);
    Result r4 = bf_potential (ag,     src, d4);
    Result r5 = bf_adapt_v1  (&csr_g, src, d5);
    Result r6 = bf_adapt_v2  (&csr_g, src, d6);

    bool ok = true;
    for(int i=0;i<n;i++)
        if(d1[i]!=d2[i]||d1[i]!=d3[i]||d1[i]!=d4[i]||
           d1[i]!=d5[i]||d1[i]!=d6[i]){ok=false;break;}

    bool all_neg = r1.neg_cycle&&r2.neg_cycle&&r3.neg_cycle&&
                   r4.neg_cycle&&r5.neg_cycle&&r6.neg_cycle;
    if (all_neg) ok = true;

    double best = r1.time_ms;
    if(r2.time_ms<best)best=r2.time_ms;
    if(r3.time_ms<best)best=r3.time_ms;
    if(r4.time_ms<best)best=r4.time_ms;
    if(r5.time_ms<best)best=r5.time_ms;
    if(r6.time_ms<best)best=r6.time_ms;

#define MARK(t) ((t)==best?" <<":"   ")

    printf("\n+------------------------------------------------------------------------------+\n");
    printf("|  %-42s  n=%-6d m=%-6d      |\n", name, n, m);
    if (all_neg)
        printf("|  %-74s|\n","Cycles negatifs -- SSSP indefini, divergence attendue");
    else
        printf("|  Coherence : %-60s|\n", ok?"OK":"DIVERGENCE ! (voir stderr)");
    printf("+----------------------------------+----------+------------+----------+-----+\n");
    printf("| Algorithme                       |Temps(ms) |Relaxations |Iters     | Neg |\n");
    printf("+----------------------------------+----------+------------+----------+-----+\n");
    printf("| BF Classique  [Bellman-Ford 58]  |%6.3f%-3s|%12lld|%10d| %-3s |\n",
           r1.time_ms,MARK(r1.time_ms),r1.relaxations,r1.iters,r1.neg_cycle?"OUI":"non");
    printf("| SPFA          [Moore/Duan 94]    |%6.3f%-3s|%12lld|%10d| %-3s |\n",
           r2.time_ms,MARK(r2.time_ms),r2.relaxations,r2.iters,r2.neg_cycle?"OUI":"non");
    printf("| BF Randomise  [Bannister 12]    |%6.3f%-3s|%12lld|%10d| %-3s |\n",
           r3.time_ms,MARK(r3.time_ms),r3.relaxations,r3.iters,r3.neg_cycle?"OUI":"non");
    printf("| BF Potentiel  [Cantone 19]       |%6.3f%-3s|%12lld|%10d| %-3s |\n",
           r4.time_ms,MARK(r4.time_ms),r4.relaxations,r4.iters,r4.neg_cycle?"OUI":"non");
    printf("| BF-ADAPT v1   [rho=|A|/n,tau=1/4]|%5.3f%-3s|%12lld|%10d| %-3s |\n",
           r5.time_ms,MARK(r5.time_ms),r5.relaxations,r5.iters,r5.neg_cycle?"OUI":"non");
    printf("| BF-ADAPT v2   [eps=arcs/m,eta*]  |%5.3f%-3s|%12lld|%10d| %-3s |\n",
           r6.time_ms,MARK(r6.time_ms),r6.relaxations,r6.iters,r6.neg_cycle?"OUI":"non");
    printf("+----------------------------------+----------+------------+----------+-----+\n");

    if(!ok&&!all_neg){
        fprintf(stderr,"[ERREUR] Divergence '%s'\n",name);
        int lim=n<6?n:6;
        for(int i=0;i<lim;i++)
            fprintf(stderr,"  d[%d] BF=%d SPFA=%d RAND=%d POT=%d V1=%d V2=%d\n",
                    i,d1[i],d2[i],d3[i],d4[i],d5[i],d6[i]);
    }
}

/* ================================================================
 *  MAIN
 * ================================================================ */
int main(int argc, char *argv[]) {
    static EdgeGraph eg;
    static AdjGraph  ag;

    /* Trier les arguments par extension */
    const char *road_files[3]  = {NULL,NULL,NULL};
    const char *road_labels[3] = {"roadNet-CA","roadNet-PA","roadNet-TX"};
    const char *btc_files[2]   = {NULL,NULL};
    const char *btc_labels[2]  = {"Bitcoin Alpha (SNAP)","Bitcoin OTC (SNAP)"};
    const char *erc20_files[3] = {NULL,NULL,NULL};  /* V1, V2, V3            */
    const char *eth_ex_files[4]= {NULL,NULL,NULL,NULL}; /* jusqu'a 4 tokens  */
    int ri=0, bi=0, ei=0, xi=0;

    for(int a=1; a<argc; a++){
        const char *p=argv[a]; size_t l=strlen(p);
        /* Routiers : *.txt */
        if(l>4&&strcmp(p+l-4,".txt")==0&&ri<3) { road_files[ri++]=p; continue; }
        /* Bitcoin : contient "bitcoin" ou "sign" */
        if(strstr(p,"bitcoin")!=NULL||strstr(p,"sign-")!=NULL) {
            if(bi<2) { btc_files[bi++]=p; continue; }
        }
        /* ERC20 stablecoins : contient "token_transfers" ou "stablecoin" */
        if(strstr(p,"token_transfers")!=NULL||strstr(p,"stablecoin")!=NULL) {
            if(ei<3) { erc20_files[ei++]=p; continue; }
        }
        /* Ethereum exchanges : tout autre .csv non-bitcoin */
        if(l>4&&strcmp(p+l-4,".csv")==0&&xi<4) { eth_ex_files[xi++]=p; continue; }
    }

    printf("\n");
    printf("+================================================================================+\n");
    printf("|  BENCHMARK BELLMAN-FORD -- 6 ALGORITHMES                                      |\n");
    printf("|  BF Classic | SPFA | BF Rand | BF Pot | BF-ADAPT v1 | BF-ADAPT v2           |\n");
    printf("|  v1: seuil rho(t)=|A|/n (noeuds), tau=1/4 fixe                              |\n");
    printf("|  v2: seuil eps(t)=arcs_A/m (arcs), tau*=m/eta adaptatif en ligne            |\n");
    printf("|  << = meilleur temps                                                          |\n");
    printf("+================================================================================+\n");

    /* ── Section 1 : Graphes synthetiques classiques ────────────── */
    printf("\n--- [1/6] GRAPHES SYNTHETIQUES CLASSIQUES ---\n");

    gen_random(&eg,&ag, 800,4000,1,100,42);
    run("Sparse (n=800, m=4000, w>0)", &eg,&ag, 0); adjgraph_free(&ag);

    gen_random(&eg,&ag, 3000,15000,1,100,42);
    run("Sparse grand (n=3000, m=15000, w>0)", &eg,&ag, 0); adjgraph_free(&ag);

    gen_grid(&eg,&ag, 25);
    run("Grille 25x25 (n=625)", &eg,&ag, 0); adjgraph_free(&ag);

    gen_grid(&eg,&ag, 60);
    run("Grille 60x60 (n=3600)", &eg,&ag, 0); adjgraph_free(&ag);

    gen_dense(&eg,&ag, 200);
    run("Dense (n=200, m~40000)", &eg,&ag, 0); adjgraph_free(&ag);

    gen_neg_weights(&eg,&ag, 800,5000);
    run("Poids negatifs (Johnson, sans cycle negatif)", &eg,&ag, 0); adjgraph_free(&ag);

    gen_road_synth(&eg,&ag, 1500);
    run("Routier synthetique (4-voisins, n=1500)", &eg,&ag, 0); adjgraph_free(&ag);

    gen_chain(&eg,&ag, 1500);
    run("Chaine inversee (pire cas BF, n=1500)", &eg,&ag, 0); adjgraph_free(&ag);

    /* ── Section 2 : Reseaux routiers reels ────────────────────── */
    printf("\n--- [2/6] RESEAUX ROUTIERS REELS (SNAP Stanford) ---\n");
    printf("    Ref: Leskovec & Krevl, SNAP, snap.stanford.edu/data\n");
    printf("    Format: u<TAB>v (non pondere, arcs bidirectionnels charges)\n");
    printf("    Limite: %d noeuds max, %d arcs max\n", MAX_V, MAX_E);
    printf("    Telecharger: snap.stanford.edu/data/roadNet-CA.txt.gz\n");
    printf("    Usage: ./bf_bench roadNet-CA.txt roadNet-PA.txt roadNet-TX.txt\n");

    bool any_road = false;
    for(int i=0;i<3;i++){
        const char *fname = road_files[i];
        if(!fname){
            /* Essayer le nom par defaut dans le rep. courant */
            static char defs[3][32];
            snprintf(defs[i],sizeof(defs[i]),"%s.txt",
                     i==0?"roadNet-CA":i==1?"roadNet-PA":"roadNet-TX");
            fname=defs[i];
        }
        printf("\n  -> Chargement %s (%s)... ", road_labels[i], fname);
        fflush(stdout);
        int ml = load_snap_road(fname, &eg, &ag);
        if(ml<0) continue;
        any_road=true;
        printf("%d noeuds, %d arcs\n", eg.n, eg.m);
        /* Fix source : SNAP roadNet trie les arcs par ID source croissant.
         * eg.n/2 n'apparait qu'en DESTINATION dans la fraction chargee.
         * Noeud 0 est toujours source dans les premieres lignes -> OK.  */
        run(road_labels[i], &eg, &ag, 0);
        adjgraph_free(&ag);
    }
    if(!any_road)
        printf("\n  Aucun fichier routier trouve (voir instructions ci-dessus).\n");

    /* ── Section 3 : Reseaux financiers Bitcoin ─────────────────── */
    printf("\n--- [3/6] RESEAUX FINANCIERS BITCOIN (SNAP Stanford) ---\n");
    printf("    Ref: Kumar et al., Edge Weight Prediction in Weighted Signed\n");
    printf("         Networks, IEEE ICDM 2016. doi:10.1109/ICDM.2016.0075\n");
    printf("    URL Alpha: snap.stanford.edu/data/soc-sign-bitcoin-alpha.html\n");
    printf("    URL OTC  : snap.stanford.edu/data/soc-sign-bitcoin-otc.html\n");
    printf("    Format: source,target,rating,timestamp (rating in [-10,+10])\n");
    printf("    Semantique SSSP: chemin de mefiance minimale entre traders\n");
    printf("    Usage: ./bf_bench soc-sign-bitcoinalpha.csv soc-sign-bitcoin-otc.csv\n");

    bool any_btc = false;
    for(int i=0;i<2;i++){
        const char *fname=btc_files[i];
        if(!fname){
            static char defs2[2][48];
            snprintf(defs2[i],sizeof(defs2[i]),"%s",
                     i==0?"soc-sign-bitcoinalpha.csv":"soc-sign-bitcoin-otc.csv");
            fname=defs2[i];
        }
        printf("\n  -> Chargement %s (%s)... ", btc_labels[i], fname);
        fflush(stdout);
        int ml = load_snap_bitcoin(fname, &eg, &ag);
        if(ml<0) continue;
        any_btc=true;
        printf("%d noeuds, %d arcs\n", eg.n, eg.m);
        run(btc_labels[i], &eg, &ag, 0);
        adjgraph_free(&ag);
    }
    if(!any_btc)
        printf("\n  Aucun fichier Bitcoin trouve.\n");

    /* ── Section 4 : ERC20 Stablecoins REEL ─────────────────────── */
    printf("\n--- [4/6] ERC20 STABLECOINS REEL (SNAP / NeurIPS 2022) ---\n");
    printf("    Ref: Shamsi et al., Chartalist, NeurIPS 2022\n");
    printf("    URL: snap.stanford.edu/data/ERC20-stablecoins.html\n");
    printf("    Format  : block_number,tx_idx,from,to,timestamp,contract,value\n");
    printf("    Tokens  : USDT, USDC, DAI, UST, PAX, WLUNA\n");
    printf("    V2      : blocs 14669683+ (env. 20M lignes)\n");
    printf("    V3      : blocs 14500001+ (env. 70M lignes, inclut crash Luna mai 2022)\n");
    printf("    Methode : top-%d adresses les plus actives (hubs/baleines/exchanges)\n", ERC20_TOP_N);
    printf("    Poids   : -round(log10(value)) | ex: 15898 tokens -> w=-4\n");
    printf("    Usage   : ./bf_bench token_transfers_V2.0.0.csv token_transfers_V3.0.0.csv\n");

    {
        /* Construire la liste des fichiers ERC20 disponibles */
        const char *candidates[6];
        int nc = 0;

        /* 1. Fichiers passes en argument (priorite) */
        for(int i=0;i<3;i++)
            if(erc20_files[i]) candidates[nc++]=erc20_files[i];

        /* 2. Noms par defaut si pas deja dans la liste */
        static const char *defaults3[] = {
            "token_transfers_V2.0.0.csv",
            "token_transfers_V3.0.0.csv",
            "token_transfers_V1.0.0.csv"
        };
        for(int i=0;i<3;i++){
            bool dup=false;
            for(int j=0;j<nc;j++)
                if(strcmp(candidates[j],defaults3[i])==0){dup=true;break;}
            if(!dup && nc<6) candidates[nc++]=defaults3[i];
        }

        /* Charger (multi-fichiers, 2 passes fusionnees) */
        int ml = load_erc20_multi(candidates, nc, &eg, &ag);
        if(ml < 0 || eg.n < 2) {
            printf("\n  Aucun fichier ERC20 trouve ou graphe trop petit.\n");
            printf("  Usage: ./bf_bench token_transfers_V2.0.0.csv token_transfers_V3.0.0.csv\n");
        } else {
            printf("     Sous-graphe final : %d noeuds, %d arcs\n", eg.n, eg.m);
            /* Source = hub avec le plus grand degre sortant */
            build_csr(&eg, &csr_g);
            int src_erc=0, best_deg=0;
            for(int u=0;u<eg.n;u++){
                int d=csr_g.offset[u+1]-csr_g.offset[u];
                if(d>best_deg){best_deg=d;src_erc=u;}
            }
            printf("     Source = noeud %d (hub, degre sortant = %d)\n",
                   src_erc, best_deg);
            char label[120];
            snprintf(label,sizeof(label),
                     "ERC20 Stablecoins top-%d hubs (V2+V3)", ERC20_TOP_N);
            run(label, &eg, &ag, src_erc);
            adjgraph_free(&ag);
        }
    }

    /* ── Section 5 : Graphes Forex synthetiques ──────────────────── */
    printf("\n--- [5/6] GRAPHES SYNTHETIQUES FINANCIERS (TYPE FOREX) ---\n");
    printf("    Modelisation : reseaux d'arbitrage multi-devises\n");
    printf("    Ref modele   : Bertimas & Lo (1998), Journal of Financial Markets\n");
    printf("    Structure    : hub-and-spoke (majors <-> mineurs),\n");
    printf("                   tranches temporelles (slots)\n");
    printf("    Poids        : -log(taux) * echelle, bases sur taux reels BCE 2024\n");
    printf("    Poids negatif= taux > 1.0 (echange profitable, arbitrage possible)\n");
    printf("    Garantie     : AUCUN cycle negatif (potentiels de Johnson)\n");
    printf("    Source       : noeud USD au slot 0 (hub principal)\n");

    gen_forex(&eg, &ag, 20, 50, 42);
    run("Forex petit (20 devises x 50 slots, n=1000)", &eg, &ag, 0);
    adjgraph_free(&ag);

    gen_forex(&eg, &ag, 35, 100, 17);
    run("Forex moyen (35 devises x 100 slots, n=3500)", &eg, &ag, 0);
    adjgraph_free(&ag);

    gen_forex(&eg, &ag, 50, 80, 99);
    run("Forex source centrale (50 devises x 80 slots)", &eg, &ag, eg.n / 2);
    adjgraph_free(&ag);

    /* ── Section 5 : Graphes Crypto / Blockchain type Bitcoin ─────── */
    printf("\n--- [6/6] GRAPHES SYNTHETIQUES CRYPTO (TYPE RESEAU DE CONFIANCE) ---\n");
    printf("    Modelisation : reseau de confiance/mefiance entre traders\n");
    printf("    Inspire de   : Bitcoin Alpha / OTC (SNAP), Kumar et al. 2016\n");
    printf("    Structure    : power-law (quelques hubs tres connectes)\n");
    printf("    Poids        : in [-10, +10] (negatif=mefiance, positif=confiance)\n");
    printf("    Cycles neg.  : POSSIBLES (mefiances mutuelles en boucle)\n");
    printf("    Cas typique  : propagation de reputation / detection de fraude\n");

    gen_crypto_trust(&eg, &ag, 2000, 12000, 42);
    run("Crypto trust (n=2000, m=12000, ~25%% arcs neg)", &eg, &ag, 0);
    adjgraph_free(&ag);

    gen_crypto_trust(&eg, &ag, 3500, 22000, 7);
    run("Crypto trust large (n=3500, m=22000, power-law)", &eg, &ag, 0);
    adjgraph_free(&ag);

    gen_crypto_trust(&eg, &ag, 5000, 35000, 13);
    run("Crypto trust XL (n=5000, m=35000)", &eg, &ag, 0);
    adjgraph_free(&ag);

    printf("\nBenchmark termine.\n\n");
    return 0;
}
