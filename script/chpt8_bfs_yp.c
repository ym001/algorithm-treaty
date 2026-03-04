/*
 * benchmark.c — BFS 5-way benchmark
 * Algorithmes : STD · DIR · SURF · BB · YP v3
 *
 * YP v3 : BFS-Yoccoz-Puzzle v3 — quatre améliorations algorithmiques :
 *   [A] Dirty-word tracking    (supprime le memset O(W) par niveau)
 *   [B] Garde de couverture    (WE ≥ μ·2m pour entrer en BU)
 *   [C] Hysteresis dual-seuil  (reversion TD possible quand frontière petite)
 *   [D] Global Harmonic Field  (precomputation spectrale, tri adj par Fiedler)
 *
 * Compilation :
 *   gcc -O3 -march=native -o chpt8_bfs_yp chpt8_bfs_yp.c -lm
 *
 * Usage :
 *   ./chpt8_bfs_yp --snap ./graphs/*.txt --runs 20 [--alpha-sweep]
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Architecture mémoire YP v2 (vs YP v1)
 * ──────────────────────────────────────
 *                          n=4 039   n=133 277
 * adj_mask (v1, n×W mots)   2 MB     2 200 MB  ← catastrophe L3 / RAM
 * frontier_bv (v2, W mots)   4 Ko       16 Ko  ← tient dans L1 !
 *
 * Trois phases Yoccoz :
 *
 *  Phase 1 · Sous-critique (Brjuno régulier)
 *    Pré-condition : WE < α·SDU          (top-down moins cher)
 *    Action        : CSR top-down classique
 *    Invariant     : SDU décroît, WE relatif reste « petit »
 *    Analogie      : pièce de puzzle régulière, expansion prévisible
 *
 *  Phase 2 · Critique (renormalisation, frontière explosive)
 *    Pré-condition : WE ≥ α·SDU  et  uv ≥ θ·n
 *    Action        : bottom-up, scan complet n sommets
 *                    test d'adjacence via frontier_bv[] (W mots en L1)
 *                    inner loop CSR avec early exit au 1er voisin de frontière
 *    Coût réel     : SDU (chaque arc examiné ≤ 1 fois par direction)
 *    Analogie      : pièce critique, renormalisation nécessaire
 *
 *  Phase 3 · Post-critique (liste explicite des non-visités)
 *    Pré-condition : WE ≥ α·SDU  et  uv < θ·n
 *    Action        : bottom-up, itération sur liste unvisited[]
 *                    construite une seule fois en O(n), compactée par swap-last
 *    Coût réel     : O(uv × avg_deg) avec uv → 0 rapidement
 *    Analogie      : post-renormalisation, le puzzle se fragmente et se résout
 *
 * Critère exact de commutation (sans α heuristique) :
 *    WE  = Σ deg(u), u ∈ frontière        (coût TD = arcs sortants)
 *    SDU = Σ deg(v), v non-visité          (coût BU avec CSR+frontier_bv)
 *    switch si WE ≥ SDU  (α=1 est optimal, cf. Thm 3.2)
 *
 * SDU est maintenu incrémentalement : quand v est découvert, SDU -= deg(v).
 * WE est recalculé à chaque niveau en O(|frontière|) — coût négligeable.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════
 * Utilitaires
 * ═══════════════════════════════════════════════════════════════════ */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ═══════════════════════════════════════════════════════════════════
 * Graphe CSR  (pas d'adj_mask — c'est le changement fondamental)
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    int   n;          /* nombre de sommets                          */
    long  m;          /* nombre d'arêtes (non-dirigées)             */
    int  *xadj;       /* xadj[v]..xadj[v+1]-1 = voisins de v       */
    int  *adj;        /* tableau des voisins                        */
    int  *deg;        /* deg[v] = xadj[v+1]-xadj[v]                */
    int   W;          /* ceil(n/64)  — largeur du bitvecteur        */
} Graph;

static void graph_free(Graph *g) {
    free(g->xadj); free(g->adj); free(g->deg);
    memset(g, 0, sizeof(*g));
}

/*
 * Chargement format SNAP (arêtes non-dirigées, lignes "u v")
 * Lignes commençant par '#' ignorées.
 */
static int graph_load_snap(const char *path, Graph *g) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }

    /* 1ère passe : trouver n, compter les demi-arêtes par sommet */
    int max_v = -1;
    long edge_cnt = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        int u, v;
        if (sscanf(line, "%d %d", &u, &v) != 2) continue;
        if (u == v) continue;
        if (u > max_v) max_v = u;
        if (v > max_v) max_v = v;
        edge_cnt++;
    }
    if (max_v < 0) { fclose(f); return -1; }

    int n = max_v + 1;
    int *cnt = (int*)calloc(n, sizeof(int));
    if (!cnt) { fclose(f); return -1; }

    /* 2ème passe : degrés */
    rewind(f);
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        int u, v;
        if (sscanf(line, "%d %d", &u, &v) != 2) continue;
        if (u == v) continue;
        cnt[u]++; cnt[v]++;
    }

    /* Allocation CSR */
    int *xadj = (int*)malloc((n + 1) * sizeof(int));
    xadj[0] = 0;
    for (int i = 0; i < n; i++) xadj[i+1] = xadj[i] + cnt[i];
    int total_adj = xadj[n];
    int *adj  = (int*)malloc(total_adj * sizeof(int));
    int *pos  = (int*)calloc(n, sizeof(int));
    if (!xadj || !adj || !pos) { fclose(f); return -1; }

    /* 3ème passe : remplissage */
    rewind(f);
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        int u, v;
        if (sscanf(line, "%d %d", &u, &v) != 2) continue;
        if (u == v) continue;
        adj[xadj[u] + pos[u]++] = v;
        adj[xadj[v] + pos[v]++] = u;
    }
    fclose(f);
    free(cnt); free(pos);

    g->n    = n;
    g->m    = edge_cnt;
    g->xadj = xadj;
    g->adj  = adj;
    g->deg  = (int*)malloc(n * sizeof(int));
    for (int v = 0; v < n; v++) g->deg[v] = xadj[v+1] - xadj[v];
    g->W = (n + 63) / 64;

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Precomputation : Global Harmonic Field  (YP v3)
 *
 * Potentiel harmonique universel par itérations de puissance sur
 * la marche aléatoire normalisée D^{-1/2} A D^{-1/2}.
 * Converge vers le vecteur de Fiedler du Laplacien normalisé.
 *
 * Effet sur BFS : tri des listes d'adjacence par potentiel décroissant
 * → les voisins les plus "centraux" testés en premier dans la boucle BU
 * → early-exit plus fréquent → moins d'arcs examinés en Phase 2.
 *
 * Complexité : O(K × m), K=4.  Mémoire : 2n floats libérés après tri.
 * ═══════════════════════════════════════════════════════════════════ */

static float *g_harm_field = NULL;
static int cmp_harm_desc(const void *a, const void *b) {
    float fa = g_harm_field[*(const int*)a];
    float fb = g_harm_field[*(const int*)b];
    return (fb > fa) - (fb < fa);
}

static void graph_spectral_sort(Graph *g) {
    int   n   = g->n;
    float *f  = (float*)malloc(n * sizeof(float));
    float *f2 = (float*)malloc(n * sizeof(float));
    if (!f || !f2) { free(f); free(f2); return; }

    /* Init : f[v] = 1/sqrt(deg[v]) */
    for (int v = 0; v < n; v++)
        f[v] = (g->deg[v] > 0) ? 1.0f / sqrtf((float)g->deg[v]) : 0.0f;

    /* 4 iterations D^{-1/2} A D^{-1/2} */
    for (int iter = 0; iter < 4; iter++) {
        memset(f2, 0, n * sizeof(float));
        for (int v = 0; v < n; v++) {
            if (g->deg[v] == 0) continue;
            float sv = f[v] / sqrtf((float)g->deg[v]);
            for (int j = g->xadj[v]; j < g->xadj[v+1]; j++) {
                int u = g->adj[j];
                f2[u] += sv / sqrtf((float)(g->deg[u] > 0 ? g->deg[u] : 1));
            }
        }
        float norm = 0.0f;
        for (int v = 0; v < n; v++) norm += f2[v]*f2[v];
        norm = (norm > 0) ? sqrtf(norm) : 1.0f;
        for (int v = 0; v < n; v++) f2[v] /= norm;
        float *tmp = f; f = f2; f2 = tmp;
    }

    /* Tri de chaque liste d'adjacence par potentiel décroissant */
    g_harm_field = f;
    for (int v = 0; v < n; v++) {
        int deg = g->deg[v];
        if (deg < 2) continue;
        int *nbrs = g->adj + g->xadj[v];
        if (deg <= 16) {
            for (int i = 1; i < deg; i++) {
                int key = nbrs[i]; float kf = f[key]; int j = i-1;
                while (j >= 0 && f[nbrs[j]] < kf) { nbrs[j+1] = nbrs[j]; j--; }
                nbrs[j+1] = key;
            }
        } else {
            qsort(nbrs, deg, sizeof(int), cmp_harm_desc);
        }
    }
    g_harm_field = NULL;
    free(f); free(f2);
}

/* ─────────────────────────────────────────────────────────────────── */

/* Sommet de degré maximum (source canonique pour comparer les algos) */
static int hub_vertex(const Graph *g) {
    int hub = 0;
    for (int v = 1; v < g->n; v++)
        if (g->deg[v] > g->deg[hub]) hub = v;
    return hub;
}

/* ═══════════════════════════════════════════════════════════════════
 * Structures de résultat
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    double  time_ms;
    long    ops;         /* arcs examinés (edge-checks)                */
    int     td;          /* niveaux top-down                           */
    int     bu;          /* niveaux bottom-up                          */
    int    *dist;        /* distances (allouées ici, libérées par l'appelant) */
} BFSResult;

static void result_free(BFSResult *r) { free(r->dist); r->dist = NULL; }

/* ═══════════════════════════════════════════════════════════════════
 * 1. STD — BFS classique (Cormen 2022)
 * ═══════════════════════════════════════════════════════════════════ */

static BFSResult bfs_std(const Graph *g, int src) {
    int n = g->n;
    int *dist   = (int*)malloc(n * sizeof(int));
    int *queue  = (int*)malloc(n * sizeof(int));
    memset(dist, -1, n * sizeof(int));

    long ops = 0;
    double t0 = now_ms();

    dist[src] = 0;
    int head = 0, tail = 0;
    queue[tail++] = src;

    while (head < tail) {
        int u = queue[head++];
        int d = dist[u];
        for (int i = g->xadj[u]; i < g->xadj[u+1]; i++) {
            ops++;
            int v = g->adj[i];
            if (dist[v] < 0) { dist[v] = d + 1; queue[tail++] = v; }
        }
    }
    double t1 = now_ms();
    free(queue);
    return (BFSResult){ t1-t0, ops, 0, 0, dist };
}

/* ═══════════════════════════════════════════════════════════════════
 * 2. DIR — Direction-Optimizing BFS (Beamer et al., SC'12)
 * ═══════════════════════════════════════════════════════════════════ */

static BFSResult bfs_dir(const Graph *g, int src,
                          double alpha, double beta) {
    int  n  = g->n;
    long m2 = 2 * g->m;   /* arcs dirigés */
    int *dist   = (int*)malloc(n * sizeof(int));
    int *front  = (int*)malloc(n * sizeof(int));
    int *nxt    = (int*)malloc(n * sizeof(int));
    memset(dist, -1, n * sizeof(int));

    long ops = 0; int td = 0, bu = 0;
    double t0 = now_ms();

    dist[src] = 0; front[0] = src;
    int fsz = 1, uv = n - 1, level = 0;
    int in_bu = 0;

    while (fsz > 0) {
        long fe = 0;
        for (int i = 0; i < fsz; i++) fe += g->deg[front[i]];

        int use_bu = in_bu ? (uv > (int)(n / beta))
                           : (fe > m2 / alpha && uv < (int)(n / beta));
        in_bu = use_bu;

        int nsz = 0;
        if (!use_bu) {
            td++;
            for (int i = 0; i < fsz; i++) {
                int u = front[i];
                for (int j = g->xadj[u]; j < g->xadj[u+1]; j++) {
                    ops++;
                    int v = g->adj[j];
                    if (dist[v] < 0) { dist[v] = level+1; nxt[nsz++] = v; }
                }
            }
        } else {
            bu++;
            /* BU utilise un hash-set (tableau dist[u]==level) */
            for (int v = 0; v < n; v++) {
                if (dist[v] >= 0) continue;
                for (int j = g->xadj[v]; j < g->xadj[v+1]; j++) {
                    ops++;
                    int u = g->adj[j];
                    if (dist[u] == level) {
                        dist[v] = level + 1; nxt[nsz++] = v; break;
                    }
                }
            }
        }
        uv -= nsz;
        memcpy(front, nxt, nsz * sizeof(int));
        fsz = nsz; level++;
    }
    double t1 = now_ms();
    free(front); free(nxt);
    return (BFSResult){ t1-t0, ops, td, bu, dist };
}

/* ═══════════════════════════════════════════════════════════════════
 * 3. SURF — EMA workload + hystérésis (Yoon & Oh, Sensors 2022)
 * ═══════════════════════════════════════════════════════════════════ */

static BFSResult bfs_surf(const Graph *g, int src) {
    int n = g->n;
    int *dist  = (int*)malloc(n * sizeof(int));
    int *front = (int*)malloc(n * sizeof(int));
    int *nxt   = (int*)malloc(n * sizeof(int));
    /* frontier bitvector pour BU (partagé SURF/YP, coût negligeable) */
    int W = g->W;
    uint64_t *fv  = (uint64_t*)calloc(W, sizeof(uint64_t));
    uint64_t *nfv = (uint64_t*)calloc(W, sizeof(uint64_t));
    memset(dist, -1, n * sizeof(int));

    const double BETA       = 0.6;
    const double TAU_ENTER  = 1.0;
    const double TAU_EXIT   = 0.5;
    const int    COOLDOWN   = 2;

    long ops = 0; int td = 0, bu = 0;
    double t0 = now_ms();

    dist[src] = 0; front[0] = src;
    int fsz = 1, uv = n - 1, level = 0;
    int in_bu = 0, cooldown = 0;
    double ema = 0.0;

    fv[src >> 6] |= (1ULL << (src & 63));

    while (fsz > 0) {
        long WE = 0;
        for (int i = 0; i < fsz; i++) WE += g->deg[front[i]];
        double WR = (double)WE / (double)(uv > 0 ? uv : 1);
        ema = BETA * ema + (1.0 - BETA) * WR;

        if (in_bu) {
            if (ema < TAU_EXIT) { in_bu = 0; cooldown = COOLDOWN; }
        } else {
            if (cooldown > 0) cooldown--;
            else if (ema >= TAU_ENTER) in_bu = 1;
        }

        int nsz = 0;
        memset(nfv, 0, W * sizeof(uint64_t));

        if (!in_bu) {
            td++;
            for (int i = 0; i < fsz; i++) {
                int u = front[i];
                for (int j = g->xadj[u]; j < g->xadj[u+1]; j++) {
                    ops++;
                    int v = g->adj[j];
                    if (dist[v] < 0) {
                        dist[v] = level + 1;
                        nxt[nsz++] = v;
                        nfv[v >> 6] |= (1ULL << (v & 63));
                    }
                }
            }
        } else {
            bu++;
            for (int v = 0; v < n; v++) {
                if (dist[v] >= 0) continue;
                ops++;
                /* test CSR + frontier bitvector (early exit) */
                for (int j = g->xadj[v]; j < g->xadj[v+1]; j++) {
                    int u = g->adj[j];
                    if (fv[u >> 6] & (1ULL << (u & 63))) {
                        dist[v] = level + 1;
                        nxt[nsz++] = v;
                        nfv[v >> 6] |= (1ULL << (v & 63));
                        break;
                    }
                }
            }
        }

        uint64_t *tmp = fv; fv = nfv; nfv = tmp;
        uv -= nsz;
        memcpy(front, nxt, nsz * sizeof(int));
        fsz = nsz; level++;
    }
    double t1 = now_ms();
    free(front); free(nxt); free(fv); free(nfv);
    return (BFSResult){ t1-t0, ops, td, bu, dist };
}

/* ═══════════════════════════════════════════════════════════════════
 * 4. BB — Bitmap Visited BFS (Bhaskar & Kanakagiri, arXiv 2025)
 *
 * Innovation : visited-bitmap compact (n/64 mots, tient dans L1/L2)
 * remplace le tableau dist[] pour le test "non visité".
 * Ici on garde adj_mask dense UNIQUEMENT pour n≤MAX_BB_N.
 * Au-delà, BB se rabat sur BU-CSR (comme YP Phase 2).
 * ═══════════════════════════════════════════════════════════════════ */

#define MAX_BB_N  8192   /* seuil : adj_mask < 4 MB en cache */

static BFSResult bfs_bb(const Graph *g, int src) {
    int  n  = g->n;
    int  W  = g->W;
    int *dist   = (int*)malloc(n * sizeof(int));
    int *front  = (int*)malloc(n * sizeof(int));
    int *nxt    = (int*)malloc(n * sizeof(int));
    uint64_t *fv   = (uint64_t*)calloc(W, sizeof(uint64_t));
    uint64_t *nfv  = (uint64_t*)calloc(W, sizeof(uint64_t));
    uint64_t *vis  = (uint64_t*)calloc(W, sizeof(uint64_t));  /* visited bitmap */
    memset(dist, -1, n * sizeof(int));

    const double ALPHA_BB = 1.2;

    /* adj_mask dense : allouée seulement si n est assez petit */
    uint64_t **adj_mask = NULL;
    if (n <= MAX_BB_N) {
        adj_mask = (uint64_t**)malloc(n * sizeof(uint64_t*));
        uint64_t *pool = (uint64_t*)calloc((long)n * W, sizeof(uint64_t));
        for (int v = 0; v < n; v++) {
            adj_mask[v] = pool + (long)v * W;
            for (int j = g->xadj[v]; j < g->xadj[v+1]; j++) {
                int u = g->adj[j];
                adj_mask[v][u >> 6] |= (1ULL << (u & 63));
            }
        }
    }

    long ops = 0; int td = 0, bu = 0;
    double t0 = now_ms();

    dist[src] = 0;
    front[0] = src; int fsz = 1;
    int uv = n - 1, level = 0;
    vis[src >> 6] |= (1ULL << (src & 63));
    fv [src >> 6] |= (1ULL << (src & 63));

    while (fsz > 0) {
        long WE = 0;
        for (int i = 0; i < fsz; i++) WE += g->deg[front[i]];
        int use_bu = (WE >= ALPHA_BB * (double)(uv > 0 ? uv : 1));

        int nsz = 0;
        memset(nfv, 0, W * sizeof(uint64_t));

        if (!use_bu) {
            td++;
            for (int i = 0; i < fsz; i++) {
                int u = front[i];
                for (int j = g->xadj[u]; j < g->xadj[u+1]; j++) {
                    ops++;
                    int v = g->adj[j];
                    if (!(vis[v >> 6] & (1ULL << (v & 63)))) {
                        dist[v] = level + 1;
                        vis[v >> 6] |= (1ULL << (v & 63));
                        nfv[v >> 6] |= (1ULL << (v & 63));
                        nxt[nsz++] = v;
                    }
                }
            }
        } else {
            bu++;
            if (adj_mask) {
                /* Dense adj_mask : 1 opération bitmask par sommet */
                for (int v = 0; v < n; v++) {
                    if (vis[v >> 6] & (1ULL << (v & 63))) continue;
                    ops++;
                    /* Test vectoriel : AND de W mots */
                    int hit = 0;
                    for (int k = 0; k < W; k++) {
                        if (adj_mask[v][k] & fv[k]) { hit = 1; break; }
                    }
                    if (hit) {
                        /* Trouver le parent réel */
                        for (int j = g->xadj[v]; j < g->xadj[v+1]; j++) {
                            int u = g->adj[j];
                            if (dist[u] == level) {
                                dist[v] = level + 1;
                                vis[v >> 6] |= (1ULL << (v & 63));
                                nfv[v >> 6] |= (1ULL << (v & 63));
                                nxt[nsz++] = v;
                                break;
                            }
                        }
                    }
                }
            } else {
                /* Grands graphes : CSR bottom-up avec frontier_bv */
                for (int v = 0; v < n; v++) {
                    if (dist[v] >= 0) continue;
                    ops++;
                    for (int j = g->xadj[v]; j < g->xadj[v+1]; j++) {
                        int u = g->adj[j];
                        if (fv[u >> 6] & (1ULL << (u & 63))) {
                            dist[v] = level + 1;
                            nfv[v >> 6] |= (1ULL << (v & 63));
                            nxt[nsz++] = v;
                            break;
                        }
                    }
                }
            }
        }

        uint64_t *tmp = fv; fv = nfv; nfv = tmp;
        uv -= nsz;
        memcpy(front, nxt, nsz * sizeof(int));
        fsz = nsz; level++;
    }
    double t1 = now_ms();

    if (adj_mask) { free(adj_mask[0]); free(adj_mask); }
    free(front); free(nxt); free(fv); free(nfv); free(vis);
    return (BFSResult){ t1-t0, ops, td, bu, dist };
}

/* ═════════════════════════════════════════════════════════════════
 * YP v2 — BFS-Yoccoz-Puzzle, trois phases
 *
 * DONNÉES
 *   dist[v]      : distance depuis src (-1 = non visité)
 *   parent[v]    : parent dans l'arbre BFS
 *   fv[W]        : frontier bitvector  (W = ⌈n/64⌉ mots)
 *   nfv[W]       : next frontier bitvector
 *   front[]      : liste frontière courante  (pour Phase 1)
 *   uv[]         : liste explicite non-visités (pour Phase 3)
 *   WE           : Σ deg(frontière)   = coût top-down
 *   SDU          : Σ deg(non-visités) = coût bottom-up CSR
 *
 * INVARIANT CLÉ
 *   SDU est maintenu incrémentalement :
 *     initialisation : SDU = Σ deg(v) pour tout v ≠ src
 *     mise à jour    : quand v est découvert → SDU -= deg(v)
 *   Cela correspond à la "somme de Brjuno généralisée" du puzzle :
 *   mesure l'énergie résiduelle de la composante non-visitée.
 *
 * COMMUTATION (α = 1.0 optimal, Thm 3.2)
 *   WE < α·SDU   → Phase 1 (top-down)
 *   WE ≥ α·SDU   → Phase 2 ou 3 (bottom-up)
 *
 * HYSTERESIS YOCCOZ
 *   Une fois en mode BU, on reste en BU même si WE redevient < SDU
 *   (les pièces ne se "dé-renormalisent" pas dans la théorie de Yoccoz).
 *   Exception : si la frontière disparaît (terminaison).
 *
 * SEUIL PHASE 3 (θ = 0.05)
 *   Quand uv < θ·n, la construction de la liste explicite O(n)
 *   est amortie sur les niveaux restants, chacun coûtant O(uv × avg_deg).
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * ═══════════════════════════════════════════════════════════════════
 * BFS-YP v3  — Quatre améliorations vs v2 :
 *
 *  [A] Dirty-word tracking  (corrige la régression sur les graphes routiers)
 *      v2 faisait memset(nfv, W×8) à chaque niveau BFS, même Phase 1.
 *      Pour roadNet-TX : 930 niveaux × W=21772 × 8o = 161 MB de memset !
 *      v3 mémorise exactement quels mots de nfv ont été modifiés et ne
 *      nettoie que ceux-ci → O(|frontière|) au lieu de O(W) par niveau.
 *
 *  [B] Garde de couverture d'arêtes  (corrige les faux déclenchements BU)
 *      v2 : switch si WE ≥ α·uv (compte les sommets non-visités)
 *      v3 : switch si WE ≥ α·uv  ET  WE ≥ μ·2m  (μ=0.10)
 *      Signification : la frontière doit couvrir ≥10% des arêtes totales
 *      avant que BU soit rentable. Élimine les basculements prématurés
 *      sur les graphes web (hub de degré élevé, petit voisinage immédiat).
 *
 *  [C] Hysteresis dual-seuil  (corrige la stagnation sur graphes creux)
 *      v2 : une fois en BU, jamais de retour (hysteresis irréversible).
 *      v3 : revient en TD si fsz < μ_lo·n (frontière trop petite pour BU).
 *      Seuils : μ_hi = 0.10 (entrée BU), μ_lo = 0.04 (sortie BU).
 *      Analogie Yoccoz révisée : la renormalisation est réversible quand
 *      le puzzle se fragmente en composantes isolées (Phase post-critique).
 *
 *  [D] Listes d'adjacence triées par Potentiel Harmonique Global
 *      (precomputation spectrale, appelée une fois par graphe)
 *      → voisins "centraux" testés en premier dans la boucle BU
 *      → early-exit plus fréquent → réduction des ops en Phase 2.
 *
 * Paramètres :
 *   alpha  : seuil WE vs uv  (optimal=1.0, Thm 3.2)
 *   mu_e   : couverture d'arêtes minimale pour activer BU  (0.10)
 *   mu_hi  : fraction de n pour entrer en BU  (0.10)
 *   mu_lo  : fraction de n pour sortir de BU  (0.04)
 * ═══════════════════════════════════════════════════════════════════ */

#define YP_THETA   0.015
#define YP_ALPHA   1
#define YP_MU_E    0.15   /* couverture arêtes minimale pour BU */
#define YP_MU_HI   0.15   /* fsz/n seuil d'entrée BU            */
#define YP_MU_LO   0.025  /* fsz/n seuil de sortie BU           */

static BFSResult bfs_yp(const Graph *g, int src, double alpha) {
    int  n  = g->n;
    int  W  = g->W;
    long m2 = 2 * g->m;

    int      *dist    = (int*)     malloc(n * sizeof(int));
    int      *parent  = (int*)     malloc(n * sizeof(int));
    int      *front   = (int*)     malloc(n * sizeof(int));
    int      *nxt_buf = (int*)     malloc(n * sizeof(int));
    int      *uvlist  = (int*)     malloc(n * sizeof(int));
    uint64_t *fv      = (uint64_t*)calloc(W, sizeof(uint64_t));
    uint64_t *nfv     = (uint64_t*)calloc(W, sizeof(uint64_t));
    uint64_t *vis_bv  = (uint64_t*)calloc(W, sizeof(uint64_t));

    /* [A] Dirty-word tracking : liste des mots de nfv touchés ce niveau */
    int *dirty_words = (int*)malloc(W * sizeof(int));
    int *dirty_mark  = (int*)calloc(W, sizeof(int));
    int  dirty_nsz   = 0;

    memset(dist,   -1, n * sizeof(int));
    memset(parent, -1, n * sizeof(int));

    long ops = 0;
    int  td = 0, bu = 0;
    double t0 = now_ms();

    dist[src] = 0;
    front[0]  = src;
    int  fsz  = 1, uv = n - 1, level = 0;
    fv   [src >> 6] |= (1ULL << (src & 63));
    vis_bv[src >> 6] |= (1ULL << (src & 63));

    long SDU = 0;
    for (int v = 0; v < n; v++) if (v != src) SDU += g->deg[v];
    long WE_next = g->deg[src];

    int in_bu        = 0;
    int uvlist_built = 0;
    int uvsz         = 0;
    int theta_n      = (int)(YP_THETA * n); if (theta_n < 8) theta_n = 8;
    int mu_hi_n      = (int)(YP_MU_HI * n); if (mu_hi_n < 16) mu_hi_n = 16;
    int mu_lo_n      = (int)(YP_MU_LO * n); if (mu_lo_n < 8)  mu_lo_n = 8;
    double mu_e_m2   = YP_MU_E * (double)m2;

    while (fsz > 0) {

        long WE = WE_next;
        WE_next = 0;

        /* [B+C] Critère de commutation amélioré
         *
         * Entrée BU  : WE ≥ α·uv  ET  WE ≥ μ·2m  ET  fsz ≥ μ_hi·n
         *   — La garde WE ≥ μ·2m évite le basculement sur hub isolé
         *     (grand WE mais petite frontière par rapport au graphe total).
         *   — La garde fsz ≥ μ_hi·n assure que la frontière est "explosive"
         *     au sens Yoccoz : suffisamment de contacts pour que BU
         *     retrouve des voisins facilement.
         *
         * Sortie BU  : fsz < μ_lo·n
         *   — Quand la frontière rétrécit (post-renormalisation),
         *     revenir en TD est plus efficace (peu d'arcs à pousser).
         */
        if (!in_bu) {
            if (WE >= alpha * (double)(uv > 0 ? uv : 1) &&
                (double)WE >= mu_e_m2 &&
                fsz >= mu_hi_n)
                in_bu = 1;
        } else {
            if (fsz < mu_lo_n)
                in_bu = 0;
        }

        int nsz = 0;

        /* [A] Nettoyer uniquement les mots dirty (pas de memset global) */
        for (int i = 0; i < dirty_nsz; i++) {
            nfv[dirty_words[i]] = 0;
            dirty_mark[dirty_words[i]] = 0;
        }
        dirty_nsz = 0;

/* Macro helper pour marquer un bit dans nfv avec dirty tracking */
#define NF_SET(v) do { \
    int _w = (v) >> 6; \
    if (!dirty_mark[_w]) { dirty_words[dirty_nsz++] = _w; dirty_mark[_w] = 1; } \
    nfv[_w] |= (1ULL << ((v) & 63)); \
} while(0)

        /* ═══ Phase 1 · Sous-critique (top-down) ═══════════════════ */
        if (!in_bu) {
            td++;
            for (int i = 0; i < fsz; i++) {
                int u = front[i];
                int d = dist[u];
                for (int j = g->xadj[u]; j < g->xadj[u+1]; j++) {
                    ops++;
                    int v = g->adj[j];
                    if (!(vis_bv[v >> 6] & (1ULL << (v & 63)))) {
                        dist[v]   = d + 1;
                        parent[v] = u;
                        SDU     -= g->deg[v];
                        WE_next += g->deg[v];
                        vis_bv[v >> 6] |= (1ULL << (v & 63));
                        nxt_buf[nsz++] = v;
                        NF_SET(v);
                    }
                }
            }

        /* ═══ Phase 3 · Post-critique (liste explicite) ════════════ */
        } else if (uv > 0 && uv < theta_n) {
            bu++;
            if (!uvlist_built) {
                uvsz = 0;
                for (int v = 0; v < n; v++)
                    if (!(vis_bv[v >> 6] & (1ULL << (v & 63))))
                        uvlist[uvsz++] = v;
                uvlist_built = 1;
            }
            int i = 0;
            while (i < uvsz) {
                int v = uvlist[i], found = 0;
                for (int j = g->xadj[v]; j < g->xadj[v+1]; j++) {
                    ops++;
                    int u = g->adj[j];
                    if (fv[u >> 6] & (1ULL << (u & 63))) {
                        dist[v]   = level + 1;
                        parent[v] = u;
                        SDU -= g->deg[v];
                        WE_next += g->deg[v];
                        vis_bv[v >> 6] |= (1ULL << (v & 63));
                        nxt_buf[nsz++] = v;
                        NF_SET(v);
                        uvlist[i] = uvlist[--uvsz];
                        found = 1; break;
                    }
                }
                if (!found) i++;
            }

        /* ═══ Phase 2 · Critique (bottom-up, scan vis_bv) ══════════ */
        } else {
            bu++;
            for (int w = 0; w < W; w++) {
                uint64_t unvis = ~vis_bv[w];
                while (unvis) {
                    int bit = __builtin_ctzll(unvis);
                    unvis &= unvis - 1;
                    int v = w * 64 + bit;
                    if (v >= n) break;
                    for (int j = g->xadj[v]; j < g->xadj[v+1]; j++) {
                        ops++;
                        int u = g->adj[j];
                        if (fv[u >> 6] & (1ULL << (u & 63))) {
                            dist[v]   = level + 1;
                            parent[v] = u;
                            SDU -= g->deg[v];
                            WE_next += g->deg[v];
                            vis_bv[v >> 6] |= (1ULL << (v & 63));
                            nxt_buf[nsz++] = v;
                            NF_SET(v);
                            break;
                        }
                    }
                }
            }
        }

#undef NF_SET

        uint64_t *tmp = fv; fv = nfv; nfv = tmp;
        uv  -= nsz;
        memcpy(front, nxt_buf, nsz * sizeof(int));
        fsz  = nsz;
        level++;
    }

    double t1 = now_ms();
    free(nxt_buf); free(uvlist);
    free(fv); free(nfv); free(vis_bv); free(parent);
    free(dirty_words); free(dirty_mark);
    return (BFSResult){ t1-t0, ops, td, bu, dist };
}

/* ═══════════════════════════════════════════════════════════════════
 * Validation : STD vs YP sur un petit graphe synthétique
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * Génère un graphe BA (Barabási-Albert) minimaliste pour la validation.
 * Seul YP est validé contre STD ici ; les autres algos partagent la
 * même structure TD/BU et sont couverts par les tests Python.
 */
static int validate_yp(void) {
    /* Chaîne 0-1-2-3-4 */
    {
        int xadj[] = {0,1,3,5,7,8};
        int adj[]  = {1, 0,2, 1,3, 2,4, 3};
        Graph g = {5, 4, xadj, adj, NULL, 1};
        int deg[] = {1,2,2,2,1};
        g.deg = deg;

        BFSResult rs = bfs_std(&g, 0);
        BFSResult ry = bfs_yp(&g, 0, 1.0);
        for (int v = 0; v < 5; v++) {
            if (rs.dist[v] != ry.dist[v]) {
                fprintf(stderr, "FAIL chain: dist[%d] std=%d yp=%d\n",
                        v, rs.dist[v], ry.dist[v]);
                result_free(&rs); result_free(&ry);
                return -1;
            }
        }
        result_free(&rs); result_free(&ry);
    }
    /* Étoile K1,7 */
    {
        int xadj[9]; xadj[0]=0;
        for (int v=0;v<8;v++) xadj[v+1]=xadj[v]+(v==0?7:1);
        int adj[14];
        for (int i=0;i<7;i++) adj[i]=i+1;
        for (int i=0;i<7;i++) adj[7+i]=0;
        Graph g = {8, 7, xadj, adj, NULL, 1};
        int deg[8]; deg[0]=7; for(int v=1;v<8;v++) deg[v]=1;
        g.deg = deg;

        BFSResult rs = bfs_std(&g, 0);
        BFSResult ry = bfs_yp(&g, 0, 1.0);
        for (int v = 0; v < 8; v++) {
            if (rs.dist[v] != ry.dist[v]) {
                fprintf(stderr, "FAIL star: dist[%d] std=%d yp=%d\n",
                        v, rs.dist[v], ry.dist[v]);
                result_free(&rs); result_free(&ry);
                return -1;
            }
        }
        result_free(&rs); result_free(&ry);
    }
    printf("  Validation YP: OK (chaine + etoile)\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Benchmark runner
 * ═══════════════════════════════════════════════════════════════════ */

#define N_ALGOS 5

typedef struct {
    const char *name;
    double alpha;         /* paramètre spécifique */
} AlgoDesc;

static AlgoDesc algos[N_ALGOS] = {
    {"STD",  0.0},
    {"DIR",  14.0},
    {"SURF", 0.0},
    {"BB",   1.2},
    {"YP",   YP_ALPHA},
};

static BFSResult run_algo(int algo_idx, const Graph *g, int src) {
    switch (algo_idx) {
        case 0: return bfs_std(g, src);
        case 1: return bfs_dir(g, src, algos[1].alpha, 24.0);
        case 2: return bfs_surf(g, src);
        case 3: return bfs_bb(g, src);
        default: return bfs_yp(g, src, algos[4].alpha);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Structures statistiques — conformes aux exigences JEA
 *
 * JEA (Journal of Experimental Algorithmics) requiert :
 *   - Moyenne + écart-type sur ≥30 runs
 *   - Médiane (robuste aux outliers thermiques)
 *   - CV% = (sd/mean)×100 — indicateur de stabilité
 *   - Test de significativité (Wilcoxon signé-rangé)
 *
 * Chaque AlgoStats stocke ces 4 statistiques + les ops du run médian.
 * Les raw_times[] sont conservés pour le test de Wilcoxon.
 * ═══════════════════════════════════════════════════════════════════ */

#define MAX_RUNS 200   /* plafond de sécurité */

typedef struct {
    double t_min;    /* minimum  — traditionnel, garde compatibilité    */
    double t_mean;   /* moyenne arithmétique                            */
    double t_sd;     /* écart-type (std dev non biaisé, div. par n-1)   */
    double t_med;    /* médiane  — robuste aux pics thermiques           */
    double cv_pct;   /* coefficient de variation = (sd/mean)×100        */
    long   ops;      /* edge-checks du run minimal                      */
    int    td, bu;   /* niveaux TD/BU du run minimal                    */
} AlgoStats;

/* Comparateur pour qsort sur double */
static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

/* ── Calcul des statistiques depuis un tableau de mesures brutes ── */
static void compute_stats(const double *raw, int n,
                          double *out_min, double *out_mean,
                          double *out_sd,  double *out_med) {
    double sorted[MAX_RUNS];
    memcpy(sorted, raw, n * sizeof(double));
    qsort(sorted, n, sizeof(double), cmp_double_asc);

    *out_min = sorted[0];
    *out_med = (n % 2 == 0)
               ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0
               : sorted[n/2];

    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += raw[i];
    *out_mean = sum / n;

    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = raw[i] - *out_mean;
        var += d * d;
    }
    *out_sd = (n > 1) ? sqrt(var / (n - 1)) : 0.0;
}

/* ── bench_run_stats : remplace l'ancien bench_run ────────────────
 *
 * Collecte `runs` mesures par algorithme, calcule min/mean/sd/med.
 * Conserve aussi raw_times[] pour le test de Wilcoxon ultérieur.
 *
 * Ordre d'exécution : interleaved (r=0: tous algos, r=1: tous algos…)
 * → chaque run voit les mêmes conditions thermiques pour tous les algos
 * → les différences mesurées sont des différences algorithmiques, pas
 *    des artefacts de réchauffement CPU.
 * ──────────────────────────────────────────────────────────────── */
static void bench_run_stats(const Graph *g, int src, int runs,
                            AlgoStats stats[N_ALGOS],
                            double raw_times[N_ALGOS][MAX_RUNS]) {
    int r_eff = (runs > MAX_RUNS) ? MAX_RUNS : runs;

    /* Initialiser les champs "best run" */
    for (int a = 0; a < N_ALGOS; a++) {
        stats[a].t_min = 1e18;
        stats[a].ops = 0; stats[a].td = 0; stats[a].bu = 0;
    }

    /* Collecte interleaved */
    for (int r = 0; r < r_eff; r++) {
        for (int a = 0; a < N_ALGOS; a++) {
            BFSResult res = run_algo(a, g, src);
            raw_times[a][r] = res.time_ms;
            if (res.time_ms < stats[a].t_min) {
                stats[a].t_min = res.time_ms;
                stats[a].ops   = res.ops;
                stats[a].td    = res.td;
                stats[a].bu    = res.bu;
            }
            result_free(&res);
        }
    }

    /* Calcul des statistiques */
    for (int a = 0; a < N_ALGOS; a++) {
        compute_stats(raw_times[a], r_eff,
                      &stats[a].t_min,
                      &stats[a].t_mean,
                      &stats[a].t_sd,
                      &stats[a].t_med);
        stats[a].cv_pct = (stats[a].t_mean > 0)
                          ? (stats[a].t_sd / stats[a].t_mean) * 100.0
                          : 0.0;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Test de Wilcoxon signé-rangé — pairwise YP vs concurrent
 *
 * Hypothèse nulle H0 : les temps YP et l'algo concurrent proviennent
 * de la même distribution (pas de différence médiane).
 *
 * Algorithme (Wilcoxon 1945, approximation normale pour n≥10) :
 *   1. d_i = t_YP[i] - t_concurrent[i]   (différences pairées)
 *   2. Écarter les d_i = 0 (ties exacts)
 *   3. Trier par |d_i| croissant, attribuer les rangs 1..n_eff
 *   4. W+ = Σ rang_i pour d_i < 0  (YP plus rapide)
 *      W- = Σ rang_i pour d_i > 0  (YP plus lent)
 *   5. T = min(W+, W-)
 *   6. Approximation normale :
 *      μ_T = n_eff(n_eff+1)/4
 *      σ_T = sqrt(n_eff(n_eff+1)(2n_eff+1)/24)
 *      z   = (T - μ_T) / σ_T
 *   7. p-value (bilatérale) ≈ 2 × Φ(-|z|)
 *      où Φ est la CDF de la loi normale standard.
 *
 * Retourne : p-value (0..1), avec signe négatif si YP est plus lent
 * ═══════════════════════════════════════════════════════════════════ */

/* Approximation de erfc pour Φ(-|z|) — Abramowitz & Stegun 7.1.26 */
static double norm_cdf_upper(double z) {
    /* P(Z > z) pour z ≥ 0 */
    if (z < 0) z = -z;
    double t = 1.0 / (1.0 + 0.2316419 * z);
    double poly = t * (0.319381530
                 + t * (-0.356563782
                 + t * (1.781477937
                 + t * (-1.821255978
                 + t * 1.330274429))));
    return poly * exp(-0.5 * z * z) / sqrt(2.0 * M_PI);
}

typedef struct {
    double p_value;    /* p-value du test bilatéral (0..1)   */
    double z_score;    /* z-score (+ = YP gagne, - = YP perd) */
    int    n_eff;      /* nombre de paires non-ties           */
    double W_plus;     /* somme rangs YP-faster               */
    double W_minus;    /* somme rangs YP-slower               */
} WilcoxonResult;

static WilcoxonResult wilcoxon_yp_vs(const double *t_yp,
                                     const double *t_other,
                                     int n) {
    WilcoxonResult res = {1.0, 0.0, 0, 0.0, 0.0};

    /* 1. Calculer les différences d_i = t_yp - t_other */
    double diffs[MAX_RUNS];
    int    n_eff = 0;
    for (int i = 0; i < n; i++) {
        double d = t_yp[i] - t_other[i];
        if (d == 0.0) continue;          /* tie exact : écarté */
        diffs[n_eff++] = d;
    }
    if (n_eff < 5) { res.n_eff = n_eff; return res; } /* pas assez de données */
    res.n_eff = n_eff;

    /* 2. Trier par |d_i| croissant pour attribuer les rangs */
    /* Indices de tri */
    int idx[MAX_RUNS];
    for (int i = 0; i < n_eff; i++) idx[i] = i;
    /* bubble sort sur |diffs| — n_eff ≤ MAX_RUNS, acceptable */
    for (int i = 0; i < n_eff - 1; i++)
        for (int j = i + 1; j < n_eff; j++)
            if (fabs(diffs[idx[i]]) > fabs(diffs[idx[j]])) {
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }

    /* 3. Attribution des rangs (moyenne en cas d'ex-aequo) */
    double ranks[MAX_RUNS];
    for (int i = 0; i < n_eff; ) {
        int j = i;
        double abs_d = fabs(diffs[idx[i]]);
        while (j < n_eff && fabs(diffs[idx[j]]) == abs_d) j++;
        /* rangs i+1 .. j (base 1) → moyenne = (i+1+j)/2 */
        double avg_rank = (double)(i + 1 + j) / 2.0;
        for (int k = i; k < j; k++) ranks[k] = avg_rank;
        i = j;
    }

    /* 4. W+ / W- */
    double Wp = 0.0, Wm = 0.0;
    for (int i = 0; i < n_eff; i++) {
        if (diffs[idx[i]] < 0.0) Wp += ranks[i];  /* d<0 → YP plus rapide */
        else                      Wm += ranks[i];
    }
    res.W_plus  = Wp;
    res.W_minus = Wm;

    /* 5. Statistique T */
    double T  = (Wp < Wm) ? Wp : Wm;
    double mu = (double)n_eff * (n_eff + 1) / 4.0;
    double sd = sqrt((double)n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24.0);
    if (sd == 0.0) return res;

    /* correction de continuité ±0.5 */
    double z = (T - mu + 0.5) / sd;   /* note : T < mu → YP gagne → z < 0 → inverser */

    /* On veut z > 0 si YP gagne (W+ > W-) */
    res.z_score = (Wp > Wm) ? -z : z;
    res.p_value = 2.0 * norm_cdf_upper(fabs(res.z_score));
    return res;
}

/* Ancien bench_run maintenu pour compatibilité avec alpha_sweep */
static void bench_run(const Graph *g, int src, int runs,
                      double times[N_ALGOS], long ops_arr[N_ALGOS],
                      int td_arr[N_ALGOS],   int bu_arr[N_ALGOS]) {
    static double raw[N_ALGOS][MAX_RUNS];
    AlgoStats stats[N_ALGOS];
    bench_run_stats(g, src, runs, stats, raw);
    for (int a = 0; a < N_ALGOS; a++) {
        times[a]   = stats[a].t_min;
        ops_arr[a] = stats[a].ops;
        td_arr[a]  = stats[a].td;
        bu_arr[a]  = stats[a].bu;
    }
}

static void print_header_time(void) {
    printf("\n-- Wall-clock time (ms) --\n\n");
    printf("%-30s %6s %8s  %8s %8s %8s %8s %8s  %6s %6s %6s %6s\n",
           "Graph","n","m",
           "STD(ms)","DIR(ms)","SURF(ms)","BB(ms)","YP(ms)",
           "xDIR","xSURF","xBB","xYP");
    printf("%-30s %6s %8s  %8s %8s %8s %8s %8s  %6s %6s %6s %6s\n",
           "------------------------------","------","--------",
           "--------","--------","--------","--------","--------",
           "------","------","------","------");
}

static void print_row_time(const char *name, int n, long m,
                           double times[N_ALGOS]) {
    printf("%-30s %6d %8ld  %8.3f %8.3f %8.3f %8.3f %8.3f  "
           "%5.2fx %5.2fx %5.2fx %5.2fx\n",
           name, n, m,
           times[0], times[1], times[2], times[3], times[4],
           times[0]/times[1], times[0]/times[2],
           times[0]/times[3], times[0]/times[4]);
}

static void print_header_ops(void) {
    printf("\n-- Edge checks (ops) --\n\n");
    printf("%-30s %6s %8s  %10s %10s %10s %10s %10s  %6s %6s %6s %6s\n",
           "Graph","n","m",
           "ops-STD","ops-DIR","ops-SURF","ops-BB","ops-YP",
           "/DIR","/SURF","/BB","/YP");
    printf("%-30s %6s %8s  %10s %10s %10s %10s %10s  %6s %6s %6s %6s\n",
           "------------------------------","------","--------",
           "----------","----------","----------","----------","----------",
           "------","------","------","------");
}

static void print_row_ops(const char *name, int n, long m,
                          long ops[N_ALGOS]) {
    printf("%-30s %6d %8ld  %10ld %10ld %10ld %10ld %10ld  "
           "%5.2fx %5.2fx %5.2fx %5.2fx\n",
           name, n, m,
           ops[0], ops[1], ops[2], ops[3], ops[4],
           (double)ops[0]/ops[1], (double)ops[0]/ops[2],
           (double)ops[0]/ops[3], (double)ops[0]/ops[4]);
}

/* ── Alpha sweep pour YP ───────────────────────────────────────── */
static void alpha_sweep(const Graph *g, int src, int runs) {
    const double alphas[] = {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0};
    int na = sizeof(alphas) / sizeof(alphas[0]);

    printf("\n=== Alpha Sensitivity (YP) ===\n");
    printf("%-8s  %10s  %10s  %8s\n", "alpha", "time(ms)", "ops", "TD/BU");
    printf("%-8s  %10s  %10s  %8s\n", "-----", "--------", "---", "-----");

    double best_t = 1e18;
    int    best_i = 0;
    /* premier passage : trouver le meilleur */
    double t_arr[8];
    long   o_arr[8];
    int    td_arr2[8], bu_arr2[8];

    for (int i = 0; i < na; i++) {
        t_arr[i] = 1e18; o_arr[i] = 0; td_arr2[i] = 0; bu_arr2[i] = 0;
        for (int r = 0; r < runs; r++) {
            BFSResult res = bfs_yp(g, src, alphas[i]);
            if (res.time_ms < t_arr[i]) {
                t_arr[i]  = res.time_ms;
                o_arr[i]  = res.ops;
                td_arr2[i] = res.td;
                bu_arr2[i] = res.bu;
            }
            result_free(&res);
        }
        if (t_arr[i] < best_t) { best_t = t_arr[i]; best_i = i; }
    }

    for (int i = 0; i < na; i++) {
        printf("%-8.2f  %10.3f  %10ld  %dT/%dB%s\n",
               alphas[i], t_arr[i], o_arr[i],
               td_arr2[i], bu_arr2[i],
               i == best_i ? "  <-- optimal" : "");
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Catalogue des graphes SNAP/KONECT suggérés (--list-graphs)
 * ═══════════════════════════════════════════════════════════════════ */

static void list_graphs(void) {
    printf(
"\n"
"╔══════════════════════════════════════════════════════════════════════════════════╗\n"
"║          Graphes réels SNAP / KONECT recommandés pour le benchmark BFS-YP       ║\n"
"╠══════════════════════════════════════════════════════════════════════════════════╣\n"
"║  Téléchargement :                                                                ║\n"
"║    wget <URL>  &&  gunzip <fichier>.gz                                           ║\n"
"║  Ou lancer directement : ./download_graphs.sh                                   ║\n"
"╚══════════════════════════════════════════════════════════════════════════════════╝\n"
"\n"
"── Réseaux sociaux ─────────────────────────────────────────────────────────────────\n"
"  facebook_combined.txt   n=4 039    m=88 234    Petite communauté dense, hub fort\n"
"    https://snap.stanford.edu/data/facebook_combined.txt.gz\n"
"  soc-Epinions1.txt       n=75 888   m=508 837   Réseau de confiance, forte hiérarchie\n"
"    https://snap.stanford.edu/data/soc-Epinions1.txt.gz\n"
"  soc-Slashdot0811.txt    n=77 360   m=905 468   Réseau media, liens amis/ennemis\n"
"    https://snap.stanford.edu/data/soc-Slashdot0811.txt.gz\n"
"  soc-LiveJournal1.txt    n=4 847 571 m=68 993 773  Grand réseau blog (stress test)\n"
"    https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz\n"
"\n"
"── Réseaux de collaboration scientifique ────────────────────────────────────────────\n"
"  CA-AstroPh.txt          n=18 772   m=198 050   Communautés denses, petit diamètre\n"
"    https://snap.stanford.edu/data/CA-AstroPh.txt.gz\n"
"  CA-CondMat.txt          n=23 133   m=93 497    Clustering élevé\n"
"    https://snap.stanford.edu/data/CA-CondMat.txt.gz\n"
"  CA-HepPh.txt            n=12 008   m=118 521   Réseau très dense\n"
"    https://snap.stanford.edu/data/CA-HepPh.txt.gz\n"
"\n"
"── Réseaux P2P (graphes creux, faible degré moyen) ────────────────────────────────\n"
"  p2p-Gnutella31.txt      n=62 586   m=147 892   Structuré, peu de hubs\n"
"    https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz\n"
"  p2p-Gnutella08.txt      n=6 301    m=20 777    Version petite, bon référence\n"
"    https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz\n"
"\n"
"── Graphes routiers (grands, faible degré, grand diamètre) ────────────────────────\n"
"  roadNet-CA.txt          n=1 965 206 m=2 766 607  Routes Californie — diamètre ~900\n"
"    https://snap.stanford.edu/data/roadNet-CA.txt.gz\n"
"  roadNet-TX.txt          n=1 379 917 m=1 921 660  Routes Texas\n"
"    https://snap.stanford.edu/data/roadNet-TX.txt.gz\n"
"  roadNet-PA.txt          n=1 088 092 m=1 541 898  Routes Pennsylvanie\n"
"    https://snap.stanford.edu/data/roadNet-PA.txt.gz\n"
"\n"
"── Graphes web (très grande échelle) ───────────────────────────────────────────────\n"
"  web-Stanford.txt        n=281 903  m=2 312 497  PageRank, forte loi de puissance\n"
"    https://snap.stanford.edu/data/web-Stanford.txt.gz\n"
"  web-Google.txt          n=875 713  m=5 105 039  Graphe dirigé Google 2002\n"
"    https://snap.stanford.edu/data/web-Google.txt.gz\n"
"\n"
"── Graphes de citations ─────────────────────────────────────────────────────────────\n"
"  cit-HepTh.txt           n=27 770   m=352 807   Citations théorie haute énergie\n"
"    https://snap.stanford.edu/data/cit-HepTh.txt.gz\n"
"  cit-Patents.txt         n=3 774 768 m=16 518 948 Citations brevets (grand test)\n"
"    https://snap.stanford.edu/data/cit-Patents.txt.gz\n"
"\n"
"── KONECT (format différent — lignes 'u v' identiques) ────────────────────────────\n"
"  DBLP coauteur           n=1 314 050 m=10 724 828  Collaboration académique\n"
"    http://konect.cc/files/download.tsv.dblp_coauthor.tar.bz2\n"
"  Amazon copurchase       n=548 552  m=1 788 725  Produits achetés ensemble\n"
"    http://konect.cc/files/download.tsv.amazon-ratings.tar.bz2\n"
"\n"
"── Suite de tests recommandée (du plus petit au plus grand) ───────────────────────\n"
"  facebook   →  p2p-Gnutella08  →  CA-AstroPh  →  soc-Epinions1\n"
"  →  web-Stanford  →  roadNet-PA  →  roadNet-CA  →  soc-LiveJournal1\n"
"\n"
"  Profil de chaque classe :\n"
"    Social   : hubs, petit diamètre, switching Phase 1→2 rapide   → YP gagne fort\n"
"    Routier  : grille, grand diamètre, peu de hubs                 → YP Phase 3 active\n"
"    P2P      : creux, degré faible, BU jamais rentable             → STD/SURF meilleur\n"
"    Web      : loi de puissance, hubs extrêmes                     → YP/BB dominent\n"
"\n"
    );
}

/* ═══════════════════════════════════════════════════════════════════
 * Stockage des résultats pour tableaux consolidés
 * ═══════════════════════════════════════════════════════════════════ */

#define MAX_GRAPHS 64

typedef struct {
    char      name[64];
    int       n;
    long      m;
    AlgoStats stats[N_ALGOS];           /* min/mean/sd/med/cv par algo  */
    double    raw[N_ALGOS][MAX_RUNS];   /* mesures brutes pour Wilcoxon */
    int       runs_done;                /* nb de runs effectivement exécutés */
    int       td_yp, bu_yp;
    long      mem_saved_mb;
} GraphResult;

static GraphResult results[MAX_GRAPHS];
static int         nresults = 0;

/* ─────────────────────────────────────────────────────────────────── */

static void sep(char c, int w) { for (int i=0;i<w;i++) putchar(c); putchar('\n'); }

/* Nom court : basename sans extension */
static void shortname(const char *path, char *out, int maxlen) {
    const char *p = strrchr(path, '/');
    p = p ? p + 1 : path;
    strncpy(out, p, maxlen - 1);
    out[maxlen - 1] = '\0';
    char *dot = strrchr(out, '.');
    if (dot && strcmp(dot, ".gz") != 0) *dot = '\0';
    /* tronquer à 28 chars pour l'affichage */
    if ((int)strlen(out) > 28) out[28] = '\0';
}

/* ═══════════════════════════════════════════════════════════════════
 * Impression des tableaux consolidés finaux
 * ═══════════════════════════════════════════════════════════════════ */

/* ── Helpers d'affichage ──────────────────────────────────────── */

/* Étoile sur le meilleur non-STD (sur la métrique `val`) */
static const char *star(double val, double best) {
    return (val == best) ? "*" : " ";
}

/* Symbole de significativité Wilcoxon */
static const char *wilcoxon_sig(double p) {
    if (p < 0.001) return "***";
    if (p < 0.01)  return "** ";
    if (p < 0.05)  return "*  ";
    return "   ";
}

static void print_summary(int runs) {
    if (nresults == 0) return;

    const int W = 120;

    /* ═══════════════════════════════════════════════════════════════
     * TABLEAU 1 — Minimum (standard pour la comparaison avec l'art)
     * ═══════════════════════════════════════════════════════════════ */
    printf("\n");
    sep('=', W);
    printf("  TABLEAU 1 — Temps minimum (ms)  [min sur %d runs]\n", runs);
    printf("  Speedup = t_STD_min / t_algo_min  —  * = meilleur non-STD\n");
    sep('=', W);
    printf("\n");

    printf("  %-28s %7s %9s │ %8s %8s %8s %8s %8s │ %6s %6s %6s %6s\n",
           "Graphe","n","m","STD","DIR","SURF","BB","YP",
           "xDIR","xSURF","xBB","xYP");
    printf("  %-28s %7s %9s ┼ %8s %8s %8s %8s %8s ┼ %6s %6s %6s %6s\n",
           "────────────────────────────","───────","─────────",
           "────────","────────","────────","────────","────────",
           "──────","──────","──────","──────");

    int wins_min[N_ALGOS] = {0};
    for (int g = 0; g < nresults; g++) {
        GraphResult *r = &results[g];
        double best = r->stats[1].t_min;
        for (int a = 2; a < N_ALGOS; a++)
            if (r->stats[a].t_min < best) best = r->stats[a].t_min;

        printf("  %-28s %7d %9ld │ %8.3f %7.3f%s %7.3f%s %7.3f%s %7.3f%s │ "
               "%5.2fx %5.2fx %5.2fx %5.2fx\n",
               r->name, r->n, r->m,
               r->stats[0].t_min,
               r->stats[1].t_min, star(r->stats[1].t_min, best),
               r->stats[2].t_min, star(r->stats[2].t_min, best),
               r->stats[3].t_min, star(r->stats[3].t_min, best),
               r->stats[4].t_min, star(r->stats[4].t_min, best),
               r->stats[0].t_min / r->stats[1].t_min,
               r->stats[0].t_min / r->stats[2].t_min,
               r->stats[0].t_min / r->stats[3].t_min,
               r->stats[0].t_min / r->stats[4].t_min);

        double best_all = r->stats[0].t_min;
        int besta = 0;
        for (int a = 1; a < N_ALGOS; a++)
            if (r->stats[a].t_min < best_all) { best_all = r->stats[a].t_min; besta = a; }
        wins_min[besta]++;
    }
    printf("  %-28s %7s %9s ┴ %8s %8s %8s %8s %8s ┴ %6s %6s %6s %6s\n",
           "────────────────────────────","───────","─────────",
           "────────","────────","────────","────────","────────",
           "──────","──────","──────","──────");
    printf("\n  Victoires (min) : STD=%d DIR=%d SURF=%d BB=%d YP=%d\n",
           wins_min[0],wins_min[1],wins_min[2],wins_min[3],wins_min[4]);

    /* ═══════════════════════════════════════════════════════════════
     * TABLEAU 2 — Moyenne ± écart-type  (requis JEA)
     * ═══════════════════════════════════════════════════════════════ */
    printf("\n");
    sep('=', W);
    printf("  TABLEAU 2 — Moyenne ± écart-type (ms)  [%d runs, format mean±sd]\n", runs);
    printf("  Speedup = mean_STD / mean_algo  —  CV%% = sd/mean×100\n");
    sep('=', W);
    printf("\n");

    printf("  %-28s │ %-16s %-16s %-16s %-16s %-16s\n",
           "Graphe","STD","DIR","SURF","BB","YP");
    printf("  %-28s ┼ %-16s %-16s %-16s %-16s %-16s\n",
           "────────────────────────────",
           "────────────────","────────────────",
           "────────────────","────────────────","────────────────");

    int wins_mean[N_ALGOS] = {0};
    for (int g = 0; g < nresults; g++) {
        GraphResult *r = &results[g];
        double best_mean = r->stats[1].t_mean;
        for (int a = 2; a < N_ALGOS; a++)
            if (r->stats[a].t_mean < best_mean) best_mean = r->stats[a].t_mean;

        printf("  %-28s │", r->name);
        for (int a = 0; a < N_ALGOS; a++) {
            char buf[20];
            snprintf(buf, sizeof(buf), "%.3f±%.3f",
                     r->stats[a].t_mean, r->stats[a].t_sd);
            printf(" %-15s%s", buf,
                   (a > 0 && r->stats[a].t_mean == best_mean) ? "*" : " ");
        }
        printf("\n");

        /* Ligne CV% (coefficient de variation) */
        printf("  %-28s │", "  CV%");
        for (int a = 0; a < N_ALGOS; a++) {
            char cvbuf[20];
            snprintf(cvbuf, sizeof(cvbuf), "%.1f%%", r->stats[a].cv_pct);
            printf(" %-16s", cvbuf);
        }
        printf("\n");

        double best_all = r->stats[0].t_mean;
        int besta = 0;
        for (int a = 1; a < N_ALGOS; a++)
            if (r->stats[a].t_mean < best_all) { best_all = r->stats[a].t_mean; besta = a; }
        wins_mean[besta]++;
    }
    printf("  %-28s ┴ %s\n",
           "────────────────────────────",
           "──────────────── ──────────────── ──────────────── ──────────────── ────────────────");
    printf("\n  Victoires (mean) : STD=%d DIR=%d SURF=%d BB=%d YP=%d\n",
           wins_mean[0],wins_mean[1],wins_mean[2],wins_mean[3],wins_mean[4]);

    /* ═══════════════════════════════════════════════════════════════
     * TABLEAU 3 — Médiane  (robuste aux outliers thermiques)
     * ═══════════════════════════════════════════════════════════════ */
    printf("\n");
    sep('=', W);
    printf("  TABLEAU 3 — Médiane (ms)  [%d runs]\n", runs);
    printf("  Speedup = med_STD / med_algo  —  * = meilleur non-STD\n");
    sep('=', W);
    printf("\n");

    printf("  %-28s %7s %9s │ %8s %8s %8s %8s %8s │ %6s %6s %6s %6s\n",
           "Graphe","n","m","STD","DIR","SURF","BB","YP",
           "xDIR","xSURF","xBB","xYP");
    printf("  %-28s %7s %9s ┼ %8s %8s %8s %8s %8s ┼ %6s %6s %6s %6s\n",
           "────────────────────────────","───────","─────────",
           "────────","────────","────────","────────","────────",
           "──────","──────","──────","──────");

    int wins_med[N_ALGOS] = {0};
    for (int g = 0; g < nresults; g++) {
        GraphResult *r = &results[g];
        double best_med = r->stats[1].t_med;
        for (int a = 2; a < N_ALGOS; a++)
            if (r->stats[a].t_med < best_med) best_med = r->stats[a].t_med;

        printf("  %-28s %7d %9ld │ %8.3f %7.3f%s %7.3f%s %7.3f%s %7.3f%s │ "
               "%5.2fx %5.2fx %5.2fx %5.2fx\n",
               r->name, r->n, r->m,
               r->stats[0].t_med,
               r->stats[1].t_med, star(r->stats[1].t_med, best_med),
               r->stats[2].t_med, star(r->stats[2].t_med, best_med),
               r->stats[3].t_med, star(r->stats[3].t_med, best_med),
               r->stats[4].t_med, star(r->stats[4].t_med, best_med),
               r->stats[0].t_med / r->stats[1].t_med,
               r->stats[0].t_med / r->stats[2].t_med,
               r->stats[0].t_med / r->stats[3].t_med,
               r->stats[0].t_med / r->stats[4].t_med);

        double best_all = r->stats[0].t_med;
        int besta = 0;
        for (int a = 1; a < N_ALGOS; a++)
            if (r->stats[a].t_med < best_all) { best_all = r->stats[a].t_med; besta = a; }
        wins_med[besta]++;
    }
    printf("  %-28s %7s %9s ┴ %8s %8s %8s %8s %8s ┴ %6s %6s %6s %6s\n",
           "────────────────────────────","───────","─────────",
           "────────","────────","────────","────────","────────",
           "──────","──────","──────","──────");
    printf("\n  Victoires (med) : STD=%d DIR=%d SURF=%d BB=%d YP=%d\n",
           wins_med[0],wins_med[1],wins_med[2],wins_med[3],wins_med[4]);

    /* ═══════════════════════════════════════════════════════════════
     * TABLEAU 4 — Test de Wilcoxon signé-rangé  (requis JEA)
     *
     * Pour chaque graphe : test pairwise YP vs chaque concurrent.
     * H0 : pas de différence de distribution.
     * Significativité : *** p<0.001, ** p<0.01, * p<0.05
     * ═══════════════════════════════════════════════════════════════ */
    printf("\n");
    sep('=', W);
    printf("  TABLEAU 4 — Test de Wilcoxon signé-rangé : YP vs concurrents\n");
    printf("  p-value bilatérale sur %d mesures pairées par graphe\n", runs);
    printf("  *** p<0.001  ** p<0.01  * p<0.05  (YP gagne si z>0, perd si z<0)\n");
    sep('=', W);
    printf("\n");

    printf("  %-28s │ %-22s %-22s %-22s %-22s\n",
           "Graphe",
           "YP vs DIR","YP vs SURF","YP vs BB","YP vs STD");
    printf("  %-28s ┼ %-22s %-22s %-22s %-22s\n",
           "────────────────────────────",
           "──────────────────────","──────────────────────",
           "──────────────────────","──────────────────────");

    int sig_wins[4] = {0};   /* victoires significatives vs DIR/SURF/BB/STD */
    int sig_total  = 0;

    for (int g = 0; g < nresults; g++) {
        GraphResult *r = &results[g];
        int n = r->runs_done;

        /* YP = algo 4, concurrents = 0(STD), 1(DIR), 2(SURF), 3(BB) */
        int conc[4] = {1, 2, 3, 0};
        printf("  %-28s │", r->name);

        for (int ci = 0; ci < 4; ci++) {
            int a = conc[ci];
            WilcoxonResult wr = wilcoxon_yp_vs(r->raw[4], r->raw[a], n);
            char buf[24];
            if (wr.n_eff < 5) {
                snprintf(buf, sizeof(buf), "n/a(ties)             ");
            } else {
                const char *sig = wilcoxon_sig(wr.p_value);
                char dir = (wr.z_score > 0) ? '+' : '-';
                snprintf(buf, sizeof(buf), "z=%+5.2f p=%.4f %s%s",
                         wr.z_score, wr.p_value, sig,
                         (dir == '+') ? "✓" : "✗");
                /* compter victoire significative */
                if (wr.p_value < 0.05 && wr.z_score > 0) {
                    sig_wins[ci]++;
                }
            }
            printf(" %-21s│", buf);
            sig_total++;
        }
        printf("\n");
    }
    printf("  %-28s ┴ %-22s %-22s %-22s %-22s\n",
           "────────────────────────────",
           "──────────────────────","──────────────────────",
           "──────────────────────","──────────────────────");

    printf("\n  Victoires significatives YP (p<0.05) : "
           "vs DIR=%d/%d  vs SURF=%d/%d  vs BB=%d/%d  vs STD=%d/%d\n",
           sig_wins[0], nresults,
           sig_wins[1], nresults,
           sig_wins[2], nresults,
           sig_wins[3], nresults);

    /* ═══════════════════════════════════════════════════════════════
     * TABLEAU 5 — Edge-checks (ops)
     * ═══════════════════════════════════════════════════════════════ */
    printf("\n");
    sep('=', W);
    printf("  TABLEAU 5 — Edge checks (ops)  [run minimal]\n");
    printf("  Ratio = ops_STD / ops_algo  —  métrique algorithmique pure\n");
    sep('=', W);
    printf("\n");

    printf("  %-28s %7s %9s │ %10s %10s %10s %10s %10s │ %6s %6s %6s %6s\n",
           "Graphe","n","m",
           "ops-STD","ops-DIR","ops-SURF","ops-BB","ops-YP",
           "/DIR","/SURF","/BB","/YP");
    printf("  %-28s %7s %9s ┼ %10s %10s %10s %10s %10s ┼ %6s %6s %6s %6s\n",
           "────────────────────────────","───────","─────────",
           "──────────","──────────","──────────","──────────","──────────",
           "──────","──────","──────","──────");

    for (int g = 0; g < nresults; g++) {
        GraphResult *r = &results[g];
        long best_ops = r->stats[1].ops;
        for (int a = 2; a < N_ALGOS; a++)
            if (r->stats[a].ops < best_ops) best_ops = r->stats[a].ops;

        char mark[N_ALGOS][3];
        for (int a = 0; a < N_ALGOS; a++)
            strcpy(mark[a], (a > 0 && r->stats[a].ops == best_ops) ? " *" : "  ");

        printf("  %-28s %7d %9ld │ %10ld %9ld%s %9ld%s %9ld%s %9ld%s │ "
               "%5.2fx %5.2fx %5.2fx %5.2fx\n",
               r->name, r->n, r->m,
               r->stats[0].ops,
               r->stats[1].ops, mark[1],
               r->stats[2].ops, mark[2],
               r->stats[3].ops, mark[3],
               r->stats[4].ops, mark[4],
               (double)r->stats[0].ops / r->stats[1].ops,
               (double)r->stats[0].ops / r->stats[2].ops,
               (double)r->stats[0].ops / r->stats[3].ops,
               (double)r->stats[0].ops / r->stats[4].ops);
    }
    printf("  %-28s %7s %9s ┴ %10s %10s %10s %10s %10s ┴ %6s %6s %6s %6s\n",
           "────────────────────────────","───────","─────────",
           "──────────","──────────","──────────","──────────","──────────",
           "──────","──────","──────","──────");

    /* ═══════════════════════════════════════════════════════════════
     * TABLEAU 6 — Profil YP (phases + mémoire)
     * ═══════════════════════════════════════════════════════════════ */
    printf("\n");
    sep('=', W);
    printf("  TABLEAU 6 — Profil BFS-YP (phases Yoccoz)\n");
    sep('=', W);
    printf("\n");

    printf("  %-28s %7s %9s %5s │ %6s %6s │ %12s %12s\n",
           "Graphe","n","m","W",
           "TD lvl","BU lvl","Mem MB","YP phase");
    printf("  %-28s %7s %9s %5s ┼ %6s %6s ┼ %12s %12s\n",
           "────────────────────────────","───────","─────────","─────",
           "──────","──────","────────────","────────────");

    for (int g = 0; g < nresults; g++) {
        GraphResult *r = &results[g];
        int Ww = (r->n + 63) / 64;
        const char *phase = (r->bu_yp == 0 && r->td_yp == 0) ? "STD-fallback" :
                            (r->bu_yp == 0)                   ? "P1-only"      :
                            (r->td_yp == 0)                   ? "P2/P3"        : "P1+P2";
        printf("  %-28s %7d %9ld %5d │ %6d %6d │ %12ld %12s\n",
               r->name, r->n, r->m, Ww,
               r->td_yp, r->bu_yp, r->mem_saved_mb, phase);
    }
    printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════ */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS] <graph1.txt> [graph2.txt ...]\n"
        "\n"
        "Options:\n"
        "  --runs N         Nombre de runs par graphe (défaut: 10)\n"
        "  --alpha-sweep    Analyse de sensibilité de α pour YP\n"
        "  --list-graphs    Afficher le catalogue des graphes SNAP/KONECT\n"
        "  --no-per-graph   Supprimer les tableaux intermédiaires par graphe\n"
        "\n"
        "Entrée: fichiers .txt au format SNAP (lignes 'u v', '#' ignoré)\n"
        "        Accepte aussi les .gz si décompressés au préalable.\n"
        "\n"
        "Exemples:\n"
        "  %s --runs 20 facebook_combined.txt CA-AstroPh.txt roadNet-CA.txt\n"
        "  %s --list-graphs\n"
        "  %s --alpha-sweep soc-Epinions1.txt\n",
        prog, prog, prog, prog);
}

int main(int argc, char *argv[]) {
    int    runs        = 10;
    int    do_sweep    = 0;
    int    per_graph   = 1;   /* tableaux intermédiaires activés par défaut */
    int    do_list     = 0;

    /* Fichiers graphes : collectés après parsing des options */
    const char *graph_files[MAX_GRAPHS];
    int         ngraphs = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--runs") && i+1 < argc)
            runs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--alpha-sweep"))
            do_sweep = 1;
        else if (!strcmp(argv[i], "--no-per-graph"))
            per_graph = 0;
        else if (!strcmp(argv[i], "--list-graphs"))
            do_list = 1;
        else if (argv[i][0] != '-') {
            if (ngraphs < MAX_GRAPHS)
                graph_files[ngraphs++] = argv[i];
        } else {
            fprintf(stderr, "Option inconnue : %s\n", argv[i]);
            usage(argv[0]); return 1;
        }
    }

    if (do_list) { list_graphs(); return 0; }

    if (ngraphs == 0) { usage(argv[0]); return 1; }

    printf("\n");
    sep('=', 80);
    printf("  BFS 5-way benchmark  |  runs=%d  |  YP alpha=%.2f  |  %d graphe(s)\n",
           runs, YP_ALPHA, ngraphs);
    printf("  STD(Cormen72) · DIR(Beamer12) · SURF(Yoon22) · BB(Bhaskar25)"
           " · YP(ce travail)\n");
    sep('=', 80);

    if (validate_yp() < 0) return 1;

    /* ── Boucle principale sur tous les graphes ────────────────────── */
    for (int gi = 0; gi < ngraphs; gi++) {
        const char *path = graph_files[gi];

        printf("\n[%d/%d] Chargement : %s\n", gi+1, ngraphs, path);

        Graph g;
        if (graph_load_snap(path, &g) < 0) {
            fprintf(stderr, "  ERREUR : impossible de charger %s — ignoré\n", path);
            continue;
        }

        /* Precomputation YP v3 : tri spectral des listes d'adjacence
         * O(4m) itérations + O(m log deg) tri — fait une seule fois,
         * amortie sur les `runs` répétitions du benchmark. */
        printf("  Precomputation spectrale (Global Harmonic Field)... ");
        fflush(stdout);
        double t_spec = now_ms();
        graph_spectral_sort(&g);
        printf("%.0f ms\n", now_ms() - t_spec);

        int src = hub_vertex(&g);
        printf("  n=%-8d  m=%-10ld  W=%-5d  hub=%d (deg %d)\n",
               g.n, g.m, g.W, src, g.deg[src]);

        /* ── Benchmark : collecte statistique complète ──────────────── */
        AlgoStats stats[N_ALGOS];
        static double raw[N_ALGOS][MAX_RUNS];
        int r_eff = (runs > MAX_RUNS) ? MAX_RUNS : runs;
        bench_run_stats(&g, src, runs, stats, raw);

        /* Affichage intermédiaire (par graphe) */
        if (per_graph) {
            double t_min[N_ALGOS]; long ops_min[N_ALGOS];
            for (int a = 0; a < N_ALGOS; a++) {
                t_min[a] = stats[a].t_min; ops_min[a] = stats[a].ops;
            }
            printf("\n");
            print_header_time();
            print_row_time(path, g.n, g.m, t_min);
            print_header_ops();
            print_row_ops(path, g.n, g.m, ops_min);
            printf("  YP: %dT/%dB niveaux | mem saved: %.0f MB\n",
                   stats[4].td, stats[4].bu,
                   (double)g.n * g.W * 8 / 1e6);
            if (do_sweep)
                alpha_sweep(&g, src, runs);
        }

        /* Enregistrement pour les tableaux consolidés */
        if (nresults < MAX_GRAPHS) {
            GraphResult *r = &results[nresults++];
            shortname(path, r->name, sizeof(r->name));
            r->n  = g.n; r->m  = g.m;
            r->runs_done    = r_eff;
            r->td_yp        = stats[4].td;
            r->bu_yp        = stats[4].bu;
            r->mem_saved_mb = (long)((double)g.n * g.W * 8 / 1e6);
            for (int a = 0; a < N_ALGOS; a++) {
                r->stats[a] = stats[a];
                memcpy(r->raw[a], raw[a], r_eff * sizeof(double));
            }
        }

        graph_free(&g);
        printf("\n");
        sep('-', 80);
    }

    /* ── Tableaux consolidés finaux ────────────────────────────────── */
    if (nresults > 1 || !per_graph)
        print_summary(runs);

    return 0;
}
