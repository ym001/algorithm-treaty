/*
 * BFS_AA.c — BFS 5-way benchmark
 * Algorithmes : STD · DIR · SURF · BB · AA (PHP-BFS v1)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * AA — PHP-BFS 
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *
 * ── Structure fondamentale ───────────────────────────────────────────────────
 *
 *   n sommets divisés en Nb = ⌈n/B⌉ pièces de taille B (B = AA_BLOCK = 512).
 *   Chaque pièce p couvre les sommets [p·B, (p+1)·B).
 *
 *   Deux bitvecteurs imbriqués :
 *     frontier_fine [W   mots] : frontière exacte au niveau sommet
 *     active_pieces [Wc  mots] : bit p = 1 ssi pièce p a encore des
 *                                 non-visités (piece_uv[p] > 0)
 *     avec Wc = ⌈Nb/64⌉  →  pour n=1M, B=512 : Wc = ⌈1954/64⌉ = 31 mots !
 *
 * ── Trois phases ────────────────────────────────────────────────────────────
 *
 *  Phase 1  [top-down]
 *    Condition : WE < α·SDU   (frontière légère, TD moins cher)
 *    Action    : expansion CSR classique depuis chaque sommet de frontière
 *    SDU mis à jour incrémentalement : SDU -= deg(v) à chaque découverte
 *
 *  Phase 2  [bottom-up hiérarchique, uv ≥ θ·n]
 *    Condition : WE ≥ α·SDU et uv grand
 *    Action    : scan à deux niveaux
 *      for cw in active_pieces            ← Wc itérations (ex: 31 au lieu de 16000)
 *        if cw == 0 → skip 64 pièces en une instruction   ← LE GAIN CLÉ
 *        for each bit set in cw           ← au plus 64
 *          for v in piece p               ← au plus B=512 sommets
 *            test CSR + frontier_fine     ← early-exit au 1er voisin frontière
 *      Pièces épuisées : bit effacé de active_pieces → jamais revisitées
 *
 *  Phase 3  [bottom-up liste, uv < θ·n]
 *    Condition : WE ≥ α·SDU et uv petit
 *    Action    : liste unvis[] construite une fois en O(n), compactée swap-last
 *    Coût      : O(uv × avg_deg) avec uv → 0
 *
 * ── Différence fondamentale vs l'art ────────────────────────────────────────
 *
 *   STD   : scan plat O(n) — aucune structure
 *   DIR   : switch TD/BU sur heuristique α/β, scan plat O(n) en BU
 *   SURF  : switch EMA+hystérésis, scan plat O(n) en BU
 *   BB    : visited-bitmap, adj_mask dense (limité aux petits graphes)
 *   AA    : scan HIÉRARCHIQUE O(Wc + pieces_actives × B) en BU, avec
 *           Wc = n/(64×B) ≪ n. Les pièces épuisées ne sont JAMAIS visitées.
 *
 *   Sur graphes sociaux/web (communautés denses) : des pièces entières
 *   s'épuisent rapidement → active_pieces devient creux → la boucle externe
 *   saute des milliers de sommets en quelques cycles.
 *
 * ── Complexité ───────────────────────────────────────────────────────────────
 *   Phase 2 : O(Wc + Σ_levels |active_pieces_scanned| × B)
 *             ≤ O(n/64B × L + m/B) dans le pire cas
 *   Phase 3 : O(uv × avg_deg) → O(m/n × n) = O(m)
 *   SDU maintenu en O(1) par découverte, WE en O(|frontière|) par niveau.
 *
 * Compilation :
 *   gcc -O3 -march=native -o benchmark_v4 benchmark_v4.c -lm
 *
 * Usage :
 *   ./benchmark_v4 graph1.txt graph2.txt --runs 20 [--alpha-sweep]
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════
 * Constantes globales
 * ═══════════════════════════════════════════════════════════════════ */

#define YP_BLOCK   512     /* taille d'une pièce de puzzle (sommets)        */
#define YP_ALPHA   1.0     /* seuil WE/SDU de basculement TD→BU (optimal=1) */
#define YP_THETA   0.14    /* seuil uv/n de basculement Phase2→Phase3        */
#define MAX_BB_N   8192    /* seuil adj_mask dense pour BB                  */
#define N_ALGOS    5
#define MAX_RUNS   200
#define MAX_GRAPHS 64

/* ═══════════════════════════════════════════════════════════════════
 * Utilitaires
 * ═══════════════════════════════════════════════════════════════════ */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ═══════════════════════════════════════════════════════════════════
 * Graphe CSR
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    int   n;        /* nombre de sommets                            */
    long  m;        /* nombre d'arêtes (non-dirigées)               */
    int  *xadj;     /* xadj[v]..xadj[v+1]-1 = voisins de v         */
    int  *adj;      /* tableau des voisins                          */
    int  *deg;      /* deg[v] = xadj[v+1]-xadj[v]                  */
    int   W;        /* ceil(n/64) — largeur bitvecteur sommet        */
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

    int  max_v = -1;
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

    int n   = max_v + 1;
    int *cnt = (int*)calloc(n, sizeof(int));
    if (!cnt) { fclose(f); return -1; }

    rewind(f);
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        int u, v;
        if (sscanf(line, "%d %d", &u, &v) != 2) continue;
        if (u == v) continue;
        cnt[u]++; cnt[v]++;
    }

    int *xadj = (int*)malloc((n + 1) * sizeof(int));
    xadj[0] = 0;
    for (int i = 0; i < n; i++) xadj[i+1] = xadj[i] + cnt[i];
    int  total_adj = xadj[n];
    int *adj  = (int*)malloc(total_adj * sizeof(int));
    int *pos  = (int*)calloc(n, sizeof(int));
    if (!xadj || !adj || !pos) { fclose(f); free(cnt); return -1; }

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

/* Sommet de degré maximum */
static int hub_vertex(const Graph *g) {
    int hub = 0;
    for (int v = 1; v < g->n; v++)
        if (g->deg[v] > g->deg[hub]) hub = v;
    return hub;
}

/* ═══════════════════════════════════════════════════════════════════
 * Résultat BFS
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    double  time_ms;
    long    ops;      /* arcs examinés (edge-checks)                   */
    int     td;       /* niveaux top-down                              */
    int     bu;       /* niveaux bottom-up                             */
    int    *dist;     /* distances allouées ici, libérées par l'appelant */
} BFSResult;

static void result_free(BFSResult *r) { free(r->dist); r->dist = NULL; }

/* ═══════════════════════════════════════════════════════════════════
 * 1. STD — BFS classique (Cormen 2022)
 * ═══════════════════════════════════════════════════════════════════ */

static BFSResult bfs_std(const Graph *g, int src) {
    int   n     = g->n;
    int  *dist  = (int*)malloc(n * sizeof(int));
    int  *queue = (int*)malloc(n * sizeof(int));
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
    long m2 = 2 * g->m;
    int *dist  = (int*)malloc(n * sizeof(int));
    int *front = (int*)malloc(n * sizeof(int));
    int *nxt   = (int*)malloc(n * sizeof(int));
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
    int  n = g->n;
    int  W = g->W;
    int *dist  = (int*)malloc(n * sizeof(int));
    int *front = (int*)malloc(n * sizeof(int));
    int *nxt   = (int*)malloc(n * sizeof(int));
    uint64_t *fv  = (uint64_t*)calloc(W, sizeof(uint64_t));
    uint64_t *nfv = (uint64_t*)calloc(W, sizeof(uint64_t));
    memset(dist, -1, n * sizeof(int));

    const double BETA_EMA  = 0.6;
    const double TAU_ENTER = 1.0;
    const double TAU_EXIT  = 0.5;
    const int    COOLDOWN  = 2;

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
        ema = BETA_EMA * ema + (1.0 - BETA_EMA) * WR;

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
 * visited-bitmap compact (n/64 mots).
 * adj_mask dense uniquement pour n ≤ MAX_BB_N.
 * ═══════════════════════════════════════════════════════════════════ */

static BFSResult bfs_bb(const Graph *g, int src) {
    int  n  = g->n;
    int  W  = g->W;
    int *dist  = (int*)malloc(n * sizeof(int));
    int *front = (int*)malloc(n * sizeof(int));
    int *nxt   = (int*)malloc(n * sizeof(int));
    uint64_t *fv  = (uint64_t*)calloc(W, sizeof(uint64_t));
    uint64_t *nfv = (uint64_t*)calloc(W, sizeof(uint64_t));
    uint64_t *vis = (uint64_t*)calloc(W, sizeof(uint64_t));
    memset(dist, -1, n * sizeof(int));

    const double ALPHA_BB = 1.2;

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
                for (int v = 0; v < n; v++) {
                    if (vis[v >> 6] & (1ULL << (v & 63))) continue;
                    ops++;
                    int hit = 0;
                    for (int k = 0; k < W; k++) {
                        if (adj_mask[v][k] & fv[k]) { hit = 1; break; }
                    }
                    if (hit) {
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

/* ═══════════════════════════════════════════════════════════════════
 * 5. AA — Almost-Reducibility BFS
 *        Inspiré d'Artur Ávila (Médaille Fields 2014)
 *        Travaux sur les systèmes dynamiques : renormalisation,
 *        presque-réductibilité des cocycles quasi-périodiques.
 *
 * ── Principe fondamental ──────────────────────────────────────────
 *
 *   Ávila démontre qu'un cocycle quasi-périodique peut être réduit
 *   à une forme plus simple à la bonne échelle (Almost Reducibility
 *   Theorem). La clé : identifier les échelles "résonantes" où la
 *   dynamique est non-triviale, et sauter les échelles "non-résonantes".
 *
 *   Transposé en BFS :
 *     - "Résonant"     = bloc contenant des sommets non-visités actifs
 *     - "Non-résonant" = bloc entièrement visité → skip O(1)
 *     - La renormalisation multi-échelle identifie en O(Wcc) quels
 *       blocs de 262 144 sommets méritent d'être explorés.
 *
 *   AA  : 3 niveaux  (Wc=⌈n/262144⌉)  →  8× moins d'itérations ext.
 *         + prefetch double en TD
 *         + phase C liste compacte
 *         + hystérésis sur le switch BU→TD
 *
 * ═══════════════════════════════════════════════════════════════════ */

#define AA_THETA       0.01   /* seuil uv/n  phase-B → phase-C          */
#define AA_SPECTRAL_CUT 8      /* seuil degré résonant/non-résonant       */
#define AA_COOLDOWN     8      /* hystérésis BU→TD                        */

static BFSResult bfs_aa(const Graph *g, int src, double alpha) {
    int   n  = g->n;
    int   W  = g->W;               /* ceil(n/64)      */
    int   Wm = (W  + 63) >> 6;    /* ceil(W/64)       */
    int   Wc = (Wm + 63) >> 6;    /* ceil(Wm/64)      */

    /* ── Allocations ──────────────────────────────────────────────── */
    int      *dist  = (int*)     malloc(n * sizeof(int));
    int      *front = (int*)     malloc(n * sizeof(int));
    int      *nxt   = (int*)     malloc(n * sizeof(int));
    /* uv0 : bitmap des sommets non-visités (toujours alloué)          */
    uint64_t *uv0   = (uint64_t*)malloc(W  * sizeof(uint64_t));
    /* uv1, uv2, fv, nfv : construction lazy au 1er passage BU        */
    uint64_t *uv1   = NULL;
    uint64_t *uv2   = NULL;
    uint64_t *fv    = NULL;
    uint64_t *nfv   = NULL;
    int      *unvis = NULL;

    if (!dist || !front || !nxt || !uv0) {
        free(dist); free(front); free(nxt); free(uv0);
        return (BFSResult){0};
    }

    memset(dist, -1, n * sizeof(int));

    /* uv0 : all 1s (all unvisited), mask tail bits */
    memset(uv0, 0xFF, W * sizeof(uint64_t));
    if (n & 63) uv0[W - 1] = (1ULL << (n & 63)) - 1;

    /* Source */
    dist[src] = 0;
    front[0]  = src;
    int fsz   = 1;
    uv0[src >> 6] &= ~(1ULL << (src & 63));

    /* SDU = sum of degrees of unvisited vertices */
    long SDU = 0;
    for (int v = 0; v < n; v++) SDU += g->deg[v];
    SDU -= g->deg[src];

    int    uv_cnt  = n - 1;
    long   ops     = 0;
    int    td = 0, bu = 0, level = 0;
    int    in_bu   = 0, cooldown = 0;
    int    bu_init = 0; /* lazy: fv/nfv/uv1/uv2 built on first BU     */
    int    fv_dirty= 1; /* fv out of sync with front[]                */
    int    phase_c = 0;
    int    unvis_sz= 0;

    double t0 = now_ms();

    while (fsz > 0) {

        /* ── WE et decision de direction ─────────────────────────── */
        long WE = 0;
        for (int i = 0; i < fsz; i++) WE += g->deg[front[i]];
        double sdu_eff = (double)(SDU > 0 ? SDU : 1);

        /* Hysteresis (Avila — spectral thresholds):
         *   Enter BU : WE >= alpha * SDU
         *   Exit  BU : WE <  0.5 * alpha * SDU  */
        int use_bu;
        if (in_bu) {
            use_bu = (WE >= 0.5 * alpha * sdu_eff);
            if (!use_bu) { cooldown = AA_COOLDOWN; in_bu = 0; }
        } else {
            if (cooldown > 0) { cooldown--; use_bu = 0; }
            else { use_bu = (WE >= alpha * sdu_eff); if (use_bu) in_bu = 1; }
        }

        int nsz = 0;

        /* ═══════════════════════════════════════════════════════════
         * Phase A/D — Top-Down
         *
         *  - uv0 as visited bitmap: working-set 16x smaller than dist[]
         *    -> better cache for large graphs (n >= 10M)
         *  - fv/nfv/uv1/uv2 NOT used or allocated in pure-TD mode
         *    -> zero overhead for road graphs (~900 TD levels)
         *  - Double prefetch (Avila spectral separation):
         *      prefetch adj[xadj[front[i+1]]]   (structure)
         *      prefetch uv0[ adj[j+2] >> 6 ]    (data)
         * ═══════════════════════════════════════════════════════════ */
        if (!use_bu) {
            td++;
            fv_dirty = 1;   /* front[] will change; fv becomes stale */
            for (int i = 0; i < fsz; i++) {
                int u = front[i];
                if (i + 1 < fsz) {
                    __builtin_prefetch(g->adj  + g->xadj[front[i+1]], 0, 1);
                    __builtin_prefetch(g->xadj + front[i+1],          0, 1);
                }
                for (int j = g->xadj[u], end = g->xadj[u+1]; j < end; j++) {
                    ops++;
                    int v = g->adj[j];
                    if (j + 2 < end)
                        __builtin_prefetch(uv0 + (g->adj[j+2] >> 6), 0, 1);
                    uint64_t bit = 1ULL << (v & 63);
                    int      w   = v >> 6;
                    if (uv0[w] & bit) {
                        uv0[w]  &= ~bit;
                        dist[v]  = level + 1;
                        nxt[nsz++] = v;
                        SDU -= g->deg[v];
                    }
                }
            }

        /* ═══════════════════════════════════════════════════════════
         * Phase B — Bottom-Up 3-level renormalization  [uv_cnt >= theta*n]
         *
         *  Lazy init on first BU entry:
         *    fv   <- front[]  O(fsz)
         *    uv1  <- uv0      O(W)
         *    uv2  <- uv1      O(Wm)
         *
         *  3-level scan (Almost Reducibility):
         *    Level 2: 1 bit = 262144 vertices  (Wc iterations)
         *    Level 1: 1 bit =   4096 vertices
         *    Level 0: 1 bit =     64 vertices
         *  Zero-word -> skip via __builtin_ctzll, no inner iterations.
         * ═══════════════════════════════════════════════════════════ */
        } else if (uv_cnt >= (int)(AA_THETA * n)) {
            bu++;

            /* Lazy allocation */
            if (!bu_init) {
                bu_init = 1;
                fv  = (uint64_t*)calloc(W,  sizeof(uint64_t));
                nfv = (uint64_t*)calloc(W,  sizeof(uint64_t));
                uv1 = (uint64_t*)calloc(Wm, sizeof(uint64_t));
                uv2 = (uint64_t*)calloc(Wc, sizeof(uint64_t));
                if (!fv || !nfv || !uv1 || !uv2) goto done_bfs;
                fv_dirty = 1;
            }

            /* Rebuild fv and uv1/uv2 from uv0 if stale */
            if (fv_dirty) {
                memset(fv,  0, W  * sizeof(uint64_t));
                for (int i = 0; i < fsz; i++)
                    fv[front[i] >> 6] |= (1ULL << (front[i] & 63));
                memset(uv1, 0, Wm * sizeof(uint64_t));
                for (int wi = 0; wi < W;  wi++)
                    if (uv0[wi]) uv1[wi>>6] |= (1ULL << (wi & 63));
                memset(uv2, 0, Wc * sizeof(uint64_t));
                for (int mi = 0; mi < Wm; mi++)
                    if (uv1[mi]) uv2[mi>>6] |= (1ULL << (mi & 63));
                fv_dirty = 0;
            }

            memset(nfv, 0, W * sizeof(uint64_t));

            for (int ci = 0; ci < Wc; ci++) {
                uint64_t c2 = uv2[ci];
                if (!c2) continue;          /* skip 262144 vertices */
                while (c2) {
                    int b1 = __builtin_ctzll(c2); c2 &= c2 - 1;
                    int mi  = (ci << 6) | b1;
                    if (mi >= Wm) break;
                    uint64_t c1 = uv1[mi];
                    while (c1) {
                        int b0 = __builtin_ctzll(c1); c1 &= c1 - 1;
                        int wi  = (mi << 6) | b0;
                        if (wi >= W) break;
                        uint64_t c0 = uv0[wi];
                        while (c0) {
                            int bv = __builtin_ctzll(c0); c0 &= c0 - 1;
                            int v   = (wi << 6) | bv;
                            if (v >= n) break;
                            for (int j = g->xadj[v]; j < g->xadj[v+1]; j++) {
                                ops++;
                                int u = g->adj[j];
                                if (fv[u >> 6] & (1ULL << (u & 63))) {
                                    dist[v] = level + 1;
                                    nxt[nsz++] = v;
                                    nfv[v >> 6] |= (1ULL << (v & 63));
                                    uv0[wi] &= ~(1ULL << bv);
                                    if (!uv0[wi]) {
                                        uv1[mi] &= ~(1ULL << b0);
                                        if (!uv1[mi])
                                            uv2[ci] &= ~(1ULL << b1);
                                    }
                                    SDU -= g->deg[v];
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            { uint64_t *tmp = fv; fv = nfv; nfv = tmp; }

        /* ═══════════════════════════════════════════════════════════
         * Phase C — Bottom-Up compact list  [uv_cnt < theta*n]
         *
         *  "Final reduction" (Avila): when the orbit is nearly fully
         *  reduced, handle the residual explicitly. List built from
         *  uv0 in O(W), compacted by swap-last. Cost O(uv*avg_deg).
         * ═══════════════════════════════════════════════════════════ */
        } else {
            bu++;
            if (!bu_init) {
                bu_init = 1;
                fv  = (uint64_t*)calloc(W, sizeof(uint64_t));
                nfv = (uint64_t*)calloc(W, sizeof(uint64_t));
                if (!fv || !nfv) goto done_bfs;
                fv_dirty = 1;
            }
            if (fv_dirty) {
                memset(fv, 0, W * sizeof(uint64_t));
                for (int i = 0; i < fsz; i++)
                    fv[front[i] >> 6] |= (1ULL << (front[i] & 63));
                fv_dirty = 0;
            }
            memset(nfv, 0, W * sizeof(uint64_t));

            if (!phase_c) {
                phase_c = 1;
                unvis = (int*)malloc((uv_cnt + 1) * sizeof(int));
                if (unvis) {
                    unvis_sz = 0;
                    for (int wi = 0; wi < W; wi++) {
                        uint64_t c0 = uv0[wi];
                        while (c0) {
                            int bv = __builtin_ctzll(c0); c0 &= c0 - 1;
                            int v  = (wi << 6) | bv;
                            if (v < n) unvis[unvis_sz++] = v;
                        }
                    }
                }
            }
            if (unvis) {
                int i = 0;
                while (i < unvis_sz) {
                    int v = unvis[i], found = 0;
                    for (int j = g->xadj[v]; j < g->xadj[v+1]; j++) {
                        ops++;
                        int u = g->adj[j];
                        if (fv[u >> 6] & (1ULL << (u & 63))) { found = 1; break; }
                    }
                    if (found) {
                        dist[v] = level + 1;
                        nxt[nsz++] = v;
                        nfv[v >> 6] |= (1ULL << (v & 63));
                        SDU -= g->deg[v];
                        unvis[i] = unvis[--unvis_sz];
                    } else { i++; }
                }
            }
            { uint64_t *tmp = fv; fv = nfv; nfv = tmp; }
        }

        /* ── Advance one level ────────────────────────────────────── */
        uv_cnt -= nsz;
        memcpy(front, nxt, nsz * sizeof(int));
        fsz = nsz;
        level++;
    }

done_bfs:;
    double t1 = now_ms();
    free(front); free(nxt);
    free(fv); free(nfv);
    free(uv0); free(uv1); free(uv2);
    free(unvis);
    return (BFSResult){ t1-t0, ops, td, bu, dist };
}
 

/* ═══════════════════════════════════════════════════════════════════
 * Validation de correctness AA vs STD
 * ═══════════════════════════════════════════════════════════════════ */

/* Graphe de test : grille 8×8 avec quelques raccourcis diagonaux */
static int validate_aa(void) {
    /* Construction manuelle d'un graphe CSR 64 sommets */
    int n = 64;
    Graph g;
    g.n = n;
    g.m = 0;
    g.W = (n + 63) / 64;
    g.xadj = (int*)calloc(n + 1, sizeof(int));
    g.deg  = (int*)calloc(n,     sizeof(int));
    /* 1ère passe : compter les arêtes */
    for (int v = 0; v < n; v++) {
        int r = v / 8, c = v % 8;
        if (r > 0)   g.deg[v]++;          /* nord       */
        if (r < 7)   g.deg[v]++;          /* sud        */
        if (c > 0)   g.deg[v]++;          /* ouest      */
        if (c < 7)   g.deg[v]++;          /* est        */
        if (r > 0 && c > 0) g.deg[v]++;   /* diag NW    */
        if (r < 7 && c < 7) g.deg[v]++;   /* diag SE    */
        g.m += g.deg[v];
    }
    g.m /= 2;
    g.xadj[0] = 0;
    for (int v = 0; v < n; v++) g.xadj[v+1] = g.xadj[v] + g.deg[v];
    g.adj = (int*)malloc(g.xadj[n] * sizeof(int));
    int *pos = (int*)calloc(n, sizeof(int));
    for (int v = 0; v < n; v++) {
        int r = v / 8, c = v % 8;
        int nbrs[6], cnt = 0;
        if (r > 0)          nbrs[cnt++] = v - 8;
        if (r < 7)          nbrs[cnt++] = v + 8;
        if (c > 0)          nbrs[cnt++] = v - 1;
        if (c < 7)          nbrs[cnt++] = v + 1;
        if (r > 0 && c > 0) nbrs[cnt++] = v - 9;
        if (r < 7 && c < 7) nbrs[cnt++] = v + 9;
        for (int k = 0; k < cnt; k++) {
            int u = nbrs[k];
            g.adj[g.xadj[v] + pos[v]++] = u;
        }
    }
    free(pos);

    int src = 0;
    BFSResult ref = bfs_std(&g, src);
    BFSResult aa  = bfs_aa( &g, src, 1.0);

    int ok = 1;
    for (int v = 0; v < n; v++) {
        if (ref.dist[v] != aa.dist[v]) {
            fprintf(stderr,
                "  [FAIL] validate_aa : dist[%d] std=%d AA=%d\n",
                v, ref.dist[v], aa.dist[v]);
            ok = 0; break;
        }
    }
    if (ok) printf("  [OK] validate_aa : grille 8×8 correcte ✓\n");
    result_free(&ref); result_free(&aa);
    free(g.xadj); free(g.adj); free(g.deg);
    return ok ? 0 : -1;
}

/* ═══════════════════════════════════════════════════════════════════
 * Benchmark runner
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    const char *name;
    double      alpha;
} AlgoDesc;

static AlgoDesc algos[N_ALGOS] = {
    {"STD",  0.0},
    {"DIR",  14.0},
    {"SURF", 0.0},
    {"BB",   1.2},
    {"AA",   YP_ALPHA},
};

static BFSResult run_algo(int algo_idx, const Graph *g, int src) {
    switch (algo_idx) {
        case 0: return bfs_std( g, src);
        case 1: return bfs_dir( g, src, algos[1].alpha, 24.0);
        case 2: return bfs_surf(g, src);
        case 3: return bfs_bb(  g, src);
        default: return bfs_aa( g, src, algos[4].alpha);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Structures statistiques (conformes JEA)
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    double t_min;
    double t_mean;
    double t_sd;
    double t_med;
    double cv_pct;
    long   ops;
    int    td, bu;
} AlgoStats;

static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

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

static void bench_run_stats(const Graph *g, int src, int runs,
                            AlgoStats stats[N_ALGOS],
                            double raw_times[N_ALGOS][MAX_RUNS]) {
    int r_eff = (runs > MAX_RUNS) ? MAX_RUNS : runs;

    for (int a = 0; a < N_ALGOS; a++) {
        stats[a].t_min = 1e18;
        stats[a].ops = 0; stats[a].td = 0; stats[a].bu = 0;
    }

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

    for (int a = 0; a < N_ALGOS; a++) {
        compute_stats(raw_times[a], r_eff,
                      &stats[a].t_min, &stats[a].t_mean,
                      &stats[a].t_sd,  &stats[a].t_med);
        stats[a].cv_pct = (stats[a].t_mean > 0)
                          ? (stats[a].t_sd / stats[a].t_mean) * 100.0
                          : 0.0;
    }
}

/* ── compat pour alpha_sweep ──────────────────────────────────────── */
__attribute__((unused))
static void bench_run(const Graph *g, int src, int runs,  /* compat alpha_sweep */
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

/* ═══════════════════════════════════════════════════════════════════
 * Test de Wilcoxon signé-rangé (Wilcoxon 1945, approx. normale)
 * ═══════════════════════════════════════════════════════════════════ */

static double norm_cdf_upper(double z) {
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
    double p_value;
    double z_score;
    int    n_eff;
    double W_plus;
    double W_minus;
} WilcoxonResult;

static WilcoxonResult wilcoxon_aa_vs(const double *t_yp,
                                     const double *t_other, int n) {
    WilcoxonResult res = {1.0, 0.0, 0, 0.0, 0.0};

    double diffs[MAX_RUNS];
    int    n_eff = 0;
    for (int i = 0; i < n; i++) {
        double d = t_yp[i] - t_other[i];
        if (d == 0.0) continue;
        diffs[n_eff++] = d;
    }
    if (n_eff < 5) { res.n_eff = n_eff; return res; }
    res.n_eff = n_eff;

    int idx[MAX_RUNS];
    for (int i = 0; i < n_eff; i++) idx[i] = i;
    for (int i = 0; i < n_eff - 1; i++)
        for (int j = i + 1; j < n_eff; j++)
            if (fabs(diffs[idx[i]]) > fabs(diffs[idx[j]])) {
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }

    double ranks[MAX_RUNS];
    for (int i = 0; i < n_eff; ) {
        int j = i;
        double abs_d = fabs(diffs[idx[i]]);
        while (j < n_eff && fabs(diffs[idx[j]]) == abs_d) j++;
        double avg_rank = (double)(i + 1 + j) / 2.0;
        for (int k = i; k < j; k++) ranks[k] = avg_rank;
        i = j;
    }

    double Wp = 0.0, Wm = 0.0;
    for (int i = 0; i < n_eff; i++) {
        if (diffs[idx[i]] < 0.0) Wp += ranks[i];
        else                      Wm += ranks[i];
    }
    res.W_plus = Wp; res.W_minus = Wm;

    double T  = (Wp < Wm) ? Wp : Wm;
    double mu = (double)n_eff * (n_eff + 1) / 4.0;
    double sd = sqrt((double)n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24.0);
    if (sd == 0.0) return res;

    double z = (T - mu + 0.5) / sd;
    res.z_score = (Wp > Wm) ? -z : z;
    res.p_value = 2.0 * norm_cdf_upper(fabs(res.z_score));
    return res;
}



/* ═══════════════════════════════════════════════════════════════════
 * Catalogue des graphes SNAP/KONECT (--list-graphs)
 * ═══════════════════════════════════════════════════════════════════ */

static void list_graphs(void) {
    printf(
"\n"
"╔═══════════════════════════════════════════════════════════════════╗\n"
"║    Graphes réels SNAP/KONECT recommandés pour le benchmark AA     ║\n"
"╠═══════════════════════════════════════════════════════════════════╣\n"
"║  wget <URL>  &&  gunzip <fichier>.gz                              ║\n"
"╚═══════════════════════════════════════════════════════════════════╝\n"
"\n"
"── Réseaux sociaux ───────────────────────────────────────────────────────\n"
"  facebook_combined.txt   n=4 039    m=88 234     dense, hub fort\n"
"    https://snap.stanford.edu/data/facebook_combined.txt.gz\n"
"  soc-Epinions1.txt       n=75 888   m=508 837    hiérarchie forte\n"
"    https://snap.stanford.edu/data/soc-Epinions1.txt.gz\n"
"  soc-LiveJournal1.txt    n=4 847 571 m=68 993 773 grand réseau blog\n"
"    https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz\n"
"\n"
"── Collaboration scientifique ────────────────────────────────────────────\n"
"  CA-AstroPh.txt          n=18 772   m=198 050    communautés denses\n"
"    https://snap.stanford.edu/data/CA-AstroPh.txt.gz\n"
"  CA-HepPh.txt            n=12 008   m=118 521    très dense\n"
"    https://snap.stanford.edu/data/CA-HepPh.txt.gz\n"
"\n"
"── P2P (creux, faible degré moyen) ──────────────────────────────────────\n"
"  p2p-Gnutella31.txt      n=62 586   m=147 892    peu de hubs\n"
"    https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz\n"
"\n"
"── Graphes routiers (grand diamètre) ────────────────────────────────────\n"
"  roadNet-CA.txt          n=1 965 206 m=2 766 607 diamètre ~900\n"
"    https://snap.stanford.edu/data/roadNet-CA.txt.gz\n"
"  roadNet-TX.txt          n=1 379 917 m=1 921 660\n"
"    https://snap.stanford.edu/data/roadNet-TX.txt.gz\n"
"\n"
"── Graphes web ───────────────────────────────────────────────────────────\n"
"  web-Stanford.txt        n=281 903  m=2 312 497  loi de puissance\n"
"    https://snap.stanford.edu/data/web-Stanford.txt.gz\n"
"\n"
"── Ordre recommandé ──────────────────────────────────────────────────────\n"
"  facebook → p2p-Gnutella31 → CA-AstroPh → soc-Epinions1\n"
"  → web-Stanford → roadNet-TX → roadNet-CA → soc-LiveJournal1\n"
"\n"
"  Profil par classe :\n"
"    Social/Web  : hubs, pièces épuisées tôt → AA gagne fort en Phase 2\n"
"    Routier     : grand diamètre, pièces lentes à s'épuiser → Phase 3 active\n"
"    P2P         : creux → BU jamais rentable → STD/SURF meilleur\n"
"\n"
    );
}

/* ═══════════════════════════════════════════════════════════════════
 * Stockage des résultats pour tableaux consolidés
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    char      name[64];
    int       n;
    long      m;
    AlgoStats stats[N_ALGOS];
    double    raw[N_ALGOS][MAX_RUNS];
    int       runs_done;
    int       td_aa, bu_aa;
    long      mem_saved_mb;  /* mémoire épargnée vs adj_mask dense */
} GraphResult;

static GraphResult results[MAX_GRAPHS];
static int         nresults = 0;

static void sep(char c, int w) { for (int i=0;i<w;i++) putchar(c); putchar('\n'); }

static void shortname(const char *path, char *out, int maxlen) {
    const char *p = strrchr(path, '/');
    p = p ? p + 1 : path;
    strncpy(out, p, maxlen - 1);
    out[maxlen - 1] = '\0';
    char *dot = strrchr(out, '.');
    if (dot && strcmp(dot, ".gz") != 0) *dot = '\0';
    if ((int)strlen(out) > 28) out[28] = '\0';
}

/* ═══════════════════════════════════════════════════════════════════
 * Helpers d'affichage
 * ═══════════════════════════════════════════════════════════════════ */

static const char *star(double val, double best) {
    return (val == best) ? "*" : " ";
}

static const char *wilcoxon_sig(double p) {
    if (p < 0.001) return "***";
    if (p < 0.01)  return "** ";
    if (p < 0.05)  return "*  ";
    return "   ";
}

static void print_header_time(void) {
    printf("\n-- Wall-clock time (ms) --\n\n");
    printf("%-30s %6s %8s  %8s %8s %8s %8s %8s  %6s %6s %6s %6s\n",
           "Graph","n","m",
           "STD(ms)","DIR(ms)","SURF(ms)","BB(ms)","AA(ms)",
           "xDIR","xSURF","xBB","xAA");
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
           "ops-STD","ops-DIR","ops-SURF","ops-BB","ops-AA",
           "/DIR","/SURF","/BB","/AA");
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

/* ═══════════════════════════════════════════════════════════════════
 * Tableaux consolidés finaux
 * ═══════════════════════════════════════════════════════════════════ */

static void print_summary(int runs) {
    if (nresults == 0) return;
    const int W = 120;

    /* TABLEAU 1 — Minimum */
    printf("\n");
    sep('=', W);
    printf("  TABLEAU 1 — Temps minimum (ms)  [min sur %d runs]\n", runs);
    printf("  Speedup = t_STD_min / t_algo_min  —  * = meilleur non-STD\n");
    sep('=', W);
    printf("\n");
    printf("  %-28s %7s %9s │ %8s %8s %8s %8s %8s │ %6s %6s %6s %6s\n",
           "Graphe","n","m","STD","DIR","SURF","BB","AA",
           "xDIR","xSURF","xBB","xAA");
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
        double best_all = r->stats[0].t_min; int besta = 0;
        for (int a = 1; a < N_ALGOS; a++)
            if (r->stats[a].t_min < best_all) { best_all = r->stats[a].t_min; besta = a; }
        wins_min[besta]++;
    }
    printf("  %-28s %7s %9s ┴ %8s %8s %8s %8s %8s ┴ %6s %6s %6s %6s\n",
           "────────────────────────────","───────","─────────",
           "────────","────────","────────","────────","────────",
           "──────","──────","──────","──────");
    printf("\n  Victoires (min) : STD=%d DIR=%d SURF=%d BB=%d AA=%d\n",
           wins_min[0],wins_min[1],wins_min[2],wins_min[3],wins_min[4]);

    /* TABLEAU 2 — Moyenne ± écart-type */
    printf("\n");
    sep('=', W);
    printf("  TABLEAU 2 — Moyenne ± écart-type (ms)  [%d runs]\n", runs);
    sep('=', W);
    printf("\n");
    printf("  %-28s │ %-16s %-16s %-16s %-16s %-16s\n",
           "Graphe","STD","DIR","SURF","BB","AA");
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
        printf("  %-28s │", "  CV%");
        for (int a = 0; a < N_ALGOS; a++) {
            char cvbuf[20];
            snprintf(cvbuf, sizeof(cvbuf), "%.1f%%", r->stats[a].cv_pct);
            printf(" %-16s", cvbuf);
        }
        printf("\n");
        double best_all = r->stats[0].t_mean; int besta = 0;
        for (int a = 1; a < N_ALGOS; a++)
            if (r->stats[a].t_mean < best_all) { best_all = r->stats[a].t_mean; besta = a; }
        wins_mean[besta]++;
    }
    printf("  %-28s ┴ %s\n",
           "────────────────────────────",
           "──────────────── ──────────────── ──────────────── ──────────────── ────────────────");
    printf("\n  Victoires (mean) : STD=%d DIR=%d SURF=%d BB=%d AA=%d\n",
           wins_mean[0],wins_mean[1],wins_mean[2],wins_mean[3],wins_mean[4]);

    /* TABLEAU 3 — Médiane */
    printf("\n");
    sep('=', W);
    printf("  TABLEAU 3 — Médiane (ms)  [%d runs]\n", runs);
    sep('=', W);
    printf("\n");
    printf("  %-28s %7s %9s │ %8s %8s %8s %8s %8s │ %6s %6s %6s %6s\n",
           "Graphe","n","m","STD","DIR","SURF","BB","AA",
           "xDIR","xSURF","xBB","xAA");
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
        double best_all = r->stats[0].t_med; int besta = 0;
        for (int a = 1; a < N_ALGOS; a++)
            if (r->stats[a].t_med < best_all) { best_all = r->stats[a].t_med; besta = a; }
        wins_med[besta]++;
    }
    printf("  %-28s %7s %9s ┴ %8s %8s %8s %8s %8s ┴ %6s %6s %6s %6s\n",
           "────────────────────────────","───────","─────────",
           "────────","────────","────────","────────","────────",
           "──────","──────","──────","──────");
    printf("\n  Victoires (med) : STD=%d DIR=%d SURF=%d BB=%d AA=%d\n",
           wins_med[0],wins_med[1],wins_med[2],wins_med[3],wins_med[4]);

    /* TABLEAU 4 — Wilcoxon */
    printf("\n");
    sep('=', W);
    printf("  TABLEAU 4 — Test de Wilcoxon signé-rangé : AA vs concurrents\n");
    printf("  p-value bilatérale sur %d mesures pairées  "
           "*** p<0.001  ** p<0.01  * p<0.05  (z>0 = AA gagne)\n", runs);
    sep('=', W);
    printf("\n");
    printf("  %-28s │ %-22s %-22s %-22s %-22s\n",
           "Graphe","AA vs DIR","AA vs SURF","AA vs BB","AA vs STD");
    printf("  %-28s ┼ %-22s %-22s %-22s %-22s\n",
           "────────────────────────────",
           "──────────────────────","──────────────────────",
           "──────────────────────","──────────────────────");

    int sig_wins[4] = {0};
    for (int g = 0; g < nresults; g++) {
        GraphResult *r = &results[g];
        int conc[4] = {1, 2, 3, 0};
        printf("  %-28s │", r->name);
        for (int ci = 0; ci < 4; ci++) {
            int a = conc[ci];
            WilcoxonResult wr = wilcoxon_aa_vs(r->raw[4], r->raw[a], r->runs_done);
            char buf[24];
            if (wr.n_eff < 5) {
                snprintf(buf, sizeof(buf), "n/a(ties)            ");
            } else {
                const char *sig = wilcoxon_sig(wr.p_value);
                snprintf(buf, sizeof(buf), "z=%+5.2f p=%.4f %s%s",
                         wr.z_score, wr.p_value, sig,
                         wr.z_score > 0 ? "✓" : "✗");
                if (wr.p_value < 0.05 && wr.z_score > 0) sig_wins[ci]++;
            }
            printf(" %-21s│", buf);
        }
        printf("\n");
    }
    printf("  %-28s ┴ %-22s %-22s %-22s %-22s\n",
           "────────────────────────────",
           "──────────────────────","──────────────────────",
           "──────────────────────","──────────────────────");
    printf("\n  Victoires significatives AA (p<0.05) : "
           "vs DIR=%d/%d  vs SURF=%d/%d  vs BB=%d/%d  vs STD=%d/%d\n",
           sig_wins[0], nresults, sig_wins[1], nresults,
           sig_wins[2], nresults, sig_wins[3], nresults);

    /* TABLEAU 5 — Edge-checks */
    printf("\n");
    sep('=', W);
    printf("  TABLEAU 5 — Edge checks (ops)  [run minimal]\n");
    printf("  Ratio = ops_STD / ops_algo\n");
    sep('=', W);
    printf("\n");
    printf("  %-28s %7s %9s │ %10s %10s %10s %10s %10s │ %6s %6s %6s %6s\n",
           "Graphe","n","m",
           "ops-STD","ops-DIR","ops-SURF","ops-BB","ops-AA",
           "/DIR","/SURF","/BB","/AA");
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

    /* TABLEAU 6 — Profil AA-PHP */
    printf("\n");
    sep('=', W);
    
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
        "  --alpha-sweep    Sensibilité de alpha pour AA-PHP\n"
        "  --list-graphs    Catalogue SNAP/KONECT\n"
        "  --no-per-graph   Supprimer tableaux intermédiaires\n"
        "\n"
        "Entrée: fichiers .txt format SNAP (lignes 'u v', '#' ignoré)\n"
        "\n"
        "Exemples:\n"
        "  %s --runs 20 facebook_combined.txt CA-AstroPh.txt\n"
        "  %s --alpha-sweep soc-Epinions1.txt\n"
        "  %s --list-graphs\n",
        prog, prog, prog, prog);
}

int main(int argc, char *argv[]) {
    int         runs      = 10;
    int         per_graph = 1;
    int         do_list   = 0;
    const char *graph_files[MAX_GRAPHS];
    int         ngraphs   = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--runs") && i+1 < argc)
            runs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-per-graph"))
            per_graph = 0;
        else if (!strcmp(argv[i], "--list-graphs"))
            do_list = 1;
        else if (argv[i][0] != '-') {
            if (ngraphs < MAX_GRAPHS) graph_files[ngraphs++] = argv[i];
        } else {
            fprintf(stderr, "Option inconnue : %s\n", argv[i]);
            usage(argv[0]); return 1;
        }
    }

    if (do_list) { list_graphs(); return 0; }
    if (ngraphs == 0) { usage(argv[0]); return 1; }

    printf("\n");
    sep('=', 80);
    printf("  PHP-BFS benchmark  |  runs=%d  |  AA alpha=%.2f  |"
           "  block=%d  |  %d graphe(s)\n",
           runs, YP_ALPHA, YP_BLOCK, ngraphs);
    printf("  STD(Cormen22) · DIR(Beamer12) · SURF(Yoon22) · BB(Bhaskar25)"
           " · AA-PHP(ce travail)\n");
    sep('=', 80);

    if (validate_aa() < 0) return 1;

    for (int gi = 0; gi < ngraphs; gi++) {
        const char *path = graph_files[gi];
        printf("\n[%d/%d] Chargement : %s\n", gi+1, ngraphs, path);

        Graph g;
        if (graph_load_snap(path, &g) < 0) {
            fprintf(stderr, "  ERREUR : impossible de charger %s — ignoré\n", path);
            continue;
        }

        int  Nb_g = (g.n + YP_BLOCK - 1) / YP_BLOCK;
        int  Wc_g = (Nb_g + 63) / 64;
        int  src  = hub_vertex(&g);
        printf("  n=%-8d  m=%-10ld  W=%-5d  Nb=%-5d  Wc=%-4d  hub=%d (deg %d)\n",
               g.n, g.m, g.W, Nb_g, Wc_g, src, g.deg[src]);
        printf("  Skip coarse : 1 mot nul = %d sommets ignorés\n",
               64 * YP_BLOCK);

        AlgoStats    stats[N_ALGOS];
        static double raw[N_ALGOS][MAX_RUNS];
        int r_eff = (runs > MAX_RUNS) ? MAX_RUNS : runs;
        bench_run_stats(&g, src, runs, stats, raw);

        if (per_graph) {
            double t_min[N_ALGOS]; long ops_min[N_ALGOS];
            for (int a = 0; a < N_ALGOS; a++) {
                t_min[a]  = stats[a].t_min;
                ops_min[a] = stats[a].ops;
            }
            printf("\n");
            print_header_time();
            print_row_time(path, g.n, g.m, t_min);
            print_header_ops();
            print_row_ops(path, g.n, g.m, ops_min);
            printf("  AA: %dT/%dB niveaux\n", stats[4].td, stats[4].bu);
        }

        if (nresults < MAX_GRAPHS) {
            GraphResult *r = &results[nresults++];
            shortname(path, r->name, sizeof(r->name));
            r->n          = g.n;
            r->m          = g.m;
            r->runs_done  = r_eff;
            r->td_aa      = stats[4].td;
            r->bu_aa      = stats[4].bu;
            /* mémoire épargnée vs adj_mask dense (BB) sur grands graphes */
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

    if (nresults > 1 || !per_graph)
        print_summary(runs);

    return 0;
}
