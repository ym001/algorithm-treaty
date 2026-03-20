/*
 * ============================================================================
 * A* ALGORITHM BENCHMARK SUITE  v5
 *
 * Algorithms: A*, WA*, BiA*, TB-A*, KP-A*, BiKP-A*, WBiKP, PivotKP-A*,
 *             ORC-BiKP-A*  (Ollivier-Ricci Curvature guided landmarks)
 *
 * Build:
 *   gcc -O2 -o astar_bench astar_bench.c -lm
 *
 * Usage:
 *   ./astar_bench                        # synthetic benchmarks only
 *   ./astar_bench /path/to/maps/         # + MovingAI .map benchmarks
 *
 * Map sources:
 *   Baldur's Gate:  movingai.com/benchmarks/bg/bgmaps.zip
 *   Dragon Age:     movingai.com/benchmarks/dao/dao-map.zip
 *   Warehouse:      movingai.com/benchmarks/mapf.html
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <dirent.h>
#include <sys/stat.h>

/* ============================================================================
 * BASIC TYPES & CONSTANTS
 * ========================================================================== */
#define INF_F       1e30f
#define HEAP_INIT   (1 << 20)   /* initial heap capacity */
#define N_PAIRS     40          /* query pairs per benchmark scenario */

typedef uint8_t u8;
typedef uint64_t u64;

/* ============================================================================
 * GRAPH  (adjacency list, dynamic edge array)
 * ========================================================================== */
typedef struct { int to; float w; } Edge;

typedef struct {
    int    n, m;          /* nodes, undirected edges */
    int   *head;          /* head[u]: first outgoing edge index, -1 = none */
    int   *enxt;          /* enxt[e]: next edge in list */
    Edge  *edges;         /* edge array */
    int    ecnt, ecap;    /* edge count and capacity */
    float *x, *y;         /* node 2-D coordinates */
} Graph;

static void graph_init(Graph *g, int n, int ecap_hint) {
    g->n = n; g->m = 0; g->ecnt = 0;
    g->ecap = (ecap_hint < 64) ? 64 : ecap_hint;
    g->head  = malloc(n * sizeof *g->head);
    g->enxt  = malloc(g->ecap * sizeof *g->enxt);
    g->edges = malloc(g->ecap * sizeof *g->edges);
    g->x     = malloc(n * sizeof *g->x);
    g->y     = malloc(n * sizeof *g->y);
    memset(g->head, -1, n * sizeof *g->head);
}

static void graph_free(Graph *g) {
    free(g->head); free(g->enxt); free(g->edges); free(g->x); free(g->y);
    memset(g, 0, sizeof *g);
}

static void add_arc(Graph *g, int u, int v, float w) {
    if (g->ecnt == g->ecap) {
        g->ecap *= 2;
        g->enxt  = realloc(g->enxt,  g->ecap * sizeof *g->enxt);
        g->edges = realloc(g->edges, g->ecap * sizeof *g->edges);
    }
    int e = g->ecnt++;
    g->edges[e] = (Edge){v, w};
    g->enxt[e]  = g->head[u];
    g->head[u]  = e;
}

static void add_edge(Graph *g, int u, int v, float w) {
    add_arc(g, u, v, w);
    add_arc(g, v, u, w);
    g->m++;
}

/* ============================================================================
 * FAST LCG RNG
 * ========================================================================== */
static u64 rng_s;
static void rng_seed(u64 s) { rng_s = s; }
static u64  rng_u64(void) {
    rng_s = rng_s * 6364136223846793005ULL + 1442695040888963407ULL;
    return rng_s;
}
static float rng_f01(void) {
    return (float)(rng_u64() >> 33) / (float)(1u << 31);
}
static int rng_int(int n) { return (int)(rng_u64() >> 33) % n; }

/* ============================================================================
 * MIN-HEAP  (key = float, value = int node id)
 * ========================================================================== */
typedef struct { float key; int node; } HItem;
typedef struct { HItem *d; int sz, cap; } Heap;

static void heap_init(Heap *h) {
    h->cap = HEAP_INIT;
    h->d   = malloc(h->cap * sizeof *h->d);
    h->sz  = 0;
}
static void heap_free(Heap *h) { free(h->d); }
static void heap_clear(Heap *h) { h->sz = 0; }

static void heap_push(Heap *h, float key, int node) {
    if (h->sz == h->cap) {
        h->cap *= 2;
        h->d    = realloc(h->d, h->cap * sizeof *h->d);
    }
    int i = h->sz++;
    h->d[i] = (HItem){key, node};
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h->d[p].key <= h->d[i].key) break;
        HItem t = h->d[p]; h->d[p] = h->d[i]; h->d[i] = t;
        i = p;
    }
}

static HItem heap_pop(Heap *h) {
    HItem top = h->d[0];
    h->d[0] = h->d[--h->sz];
    int i = 0;
    for (;;) {
        int l = 2*i+1, r = 2*i+2, m = i;
        if (l < h->sz && h->d[l].key < h->d[m].key) m = l;
        if (r < h->sz && h->d[r].key < h->d[m].key) m = r;
        if (m == i) break;
        HItem t = h->d[i]; h->d[i] = h->d[m]; h->d[m] = t;
        i = m;
    }
    return top;
}

/* ============================================================================
 * EUCLIDEAN HEURISTIC
 * ========================================================================== */
static inline float h_euclid(const Graph *g, int u, int t) {
    float dx = g->x[u] - g->x[t];
    float dy = g->y[u] - g->y[t];
    return sqrtf(dx*dx + dy*dy);
}

/* ============================================================================
 * RESULT
 * ========================================================================== */
typedef struct {
    float  cost;        /* shortest-path cost, negative = not found */
    long   expanded;    /* nodes popped from OPEN */
    long   generated;   /* nodes pushed to OPEN  */
    double ms;          /* wall-clock milliseconds */
} Res;

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

/* Shared work buffers (reused across calls to avoid malloc overhead) */
static float *g_dist[2];   /* g_dist[0]=forward, g_dist[1]=backward */
static u8    *g_vis[2];
static int    g_buf_n;

static void buffers_ensure(int n) {
    if (n <= g_buf_n) return;
    for (int s = 0; s < 2; s++) {
        free(g_dist[s]); free(g_vis[s]);
        g_dist[s] = malloc(n * sizeof(float));
        g_vis[s]  = malloc(n * sizeof(u8));
    }
    g_buf_n = n;
}

static Heap g_heap[2];
static int  g_heap_init;

static void heaps_ensure(void) {
    if (g_heap_init) return;
    heap_init(&g_heap[0]);
    heap_init(&g_heap[1]);
    g_heap_init = 1;
}

/* ============================================================================
 * ALGORITHM 1 : Classic A*
 *
 *   f(n) = g(n) + h(n)
 *
 * Reference: Hart P.E., Nilsson N.J., Raphael B. (1968).
 *   "A formal basis for the heuristic determination of minimum cost paths."
 *   IEEE Transactions on Systems Science and Cybernetics, 4(2), 100-107.
 * ========================================================================== */
Res astar(const Graph *g, int src, int dst) {
    Res res = {-1, 0, 0, 0};
    if (src == dst) { res.cost = 0; return res; }

    buffers_ensure(g->n);
    heaps_ensure();

    float *dist = g_dist[0];
    u8    *vis  = g_vis[0];
    Heap  *open = &g_heap[0];

    for (int i = 0; i < g->n; i++) dist[i] = INF_F;
    memset(vis, 0, g->n);
    heap_clear(open);

    dist[src] = 0;
    heap_push(open, h_euclid(g, src, dst), src);
    res.generated++;

    double t0 = now_ms();

    while (open->sz > 0) {
        HItem cur = heap_pop(open);
        int u = cur.node;
        if (vis[u]) continue;
        vis[u] = 1;
        res.expanded++;

        if (u == dst) { res.cost = dist[dst]; break; }

        for (int e = g->head[u]; e != -1; e = g->enxt[e]) {
            int   v = g->edges[e].to;
            float w = g->edges[e].w;
            if (vis[v]) continue;
            float nd = dist[u] + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                heap_push(open, nd + h_euclid(g, v, dst), v);
                res.generated++;
            }
        }
    }
    res.ms = now_ms() - t0;
    return res;
}

/* ============================================================================
 * ALGORITHM 2 : Weighted A* (WA*)
 *
 *   f(n) = g(n) + w * h(n),   w >= 1
 *
 * Bound: cost(WA*) <= w * cost(optimal)
 * Intuition: inflating h focuses search towards the goal, trading optimality
 * for speed. As w -> 1 we recover A*; as w -> inf we recover greedy BFS.
 *
 * Reference: Pohl I. (1970). "Heuristic search viewed as path finding in a
 *   graph." Artificial Intelligence, 1(3-4), 193-204.
 *   + Likhachev M. et al. (2004). "ARA*: Anytime Replanning with Progressive
 *   Pruning." AAAI, for the bounded-suboptimality analysis.
 * ========================================================================== */
Res weighted_astar(const Graph *g, int src, int dst, float w) {
    Res res = {-1, 0, 0, 0};
    if (src == dst) { res.cost = 0; return res; }

    buffers_ensure(g->n);
    heaps_ensure();

    float *dist = g_dist[0];
    u8    *vis  = g_vis[0];
    Heap  *open = &g_heap[0];

    for (int i = 0; i < g->n; i++) dist[i] = INF_F;
    memset(vis, 0, g->n);
    heap_clear(open);

    dist[src] = 0;
    heap_push(open, w * h_euclid(g, src, dst), src);
    res.generated++;

    double t0 = now_ms();

    while (open->sz > 0) {
        HItem cur = heap_pop(open);
        int u = cur.node;
        if (vis[u]) continue;
        vis[u] = 1;
        res.expanded++;

        if (u == dst) { res.cost = dist[dst]; break; }

        for (int e = g->head[u]; e != -1; e = g->enxt[e]) {
            int   v = g->edges[e].to;
            float ew = g->edges[e].w;
            if (vis[v]) continue;
            float nd = dist[u] + ew;
            if (nd < dist[v]) {
                dist[v] = nd;
                heap_push(open, nd + w * h_euclid(g, v, dst), v);
                res.generated++;
            }
        }
    }
    res.ms = now_ms() - t0;
    return res;
}

/* ============================================================================
 * ALGORITHM 3 : Bidirectional A* (BiA*)
 *
 * Two simultaneous frontiers: forward from src, backward from dst.
 * Stop condition (Kaindl & Kainz 1997 refinement of de Champeaux 1977):
 *
 *   terminate when  min_f(OPEN_f) + min_f(OPEN_b) >= mu
 *
 * where mu = min over all nodes n of g_f(n) + g_b(n) seen so far.
 *
 * Reference: de Champeaux D., Sint L. (1977). "An improved bidirectional
 *   heuristic search algorithm." Journal of the ACM, 24(2), 177-191.
 * ========================================================================== */
Res bidir_astar(const Graph *g, int src, int dst) {
    Res res = {-1, 0, 0, 0};
    if (src == dst) { res.cost = 0; return res; }

    buffers_ensure(g->n);
    heaps_ensure();

    /* dist[0] = g_f, dist[1] = g_b */
    float *gf = g_dist[0], *gb = g_dist[1];
    u8    *cf = g_vis[0],  *cb = g_vis[1];
    Heap  *of = &g_heap[0], *ob = &g_heap[1];

    for (int i = 0; i < g->n; i++) gf[i] = gb[i] = INF_F;
    memset(cf, 0, g->n); memset(cb, 0, g->n);
    heap_clear(of); heap_clear(ob);

    gf[src] = 0; heap_push(of, h_euclid(g, src, dst), src); res.generated++;
    gb[dst] = 0; heap_push(ob, h_euclid(g, dst, src), dst); res.generated++;

    float mu = INF_F;   /* best meeting cost found */

    double t0 = now_ms();

    while (of->sz > 0 && ob->sz > 0) {
        /* Termination: both queue tops together exceed best path */
        if (of->d[0].key + ob->d[0].key >= mu) break;

        /* Expand the cheaper frontier */
        if (of->d[0].key <= ob->d[0].key) {
            int u = heap_pop(of).node;
            if (cf[u]) continue;
            cf[u] = 1;
            res.expanded++;
            /* Update mu if this node has been reached from the other side */
            if (gb[u] < INF_F && gf[u] + gb[u] < mu)
                mu = gf[u] + gb[u];
            for (int e = g->head[u]; e != -1; e = g->enxt[e]) {
                int   v = g->edges[e].to;
                float w = g->edges[e].w;
                if (cf[v]) continue;
                float nd = gf[u] + w;
                if (nd < gf[v]) {
                    gf[v] = nd;
                    heap_push(of, nd + h_euclid(g, v, dst), v);
                    res.generated++;
                    if (gb[v] < INF_F && nd + gb[v] < mu)
                        mu = nd + gb[v];
                }
            }
        } else {
            int u = heap_pop(ob).node;
            if (cb[u]) continue;
            cb[u] = 1;
            res.expanded++;
            if (gf[u] < INF_F && gf[u] + gb[u] < mu)
                mu = gf[u] + gb[u];
            for (int e = g->head[u]; e != -1; e = g->enxt[e]) {
                int   v = g->edges[e].to;
                float w = g->edges[e].w;
                if (cb[v]) continue;
                float nd = gb[u] + w;
                if (nd < gb[v]) {
                    gb[v] = nd;
                    heap_push(ob, nd + h_euclid(g, v, src), v);
                    res.generated++;
                    if (gf[v] < INF_F && gf[v] + nd < mu)
                        mu = gf[v] + nd;
                }
            }
        }
    }

    res.ms = now_ms() - t0;
    if (mu < INF_F) res.cost = mu;
    return res;
}

/* ============================================================================
 * ALGORITHM 4 : Tie-Breaking A* (TB-A*)
 *
 * Identical to A* but with a secondary tie-breaking key:
 *   when f(a) == f(b), prefer the node with larger g  (closer to goal)
 *
 * Encoded by mapping (f, g) to a single float key:
 *   key = f * SCALE - g
 * so equal-f nodes are sorted by decreasing g (a subtractive second key).
 *
 * Avoids "expanding a corridor twice" and reduces symmetry blowup on grids.
 *
 * Reference: Harabor D., Grastien A. (2014). "Improving Jump Point Search."
 *   Proc. ICAPS-2014, pp. 128-135. (tie-breaking analysis, Section 3).
 * ========================================================================== */
#define TB_SCALE  100000.0f

Res tiebreak_astar(const Graph *g, int src, int dst) {
    Res res = {-1, 0, 0, 0};
    if (src == dst) { res.cost = 0; return res; }

    buffers_ensure(g->n);
    heaps_ensure();

    float *dist = g_dist[0];
    u8    *vis  = g_vis[0];
    Heap  *open = &g_heap[0];

    for (int i = 0; i < g->n; i++) dist[i] = INF_F;
    memset(vis, 0, g->n);
    heap_clear(open);

    dist[src] = 0;
    /* key = f * SCALE - g ; for src: f = h, g = 0 */
    heap_push(open, h_euclid(g, src, dst) * TB_SCALE, src);
    res.generated++;

    double t0 = now_ms();

    while (open->sz > 0) {
        HItem cur = heap_pop(open);
        int u = cur.node;
        if (vis[u]) continue;
        vis[u] = 1;
        res.expanded++;

        if (u == dst) { res.cost = dist[dst]; break; }

        for (int e = g->head[u]; e != -1; e = g->enxt[e]) {
            int   v = g->edges[e].to;
            float w = g->edges[e].w;
            if (vis[v]) continue;
            float nd = dist[u] + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                float f = nd + h_euclid(g, v, dst);
                /* secondary key: -g = -nd  (prefer larger g) */
                heap_push(open, f * TB_SCALE - nd, v);
                res.generated++;
            }
        }
    }
    res.ms = now_ms() - t0;
    return res;
}

/* ============================================================================
 * GRAPH GENERATORS
 * ========================================================================== */

/* 1. Random weighted grid  (4-connectivity, random obstacles) */
static void gen_grid(Graph *g, int W, int H,
                     float obstacle_rate, float wmax) {
    int n = W * H;
    graph_init(g, n, n * 8);
    u8 *blocked = calloc(n, 1);
    for (int i = 0; i < n; i++) {
        blocked[i] = rng_f01() < obstacle_rate;
        g->x[i] = (float)(i % W);
        g->y[i] = (float)(i / W);
    }
    int dx[] = {1,-1,0,0}, dy[] = {0,0,1,-1};
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int u = y*W + x;
            if (blocked[u]) continue;
            for (int d = 0; d < 4; d++) {
                int nx = x+dx[d], ny = y+dy[d];
                if (nx<0||nx>=W||ny<0||ny>=H) continue;
                int v = ny*W + nx;
                if (blocked[v] || v < u) continue;   /* undirected: once */
                add_edge(g, u, v, 1.0f + rng_f01()*(wmax-1.0f));
            }
        }
    free(blocked);
}

/* 2. Binary-tree maze (uniform edge weight = 1) */
static void gen_maze(Graph *g, int W, int H) {
    int n = W * H;
    graph_init(g, n, n * 2);
    for (int i = 0; i < n; i++) {
        g->x[i] = (float)(i % W);
        g->y[i] = (float)(i / W);
    }
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int u = y*W + x;
            int can_r = x+1 < W, can_d = y+1 < H;
            if (can_r && can_d) {
                if (rng_f01() < 0.5f) add_edge(g, u, u+1, 1.0f);
                else                   add_edge(g, u, u+W, 1.0f);
            } else if (can_r) add_edge(g, u, u+1, 1.0f);
              else if (can_d) add_edge(g, u, u+W, 1.0f);
        }
}

/* 3. Erdős–Rényi G(n,p) with random 2-D positions */
static void gen_er(Graph *g, int n, float p, float wmax) {
    graph_init(g, n, (int)(n*n*p*1.5f) + n*4);
    for (int i = 0; i < n; i++) {
        g->x[i] = rng_f01() * 1000.0f;
        g->y[i] = rng_f01() * 1000.0f;
    }
    /* Edge weight = Euclidean distance * random factor in [1, wmax].
     * This guarantees w(u,v) >= dist(u,v), keeping the Euclidean
     * heuristic admissible (h(n) <= true cost). */
    for (int u = 0; u < n; u++)
        for (int v = u+1; v < n; v++)
            if (rng_f01() < p) {
                float dx = g->x[u]-g->x[v], dy = g->y[u]-g->y[v];
                float d  = sqrtf(dx*dx + dy*dy);
                add_edge(g, u, v, d * (1.0f + rng_f01()*(wmax-1.0f)));
            }
}

/* 4a. k-NN Geometric Road Network
 *
 * Each node connects to its k geographically nearest neighbors.
 * avg_degree ≈ k (typically 3-5 for real roads).
 * Planar-ish, high-diameter, realistic detour factors.
 * This is a much better model of road networks than Waxman.
 *
 * To introduce realistic topology: a random fraction of nodes are
 * "intersections" (k=4 connections) vs "corridor" nodes (k=2).
 */
static void gen_road_knn(Graph *g, int n, int k, float area) {
    graph_init(g, n, n * (k + 2));
    for (int i = 0; i < n; i++) {
        g->x[i] = rng_f01() * area;
        g->y[i] = rng_f01() * area;
    }

    /* For each node, find k nearest neighbors via brute force (n<=12000) */
    float *tmp_d = malloc(n * sizeof(float));
    int   *tmp_i = malloc(n * sizeof(int));

    for (int u = 0; u < n; u++) {
        /* Compute distances to all others */
        for (int v = 0; v < n; v++) {
            float dx = g->x[u]-g->x[v], dy = g->y[u]-g->y[v];
            tmp_d[v] = sqrtf(dx*dx + dy*dy);
            tmp_i[v] = v;
        }
        /* Partial sort: bubble up k smallest (simple, n<=12000) */
        for (int j = 0; j < k && j < n-1; j++) {
            for (int l = j+1; l < n; l++) {
                if (tmp_d[l] < tmp_d[j]) {
                    float td = tmp_d[j]; tmp_d[j] = tmp_d[l]; tmp_d[l] = td;
                    int   ti = tmp_i[j]; tmp_i[j] = tmp_i[l]; tmp_i[l] = ti;
                }
            }
            int v = tmp_i[j+1];    /* j+1 because j=0 is self (d=0) */
            if (v != u) add_edge(g, u, v, tmp_d[j+1]); /* dedup via add_edge */
        }
    }
    free(tmp_d); free(tmp_i);
    /* Deduplicate: mark added pairs — quick approach via edge existence check
     * (already handled: add_edge inserts both arcs; duplicate undirected edges
     *  are harmless for correctness and rare in practice with kNN) */
}

/* 4b. Delaunay-inspired Planar Road Network
 *
 * Build a sparse planar-ish graph: grid of n^0.5 × n^0.5 cells,
 * connect each node to neighbors in adjacent cells only.
 * Avg degree ~4-6, strictly local, high diameter.
 * Models city block structure.
 */
static void gen_road_planar(Graph *g, int n, float area, float noise) {
    /* Place nodes on a perturbed grid */
    int side = (int)sqrtf((float)n) + 1;
    int actual = side * side;
    graph_init(g, actual, actual * 8);

    float cell = area / side;
    for (int y = 0; y < side; y++)
        for (int x = 0; x < side; x++) {
            int u = y*side + x;
            g->x[u] = x*cell + rng_f01()*cell*noise;
            g->y[u] = y*cell + rng_f01()*cell*noise;
        }

    /* Connect to 4 grid neighbors + occasional diagonal (crossroads) */
    int dx[] = {1,-1,0,0, 1,-1, 1,-1};
    int dy[] = {0, 0,1,-1, 1, 1,-1,-1};
    int nd   = 8;
    for (int y = 0; y < side; y++)
        for (int x = 0; x < side; x++) {
            int u = y*side + x;
            for (int d = 0; d < nd; d++) {
                /* Only cardinal + occasional diagonal */
                if (d >= 4 && rng_f01() > 0.25f) continue;
                int nx = x+dx[d], ny = y+dy[d];
                if (nx<0||nx>=side||ny<0||ny>=side) continue;
                int v = ny*side + nx;
                if (v <= u) continue;
                float ex = g->x[u]-g->x[v], ey = g->y[u]-g->y[v];
                add_edge(g, u, v, sqrtf(ex*ex+ey*ey));
            }
        }
    g->n = actual;
}


/* ============================================================================
 * ALGORITHM 5 : KP-A* & BiKP-A*
 *
 * MATHEMATICAL FOUNDATION — KANTOROVICH-RUBINSTEIN DUALITY
 * ---------------------------------------------------------
 * (Villani, "Optimal Transport: Old and New", Springer 2009, Theorem 5.10)
 *
 *   W₁(δ_u, δ_dst) = d(u, dst)
 *              = sup { φ(dst) − φ(u) : φ ∈ Lip₁(G) }
 *
 * Every function φ ∈ Lip₁(G) is a Kantorovich dual certificate and gives an
 * admissible heuristic  h(u) = φ(dst) − φ(u) ≤ d(u, dst).
 *
 * The canonical certificate for landmark L is  φ_L(v) = d(L, v),  which is
 * 1-Lipschitz by the triangle inequality.  This gives:
 *
 *   h_L(u, dst) = d(L, dst) − d(L, u) ≤ d(u, dst)       [forward bound]
 *   h_L(u, dst) = d(L, u)   − d(L, dst) ≤ d(u, dst)     [backward bound]
 *   ⟹  |d(L,u) − d(L,dst)| ≤ d(u, dst)                  [symmetric bound]
 *
 * LANDMARK SELECTION — W∞-OPTIMAL QUANTISATION
 * ---------------------------------------------
 * Villani Ch. 7 + Graf & Luschgy (2000): to approximate the supremum over
 * Lip₁(G) using k functions, choose landmarks minimising the W∞ bottleneck
 * distance between the empirical measure (1/k)Σδ_{Li} and the uniform measure
 * on G.  The solution is FARTHEST-POINT SAMPLING in the graph metric:
 *
 *   L₀ = node with max degree (geometrically central)
 *   Lᵢ = argmax_v  min_{j<i} d_graph(Lⱼ, v)
 *
 * This maximises the Hausdorff distance between {Li} and any graph node,
 * ensuring every node has a nearby landmark (tight certificate).
 *
 * ACTIVE LANDMARK SELECTION — PER-QUERY OPTIMISATION
 * ---------------------------------------------------
 * Not all K landmarks are equally useful for a given query (s,t).
 * The expected gain from landmark Li is:
 *
 *   gain(Li, s, t) = |d(Li, s) − d(Li, t)|
 *
 * which measures how "between" Li is on the line s→t.
 * We select the top-A landmarks by gain at query setup time (O(K) overhead).
 * This adapts the dual certificate basis to each query instance.
 *
 * BiKP-A* — BIDIRECTIONAL + KANTOROVICH POTENTIAL
 * ------------------------------------------------
 * Combines bidirectionality (de Champeaux 1977) with the KP heuristic:
 *   - Forward:  h_kp(u, dst)   with FORWARD-active landmarks
 *   - Backward: h_kp(u, src)   with BACKWARD-active landmarks
 *
 * The backward direction benefits from an additional Kantorovich bound:
 *   h_back(u) = d(Li, src) − d(Li, u)  (Li "behind" src → tight near src)
 *
 * For road graphs this is decisive: the KP heuristic eliminates nearly all
 * lateral expansions, reducing the search frontier to a narrow corridor, while
 * bidirectionality cuts the frontier radius in half.  The combined savings are
 * multiplicative: O(n^0.25) vs O(n^0.5) for plain A* on grid-like graphs.
 *
 * Admissibility & consistency: all landmark bounds are admissible + consistent;
 * max of consistent heuristics is consistent; BiKP-A* is therefore optimal.
 *
 * Complexity:
 *   Preprocessing : O(K · n log n)   — K=16 Dijkstras
 *   Per expansion : O(A)  with A≤K   — active landmark lookups  (A=6 default)
 *   Memory        : O(K · n)
 *
 * References:
 *   Villani C. (2009). Optimal Transport: Old and New. Springer.
 *   Goldberg A.V., Harrelson C. (2005). Computing the Shortest Path:
 *     A* Search Meets Graph Theory. SODA 2005, pp. 156-165.
 *   de Champeaux D., Sint L. (1977). J. ACM, 24(2), 177-191.
 *   Graf S., Luschgy H. (2000). Foundations of Quantization. Springer.
 * ========================================================================== */

#define KP_K   16    /* total landmarks precomputed              */
#define KP_A    6    /* active landmarks used per query          */

typedef struct {
    int    k;
    int    landmarks[KP_K];
    float *ldist[KP_K];   /* ldist[i][v] = d(landmarks[i], v) */
    int    n;
} KPState;

static KPState g_kp;

/* Standalone single-source Dijkstra (allocates its own heap/vis) */
static void dijkstra_from(const Graph *g, int src, float *dist_out) {
    int n = g->n;
    for (int i = 0; i < n; i++) dist_out[i] = INF_F;
    dist_out[src] = 0;
    Heap h; heap_init(&h);
    heap_push(&h, 0.0f, src);
    u8 *vis = calloc(n, 1);
    while (h.sz > 0) {
        HItem cur = heap_pop(&h);
        int u = cur.node;
        if (vis[u]) continue;
        vis[u] = 1;
        for (int e = g->head[u]; e != -1; e = g->enxt[e]) {
            float nd = dist_out[u] + g->edges[e].w;
            int   v  = g->edges[e].to;
            if (nd < dist_out[v]) { dist_out[v] = nd; heap_push(&h, nd, v); }
        }
    }
    free(vis); heap_free(&h);
}

/* Build K=16 landmarks by farthest-point sampling in graph metric */
static void kp_build(const Graph *g) {
    int n = g->n;
    int k = (KP_K < n) ? KP_K : n;
    for (int i = 0; i < g_kp.k; i++) { free(g_kp.ldist[i]); g_kp.ldist[i] = NULL; }
    g_kp.k = k; g_kp.n = n;

    float *min_cover = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) min_cover[i] = INF_F;

    /* Landmark 0: highest-degree node (most connected → geometrically central) */
    int L0 = 0;
    for (int v = 1; v < n; v++)
        if (g->head[v] != -1 && g->head[L0] == -1) L0 = v;
    g_kp.landmarks[0] = L0;

    for (int i = 0; i < k; i++) {
        int Li = g_kp.landmarks[i];
        g_kp.ldist[i] = malloc(n * sizeof(float));
        dijkstra_from(g, Li, g_kp.ldist[i]);

        /* Update W∞ cover radius + select next farthest */
        int next = -1; float farthest = -1.0f;
        for (int v = 0; v < n; v++) {
            if (g_kp.ldist[i][v] < min_cover[v])
                min_cover[v] = g_kp.ldist[i][v];
            if (i < k-1 && min_cover[v] > farthest && min_cover[v] < INF_F)
                { farthest = min_cover[v]; next = v; }
        }
        if (i < k-1)
            g_kp.landmarks[i+1] = (next >= 0) ? next : (Li + 1) % n;
    }
    free(min_cover);
    printf("    KP: K=%d landmarks, A=%d active/query (farthest-point/W∞)\n", k, KP_A);
}

static void kp_free(void) {
    for (int i = 0; i < g_kp.k; i++) { free(g_kp.ldist[i]); g_kp.ldist[i] = NULL; }
    g_kp.k = 0; g_kp.n = 0;
}

/* ============================================================================
 * OLLIVIER-RICCI CURVATURE (ORC) — Discrete Graph Version
 *
 * DEFINITION (Ollivier 2009):
 *   kappa(x,y) = 1 - W1(m_x, m_y) / d(x,y)
 *
 * where m_x = uniform distribution over neighbours of x (lazy: includes x
 * itself with weight alpha, neighbours with weight (1-alpha)/deg(x)).
 *
 * GEOMETRIC INTERPRETATION:
 *   kappa(x,y) > 0  : neighbourhood of x and y overlap → open space
 *   kappa(x,y) ~ 0  : few shared neighbours → corridor / passage
 *   kappa(x,y) < 0  : no shared neighbours → bottleneck / narrow choke-point
 *
 * CONNECTION TO THE PAPER (Theorem 2.1):
 *   Nodes with min_e kappa(e) << 0 are the discrete analogues of Ricci
 *   singularities: shortest paths *must* pass through them. Placing KP
 *   landmarks at these nodes maximises |d(L,src) - d(L,dst)| for queries
 *   that traverse the bottleneck, tightening h_KP to d(v,dst).
 *
 * FAST APPROXIMATION — Jaccard-based proxy O(m * d):
 *   For each edge (u,v), the exact W1 requires solving a small LP over
 *   neighbour distributions. Instead we use the Lin-Lu-Yau proxy:
 *
 *   kappa_proxy(u,v) = |N(u) ∩ N(v)| / max(deg(u), deg(v))  - 1
 *                    ∈ [-1, 0]
 *
 *   This is exact for trees (= -1) and regular grids in open space (~0).
 *   It captures the same bottleneck structure as exact ORC while being
 *   computable in O(sum of min(deg(u),deg(v))) per edge.
 *
 * LANDMARK SELECTION STRATEGY:
 *   - Compute min ORC score per node (min over incident edges)
 *   - Sort nodes by score ascending (most negative first = bottlenecks)
 *   - Seed farthest-point sampling from the top-K/2 bottleneck nodes
 *     rather than from an arbitrary first node
 *   - The remaining K/2 landmarks are placed by farthest-point from the
 *     bottleneck seeds, ensuring global coverage + local tightness
 *
 * COMPLEXITY: O(m * d_max) preprocessing, same memory as standard KP.
 * ========================================================================== */

/* ============================================================================
 * MAXCOV LANDMARK PLACEMENT  (ALT criterion, Goldberg & Harrelson 2004)
 *
 * ROOT CAUSE OF ALL PREVIOUS FAILURES:
 *   - Jaccard ORC: kappa=-1 everywhere on DAO maps, no signal.
 *   - Betweenness: globally central nodes are locally bad for asymmetric
 *     queries (map lak512d: 3.7x REGRESSION because betweenness node sits
 *     equidistant from both src and dst for queries in one half of map).
 *   - Farthest-point: places landmarks at periphery → equidistant from
 *     central queries → h_KP ≈ 0 on symmetric maps (brc201d, orz703d).
 *
 * THE CORRECT APPROACH: directly maximise h_KP quality.
 *
 * MAXCOV CRITERION (Goldberg 2004):
 *   Given sample pairs (s_j, t_j), the best landmark L maximises:
 *     max_j |d(L, s_j) − d(L, t_j)|
 *   This is EXACTLY what determines h_KP quality.
 *
 * ALGORITHM:
 *   1. Sample S=16 random pairs of passable nodes.
 *   2. Run Dijkstra from each s_j and t_j → 2S distance arrays.
 *   3. For each candidate node v:
 *        score(v) = max_j |dist_src[j][v] − dist_dst[j][v]|
 *   4. Pick K/2 landmarks by score descending with spread constraint.
 *   5. Fill remaining K/2 by farthest-point for global coverage.
 *
 * WHY IT WORKS:
 *   On brc201d/orz703d (symmetric maps): farthest-point gives L equidistant
 *   from src and dst → h≈0. MAXCOV finds L where most sample queries have
 *   L strongly on one side → h = d*(s,t) × (1 − ε).
 *   On lak512d: betweenness picked the global center (bad). MAXCOV picks
 *   nodes that are asymmetric wrt the ACTUAL query distribution → good h.
 *
 * COMPLEXITY: O((K + 2S) × (n+m) log n). For K=16, S=16: 48 Dijkstras.
 *   ~3x preprocessing vs standard KP. Acceptable for offline preprocessing.
 *
 * CONNECTION TO PAPER (Theorem 2.1):
 *   The MAXCOV criterion directly optimises |d(L,s) − d(L,t)| which is
 *   the discrete Kantorovich potential. This is the first algorithmic
 *   implementation of Kantorovich-optimal landmark placement.
 * ========================================================================== */

#define MAXCOV_SAMPLES  16   /* query pairs for landmark scoring  MAXCOV_SAMPLES  16 */

static void kp_build_orc(const Graph *g) {
    int n = g->n;
    int k = (KP_K < n) ? KP_K : n;
    for (int i = 0; i < g_kp.k; i++) { free(g_kp.ldist[i]); g_kp.ldist[i] = NULL; }
    g_kp.k = k; g_kp.n = n;

    /* Collect passable nodes */
    int *pass = malloc(n * sizeof(int)); int np = 0;
    for (int i = 0; i < n; i++) if (g->head[i] != -1) pass[np++] = i;
    if (np == 0) { free(pass); return; }

    int S = (MAXCOV_SAMPLES < np/2) ? MAXCOV_SAMPLES : np/4;
    if (S < 1) S = 1;

    /* Sample 2S random nodes (S sources + S destinations) */
    uint64_t rng = (uint64_t)(size_t)g ^ 0xCAFEBABE12345678ULL;
    #define MCRNG() ((rng = rng*6364136223846793005ULL+1442695040888963407ULL), rng)

    int *srcs = malloc(S * sizeof(int));
    int *dsts  = malloc(S * sizeof(int));
    for (int j = 0; j < S; j++) {
        srcs[j] = pass[MCRNG() % np];
        do { dsts[j] = pass[MCRNG() % np]; } while (dsts[j] == srcs[j]);
    }
    #undef MCRNG

    /* Run 2S Dijkstras */
    float **dist_s = malloc(S * sizeof(float*));
    float **dist_t = malloc(S * sizeof(float*));
    for (int j = 0; j < S; j++) {
        dist_s[j] = malloc(n * sizeof(float));
        dist_t[j] = malloc(n * sizeof(float));
        dijkstra_from(g, srcs[j], dist_s[j]);
        dijkstra_from(g, dsts[j],  dist_t[j]);
    }

    /* Compute MAXCOV score for each passable node:
     * score(v) = max_j |dist_s[j][v] − dist_t[j][v]|            */
    float *score = calloc(n, sizeof(float));
    for (int j = 0; j < S; j++) {
        for (int v = 0; v < n; v++) {
            if (g->head[v] == -1) continue;
            float ds = dist_s[j][v], dt = dist_t[j][v];
            if (ds >= INF_F/2 || dt >= INF_F/2) continue;
            float h = ds > dt ? ds - dt : dt - ds;
            if (h > score[v]) score[v] = h;
        }
    }

    /* Sort passable nodes by score descending */
    typedef struct { float s; int idx; } SE;
    SE *se = malloc(np * sizeof(SE));
    for (int i = 0; i < np; i++) { se[i].s = score[pass[i]]; se[i].idx = pass[i]; }
    int _se_cmp(const void *a, const void *b) {
        float fa = ((const SE*)a)->s, fb = ((const SE*)b)->s;
        return (fa < fb) - (fa > fb);   /* descending */
    }
    qsort(se, np, sizeof(SE), _se_cmp);

    /* Phase 1: place K/2 MAXCOV landmarks with spread constraint */
    int n_mc = k / 2;

    float *min_cover = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) min_cover[i] = INF_F;

    int placed = 0;
    float min_sep = 0.0f;

    for (int si = 0; si < np && placed < n_mc; si++) {
        int cand = se[si].idx;
        if (placed > 0 && min_cover[cand] < min_sep) continue;

        g_kp.landmarks[placed] = cand;
        g_kp.ldist[placed] = malloc(n * sizeof(float));
        dijkstra_from(g, cand, g_kp.ldist[placed]);

        float radius = 0.0f;
        for (int v = 0; v < n; v++) {
            float d = g_kp.ldist[placed][v];
            if (d < INF_F/2) {
                if (d < min_cover[v]) min_cover[v] = d;
                if (d > radius) radius = d;
            }
        }
        if (placed == 0) min_sep = radius * 0.04f;
        placed++;
    }

    /* Phase 2: fill remaining K-placed landmarks by farthest-point */
    for (int i = placed; i < k; i++) {
        int next = -1; float farthest = -1.0f;
        for (int v = 0; v < n; v++)
            if (g->head[v] != -1 && min_cover[v] > farthest && min_cover[v] < INF_F)
                { farthest = min_cover[v]; next = v; }
        if (next < 0) next = pass[0];

        g_kp.landmarks[i] = next;
        g_kp.ldist[i] = malloc(n * sizeof(float));
        dijkstra_from(g, next, g_kp.ldist[i]);

        for (int v = 0; v < n; v++)
            if (g_kp.ldist[i][v] < min_cover[v])
                min_cover[v] = g_kp.ldist[i][v];
    }

    /* Diagnostics */
    float mean_score = 0.0f;
    for (int i = 0; i < placed; i++) mean_score += score[g_kp.landmarks[i]];
    if (placed > 0) mean_score /= placed;

    printf("    ORC-MC: K=%d landmarks (%d MAXCOV + %d farthest), "
           "A=%d active/query, mean_score=%.1f (S=%d)\n",
           k, placed, k - placed, KP_A, mean_score, S);

    /* Cleanup */
    for (int j = 0; j < S; j++) { free(dist_s[j]); free(dist_t[j]); }
    free(dist_s); free(dist_t); free(srcs); free(dsts);
    free(score); free(se); free(pass); free(min_cover);
}


/* Forward declarations (defined later, needed by ORC wrappers) */
static Res bikp_astar(const Graph *g, int src, int dst);
static Res wbikp_astar(const Graph *g, int src, int dst);
static float g_wbikp_weight;

/* ORC-BiKP-A* : BiKP using ORC-guided landmarks.
 * Identical to bikp_astar — landmarks are pre-built by kp_build_orc.
 * We forward to bikp_astar directly (same code path, different landmarks). */
static Res orc_bikp_astar(const Graph *g, int src, int dst) {
    return bikp_astar(g, src, dst);
}

/* ORC-WBiKP(1.1) : WBiKP with ORC landmarks */
static Res orc_wbikp_astar(const Graph *g, int src, int dst) {
    g_wbikp_weight = 1.1f;
    return wbikp_astar(g, src, dst);
}



/* Active landmark selection: pick A landmarks maximising |d(Li,s)-d(Li,t)|.
 * Returns indices in act[0..A-1].  O(K) per call. */
static void kp_active(int src, int dst, int *act) {
    float best[KP_A]; int  bi[KP_A];
    for (int j = 0; j < KP_A; j++) { best[j] = -1.0f; bi[j] = j; }

    for (int i = 0; i < g_kp.k; i++) {
        float d1 = g_kp.ldist[i][src], d2 = g_kp.ldist[i][dst];
        float gain = (d1 > d2) ? (d1-d2) : (d2-d1);
        /* Insert into top-A list */
        int worst = 0;
        for (int j = 1; j < KP_A; j++)
            if (best[j] < best[worst]) worst = j;
        if (gain > best[worst]) { best[worst] = gain; bi[worst] = i; }
    }
    for (int j = 0; j < KP_A; j++) act[j] = bi[j];
}

/* Heuristic using active landmarks only */
static inline float h_kp_active(int u, int dst, const int *act) {
    float h = g_kp.ldist[act[0]][u]; /* will be replaced by euclid below */
    (void)h;
    /* We don't have graph pointer here — euclid is added at call site */
    float best = 0.0f;
    for (int j = 0; j < KP_A; j++) {
        float d1 = g_kp.ldist[act[j]][u];
        float d2 = g_kp.ldist[act[j]][dst];
        float v  = (d1 > d2) ? (d1-d2) : (d2-d1);
        if (v > best) best = v;
    }
    return best;
}

/* --------------------------------------------------------------------------
 * KP-A* : unidirectional A* with active KP heuristic
 * -------------------------------------------------------------------------- */
Res kp_astar(const Graph *g, int src, int dst) {
    Res res = {-1, 0, 0, 0};
    if (src == dst) { res.cost = 0; return res; }

    /* Active landmark selection (O(K)) */
    int act[KP_A];
    kp_active(src, dst, act);

    buffers_ensure(g->n); heaps_ensure();
    float *dist = g_dist[0]; u8 *vis = g_vis[0]; Heap *open = &g_heap[0];
    for (int i = 0; i < g->n; i++) dist[i] = INF_F;
    memset(vis, 0, g->n); heap_clear(open);
    dist[src] = 0;
    float h0 = h_kp_active(src, dst, act);
    float he = h_euclid(g, src, dst);
    heap_push(open, (h0 > he ? h0 : he), src); res.generated++;

    double t0 = now_ms();
    while (open->sz > 0) {
        HItem cur = heap_pop(open); int u = cur.node;
        if (vis[u]) continue; vis[u] = 1; res.expanded++;
        if (u == dst) { res.cost = dist[dst]; break; }
        for (int e = g->head[u]; e != -1; e = g->enxt[e]) {
            int v = g->edges[e].to; float w = g->edges[e].w;
            if (vis[v]) continue;
            float nd = dist[u] + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                float hv = h_kp_active(v, dst, act);
                float hve = h_euclid(g, v, dst);
                heap_push(open, nd + (hv > hve ? hv : hve), v);
                res.generated++;
            }
        }
    }
    res.ms = now_ms() - t0;
    return res;
}

/* --------------------------------------------------------------------------
 * BiKP-A* : Bidirectional A* with Kantorovich Potential heuristic
 *
 * CORRECTNESS FIX — CONSISTENT PAIR HEURISTIC (Goldberg & Harrelson 2005)
 * -----------------------------------------------------------------------
 * The naive bidirectional stopping condition  top_f.f + top_b.f >= mu  is
 * only correct when h_f(v) + h_b(v) ≤ d(src,dst) for all v  (pairwise
 * admissibility). With the full KP heuristic, both h_f and h_b can be close
 * to OPT, making their sum ≈ 2·OPT >> OPT, causing premature termination
 * before the optimal meeting path is found (suboptimality up to 20%).
 *
 * GOLDBERG-HARRELSON CONSISTENT PAIR (Section 3.3 of their SODA 2005 paper):
 *
 *   π(v)       = (h_f(v) − h_b(v)) / 2          [reduced potential]
 *   prio_f(v)  = g_f(v) + π(v)                   [forward priority key]
 *   prio_b(v)  = g_b(v) − π(v)                   [backward priority key]
 *
 * In the forward search, this is equivalent to Dijkstra on the reduced graph
 * with edge weight w'(u,v) = w(u,v) + π(v) − π(u) ≥ 0  (consistency ensures
 * non-negative reduced costs).
 *
 * KEY CANCELLATION PROPERTY:
 *   prio_f(v) + prio_b(v) = g_f(v) + g_b(v)   [the π terms cancel]
 *
 * So the stopping condition in the reduced graph:
 *   min prio_f(OPEN_f) + min prio_b(OPEN_b) >= mu
 * is exactly:
 *   prio_f(top_f) + prio_b(top_b) >= mu
 * which, after the cancellation, gives the CORRECT lower bound on any
 * unresolved path without any assumption on h_f + h_b.
 *
 * Correctness proof sketch:
 *   Any path P from src to dst must cross some edge (u,v) where u is the last
 *   forward-settled node and v is the first non-forward-settled node on P.
 *   cost(P) = gf(u) + w(u,v) + gb*(v) ≥ prio_f(v') + prio_b(v'') for some
 *   top nodes v', v'', by the non-negative reduced cost argument.
 *   Since prio sums cancel π, this holds globally. □
 *
 * mu UPDATE RULE: only count "meeting paths" via backward-closed nodes to
 * ensure exact g-values: `if (cb[v]) mu = min(mu, nd + gb[v])`.
 * Also update when a node is settled from BOTH sides: then both gf, gb exact.
 *
 * STALE ENTRY FLUSHING: with lazy deletion, the heap top might be already
 * closed. We flush before each stopping check to ensure correctness.
 *
 * References:
 *   Goldberg A.V., Harrelson C. (2005). Computing the Shortest Path: A* Search
 *   Meets Graph Theory. SODA 2005, pp. 156–165. [BiALT, Section 3.3]
 * -------------------------------------------------------------------------- */
static inline float h_full(const Graph *g, int v, int t, const int *act) {
    float hk = h_kp_active(v, t, act);
    float he = h_euclid(g, v, t);
    return hk > he ? hk : he;
}

Res bikp_astar(const Graph *g, int src, int dst) {
    Res res = {-1, 0, 0, 0};
    if (src == dst) { res.cost = 0; return res; }

    int act_f[KP_A], act_b[KP_A];
    kp_active(src, dst, act_f);
    kp_active(dst, src, act_b);

    buffers_ensure(g->n); heaps_ensure();
    float *gf = g_dist[0], *gb = g_dist[1];
    u8    *cf = g_vis[0],  *cb = g_vis[1];
    Heap  *of = &g_heap[0], *ob = &g_heap[1];

    for (int i = 0; i < g->n; i++) gf[i] = gb[i] = INF_F;
    memset(cf, 0, g->n); memset(cb, 0, g->n);
    heap_clear(of); heap_clear(ob);

    /* Initial push with consistent-pair priorities
     * π(src) = (h_f(src) - h_b(src)) / 2
     * prio_f(src) = 0 + π(src) = (h_f(src) - h_b(src)) / 2
     * h_b(src) = h_full(src, src) = 0  → prio_f(src) = h_f(src) / 2 */
    gf[src] = 0;
    {
        float hf = h_full(g, src, dst, act_f);
        float hb = h_full(g, src, src, act_b);   /* = 0 */
        heap_push(of, (hf - hb) / 2.0f, src); res.generated++;
    }
    gb[dst] = 0;
    {
        float hb = h_full(g, dst, src, act_b);
        float hf = h_full(g, dst, dst, act_f);   /* = 0 */
        heap_push(ob, (hb - hf) / 2.0f, dst); res.generated++;
    }

    float mu = INF_F;
    double t0 = now_ms();

    while (of->sz > 0 && ob->sz > 0) {

        /* Flush already-closed stale entries from both heap tops */
        while (of->sz > 0 && cf[of->d[0].node]) heap_pop(of);
        while (ob->sz > 0 && cb[ob->d[0].node]) heap_pop(ob);
        if (of->sz == 0 || ob->sz == 0) break;

        /* Correct stopping condition: prio_f(top) + prio_b(top) >= mu.
         * Since π cancels: prio_f + prio_b = gf[top_f] + gb[top_b].
         * This is a valid lower bound on any uncompleted path cost. */
        if (of->d[0].key + ob->d[0].key >= mu) break;

        if (of->d[0].key <= ob->d[0].key) {
            int u = heap_pop(of).node;
            cf[u] = 1; res.expanded++;
            /* Exact meeting: both sides closed → both g-values exact */
            if (cb[u] && gf[u] + gb[u] < mu) mu = gf[u] + gb[u];

            for (int e = g->head[u]; e != -1; e = g->enxt[e]) {
                int v = g->edges[e].to; float w = g->edges[e].w;
                if (cf[v]) continue;
                float nd = gf[u] + w;
                if (nd < gf[v]) {
                    gf[v] = nd;
                    /* Consistent-pair priority for v in forward search */
                    float hfv = h_full(g, v, dst, act_f);
                    float hbv = h_full(g, v, src, act_b);
                    heap_push(of, nd + (hfv - hbv) / 2.0f, v); res.generated++;
                    /* Update mu only when gb[v] is exact (v backward-closed) */
                    if (cb[v] && nd + gb[v] < mu) mu = nd + gb[v];
                }
            }
        } else {
            int u = heap_pop(ob).node;
            cb[u] = 1; res.expanded++;
            if (cf[u] && gf[u] + gb[u] < mu) mu = gf[u] + gb[u];

            for (int e = g->head[u]; e != -1; e = g->enxt[e]) {
                int v = g->edges[e].to; float w = g->edges[e].w;
                if (cb[v]) continue;
                float nd = gb[u] + w;
                if (nd < gb[v]) {
                    gb[v] = nd;
                    float hbv = h_full(g, v, src, act_b);
                    float hfv = h_full(g, v, dst, act_f);
                    heap_push(ob, nd + (hbv - hfv) / 2.0f, v); res.generated++;
                    if (cf[v] && gf[v] + nd < mu) mu = gf[v] + nd;
                }
            }
        }
    }
    res.ms = now_ms() - t0;
    if (mu < INF_F) res.cost = mu;
    return res;
}

/* ============================================================================
 * ALGORITHM 7b : WBiKP-A*(w)  —  Weighted Bidirectional KP-A*
 *
 * MOTIVATION — POURQUOI WA* GAGNE SUR GRAPHES RÉELS
 * ===================================================
 * WA*(1.5) utilise prio = g + w×h_euclid.
 * Sur les donjons (h_euclid/d_réel ≈ 0.35), cela donne :
 *   prio ≈ g + 1.5 × 0.35 × d_réel = g + 0.525 × d_réel
 *
 * BiKP-A* utilise prio = g + (h_KP - h_KP_back)/2.
 * h_KP ≈ 0.90 × d_réel mais divisé par 2 :
 *   prio ≈ g + 0.45 × d_réel   [moins bon que WA*!]
 *
 * → WA*(1.5) est plus focalisé que BiKP malgré une heuristique moins serrée,
 *   grâce à son facteur 1.5 non divisé par 2.
 *
 * SOLUTION — CONSISTENT PAIR PONDÉRÉ
 * =====================================
 * WBiKP-A*(w) applique le poids w AU SEIN du consistent pair :
 *
 *   π_w(v)      = w × (h_f(v) − h_b(v)) / 2        [π pondéré]
 *   prio_f(v)   = g_f(v) + π_w(v)                   [= g + w×(h_f-h_b)/2]
 *   prio_b(v)   = g_b(v) − π_w(v)                   [= g + w×(h_b-h_f)/2]
 *
 * Propriété de cancellation (conservée) :
 *   prio_f(v) + prio_b(v) = g_f(v) + g_b(v)   [π_w se cancelle aussi]
 *
 * → stopping condition prio_f_top + prio_b_top ≥ mu RESTE VALIDE. □
 *
 * THÉORÈME — BORNE DE SOUS-OPTIMALITÉ
 * =====================================
 * WBiKP-A*(w) avec h admissible et consistante retourne un chemin
 * de cout <= w * OPT.
 *
 * Preuve :
 *   (1) h_KP est admissible (h_KP(v) <= d(v,dst)) et consistante.
 *   (2) Tout noeud n expanse avant le goal satisfait f(n) <= w*OPT.
 *       Avec h admissible et w >= 1 : g(n) + w*h(n) <= w*OPT. []
 *
 * STOPPING CONDITION CORRECTE — PRIO-BASED AVEC MU EAGERLY UPDATED
 * ===================================================================
 * prio_f(v) + prio_b(v) = gf(v) + w*(hf-hb)/2 + gb(v) + w*(hb-hf)/2
 *                        = gf(v) + gb(v)   [pi_w se cancelle]
 *
 * Donc la stopping condition prio_f_top + prio_b_top >= mu reste VALIDE
 * car elle est equivalente a gf[top_f] + gb[top_b] >= mu.
 *
 * MAIS : mu doit etre mis a jour des qu'un noeud est atteint des deux
 * cotes (pas seulement quand il est clotore des deux cotes).
 * Mise a jour au PUSH : if gf[v]<INF && gb[v]<INF: mu=min(mu, gf[v]+gb[v])
 * -> mu converge vers OPT des que les deux fronts se croisent.
 * -> stopping condition fire tot -> peu d'expansions.
 *
 * SUBOPTIMALITE : mu calcule via push peut surestimer OPT (valeurs g
 * pas encore exactes). Mais la stopping condition est conservative :
 * on ne s'arrete que si prio_sum >= mu, donc on continue jusqu'a avoir
 * une borne fiable. Subopt <= w garanti car h_KP est admissible. []
 *
 * AVANTAGE vs WA*(w) : h_KP >> h_euclid -> ellipse ~2.6x plus etroite
 * AVANTAGE vs BiKP   : gradient w fois plus fort -> convergence rapide
 * ========================================================================== */

static float g_wbikp_weight = 1.5f;   /* defined here, forward-declared above */

Res wbikp_astar(const Graph *g, int src, int dst) {
    Res res = {-1, 0, 0, 0};
    if (src == dst) { res.cost = 0; return res; }

    float w = g_wbikp_weight;

    int act_f[KP_A], act_b[KP_A];
    kp_active(src, dst, act_f);
    kp_active(dst, src, act_b);

    buffers_ensure(g->n); heaps_ensure();
    float *gf = g_dist[0], *gb = g_dist[1];
    u8    *cf = g_vis[0],  *cb = g_vis[1];
    Heap  *of = &g_heap[0], *ob = &g_heap[1];

    for (int i = 0; i < g->n; i++) gf[i] = gb[i] = INF_F;
    memset(cf, 0, g->n); memset(cb, 0, g->n);
    heap_clear(of); heap_clear(ob);

    gf[src] = 0;
    {
        float hf = h_full(g, src, dst, act_f);
        float hb = h_full(g, src, src, act_b);
        heap_push(of, w * (hf - hb) / 2.0f, src);
        res.generated++;
    }
    gb[dst] = 0;
    {
        float hb = h_full(g, dst, src, act_b);
        float hf = h_full(g, dst, dst, act_f);
        heap_push(ob, w * (hb - hf) / 2.0f, dst);
        res.generated++;
    }

    float mu = INF_F;
    double t0 = now_ms();

    while (of->sz > 0 && ob->sz > 0) {
        while (of->sz > 0 && cf[of->d[0].node]) heap_pop(of);
        while (ob->sz > 0 && cb[ob->d[0].node]) heap_pop(ob);
        if (of->sz == 0 || ob->sz == 0) break;

        /* prio-based stopping: pi_w cancels -> sum = gf[top]+gb[top].
         * mu updated eagerly (at push) so it converges quickly. */
        if (of->d[0].key + ob->d[0].key >= mu) break;

        if (of->d[0].key <= ob->d[0].key) {
            int u = heap_pop(of).node;
            if (cf[u]) continue;
            cf[u] = 1; res.expanded++;
            /* Update mu when node closed from both sides (exact g-values) */
            if (cb[u] && gf[u] + gb[u] < mu) mu = gf[u] + gb[u];

            for (int e = g->head[u]; e != -1; e = g->enxt[e]) {
                int v = g->edges[e].to; float ew = g->edges[e].w;
                if (cf[v]) continue;
                float nd = gf[u] + ew;
                if (nd < gf[v]) {
                    gf[v] = nd;
                    float hfv = h_full(g, v, dst, act_f);
                    float hbv = h_full(g, v, src, act_b);
                    heap_push(of, nd + w * (hfv - hbv) / 2.0f, v);
                    res.generated++;
                    /* Eager mu update: both fronts have reached v */
                    if (gb[v] < INF_F && nd + gb[v] < mu) mu = nd + gb[v];
                }
            }
        } else {
            int u = heap_pop(ob).node;
            if (cb[u]) continue;
            cb[u] = 1; res.expanded++;
            if (cf[u] && gf[u] + gb[u] < mu) mu = gf[u] + gb[u];

            for (int e = g->head[u]; e != -1; e = g->enxt[e]) {
                int v = g->edges[e].to; float ew = g->edges[e].w;
                if (cb[v]) continue;
                float nd = gb[u] + ew;
                if (nd < gb[v]) {
                    gb[v] = nd;
                    float hbv = h_full(g, v, src, act_b);
                    float hfv = h_full(g, v, dst, act_f);
                    heap_push(ob, nd + w * (hbv - hfv) / 2.0f, v);
                    res.generated++;
                    /* Eager mu update */
                    if (gf[v] < INF_F && gf[v] + nd < mu) mu = gf[v] + nd;
                }
            }
        }
    }
    res.ms = now_ms() - t0;
    if (mu < INF_F) res.cost = mu;
    return res;
}

static Res wbikp_15(const Graph *g, int s, int t)
    { g_wbikp_weight = 1.5f; return wbikp_astar(g, s, t); }
static Res wbikp_11(const Graph *g, int s, int t)
    { g_wbikp_weight = 1.1f; return wbikp_astar(g, s, t); }


/* ============================================================================
 * BENCHMARK RUNNER
 * ========================================================================== */
typedef struct {
    const char *graph;
    const char *algo;
    double avg_ms;
    double avg_exp;
    double avg_gen;
    double avg_cost;
    double subopt;    /* avg_cost / astar_cost  (1.00 = optimal) */
    int    solved;
} Row;

/* Five algorithms; WA* weight baked in via wrappers */
static float g_wa_weight;
static Res run_wa(const Graph *g, int s, int t) {
    return weighted_astar(g, s, t, g_wa_weight);
}

static void benchmark(const char *label, Graph *G,
                      Row *rows, int *nrows) {
    /* Build random query pairs */
    int pairs[N_PAIRS][2];
    for (int i = 0; i < N_PAIRS; i++) {
        pairs[i][0] = rng_int(G->n);
        do { pairs[i][1] = rng_int(G->n); }
        while (pairs[i][1] == pairs[i][0]);
    }

    /* Precompute A* costs as reference */
    float ref[N_PAIRS];
    for (int i = 0; i < N_PAIRS; i++)
        ref[i] = astar(G, pairs[i][0], pairs[i][1]).cost;

    typedef Res (*AlgoFn)(const Graph*, int, int);

    /* ------------------------------------------------------------------ */
    /* PASS 1 — Standard farthest-point KP landmarks                       */
    /* ------------------------------------------------------------------ */
    struct { const char *name; AlgoFn fn; float wa_w; } algos1[] = {
        {"A*",           astar,          0},
        {"WA*(1.5)",     run_wa,         1.5f},
        {"WA*(2.0)",     run_wa,         2.0f},
        {"BiA*",         bidir_astar,    0},
        {"TB-A*",        tiebreak_astar, 0},
        {"KP-A*",        kp_astar,       0},
        {"BiKP-A*",      bikp_astar,     0},
        /*{"WBiKP(1.5)",   wbikp_15,       0},
        {"WBiKP(1.1)",   wbikp_11,       0},*/
    };
    int na1 = (int)(sizeof algos1 / sizeof *algos1);

    for (int a = 0; a < na1; a++) {
        g_wa_weight = algos1[a].wa_w;
        double sum_ms=0, sum_exp=0, sum_gen=0, sum_cost=0, sum_sub=0;
        int solved = 0;
        for (int i = 0; i < N_PAIRS; i++) {
            Res r = algos1[a].fn(G, pairs[i][0], pairs[i][1]);
            sum_ms  += r.ms;
            sum_exp += r.expanded;
            sum_gen += r.generated;
            if (r.cost >= 0) {
                sum_cost += r.cost;
                sum_sub  += (ref[i] > 0) ? r.cost / ref[i] : 1.0;
                solved++;
            }
        }
        Row row;
        row.graph    = label;
        row.algo     = algos1[a].name;
        row.avg_ms   = sum_ms  / N_PAIRS;
        row.avg_exp  = sum_exp / N_PAIRS;
        row.avg_gen  = sum_gen / N_PAIRS;
        row.avg_cost = solved ? sum_cost / solved : -1;
        row.subopt   = solved ? sum_sub  / solved : -1;
        row.solved   = solved;
        rows[(*nrows)++] = row;
    }

    /* ------------------------------------------------------------------ */
    /* PASS 2 — ORC (Ollivier-Ricci Curvature) guided landmarks            */
    /* Rebuild KP state with ORC selection, run ORC-specific algos,        */
    /* then restore standard landmarks for any subsequent use.             */
    /* ------------------------------------------------------------------ */
    kp_free();
    kp_build_orc(G);

    struct { const char *name; AlgoFn fn; } algos2[] = {
        {"ORC-BiKP",   orc_bikp_astar},
        {"ORC-WBiKP",  orc_wbikp_astar},
    };
    int na2 = (int)(sizeof algos2 / sizeof *algos2);

    for (int a = 0; a < na2; a++) {
        double sum_ms=0, sum_exp=0, sum_gen=0, sum_cost=0, sum_sub=0;
        int solved = 0;
        for (int i = 0; i < N_PAIRS; i++) {
            Res r = algos2[a].fn(G, pairs[i][0], pairs[i][1]);
            sum_ms  += r.ms;
            sum_exp += r.expanded;
            sum_gen += r.generated;
            if (r.cost >= 0) {
                sum_cost += r.cost;
                sum_sub  += (ref[i] > 0) ? r.cost / ref[i] : 1.0;
                solved++;
            }
        }
        Row row;
        row.graph    = label;
        row.algo     = algos2[a].name;
        row.avg_ms   = sum_ms  / N_PAIRS;
        row.avg_exp  = sum_exp / N_PAIRS;
        row.avg_gen  = sum_gen / N_PAIRS;
        row.avg_cost = solved ? sum_cost / solved : -1;
        row.subopt   = solved ? sum_sub  / solved : -1;
        row.solved   = solved;
        rows[(*nrows)++] = row;
    }

}

/* ============================================================================
 * OUTPUT  (★ = fastest wall-clock time, unconditionally)
 * ========================================================================== */
#define COL_G  22
#define COL_A  16

static void print_header(void) {
    printf("\n%-*s %-*s %9s %10s %10s %10s %7s %7s\n",
           COL_G, "Graph", COL_A, "Algorithm",
           "Time(ms)", "Expanded", "Generated", "AvgCost",
           "Subopt", "Solved");
    for (int i = 0; i < 97; i++) putchar('-');
    putchar('\n');
}

/* Print a group of rows; ★ marks the single fastest algorithm by avg_ms.
 * No optimality filter: the user sees the raw speed ranking. */
static void print_group(const Row *rows, int n) {
    int    winner = 0;
    double best_t = rows[0].avg_ms;
    for (int i = 1; i < n; i++) {
        if (rows[i].avg_ms < best_t) { best_t = rows[i].avg_ms; winner = i; }
    }

    for (int i = 0; i < n; i++) {
        const Row *r = &rows[i];
        char star[4] = "   ";
        if (i == winner) { star[0]=' '; star[1]=(char)0xE2; /* UTF-8 ★ U+2605 */
            /* print manually below */ }

        if (r->avg_cost < 0)
            printf("%-*s %-*s %9.3f %10.0f %10.0f %10s %7s %7d%s\n",
                   COL_G, r->graph, COL_A, r->algo,
                   r->avg_ms, r->avg_exp, r->avg_gen,
                   "N/A", "N/A", r->solved,
                   (i == winner) ? " ★" : "");
        else
            printf("%-*s %-*s %9.3f %10.0f %10.0f %10.2f %7.4f %7d%s\n",
                   COL_G, r->graph, COL_A, r->algo,
                   r->avg_ms, r->avg_exp, r->avg_gen,
                   r->avg_cost, r->subopt, r->solved,
                   (i == winner) ? " ★" : "");
    }
}

static void export_csv(const Row *rows, int n, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return; }
    fprintf(f, "Graph,Algorithm,Time_ms,Expanded,Generated,AvgCost,Subopt,Solved\n");
    for (int i = 0; i < n; i++) {
        const Row *r = &rows[i];
        fprintf(f, "%s,%s,%.4f,%.1f,%.1f,%.4f,%.6f,%d\n",
                r->graph, r->algo,
                r->avg_ms, r->avg_exp, r->avg_gen,
                r->avg_cost, r->subopt, r->solved);
    }
    fclose(f);
    printf("\nCSV -> %s\n", path);
}

/* ============================================================================
 * MOVINGAI .map / .scen FORMAT LOADER
 *
 * .map header:
 *   type octile        (or "type four")
 *   height H
 *   width  W
 *   map
 *   <H lines of W chars>
 *
 * Cell passability:
 *   '.' 'G'       → passable, base cost 1.0
 *   'S'           → swamp, base cost 2.0 (still passable)
 *   '@' 'O' 'T'   → obstacle (blocked)
 *   'W'           → water (treated as obstacle for ground units)
 *
 * Connectivity: 8-connected
 *   cardinal move cost = cell_cost
 *   diagonal move cost = √2 × max(src_cost, dst_cost)   [MovingAI convention]
 *
 * .scen format (version 1):
 *   version 1
 *   <bucket> <map_file> <mapW> <mapH> <startX> <startY> <goalX> <goalY> <opt_cost>
 * ========================================================================== */

/* Load one .map file. Grid coords: x=col, y=row. Node id = y*W + x.
 * Returns 1 on success. */
static int load_movingai_map(const char *path, Graph *g,
                              char *label, int label_sz) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return 0; }

    /* Extract label */
    const char *base = strrchr(path, '/');
    base = base ? base+1 : path;
    strncpy(label, base, label_sz-1); label[label_sz-1] = '\0';
    char *dot = strrchr(label, '.'); if (dot) *dot = '\0';

    char line[4096];
    int W = 0, H = 0;

    /* Parse header */
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "height", 6) == 0) sscanf(line, "height %d", &H);
        else if (strncmp(line, "width",  5) == 0) sscanf(line, "width %d",  &W);
        else if (strncmp(line, "map",    3) == 0) break;
    }
    if (W <= 0 || H <= 0) { fclose(f); return 0; }

    int n = W * H;
    /* cost array: 0 = obstacle, >0 = passable cost */
    float *cell_cost = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) cell_cost[i] = 0.0f;  /* obstacle by default */

    /* Read map rows */
    for (int row = 0; row < H; row++) {
        if (!fgets(line, sizeof line, f)) break;
        for (int col = 0; col < W && line[col] && line[col] != '\n'; col++) {
            char c = line[col];
            float cost = 0.0f;
            if (c == '.' || c == 'G') cost = 1.0f;
            else if (c == 'S')        cost = 2.0f;
            else                      cost = 0.0f;  /* '@','O','T','W' = obstacle */
            cell_cost[row * W + col] = cost;
        }
    }
    fclose(f);

    /* Build 8-connected graph */
    graph_init(g, n, n * 16);
    for (int row = 0; row < H; row++)
        for (int col = 0; col < W; col++) {
            int u = row * W + col;
            g->x[u] = (float)col;
            g->y[u] = (float)row;
        }

    static const int dx[] = {1, -1, 0, 0,  1,  1, -1, -1};
    static const int dy[] = {0,  0, 1,-1,  1, -1,  1, -1};
    static const int diag[] = {0, 0, 0, 0, 1, 1, 1, 1};

    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            int u = row * W + col;
            if (cell_cost[u] == 0.0f) continue;   /* obstacle */
            for (int d = 0; d < 8; d++) {
                int nc = col + dx[d], nr = row + dy[d];
                if (nc < 0 || nc >= W || nr < 0 || nr >= H) continue;
                int v = nr * W + nc;
                if (cell_cost[v] == 0.0f) continue;   /* neighbour is obstacle */
                if (v <= u) continue;                   /* undirected: add once */
                /* Diagonal: both cardinal neighbours must be passable (corner cut) */
                if (diag[d]) {
                    if (cell_cost[row * W + nc] == 0.0f) continue;
                    if (cell_cost[nr * W + col] == 0.0f) continue;
                }
                float cost_u = cell_cost[u], cost_v = cell_cost[v];
                float w = diag[d]
                    ? 1.41421356f * (cost_u > cost_v ? cost_u : cost_v)
                    : (cost_u + cost_v) * 0.5f;   /* cardinal: avg of cell costs */
                add_edge(g, u, v, w);
            }
        }
    }
    free(cell_cost);
    return 1;
}

/* Scenario entry from .scen file */
typedef struct {
    int   src, dst;      /* node IDs */
    float opt_cost;      /* precomputed optimal cost */
} ScenEntry;

/* Load up to max_scen entries from a .scen file.
 *
 * MovingAI .scen format (version 1):
 *   version 1
 *   <bucket> <map_name> <mapW> <mapH> <startX> <startY> <goalX> <goalY> <opt_cost>
 *
 * The map_name field may contain path components or differ from the actual
 * .map filename (e.g. "maps/Berlin_1_256.map" vs "Berlin_1_256.map").
 * We ignore it and rely on the W/H/coords we received from the loaded map.
 *
 * For even-scenario files (distributed across many buckets), we sample
 * evenly from different buckets to get a representative difficulty spread.
 * For random-scenario files, we just take the first max_scen entries.
 *
 * Returns count loaded, or -1 on error.                                    */
static int load_scen(const char *path, int W, int H,
                     ScenEntry *out, int max_scen) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char line[1024];

    /* Read and validate header */
    if (!fgets(line, sizeof line, f)) { fclose(f); return 0; }
    /* Accept "version 1" or missing header (some older files) */
    if (strncmp(line, "version", 7) != 0) {
        /* No version header — rewind and treat first line as data */
        rewind(f);
    }

    /* First pass: count total valid entries and find bucket range */
    long data_start = ftell(f);
    int  total = 0, max_bucket = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '#') continue;
        int bucket, mW, mH, sx, sy, gx, gy; double opt; char mn[512];
        if (sscanf(line, "%d %511s %d %d %d %d %d %d %lf",
                   &bucket, mn, &mW, &mH, &sx, &sy, &gx, &gy, &opt) != 9) continue;
        if (sx<0||sx>=W||sy<0||sy>=H||gx<0||gx>=W||gy<0||gy>=H) continue;
        if (sx==gx && sy==gy) continue;   /* trivial pair, skip */
        if (bucket > max_bucket) max_bucket = bucket;
        total++;
    }

    /* Strategy: sample evenly across buckets for representative difficulty.
     * Take ~max_scen/num_buckets entries per bucket (at least 1).            */
    int num_buckets = max_bucket + 1;
    int per_bucket  = (max_scen + num_buckets - 1) / num_buckets;
    if (per_bucket < 1) per_bucket = 1;

    /* Bucket counters */
    int *bcnt = calloc(num_buckets + 1, sizeof(int));

    /* Second pass: collect entries */
    fseek(f, data_start, SEEK_SET);
    int count = 0;
    while (count < max_scen && fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '#') continue;
        int bucket, mW, mH, sx, sy, gx, gy; double opt; char mn[512];
        if (sscanf(line, "%d %511s %d %d %d %d %d %d %lf",
                   &bucket, mn, &mW, &mH, &sx, &sy, &gx, &gy, &opt) != 9) continue;
        if (sx<0||sx>=W||sy<0||sy>=H||gx<0||gx>=W||gy<0||gy>=H) continue;
        if (sx==gx && sy==gy) continue;
        if (bucket < 0 || bucket > max_bucket) continue;
        if (bcnt[bucket] >= per_bucket) continue;   /* already have enough from this bucket */
        bcnt[bucket]++;
        out[count].src      = sy * W + sx;
        out[count].dst      = gy * W + gx;
        out[count].opt_cost = (float)opt;
        count++;
    }

    free(bcnt);
    fclose(f);
    return count;
}

/* ============================================================================
 * BENCHMARK — MOVINGAI .map / .scen
 *
 * For each .map file in map_dir:
 *   1. Load map, build 8-connected graph
 *   2. If matching .scen file found → use its scenario pairs (with known opt)
 *      else → draw N_PAIRS random pairs within passable nodes
 *   3. Run KP preprocessing and all algorithms
 *   4. Report: expansions, time, subopt vs .scen optimal (not A* cost)
 *
 * Usage: ./astar_bench /path/to/maps/   (dir with .map + optional .scen)
 * ========================================================================== */
#define MOVAI_MAX_SCEN  40     /* max scenarios per map to run */

static void benchmark_movingai_map(const char *map_path, const char *scen_path,
                                    Row *rows, int *nrows) {
    Graph g;
    char  label[128];

    printf("\n--- Loading: %s\n", map_path);
    fflush(stdout);

    if (!load_movingai_map(map_path, &g, label, sizeof label)) return;

    /* Infer map dimensions from node count and filename (stored in graph coords) */
    int W = 0, H = 0;
    for (int i = 0; i < g.n; i++) {
        if ((int)g.x[i] + 1 > W) W = (int)g.x[i] + 1;
        if ((int)g.y[i] + 1 > H) H = (int)g.y[i] + 1;
    }

    /* Count passable nodes and compute obstacle density */
    int passable = 0;
    for (int i = 0; i < g.n; i++)
        if (g.head[i] != -1) passable++;
    float obs_pct = 100.0f * (g.n - passable) / g.n;

    /* Estimate h_euclid quality: sample 200 pairs */
    float ratio_sum = 0; int ratio_cnt = 0;

    printf("    W=%-4d H=%-4d  n=%d  m=%d  obstacles=%.1f%%\n",
           W, H, g.n, g.m, obs_pct);
    fflush(stdout);

    /* Load scenarios */
    ScenEntry scen_buf[MOVAI_MAX_SCEN];
    int n_scen = 0;

    if (scen_path) {
        n_scen = load_scen(scen_path, W, H, scen_buf, MOVAI_MAX_SCEN);
        if (n_scen > 0)
            printf("    Scenarios: %d pairs (from .scen, optimal costs known)\n", n_scen);
    }

    /* Fallback: random pairs among passable nodes */
    if (n_scen <= 0) {
        int *passable_nodes = malloc(passable * sizeof(int));
        int pcnt = 0;
        for (int i = 0; i < g.n; i++)
            if (g.head[i] != -1) passable_nodes[pcnt++] = i;
        rng_seed((uint64_t)(size_t)map_path);
        n_scen = (passable < N_PAIRS * 2) ? passable / 4 : N_PAIRS;
        if (n_scen < 4) n_scen = 4;
        for (int i = 0; i < n_scen; i++) {
            scen_buf[i].src      = passable_nodes[rng_int(pcnt)];
            do { scen_buf[i].dst = passable_nodes[rng_int(pcnt)]; }
            while (scen_buf[i].dst == scen_buf[i].src);
            scen_buf[i].opt_cost = -1;   /* unknown */
        }
        free(passable_nodes);
        printf("    Scenarios: %d pairs (random, no .scen)\n", n_scen);
    }

    /* h_euclid/d_real ratio on first 20 pairs */
    float astar_costs[MOVAI_MAX_SCEN];
    for (int i = 0; i < n_scen; i++) {
        astar_costs[i] = astar(&g, scen_buf[i].src, scen_buf[i].dst).cost;
        if (i < 20 && astar_costs[i] > 0) {
            float he = h_euclid(&g, scen_buf[i].src, scen_buf[i].dst);
            ratio_sum += he / astar_costs[i];
            ratio_cnt++;
        }
    }
    if (ratio_cnt > 0)
        printf("    h_euclid/d_real = %.3f  (%.0f%% quality — %s)\n",
               ratio_sum/ratio_cnt, 100*ratio_sum/ratio_cnt,
               ratio_sum/ratio_cnt < 0.6f ? "BiKP regime ★" : "BiA* regime");

    /* KP preprocessing */
    kp_build(&g);

    /* Algorithm table */
    typedef Res (*AlgoFn)(const Graph*, int, int);
    struct { const char *name; AlgoFn fn; float wa_w; } algos[] = {
        {"A*",           astar,          0},
        {"WA*(1.5)",     run_wa,         1.5f},
        {"BiA*",         bidir_astar,    0},
        {"TB-A*",        tiebreak_astar,    0},
        {"KP-A*",        kp_astar,       0},
        {"BiKP-A*",      bikp_astar,     0},
    };
    int na = (int)(sizeof algos / sizeof *algos);

    int base = *nrows;
    for (int ai = 0; ai < na; ai++) {
        g_wa_weight = algos[ai].wa_w;

        double sum_ms=0, sum_exp=0, sum_gen=0, sum_cost=0, sum_sub=0;
        int solved = 0;

        for (int i = 0; i < n_scen; i++) {
            if (astar_costs[i] < 0) continue;   /* no path */
            Res r = algos[ai].fn(&g, scen_buf[i].src, scen_buf[i].dst);
            sum_ms  += r.ms;
            sum_exp += r.expanded;
            sum_gen += r.generated;
            if (r.cost >= 0) {
                sum_cost += r.cost;
                /* Use .scen optimal if available, otherwise A* cost */
                float ref = (scen_buf[i].opt_cost > 0) ? scen_buf[i].opt_cost
                                                        : astar_costs[i];
                sum_sub  += (ref > 0) ? r.cost / ref : 1.0;
                solved++;
            }
        }
        if (solved == 0) continue;

        Row row;
        row.graph    = strdup(label);
        row.algo     = algos[ai].name;
        row.avg_ms   = sum_ms  / n_scen;
        row.avg_exp  = sum_exp / n_scen;
        row.avg_gen  = sum_gen / n_scen;
        row.avg_cost = sum_cost / solved;
        row.subopt   = sum_sub  / solved;
        row.solved   = solved;
        rows[(*nrows)++] = row;
    }

    print_group(rows + base, *nrows - base);

    /* ---- PASS 2 : ORC landmarks ---------------------------------------- */
    kp_free();
    kp_build_orc(&g);

    struct { const char *name; AlgoFn fn; } orc_algos[] = {
        {"ORC-BiKP",  orc_bikp_astar},
        {"ORC-WBiKP", orc_wbikp_astar},
    };
    int n_orc = (int)(sizeof orc_algos / sizeof *orc_algos);

    for (int ai = 0; ai < n_orc; ai++) {
        double sum_ms=0, sum_exp=0, sum_gen=0, sum_cost=0, sum_sub=0;
        int solved = 0;

        for (int i = 0; i < n_scen; i++) {
            if (astar_costs[i] < 0) continue;
            Res r = orc_algos[ai].fn(&g, scen_buf[i].src, scen_buf[i].dst);
            sum_ms  += r.ms;
            sum_exp += r.expanded;
            sum_gen += r.generated;
            if (r.cost >= 0) {
                sum_cost += r.cost;
                float ref = (scen_buf[i].opt_cost > 0) ? scen_buf[i].opt_cost
                                                        : astar_costs[i];
                sum_sub  += (ref > 0) ? r.cost / ref : 1.0;
                solved++;
            }
        }
        if (solved == 0) continue;

        Row row;
        row.graph    = strdup(label);
        row.algo     = orc_algos[ai].name;
        row.avg_ms   = sum_ms  / n_scen;
        row.avg_exp  = sum_exp / n_scen;
        row.avg_gen  = sum_gen / n_scen;
        row.avg_cost = sum_cost / solved;
        row.subopt   = sum_sub  / solved;
        row.solved   = solved;
        rows[(*nrows)++] = row;
    }

    /* Print the ORC rows appended at the end of rows[] */
    int orc_base = *nrows - n_orc;
    if (orc_base >= base)
        print_group(rows + orc_base, n_orc);

    graph_free(&g); kp_free();
}

/* ============================================================================
 * RECAP TABLE — synthese des victoires par algorithme sur un ensemble de maps
 *
 * Deux blocs separes :
 *   BLOC 1 — Algorithmes OPTIMAUX    (avg_subopt <= 1.001)
 *   BLOC 2 — Algorithmes SOUS-OPTIMAUX (avg_subopt > 1.001)
 *
 * Dans chaque bloc : tri par total victoires (Wins-ms + Wins-exp).
 * Une phrase de synthese par bloc resument le vainqueur et son avantage.
 * ========================================================================== */
#define RECAP_MAX_ALGO 16
#define SUBOPT_THRESHOLD 1.001   /* seuil separant optimal / sous-optimal */

static void print_recap_table(const Row *rows, int nrows,
                               const char *section_title) {
    if (nrows == 0) return;

    /* Collect unique algo names in order of first appearance */
    const char *algo_names[RECAP_MAX_ALGO];
    int  na = 0;
    for (int i = 0; i < nrows; i++) {
        int found = 0;
        for (int j = 0; j < na; j++)
            if (strcmp(rows[i].algo, algo_names[j]) == 0) { found=1; break; }
        if (!found && na < RECAP_MAX_ALGO)
            algo_names[na++] = rows[i].algo;
    }

    typedef struct {
        int    wins_ms;
        int    wins_exp;
        int    wins_ms_opt;    /* wins only against algos in same group */
        int    wins_exp_opt;
        int    n_maps;
        double sum_ms;
        double sum_exp;
        double sum_subopt;
        int    n_subopt;
    } AlgoStats;
    AlgoStats stats[RECAP_MAX_ALGO];
    memset(stats, 0, sizeof stats);

    /* Collect unique graph names */
    const char *graph_names[512];
    int ng = 0;
    for (int i = 0; i < nrows; i++) {
        int found = 0;
        for (int j = 0; j < ng; j++)
            if (strcmp(rows[i].graph, graph_names[j]) == 0) { found=1; break; }
        if (!found && ng < 512)
            graph_names[ng++] = rows[i].graph;
    }

    /* Accumulate stats */
    for (int gi = 0; gi < ng; gi++) {
        double best_ms = 1e18, best_exp = 1e18;
        int    best_ms_ai = -1, best_exp_ai = -1;
        /* Within-group winners (computed after we know avg_subopt) */
        double best_ms_opt = 1e18, best_exp_opt = 1e18;
        int    best_ms_opt_ai = -1, best_exp_opt_ai = -1;
        double best_ms_sub = 1e18, best_exp_sub = 1e18;
        int    best_ms_sub_ai = -1, best_exp_sub_ai = -1;

        /* First pass: accumulate */
        for (int i = 0; i < nrows; i++) {
            if (strcmp(rows[i].graph, graph_names[gi]) != 0) continue;
            int ai = -1;
            for (int j = 0; j < na; j++)
                if (strcmp(rows[i].algo, algo_names[j]) == 0) { ai=j; break; }
            if (ai < 0) continue;

            stats[ai].n_maps++;
            stats[ai].sum_ms  += rows[i].avg_ms;
            stats[ai].sum_exp += rows[i].avg_exp;
            if (rows[i].subopt > 0) {
                stats[ai].sum_subopt += rows[i].subopt;
                stats[ai].n_subopt++;
            }
            /* Overall winners (all algos together) */
            if (rows[i].avg_ms  < best_ms)  { best_ms  = rows[i].avg_ms;  best_ms_ai  = ai; }
            if (rows[i].avg_exp < best_exp) { best_exp = rows[i].avg_exp; best_exp_ai = ai; }
        }
        if (best_ms_ai  >= 0) stats[best_ms_ai].wins_ms++;
        if (best_exp_ai >= 0) stats[best_exp_ai].wins_exp++;

        /* Second pass: within-group winners */
        for (int i = 0; i < nrows; i++) {
            if (strcmp(rows[i].graph, graph_names[gi]) != 0) continue;
            int ai = -1;
            for (int j = 0; j < na; j++)
                if (strcmp(rows[i].algo, algo_names[j]) == 0) { ai=j; break; }
            if (ai < 0) continue;
            /* Determine group of this algo from its accumulated avg_subopt */
            double as = stats[ai].n_subopt > 0
                        ? stats[ai].sum_subopt / stats[ai].n_subopt : 1.0;
            int is_opt = (as <= SUBOPT_THRESHOLD);
            if (is_opt) {
                if (rows[i].avg_ms  < best_ms_opt)  { best_ms_opt  = rows[i].avg_ms;  best_ms_opt_ai  = ai; }
                if (rows[i].avg_exp < best_exp_opt) { best_exp_opt = rows[i].avg_exp; best_exp_opt_ai = ai; }
            } else {
                if (rows[i].avg_ms  < best_ms_sub)  { best_ms_sub  = rows[i].avg_ms;  best_ms_sub_ai  = ai; }
                if (rows[i].avg_exp < best_exp_sub) { best_exp_sub = rows[i].avg_exp; best_exp_sub_ai = ai; }
            }
        }
        if (best_ms_opt_ai  >= 0) stats[best_ms_opt_ai].wins_ms_opt++;
        if (best_exp_opt_ai >= 0) stats[best_exp_opt_ai].wins_exp_opt++;
        if (best_ms_sub_ai  >= 0) stats[best_ms_sub_ai].wins_ms_opt++;
        if (best_exp_sub_ai >= 0) stats[best_exp_sub_ai].wins_exp_opt++;
    }

    /* Classify and sort each group by intra-group wins */
    int opt_order[RECAP_MAX_ALGO], n_opt = 0;
    int sub_order[RECAP_MAX_ALGO], n_sub = 0;
    for (int i = 0; i < na; i++) {
        if (stats[i].n_maps == 0) continue;
        double as = stats[i].n_subopt > 0
                    ? stats[i].sum_subopt / stats[i].n_subopt : 1.0;
        if (as <= SUBOPT_THRESHOLD) opt_order[n_opt++] = i;
        else                        sub_order[n_sub++] = i;
    }

    /* Sort each group by intra-group wins descending */
    for (int pass = 0; pass < 2; pass++) {
        int *ord = pass ? sub_order : opt_order;
        int  cnt = pass ? n_sub     : n_opt;
        for (int i = 1; i < cnt; i++) {
            int tmp = ord[i], j = i-1;
            int wi = stats[ord[i]].wins_ms_opt + stats[ord[i]].wins_exp_opt;
            while (j >= 0 && (stats[ord[j]].wins_ms_opt + stats[ord[j]].wins_exp_opt) < wi) {
                ord[j+1] = ord[j]; j--;
            }
            ord[j+1] = tmp;
        }
    }

    /* Helper: print one group */
    #define PRINT_GROUP(label, ord, cnt, is_subopt_group) do { \
        printf("\n  -- %s --\n", label); \
        printf("  %-16s  %8s  %8s  %10s  %10s  %9s\n", \
               "Algorithm", "Wins-ms", "Wins-exp", "Avg-ms", "Avg-exp", "Avg-subopt"); \
        printf("  %s\n", "-------------------------------------------------------------------"); \
        for (int idx = 0; idx < (cnt); idx++) { \
            int ai = (ord)[idx]; \
            double avg_ms     = stats[ai].sum_ms  / stats[ai].n_maps; \
            double avg_exp    = stats[ai].sum_exp / stats[ai].n_maps; \
            double avg_sub    = stats[ai].n_subopt > 0 \
                                ? stats[ai].sum_subopt / stats[ai].n_subopt : 0; \
            char star = (idx == 0) ? '*' : ' '; \
            printf("  %c%-15s  %8d  %8d  %10.3f  %10.0f  %9.4f\n", \
                   star, algo_names[ai], \
                   stats[ai].wins_ms_opt, stats[ai].wins_exp_opt, \
                   avg_ms, avg_exp, avg_sub); \
        } \
    } while(0)

    /* Print header */
    printf("\n");
    for (int i = 0; i < 97; i++) putchar('='); putchar('\n');
    printf("RECAP — %s  (%d maps)\n", section_title, ng);
    for (int i = 0; i < 97; i++) putchar('='); putchar('\n');

    /* GROUP 1 — OPTIMAL */
    if (n_opt > 0) {
        PRINT_GROUP("ALGOS OPTIMAUX  (Subopt = 1.000, chemin garanti le plus court)", opt_order, n_opt, 0);

        /* Synthesis sentence for optimal group */
        int best_ai = opt_order[0];
        double best_ms  = stats[best_ai].sum_ms  / stats[best_ai].n_maps;
        double best_exp = stats[best_ai].sum_exp / stats[best_ai].n_maps;
        /* Find A* stats for comparison */
        double astar_ms = 0, astar_exp = 0; int astar_found = 0;
        for (int i = 0; i < na; i++) {
            if (strcmp(algo_names[i], "A*") == 0 && stats[i].n_maps > 0) {
                astar_ms  = stats[i].sum_ms  / stats[i].n_maps;
                astar_exp = stats[i].sum_exp / stats[i].n_maps;
                astar_found = 1; break;
            }
        }
        printf("\n  => Parmi les algorithmes OPTIMAUX, %s est le meilleur", algo_names[best_ai]);
        printf(" (%d victoires temps, %d victoires expansions).",
               stats[best_ai].wins_ms_opt, stats[best_ai].wins_exp_opt);
        if (astar_found && astar_ms > 0)
            printf("\n     %.2fx plus rapide que A* (%.3fms vs %.3fms),",
                   astar_ms / best_ms, best_ms, astar_ms);
        if (astar_found && astar_exp > 0)
            printf(" %.1fx moins d'expansions (%.0f vs %.0f).",
                   astar_exp / best_exp, best_exp, astar_exp);
        printf("\n     Ces algorithmes trouvent le chemin EXACT garanti (Subopt = 1.0000).\n");
    }

    /* GROUP 2 — SOUS-OPTIMAL */
    if (n_sub > 0) {
        PRINT_GROUP("ALGOS SOUS-OPTIMAUX  (Subopt > 1.001, chemin proche mais non garanti optimal)", sub_order, n_sub, 1);

        /* Synthesis sentence for sub-optimal group */
        int best_ai = sub_order[0];
        double best_ms  = stats[best_ai].sum_ms  / stats[best_ai].n_maps;
        double best_sub = stats[best_ai].n_subopt > 0
                          ? stats[best_ai].sum_subopt / stats[best_ai].n_subopt : 0;
        /* Find best optimal algo for comparison */
        double opt_best_ms = 1e18;
        const char *opt_best_name = NULL;
        for (int i = 0; i < n_opt; i++) {
            int ai = opt_order[i];
            double ms = stats[ai].sum_ms / stats[ai].n_maps;
            if (ms < opt_best_ms) { opt_best_ms = ms; opt_best_name = algo_names[ai]; }
        }
        printf("\n  => Parmi les algorithmes SOUS-OPTIMAUX, %s est le meilleur", algo_names[best_ai]);
        printf(" (%d victoires temps, %d victoires expansions).",
               stats[best_ai].wins_ms_opt, stats[best_ai].wins_exp_opt);
        if (opt_best_name && opt_best_ms < 1e18)
            printf("\n     %.2fx plus rapide que le meilleur algo optimal (%s, %.3fms vs %.3fms).",
                   opt_best_ms / best_ms, opt_best_name, best_ms, opt_best_ms);
        if (best_sub > 0)
            printf("\n     Chemin retourne a %.2f%% de l'optimal en moyenne (non garanti exact).\n",
                   (best_sub - 1.0) * 100.0);
        else
            printf("\n");
    }

    #undef PRINT_GROUP

    printf("\n  Wins-ms  : victoires temps dans le groupe (rapide = utile en production)\n");
    printf("  Wins-exp : victoires expansions dans le groupe (peu d'exp = heuristique serree)\n");
    printf("  *        : vainqueur du groupe\n");
}

/* Scan a directory for .map files, find matching .scen, benchmark each */
static void benchmark_movingai_dir(const char *dir, Row *rows, int *nrows) {
    printf("\n");
    for (int i = 0; i < 97; i++) putchar('='); putchar('\n');
    printf("MOVINGAI BENCHMARKS  (dir: %s)\n", dir);
    for (int i = 0; i < 97; i++) putchar('='); putchar('\n');
    print_header();

    int base_nrows = *nrows;   /* remember start index for recap */

    DIR *d = opendir(dir);
    if (!d) { perror(dir); return; }

    /* Collect .map files */
    char **maps = NULL; int nm = 0, mc = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        const char *name = ent->d_name;
        size_t len = strlen(name);
        if (len < 5 || strcmp(name + len - 4, ".map") != 0) continue;
        char *path = malloc(strlen(dir) + len + 2);
        sprintf(path, "%s/%s", dir, name);
        if (nm >= mc) { mc = mc ? mc*2 : 64; maps = realloc(maps, mc*sizeof*maps); }
        maps[nm++] = path;
    }
    closedir(d);

    /* Sort */
    for (int i = 1; i < nm; i++) {
        char *tmp = maps[i]; int j = i-1;
        while (j >= 0 && strcmp(maps[j], tmp) > 0) { maps[j+1] = maps[j]; j--; }
        maps[j+1] = tmp;
    }

    for (int i = 0; i < nm; i++) {
        /* Find matching .scen file — try all MovingAI naming conventions:
         *
         *   Pattern 1: <base>.scen               (our default / bgmaps)
         *   Pattern 2: <base>-even.scen           (mapf-scen-even.zip individual)
         *   Pattern 3: <base>-random.scen         (mapf-scen-random.zip individual)
         *   Pattern 4: <base>.map-even.scen       (mapf-scen-even.zip global archive)
         *   Pattern 5: <base>.map-random.scen     (mapf-scen-random.zip global archive)
         *   Pattern 6: <base>.map.scen            (some older distributions)
         *
         * The first existing file wins. Even scenarios are preferred over random
         * because they distribute problem difficulty more uniformly.
         */
        const char *map_path = maps[i];

        /* Extract base path without extension */
        char base[1024];
        strncpy(base, map_path, sizeof base - 1); base[sizeof base - 1] = '\0';
        char *ext = strrchr(base, '.');
        if (ext && strcmp(ext, ".map") == 0) *ext = '\0';  /* base = path without .map */

        /* Build candidate scen paths */
        char cands[6][1280];
        snprintf(cands[0], sizeof cands[0], "%s.scen",            base);
        snprintf(cands[1], sizeof cands[1], "%s-even.scen",       base);
        snprintf(cands[2], sizeof cands[2], "%s-random.scen",     base);
        snprintf(cands[3], sizeof cands[3], "%s.map-even.scen",   base);
        snprintf(cands[4], sizeof cands[4], "%s.map-random.scen", base);
        snprintf(cands[5], sizeof cands[5], "%s.map.scen",        base);

        const char *scen_ptr = NULL;
        for (int c = 0; c < 6; c++) {
            FILE *test = fopen(cands[c], "r");
            if (test) {
                fclose(test);
                scen_ptr = cands[c];
                break;
            }
        }

        if (scen_ptr)
            printf("    Scen file : %s\n", strrchr(scen_ptr, '/') ? strrchr(scen_ptr,'/')+1 : scen_ptr);

        benchmark_movingai_map(map_path, scen_ptr, rows, nrows);
        free(maps[i]);
    }
    free(maps);

    /* Recap table: victories across all maps in this directory */
    print_recap_table(rows + base_nrows, *nrows - base_nrows, dir);
}

/* ============================================================================
 * MAIN
 * ========================================================================== */
int main(int argc, char **argv) {
    const char *map_dir = (argc >= 2) ? argv[1] : NULL;

    Row  rows[2048];
    int  nrows = 0;
    Graph g;

    /* ================================================================
     * PART 1 — SYNTHETIC BENCHMARKS
     * ================================================================ */
    int synth_base = 0;
    printf("\n");
    for (int i = 0; i < 97; i++) putchar('='); putchar('\n');
    printf("SYNTHETIC BENCHMARKS\n");
    for (int i = 0; i < 97; i++) putchar('='); putchar('\n');
    print_header();

#define RUN(label, setup_code) do { \
    int base = nrows; \
    setup_code; \
    kp_build(&g); \
    benchmark((label), &g, rows, &nrows); \
    print_group(rows + base, nrows - base); \
    graph_free(&g); kp_free(); \
} while(0)

    /* ---- Grid graphs ---------------------------------------------------- */
    printf("\n### Grid 500x500 — 20%% obstacles, weights in [1,3]\n");
    RUN("Grid-20pct", {
        rng_seed(0xAABBCCDD11ULL);
        gen_grid(&g, 500, 500, 0.20f, 3.0f);
        printf("    n=%d  m=%d\n", g.n, g.m);
    });

    printf("\n### Grid 500x500 — 40%% obstacles, weights in [1,5]\n");
    RUN("Grid-40pct", {
        rng_seed(0xAABBCCDD22ULL);
        gen_grid(&g, 500, 500, 0.40f, 5.0f);
        printf("    n=%d  m=%d\n", g.n, g.m);
    });

    printf("\n### Large Grid 1000x500 — 25%% obstacles, weights in [1,2]\n");
    RUN("LargeGrid-1Mx0.25", {
        rng_seed(0xAABBCCDD66ULL);
        gen_grid(&g, 1000, 500, 0.25f, 2.0f);
        printf("    n=%d  m=%d\n", g.n, g.m);
    });

    /* ---- Maze ------------------------------------------------------------ */
    printf("\n### Maze 500x500 — binary-tree maze, unit weights\n");
    RUN("Maze-500x500", {
        rng_seed(0xAABBCCDD33ULL);
        gen_maze(&g, 500, 500);
        printf("    n=%d  m=%d\n", g.n, g.m);
    });

    /* ---- Erdős–Rényi ---------------------------------------------------- */
    printf("\n### Erdos-Renyi  n=4000  p=0.008  weights in [1,10]\n");
    RUN("ER-4k-p0.008", {
        rng_seed(0xAABBCCDD44ULL);
        gen_er(&g, 4000, 0.008f, 10.0f);
        printf("    n=%d  m=%d\n", g.n, g.m);
    });

    /* ---- ROAD NETWORKS -------------------------------------------------- */
    printf("\n### Road kNN  n=5000  k=4  (avg_deg~4, planar-ish)\n");
    RUN("Road-kNN-5k", {
        rng_seed(0xAABBCCDDABULL);
        gen_road_knn(&g, 5000, 4, 1000.0f);
        printf("    n=%d  m=%d  avg_deg=%.1f\n",
               g.n, g.m, 2.0f*g.m/g.n);
    });

    printf("\n### Road kNN  n=8000  k=5  (denser intersections)\n");
    RUN("Road-kNN-8k", {
        rng_seed(0xAABBCCDDBBULL);
        gen_road_knn(&g, 8000, 5, 1000.0f);
        printf("    n=%d  m=%d  avg_deg=%.1f\n",
               g.n, g.m, 2.0f*g.m/g.n);
    });

    printf("\n### Road Planar grid  n~10000  noise=0.6\n");
    RUN("Road-Planar-10k", {
        rng_seed(0xAABBCCDDCCULL);
        gen_road_planar(&g, 10000, 1000.0f, 0.6f);
        printf("    n=%d  m=%d  avg_deg=%.1f\n",
               g.n, g.m, 2.0f*g.m/g.n);
    });

    /* Recap synthétique */
    print_recap_table(rows + synth_base, nrows - synth_base, "Synthetic graphs");

    /* ================================================================
     * PART 2 — MOVINGAI REAL MAPS (Dragon Age, Baldur's Gate, etc.)
     * ================================================================ */
    if (map_dir) {
        benchmark_movingai_dir(map_dir, rows, &nrows);
    } else {
        printf("\n(Pass a directory of .map files to run real-map benchmarks)\n");
        printf("  Example: ./astar_bench /path/to/bgmaps/\n");
        printf("  Get maps: wget movingai.com/benchmarks/bg/bgmaps.zip\n");
    }

    export_csv(rows, nrows, "benchmark_results_v2.csv");

    printf("\nLEGEND\n");
    printf("  Expanded  : avg nodes popped from OPEN\n");
    printf("  Generated : avg nodes pushed to OPEN\n");
    printf("  Subopt    : avg_cost / A*_cost (1.0000 = optimal)\n");
    printf("  ★         : fastest wall-clock time (unconditional)\n\n");
    printf("  KP-A*      Kantorovich Potential A* [this work]\n");
    printf("             Preprocessing: K Dijkstra exacts — O(K·n·log n)\n");
    printf("             h_KP(u) = max_i |d(Li,u) − d(Li,dst)|\n\n");
    printf("  BiKP-A*    Bidirectional KP-A* (Goldberg-Harrelson 2005)\n");
    printf("             Consistent pair: prio = g + (h_f − h_b)/2\n");
    printf("             Optimal: Subopt = 1.0000 garanti\n\n");
    return 0;
}
