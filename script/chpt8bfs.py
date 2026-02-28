# -*- coding: utf-8 -*-
"""
Breadth-First Search (BFS)
==========================
Parcours en largeur d'abord d'un graphe non-orienté ou orienté.

Applications couvertes :
  - Parcours complet d'un graphe (composantes connexes)
  - Plus court chemin en nombre d'arêtes (graphe non pondéré)
  - Détection de bipartition (2-coloration)
  - Distance et arbre BFS
  - BFS multi-source

Complexité :
  - Temporelle : O(V + E)
  - Spatiale   : O(V)

Représentation : liste d'adjacence (dict[int, list[int]])
Compatible graphes orientés et non-orientés, connexes et déconnectés.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Generator, Optional


# ══════════════════════════════════════════════════════════════════════
# Type de base : graphe par liste d'adjacence
# ══════════════════════════════════════════════════════════════════════

Graph = dict[int, list[int]]


def make_graph(n: int, edges: list[tuple[int, int]],
               directed: bool = False) -> Graph:
    """
    Construit une liste d'adjacence à partir d'une liste d'arêtes.

    Paramètres
    ──────────
    n        : nombre de sommets (indices 0 … n-1)
    edges    : liste de paires (u, v)
    directed : si False, ajoute l'arête dans les deux sens
    """
    g: Graph = {v: [] for v in range(n)}
    for u, v in edges:
        g[u].append(v)
        if not directed:
            g[v].append(u)
    return g


# ══════════════════════════════════════════════════════════════════════
# Structure de résultat
# ══════════════════════════════════════════════════════════════════════

@dataclass
class BFSResult:
    """
    Résultat d'un parcours BFS depuis une (ou plusieurs) sources.

    Champs
    ──────
    source    : sommet(s) de départ
    dist      : dist[v] = distance (en arêtes) depuis source, None si non atteint
    parent    : parent[v] = prédécesseur de v dans l'arbre BFS, None si racine
    order     : ordre de visite des sommets
    n_visited : nombre de sommets atteints
    """
    source:    int | list[int]
    dist:      dict[int, Optional[int]]
    parent:    dict[int, Optional[int]]
    order:     list[int]
    n_visited: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_visited = sum(1 for d in self.dist.values() if d is not None)

    def reached(self, v: int) -> bool:
        """True si v a été atteint depuis la source."""
        return self.dist.get(v) is not None

    def path(self, dst: int) -> Optional[list[int]]:
        """
        Reconstruit le plus court chemin source → dst.
        Retourne None si dst n'est pas atteignable.
        """
        if not self.reached(dst):
            return None
        route: list[int] = []
        cur: Optional[int] = dst
        while cur is not None:
            route.append(cur)
            cur = self.parent[cur]
        route.reverse()
        return route

    def distance(self, dst: int) -> Optional[int]:
        """Distance (en arêtes) depuis la source jusqu'à dst."""
        return self.dist.get(dst)

    def print_summary(self) -> None:
        sep = "─" * 44
        src_str = str(self.source)
        print(f"\n{sep}")
        print(f"  BFS depuis {src_str}")
        print(sep)
        print(f"  Sommets visités : {self.n_visited}")
        print(f"  Ordre de visite : {self.order}")
        print(sep)
        for v in sorted(self.dist):
            d    = self.dist[v]
            p    = self.parent[v]
            dist_str   = str(d) if d is not None else "∞"
            parent_str = str(p) if p is not None else "—"
            print(f"  v={v:>3}  dist={dist_str:>4}  parent={parent_str:>4}")
        print(sep + "\n")


# ══════════════════════════════════════════════════════════════════════
# 1. BFS depuis une source  (cœur de l'algorithme)
# ══════════════════════════════════════════════════════════════════════

def bfs(graph: Graph, source: int,
        callback: Optional[Callable[[int, int], None]] = None) -> BFSResult:
    """
    BFS classique depuis un sommet source.

    Paramètres
    ──────────
    graph    : liste d'adjacence
    source   : sommet de départ
    callback : fonction optionnelle appelée à chaque visite
               callback(vertex, distance)

    Implémentation
    ──────────────
    Utilise collections.deque pour des opérations popleft() en O(1).
    Un dict `dist` sert simultanément de marqueur "visité" et de
    stockage des distances, évitant un ensemble `visited` séparé.
    """
    dist:   dict[int, Optional[int]] = {v: None for v in graph}
    parent: dict[int, Optional[int]] = {v: None for v in graph}
    order:  list[int] = []

    dist[source] = 0
    queue: deque[int] = deque([source])

    while queue:
        u = queue.popleft()          # O(1) grâce à deque
        order.append(u)

        if callback:
            callback(u, dist[u])     # type: ignore[arg-type]

        for v in graph[u]:
            if dist[v] is None:      # non encore visité
                dist[v]   = dist[u] + 1  # type: ignore[operator]
                parent[v] = u
                queue.append(v)

    return BFSResult(source, dist, parent, order)


# ══════════════════════════════════════════════════════════════════════
# 2. BFS multi-source
# ══════════════════════════════════════════════════════════════════════

def bfs_multi_source(graph: Graph, sources: list[int]) -> BFSResult:
    """
    BFS depuis plusieurs sources simultanément.

    Initialise la file avec toutes les sources à distance 0.
    Utile pour :
      - Plus proche source de chaque sommet
      - 0-1 BFS (nœuds "frontière" pré-placés)
      - Propagation d'onde depuis plusieurs origines

    Complexité : O(V + E), identique au BFS mono-source.
    """
    if not sources:
        raise ValueError("La liste de sources ne peut pas être vide.")

    dist:   dict[int, Optional[int]] = {v: None for v in graph}
    parent: dict[int, Optional[int]] = {v: None for v in graph}
    order:  list[int] = []

    queue: deque[int] = deque()
    for s in sources:
        dist[s] = 0
        queue.append(s)

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            if dist[v] is None:
                dist[v]   = dist[u] + 1  # type: ignore[operator]
                parent[v] = u
                queue.append(v)

    return BFSResult(sources, dist, parent, order)


# ══════════════════════════════════════════════════════════════════════
# 3. BFS complet (composantes connexes d'un graphe déconnecté)
# ══════════════════════════════════════════════════════════════════════

def bfs_all_components(graph: Graph) -> list[BFSResult]:
    """
    Lance un BFS depuis chaque composante connexe non encore visitée.
    Retourne une liste de BFSResult, un par composante.
    Adapté aux graphes déconnectés.
    """
    visited:    set[int]       = set()
    components: list[BFSResult] = []

    for start in graph:
        if start not in visited:
            result = bfs(graph, start)
            components.append(result)
            visited.update(v for v, d in result.dist.items() if d is not None)

    return components


# ══════════════════════════════════════════════════════════════════════
# 4. Détection de bipartition (2-coloration par BFS)
# ══════════════════════════════════════════════════════════════════════

def is_bipartite(graph: Graph) -> tuple[bool, Optional[dict[int, int]]]:
    """
    Détecte si le graphe est biparti via BFS (2-coloration).

    Un graphe est biparti si et seulement si il ne contient pas
    de cycle de longueur impaire.

    Retourne
    ────────
    (True, color) si biparti : color[v] ∈ {0, 1}
    (False, None) si non biparti (cycle impair trouvé)

    Complexité : O(V + E)
    """
    color: dict[int, int] = {}

    for start in graph:
        if start in color:
            continue
        color[start] = 0
        queue: deque[int] = deque([start])

        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in color:
                    color[v] = 1 - color[u]   # couleur opposée
                    queue.append(v)
                elif color[v] == color[u]:    # conflit → cycle impair
                    return False, None

    return True, color


# ══════════════════════════════════════════════════════════════════════
# 5. Générateur BFS (itération paresseuse)
# ══════════════════════════════════════════════════════════════════════

def bfs_iter(graph: Graph, source: int) -> Generator[tuple[int, int], None, None]:
    """
    Générateur BFS : produit (sommet, distance) un par un.
    Permet un arrêt anticipé sans calculer l'ensemble du parcours.

    Exemple :
        for v, d in bfs_iter(g, 0):
            if d > 3:
                break   # n'explore pas au-delà de la profondeur 3
    """
    dist: dict[int, Optional[int]] = {v: None for v in graph}
    dist[source] = 0
    queue: deque[int] = deque([source])

    while queue:
        u = queue.popleft()
        yield u, dist[u]     # type: ignore[misc]
        for v in graph[u]:
            if dist[v] is None:
                dist[v] = dist[u] + 1   # type: ignore[operator]
                queue.append(v)


# ══════════════════════════════════════════════════════════════════════
# Suite de tests de vérification
# ══════════════════════════════════════════════════════════════════════

def _run_tests() -> None:
    passed = failed = 0

    def check(name: str, got, expected) -> None:
        nonlocal passed, failed
        ok = (got == expected)
        if ok:
            print(f"  ✔  {name}")
            passed += 1
        else:
            print(f"  ✘  {name}")
            print(f"       attendu : {expected}")
            print(f"       obtenu  : {got}")
            failed += 1

    print("\n══ Tests BFS ══\n")

    # ── Test 1 : chemin linéaire 0-1-2-3 ─────────────────────────────
    g1 = make_graph(4, [(0,1),(1,2),(2,3)])
    r1 = bfs(g1, 0)
    check("Chaîne : dist[3] = 3",      r1.distance(3), 3)
    check("Chaîne : dist[1] = 1",      r1.distance(1), 1)
    check("Chaîne : chemin 0→3",       r1.path(3), [0, 1, 2, 3])
    check("Chaîne : ordre visite",     r1.order, [0, 1, 2, 3])
    check("Chaîne : parent[3] = 2",    r1.parent[3], 2)

    # ── Test 2 : étoile (source = centre) ────────────────────────────
    g2 = make_graph(5, [(0,1),(0,2),(0,3),(0,4)])
    r2 = bfs(g2, 0)
    check("Étoile : tous à dist 1",    all(r2.distance(v)==1 for v in [1,2,3,4]), True)
    check("Étoile : n_visited = 5",    r2.n_visited, 5)

    # ── Test 3 : graphe déconnecté ───────────────────────────────────
    g3 = make_graph(6, [(0,1),(1,2),(3,4),(4,5)])
    r3 = bfs(g3, 0)
    check("Déco : 3 non atteint",      r3.reached(3), False)
    check("Déco : dist[2] = 2",        r3.distance(2), 2)
    check("Déco : n_visited = 3",      r3.n_visited, 3)

    # ── Test 4 : composantes connexes ────────────────────────────────
    comps = bfs_all_components(g3)
    check("Composantes : 2 trouvées",  len(comps), 2)
    sizes = sorted(c.n_visited for c in comps)
    check("Composantes : tailles [3,3]", sizes, [3, 3])

    # ── Test 5 : graphe complet K4 ───────────────────────────────────
    g4 = make_graph(4, [(i,j) for i in range(4) for j in range(i+1,4)])
    r4 = bfs(g4, 0)
    check("K4 : tous à dist 1",        all(r4.distance(v)==1 for v in [1,2,3]), True)

    # ── Test 6 : cycle pair → biparti ────────────────────────────────
    g5 = make_graph(6, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)])
    bip, col = is_bipartite(g5)
    check("Cycle C6 biparti",          bip, True)
    # vérification des couleurs alternées
    alt_ok = all(col[u] != col[v]
                 for u in g5 for v in g5[u])        # type: ignore[index]
    check("C6 couleurs alternées",     alt_ok, True)

    # ── Test 7 : cycle impair → non biparti ──────────────────────────
    g6 = make_graph(3, [(0,1),(1,2),(2,0)])
    bip2, _ = is_bipartite(g6)
    check("Triangle non biparti",      bip2, False)

    # ── Test 8 : multi-source ────────────────────────────────────────
    g7 = make_graph(7, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)])
    rm = bfs_multi_source(g7, [0, 6])
    check("Multi-source : dist[3] = 3",  rm.distance(3), 3)
    check("Multi-source : dist[0] = 0",  rm.distance(0), 0)
    check("Multi-source : dist[6] = 0",  rm.distance(6), 0)
    check("Multi-source : dist[1] = 1",  rm.distance(1), 1)
    check("Multi-source : dist[5] = 1",  rm.distance(5), 1)

    # ── Test 9 : graphe orienté ───────────────────────────────────────
    gd = make_graph(4, [(0,1),(1,2),(0,3)], directed=True)
    rd = bfs(gd, 0)
    check("Orienté : dist[2] = 2",     rd.distance(2), 2)
    check("Orienté : 3 non atteint depuis 2", bfs(gd, 2).reached(0), False)

    # ── Test 10 : sommet isolé ────────────────────────────────────────
    gi = make_graph(3, [(0,1)])
    ri = bfs(gi, 2)
    check("Isolé : n_visited = 1",     ri.n_visited, 1)
    check("Isolé : chemin vers 0 = None", ri.path(0), None)

    # ── Test 11 : générateur avec arrêt anticipé ──────────────────────
    g8 = make_graph(5, [(0,1),(1,2),(2,3),(3,4)])
    visited_at_most_2 = [v for v, d in bfs_iter(g8, 0) if d <= 2]
    check("Générateur : profondeur ≤2", visited_at_most_2, [0, 1, 2])

    # ── Test 12 : plus court chemin dans grille 3×3 ───────────────────
    def grid(r, c):
        edges = []
        for i in range(r):
            for j in range(c):
                v = i*c+j
                if j+1 < c: edges.append((v, v+1))
                if i+1 < r: edges.append((v, v+c))
        return make_graph(r*c, edges)
    g9 = grid(3, 3)  # sommets 0..8
    r9 = bfs(g9, 0)
    check("Grille 3×3 : dist[8] = 4",  r9.distance(8), 4)
    check("Grille 3×3 : chemin 0→8",   r9.path(8), [0, 1, 2, 5, 8])

    print(f"\n  Résultat : {passed} réussi(s), {failed} échoué(s)\n")


# ══════════════════════════════════════════════════════════════════════
# Démonstration
# ══════════════════════════════════════════════════════════════════════

def demo() -> None:
    print("══ Démonstration BFS ══\n")

    # Graphe CLRS Figure 22.3
    edges = [
        (0,1),(0,4),
        (1,5),
        (2,3),(2,5),(2,6),
        (3,6),(3,7),
        (5,6),
        (6,7),
    ]
    g = make_graph(8, edges)
    r = bfs(g, 1)
    r.print_summary()

    print("  Plus courts chemins depuis 1 :")
    for dst in range(8):
        p = r.path(dst)
        d = r.distance(dst)
        if p is not None:
            print(f"    1 → {dst}  (dist={d})  chemin : {' → '.join(map(str,p))}")

    print()
    bip, col = is_bipartite(g)
    print(f"  Graphe biparti : {bip}")
    if bip and col:
        A = [v for v, c in col.items() if c == 0]
        B = [v for v, c in col.items() if c == 1]
        print(f"  Partition A = {sorted(A)}")
        print(f"  Partition B = {sorted(B)}")


if __name__ == "__main__":
    _run_tests()
    demo()
