# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  Dijkstra — Plus court chemin
  Implémentation Python élégante, performante et vérifiée
═══════════════════════════════════════════════════════════════════════

  Complexité  : O((V + E) log V)  via heap binaire (heapq)
  Espace      : O(V + E)
  Contraintes : poids ≥ 0 (requis par Dijkstra)
"""

from __future__ import annotations
import heapq
from collections import defaultdict
from typing import Any, Iterator


# ═══════════════════════════════════════════════════════
#  Structure de graphe
# ═══════════════════════════════════════════════════════

class Graph:
    """
    Graphe orienté pondéré (liste d'adjacence).
    Supporte tout hashable comme nœud (str, int, tuple…).
    """

    def __init__(self):
        self._adj: dict[Any, list[tuple[float, Any]]] = defaultdict(list)
        self._nodes: set = set()

    # ── Construction ──────────────────────────────────────

    def add_edge(self, u, v, weight: float, bidirectional: bool = False) -> "Graph":
        """Ajoute une arête u→v de poids `weight`. Chainable."""
        if weight < 0:
            raise ValueError(f"Poids négatif interdit pour Dijkstra : {u}→{v} ({weight})")
        self._adj[u].append((weight, v))
        self._nodes |= {u, v}
        if bidirectional:
            self._adj[v].append((weight, u))
        return self

    def add_node(self, node) -> "Graph":
        """Ajoute un nœud isolé."""
        self._nodes.add(node)
        return self

    @property
    def nodes(self) -> set:
        return self._nodes

    @property
    def edges(self) -> Iterator[tuple]:
        for u, neighbors in self._adj.items():
            for w, v in neighbors:
                yield u, v, w

    def __repr__(self) -> str:
        return f"Graph({len(self._nodes)} nœuds, {sum(len(v) for v in self._adj.values())} arêtes)"


# ═══════════════════════════════════════════════════════
#  Algorithme de Dijkstra
# ═══════════════════════════════════════════════════════

def dijkstra(
    graph: Graph,
    source: Any,
    target: Any = None,
) -> tuple[dict, dict]:
    """
    Algorithme de Dijkstra depuis `source`.

    Paramètres
    ──────────
    graph   : graphe orienté pondéré (poids ≥ 0)
    source  : nœud de départ
    target  : nœud cible optionnel — arrêt anticipé si atteint

    Retourne
    ────────
    dist    : dict {nœud: distance minimale depuis source}
              (inf pour les nœuds non atteignables)
    prev    : dict {nœud: prédécesseur sur le chemin optimal}
              (None pour source et nœuds non atteignables)

    Complexité
    ──────────
    O((V + E) log V) — V nœuds, E arêtes
    """
    if source not in graph.nodes:
        raise KeyError(f"Nœud source inconnu : {source!r}")

    INF = float("inf")

    # Initialisation
    dist = {node: INF for node in graph.nodes}
    prev = {node: None for node in graph.nodes}
    dist[source] = 0.0

    # Tas min : (distance, nœud)
    heap = [(0.0, source)]

    # Ensemble des nœuds définitivement traités
    visited: set = set()

    while heap:
        d_u, u = heapq.heappop(heap)

        # Entrée périmée dans le tas (distance déjà améliorée)
        if u in visited:
            continue
        visited.add(u)

        # Arrêt anticipé si on a atteint la cible
        if u is target or u == target:
            break

        # Relaxation des voisins
        for w, v in graph._adj[u]:
            if v in visited:
                continue
            d_v = d_u + w
            if d_v < dist[v]:
                dist[v] = d_v
                prev[v] = u
                heapq.heappush(heap, (d_v, v))

    return dist, prev


def reconstruct_path(prev: dict, source: Any, target: Any) -> list | None:
    """
    Reconstruit le chemin optimal depuis `source` vers `target`
    à partir du dictionnaire `prev` retourné par dijkstra().

    Retourne la liste des nœuds du chemin, ou None si inaccessible.
    """
    if prev.get(target) is None and target != source:
        return None  # nœud non atteignable

    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]

    path.reverse()
    return path if path[0] == source else None


def shortest_path(
    graph: Graph,
    source: Any,
    target: Any,
) -> tuple[float, list | None]:
    """
    API haut niveau : retourne (distance, chemin) directement.

    Exemple
    ───────
    >>> d, path = shortest_path(g, "A", "D")
    >>> print(d, path)
    4.0 ['A', 'B', 'D']
    """
    dist, prev = dijkstra(graph, source, target)
    distance = dist.get(target, float("inf"))
    path = reconstruct_path(prev, source, target)
    return distance, path


# ═══════════════════════════════════════════════════════
#  Tests de correction
# ═══════════════════════════════════════════════════════

def _run_tests():
    OK   = "\033[92m✓\033[0m"
    FAIL = "\033[91m✗\033[0m"
    errors = []

    def check(name: str, cond: bool):
        print(f"  {'  OK' if cond else FAIL}  {name}")
        if not cond: errors.append(name)

    def near(a, b, eps=1e-9): return abs(a - b) < eps

    print("\n══════════════════════════════════════════════")
    print("  Tests Dijkstra")
    print("══════════════════════════════════════════════\n")

    # ── Test 1 : graphe simple ────────────────────────────
    print("▸ Graphe triangulaire A→B→C")
    #   A ──2── B ──1── C
    #    ╲              ↗
    #     ────── 4 ─────
    g = Graph()
    g.add_edge("A", "B", 2).add_edge("B", "C", 1).add_edge("A", "C", 4)
    d, p = dijkstra(g, "A")
    check("A→A = 0",             near(d["A"], 0))
    check("A→B = 2",             near(d["B"], 2))
    check("A→C = 3 (via B)",     near(d["C"], 3))
    check("chemin A→C = [A,B,C]", reconstruct_path(p, "A", "C") == ["A","B","C"])

    # ── Test 2 : arrêt anticipé ───────────────────────────
    print("\n▸ Arrêt anticipé (target)")
    d2, p2 = dijkstra(g, "A", target="B")
    check("A→B avec target",     near(d2["B"], 2))

    # ── Test 3 : nœud inaccessible ────────────────────────
    print("\n▸ Nœud inaccessible")
    g2 = Graph()
    g2.add_edge("X", "Y", 5).add_node("Z")
    d3, _ = dijkstra(g2, "X")
    check("Z inaccessible = inf", d3["Z"] == float("inf"))

    # ── Test 4 : graphe non orienté ───────────────────────
    print("\n▸ Graphe non orienté")
    g3 = Graph()
    g3.add_edge("A", "B", 1, bidirectional=True)
    g3.add_edge("B", "C", 2, bidirectional=True)
    g3.add_edge("A", "C", 10, bidirectional=True)
    dist, prev = dijkstra(g3, "C")
    check("C→A = 3 (bidirectionnel)", near(dist["A"], 3))

    # ── Test 5 : graphe plus large ────────────────────────
    print("\n▸ Graphe classique ABCDEF")
    #    A──1──B──2──C
    #    |     |     |
    #    4     3     1
    #    |     |     |
    #    D──5──E──1──F
    g4 = Graph()
    edges = [
        ("A","B",1),("B","C",2),("A","D",4),
        ("B","E",3),("C","F",1),("D","E",5),("E","F",1),
    ]
    for u, v, w in edges:
        g4.add_edge(u, v, w, bidirectional=True)

    d4, p4 = dijkstra(g4, "A")
    check("A→F = 4 (A→B→C→F)",  near(d4["F"], 4))
    check("A→E = 4 (A→B→C→F→E)", near(d4["E"], 4))  # A→B→E=4 aussi
    check("chemin A→F correct",
          reconstruct_path(p4, "A", "F") in [["A","B","C","F"]])

    # ── Test 6 : nœuds entiers ────────────────────────────
    print("\n▸ Nœuds entiers")
    g5 = Graph()
    g5.add_edge(0, 1, 7).add_edge(0, 2, 3).add_edge(2, 1, 2).add_edge(1, 3, 1)
    d5, _ = dijkstra(g5, 0)
    check("0→1 = 5 (via 2)",     near(d5[1], 5))
    check("0→3 = 6",             near(d5[3], 6))

    # ── Test 7 : source = target ──────────────────────────
    print("\n▸ Source = target")
    dist_self, path_self = shortest_path(g4, "A", "A")
    check("distance = 0",        near(dist_self, 0))
    check("chemin = ['A']",      path_self == ["A"])

    # ── Test 8 : poids négatif → ValueError ───────────────
    print("\n▸ Poids négatif → ValueError")
    try:
        g_neg = Graph(); g_neg.add_edge("X", "Y", -1)
        check("ValueError levée", False)
    except ValueError:
        check("ValueError levée", True)

    # ── Test 9 : source inconnue → KeyError ───────────────
    print("\n▸ Source inconnue → KeyError")
    try:
        dijkstra(g4, "Z")
        check("KeyError levée", False)
    except KeyError:
        check("KeyError levée", True)

    # ── Test 10 : graphe en étoile (perf) ─────────────────
    print("\n▸ Graphe en étoile (1000 nœuds)")
    g6 = Graph()
    for i in range(1, 1001):
        g6.add_edge(0, i, i)
    d6, _ = dijkstra(g6, 0)
    check("0→500 = 500",         near(d6[500], 500))
    check("0→1000 = 1000",       near(d6[1000], 1000))

    # ── Résultat ──────────────────────────────────────────
    print(f"\n══════════════════════════════════════════════")
    if not errors:
        print(f"  \033[92mTous les tests passés ✓\033[0m")
    else:
        print(f"  \033[91m{len(errors)} échec(s) : {errors}\033[0m")
    print(f"══════════════════════════════════════════════\n")
    return not errors


# ═══════════════════════════════════════════════════════
#  Démonstration
# ═══════════════════════════════════════════════════════

def _demo():
    print("════════════════════════════════════")
    print("  Démonstration")
    print("════════════════════════════════════\n")

    g = Graph()
    g.add_edge("Paris",    "Lyon",      466, bidirectional=True)
    g.add_edge("Paris",    "Bordeaux",  579, bidirectional=True)
    g.add_edge("Lyon",     "Marseille", 315, bidirectional=True)
    g.add_edge("Bordeaux", "Toulouse",  244, bidirectional=True)
    g.add_edge("Toulouse", "Marseille", 405, bidirectional=True)
    g.add_edge("Lyon",     "Toulouse",  535, bidirectional=True)

    pairs = [
        ("Paris", "Marseille"),
        ("Paris", "Toulouse"),
        ("Bordeaux", "Marseille"),
    ]
    for src, tgt in pairs:
        dist, path = shortest_path(g, src, tgt)
        print(f"  {src} → {tgt}")
        print(f"    Distance : {dist} km")
        print(f"    Chemin   : {' → '.join(path)}\n")


# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    ok = _run_tests()
    if ok:
        _demo()
