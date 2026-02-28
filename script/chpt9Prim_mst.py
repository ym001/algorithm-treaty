"""
Algorithme de Prim — Arbre Couvrant Minimal (Minimum Spanning Tree)
====================================================================
Complexité : O(E log V)  avec un tas binaire + liste d'adjacence
Auteur      : généré et vérifié

Principe :
  1. Partir d'un sommet arbitraire.
  2. Maintenir un tas-min des arêtes « de frontière » (qui relient
     l'arbre en construction à un sommet non encore visité).
  3. Extraire l'arête de poids minimal, ajouter le sommet cible à
     l'arbre, pousser ses nouvelles arêtes de frontière dans le tas.
  4. Répéter jusqu'à avoir V sommets dans l'arbre.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Generator


# ── Types ─────────────────────────────────────────────────────────────────────

Graph = dict[int, list[tuple[int, float]]]   # sommet → [(voisin, poids)]


# ── Construction du graphe ────────────────────────────────────────────────────

def build_graph(edges: list[tuple[int, int, float]]) -> Graph:
    """
    Construit un graphe non-orienté à partir d'une liste d'arêtes.

    Paramètres
    ----------
    edges : liste de (u, v, poids)

    Retourne
    --------
    Dictionnaire d'adjacence {sommet: [(voisin, poids), ...]}
    """
    graph: Graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    return graph


# ── Algorithme de Prim ────────────────────────────────────────────────────────

@dataclass(order=True)
class _HeapEntry:
    """Entrée du tas-min : comparaison sur le poids uniquement."""
    weight: float
    u: int = field(compare=False)
    v: int = field(compare=False)


def prim(graph: Graph, start: int = 0) -> tuple[list[tuple[int, int, float]], float]:
    """
    Calcule l'Arbre Couvrant Minimal d'un graphe connexe non-orienté pondéré.

    Paramètres
    ----------
    graph : graphe sous forme de liste d'adjacence (voir `build_graph`)
    start : sommet de départ (par défaut 0)

    Retourne
    --------
    (arêtes_mst, poids_total)
        arêtes_mst : liste de triplets (u, v, poids) formant l'ACM
        poids_total : somme des poids de l'ACM

    Lève
    ----
    ValueError  si le graphe est vide ou si le sommet de départ est absent.
    ValueError  si le graphe n'est pas connexe.

    Complexité
    ----------
    Temps  : O(E log V)
    Espace : O(V + E)
    """
    if not graph:
        raise ValueError("Le graphe est vide.")
    if start not in graph:
        raise ValueError(f"Le sommet de départ {start!r} n'existe pas dans le graphe.")

    visited: set[int] = {start}
    mst_edges: list[tuple[int, int, float]] = []
    total_weight: float = 0.0

    # Initialisation du tas avec les arêtes issues du sommet de départ
    heap: list[_HeapEntry] = [
        _HeapEntry(w, start, v) for v, w in graph[start]
    ]
    heapq.heapify(heap)

    while heap:
        entry = heapq.heappop(heap)
        if entry.v in visited:          # arête obsolète : sommet déjà dans l'arbre
            continue

        # Accepter l'arête de poids minimal
        visited.add(entry.v)
        mst_edges.append((entry.u, entry.v, entry.weight))
        total_weight += entry.weight

        # Pousser les nouvelles arêtes de frontière
        for neighbor, w in graph[entry.v]:
            if neighbor not in visited:
                heapq.heappush(heap, _HeapEntry(w, entry.v, neighbor))

    # Vérification de la connexité
    all_vertices = set(graph.keys())
    if visited != all_vertices:
        unreachable = all_vertices - visited
        raise ValueError(
            f"Le graphe n'est pas connexe. "
            f"Sommets non atteints : {sorted(unreachable)}"
        )

    return mst_edges, total_weight


# ── Affichage ─────────────────────────────────────────────────────────────────

def print_mst(mst_edges: list[tuple[int, int, float]], total_weight: float) -> None:
    """Affiche l'ACM de manière lisible."""
    print("╔══════════════════════════════════════╗")
    print("║   Arbre Couvrant Minimal (Prim)      ║")
    print("╠══════════════╦═══════════╦═══════════╣")
    print("║    De        ║    Vers   ║   Poids   ║")
    print("╠══════════════╬═══════════╬═══════════╣")
    for u, v, w in mst_edges:
        print(f"║  {u:>10}  ║  {v:>7}  ║  {w:>7.2f}  ║")
    print("╠══════════════╩═══════════╩═══════════╣")
    print(f"║  Poids total : {total_weight:>22.2f}  ║")
    print("╚══════════════════════════════════════╝")


# ── Tests unitaires ───────────────────────────────────────────────────────────

def _run_tests() -> None:
    """Suite de tests vérifiant la correction de l'implémentation."""
    import math

    # ── Test 1 : exemple classique ──────────────────────────────────────────
    # Graphe de 5 sommets, ACM connu = poids 11
    #
    #   0 ──2── 1
    #   |  \    |
    #   6    8  3
    #   |      \|
    #   3 ──5── 4
    #
    edges_1 = [(0,1,2),(0,2,6),(0,3,0),(1,4,3),(2,4,8),(3,4,5)]
    g1 = build_graph(edges_1)
    mst1, w1 = prim(g1, start=0)
    assert len(mst1) == 4,            f"Test 1 – nombre d'arêtes : attendu 4, obtenu {len(mst1)}"
    assert math.isclose(w1, 11.0),    f"Test 1 – poids total : attendu 11.0, obtenu {w1}"
    print("✅ Test 1 passé — graphe classique 5 sommets")

    # ── Test 2 : triangle (un seul ACM possible) ────────────────────────────
    edges_2 = [(0,1,1),(1,2,2),(0,2,5)]
    g2 = build_graph(edges_2)
    mst2, w2 = prim(g2, start=0)
    assert len(mst2) == 2,            f"Test 2 – nombre d'arêtes : attendu 2, obtenu {len(mst2)}"
    assert math.isclose(w2, 3.0),     f"Test 2 – poids total : attendu 3.0, obtenu {w2}"
    print("✅ Test 2 passé — triangle")

    # ── Test 3 : arbre linéaire (chaîne) ────────────────────────────────────
    edges_3 = [(0,1,1),(1,2,1),(2,3,1),(3,4,1)]
    g3 = build_graph(edges_3)
    mst3, w3 = prim(g3, start=0)
    assert len(mst3) == 4,            f"Test 3 – nombre d'arêtes : attendu 4, obtenu {len(mst3)}"
    assert math.isclose(w3, 4.0),     f"Test 3 – poids total : attendu 4.0, obtenu {w3}"
    print("✅ Test 3 passé — chaîne linéaire")

    # ── Test 4 : graphe non connexe → exception ──────────────────────────────
    edges_4 = [(0,1,1),(2,3,1)]       # deux composantes disjointes
    g4 = build_graph(edges_4)
    try:
        prim(g4, start=0)
        assert False, "Test 4 – une ValueError aurait dû être levée"
    except ValueError:
        pass
    print("✅ Test 4 passé — graphe non connexe détecté")

    # ── Test 5 : graphe à deux sommets ──────────────────────────────────────
    edges_5 = [(0,1,42.5)]
    g5 = build_graph(edges_5)
    mst5, w5 = prim(g5, start=0)
    assert len(mst5) == 1 and math.isclose(w5, 42.5), "Test 5 – graphe à deux sommets"
    print("✅ Test 5 passé — graphe minimal (2 sommets)")

    # ── Test 6 : poids flottants ────────────────────────────────────────────
    edges_6 = [(0,1,1.5),(0,2,2.3),(1,2,0.7),(1,3,3.1),(2,3,1.1)]
    g6 = build_graph(edges_6)
    mst6, w6 = prim(g6, start=0)
    assert len(mst6) == 3,            f"Test 6 – nombre d'arêtes : attendu 3, obtenu {len(mst6)}"
    assert math.isclose(w6, 3.3, rel_tol=1e-9), f"Test 6 – poids : attendu 3.3, obtenu {w6}"
    print("✅ Test 6 passé — poids flottants")

    print("\n🎉 Tous les tests sont passés !")


# ── Démonstration ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 42)
    print("  Exemple d'utilisation — Prim MST")
    print("=" * 42)

    #  Graphe de démonstration (7 sommets)
    #
    #      4       8
    #  0 ──── 1 ──── 2
    #  |    / |    / |
    # 11  8   |  2   |
    #  |/     4    \ 9
    #  7      |    \ |
    #  |    \ 7     \|
    #  8      |      3
    #  |       \     |
    #  6 ──7── 5 ─10─ 4
    #      2           14

    edges = [
        (0, 1, 4),  (0, 7, 8),
        (1, 2, 8),  (1, 7, 11),
        (2, 3, 7),  (2, 5, 4),  (2, 8, 2),
        (3, 4, 9),  (3, 5, 14),
        (4, 5, 10),
        (5, 6, 2),
        (6, 7, 1),  (6, 8, 6),
        (7, 8, 7),
    ]

    graph = build_graph(edges)
    mst_edges, total = prim(graph, start=0)
    print_mst(mst_edges, total)

    print("\n--- Lancement des tests unitaires ---\n")
    _run_tests()


if __name__ == "__main__":
    main()
