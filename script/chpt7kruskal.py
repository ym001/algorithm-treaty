# -*- coding: utf-8 -*-
"""
Algorithme de Kruskal — Arbre Couvrant Minimum (ACM)
=====================================================

Trouve l'arbre couvrant de poids minimum (Minimum Spanning Tree)
d'un graphe non-orienté pondéré connexe, ou une forêt couvrante
minimum si le graphe est déconnecté.

Complexité :
  - Tri des arêtes     : O(m log m)
  - Union-Find (total) : O(m · α(n))  ← quasi-linéaire, α = inverse d'Ackermann
  - Globale            : O(m log m)

Structure Union-Find :
  - Union par rang     : garantit des arbres de hauteur O(log n)
  - Compression de chemin : amortit find() à O(α(n)) ≈ O(1) en pratique
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# Union-Find (Disjoint Set Union) avec rang + compression de chemin
# ══════════════════════════════════════════════════════════════════════

class UnionFind:
    """
    Structure Union-Find (DSU) avec :
      - Union par rang  : O(log n) garanti sur find sans compression
      - Compression de chemin : aplatie l'arbre lors de chaque find
      → Complexité amortie par opération : O(α(n)) ≈ O(1) pratique

    Usage :
      uf = UnionFind(n)
      uf.find(i)          → représentant de la composante de i
      uf.union(u, v)      → fusionne les composantes, True si elles étaient distinctes
      uf.connected(u, v)  → True si u et v sont dans la même composante
    """
    __slots__ = ("parent", "rank", "n_components")

    def __init__(self, n: int) -> None:
        self.parent: list[int] = list(range(n))
        self.rank:   list[int] = [0] * n
        self.n_components: int = n

    def find(self, x: int) -> int:
        """Retourne la racine de la composante de x (avec compression de chemin)."""
        # Compression de chemin itérative (évite la récursion Python)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Aplati : pointe tous les nœuds visités directement vers la racine
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, u: int, v: int) -> bool:
        """
        Fusionne les composantes de u et v.
        Retourne True si elles étaient distinctes (arête utile pour l'ACM).
        Union par rang : attache le petit arbre sous le grand.
        """
        ru, rv = self.find(u), self.find(v)
        if ru == rv:
            return False          # même composante → cycle → arête rejetée
        # Attacher le rang inférieur sous le rang supérieur
        if self.rank[ru] < self.rank[rv]:
            ru, rv = rv, ru
        self.parent[rv] = ru
        if self.rank[ru] == self.rank[rv]:
            self.rank[ru] += 1
        self.n_components -= 1
        return True

    def connected(self, u: int, v: int) -> bool:
        return self.find(u) == self.find(v)


# ══════════════════════════════════════════════════════════════════════
# Types de données
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, order=True)
class Edge:
    """
    Arête non-orientée pondérée.
    Ordonnée par poids (pour le tri de Kruskal).
    frozen=True → hashable, utilisable dans des sets.
    order=True  → comparaison directe par poids grâce à @dataclass.
    """
    weight: float
    u: int = field(compare=False)
    v: int = field(compare=False)

    def __repr__(self) -> str:
        return f"Edge({self.u}─{self.v}, w={self.weight})"

    def endpoints(self) -> tuple[int, int]:
        return self.u, self.v


@dataclass
class KruskalResult:
    """Résultat complet de l'algorithme de Kruskal."""
    mst_edges:    list[Edge]
    total_weight: float
    n:            int
    n_components: int
    is_spanning_tree: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_spanning_tree = (self.n_components == 1)

    def adjacency(self) -> dict[int, list[tuple[int, float]]]:
        """Retourne la liste d'adjacence de l'ACM."""
        adj: dict[int, list[tuple[int, float]]] = {v: [] for v in range(self.n)}
        for e in self.mst_edges:
            adj[e.u].append((e.v, e.weight))
            adj[e.v].append((e.u, e.weight))
        return adj

    def print_summary(self) -> None:
        sep = "─" * 44
        print(f"\n{sep}")
        print(f"  Résultat Kruskal  ({self.n} sommets)")
        print(sep)
        print(f"  Arêtes ACM    : {len(self.mst_edges)}")
        print(f"  Poids total   : {self.total_weight:.4f}")
        status = "Arbre couvrant ✔" if self.is_spanning_tree else f"Forêt ({self.n_components} composantes)"
        print(f"  Statut        : {status}")
        print(sep)
        for e in self.mst_edges:
            print(f"  {e.u:>4} ─ {e.v:<4}  w = {e.weight:.4f}")
        print(sep + "\n")


# ══════════════════════════════════════════════════════════════════════
# Algorithme de Kruskal
# ══════════════════════════════════════════════════════════════════════

def kruskal(
    n: int,
    edges: list[Edge | tuple[int, int, float]],
) -> KruskalResult:
    """
    Algorithme de Kruskal — Arbre Couvrant Minimum.

    Paramètres
    ──────────
    n     : nombre de sommets (indices 0 … n-1)
    edges : arêtes sous forme de Edge ou (u, v, poids)
            Les arêtes en double sont acceptées (poids minimum retenu).

    Retour
    ──────
    KruskalResult contenant les arêtes de l'ACM et le poids total.

    Algorithme
    ──────────
    1. Normaliser et dédupliquer les arêtes.
    2. Trier par poids croissant  →  O(m log m).
    3. Pour chaque arête (u, v, w) dans l'ordre :
         - Si find(u) ≠ find(v) : ajouter à l'ACM, union(u, v).
         - Sinon : rejeter (formerait un cycle).
    4. Arrêter dès que n-1 arêtes acceptées (graphe connexe).
    """
    # ── 1. Normalisation ──────────────────────────────────────────────
    normalized: dict[tuple[int, int], float] = {}
    for e in edges:
        if isinstance(e, Edge):
            u, v, w = e.u, e.v, e.weight
        else:
            u, v, w = int(e[0]), int(e[1]), float(e[2])
        # Clé canonique : (min, max) pour ignorer la direction
        key = (min(u, v), max(u, v))
        if key not in normalized or w < normalized[key]:
            normalized[key] = w

    sorted_edges: list[Edge] = sorted(
        Edge(w, u, v) for (u, v), w in normalized.items()
    )

    # ── 2. Kruskal avec Union-Find ────────────────────────────────────
    uf = UnionFind(n)
    mst: list[Edge] = []
    total_w = 0.0
    target = n - 1          # nombre d'arêtes d'un arbre couvrant

    for edge in sorted_edges:
        if uf.union(edge.u, edge.v):
            mst.append(edge)
            total_w += edge.weight
            if len(mst) == target:
                break       # arbre complet → arrêt anticipé

    return KruskalResult(
        mst_edges    = mst,
        total_weight = total_w,
        n            = n,
        n_components = uf.n_components,
    )


# ══════════════════════════════════════════════════════════════════════
# Helpers de construction de graphes
# ══════════════════════════════════════════════════════════════════════

def make_edge(u: int, v: int, w: float) -> Edge:
    return Edge(w, u, v)

def complete_graph(n: int, weights: list[list[float]]) -> list[Edge]:
    """Construit la liste d'arêtes d'un graphe complet depuis une matrice de poids."""
    return [Edge(weights[i][j], i, j) for i in range(n) for j in range(i+1, n)]


# ══════════════════════════════════════════════════════════════════════
# Suite de tests de vérification
# ══════════════════════════════════════════════════════════════════════

def _run_tests() -> None:
    passed = failed = 0

    def check(name: str, got, expected) -> None:
        nonlocal passed, failed
        if isinstance(expected, float):
            ok = abs(got - expected) < 1e-9
        else:
            ok = (got == expected)
        if ok:
            print(f"  ✔  {name}")
            passed += 1
        else:
            print(f"  ✘  {name}")
            print(f"       attendu : {expected}")
            print(f"       obtenu  : {got}")
            failed += 1

    print("\n══ Tests Kruskal ══\n")

    # ── Test 1 : exemple classique CLRS (Introduction to Algorithms) ──
    # Graphe à 9 sommets, résultat connu : poids ACM = 37
    edges_clrs = [
        (0,1,4),(0,7,8),(1,2,8),(1,7,11),(2,3,7),(2,5,4),(2,8,2),
        (3,4,9),(3,5,14),(4,5,10),(5,6,2),(6,7,1),(6,8,6),(7,8,7),
    ]
    r = kruskal(9, edges_clrs)
    check("CLRS : poids total ACM = 37",     r.total_weight, 37.0)
    check("CLRS : nombre d'arêtes = n-1",    len(r.mst_edges), 8)
    check("CLRS : arbre couvrant",           r.is_spanning_tree, True)
    check("CLRS : 1 composante",             r.n_components, 1)

    # ── Test 2 : triangle simple ──────────────────────────────────────
    r2 = kruskal(3, [(0,1,1.0),(1,2,2.0),(0,2,3.0)])
    check("Triangle : poids = 3.0",         r2.total_weight, 3.0)
    check("Triangle : 2 arêtes",            len(r2.mst_edges), 2)

    # ── Test 3 : graphe déconnecté (forêt) ───────────────────────────
    r3 = kruskal(4, [(0,1,5.0),(2,3,3.0)])
    check("Déconnecté : forêt (non spanning)", r3.is_spanning_tree, False)
    check("Déconnecté : 2 composantes",     r3.n_components, 2)
    check("Déconnecté : poids = 8.0",       r3.total_weight, 8.0)

    # ── Test 4 : graphe à 1 sommet ───────────────────────────────────
    r4 = kruskal(1, [])
    check("1 sommet : poids = 0",           r4.total_weight, 0.0)
    check("1 sommet : arbre trivial",       r4.is_spanning_tree, True)

    # ── Test 5 : arêtes en double → poids minimum retenu ─────────────
    r5 = kruskal(2, [(0,1,10.0),(0,1,2.0),(1,0,5.0)])
    check("Doublons : poids minimum = 2.0", r5.total_weight, 2.0)

    # ── Test 6 : poids négatifs (autorisés par Kruskal) ──────────────
    r6 = kruskal(3, [(0,1,-3.0),(1,2,-1.0),(0,2,-5.0)])
    check("Négatifs : poids = -8.0",        r6.total_weight, -8.0)
    check("Négatifs : 2 arêtes",            len(r6.mst_edges), 2)

    # ── Test 7 : chaîne linéaire ─────────────────────────────────────
    n = 5
    chain = [(i, i+1, float(i+1)) for i in range(n-1)]
    r7 = kruskal(n, chain)
    check("Chaîne : n-1 arêtes",            len(r7.mst_edges), n-1)
    check("Chaîne : poids = 1+2+3+4 = 10", r7.total_weight, 10.0)

    # ── Test 8 : graphe complet K4 ───────────────────────────────────
    # K4 avec poids = distance euclidienne (pts sur carré unité)
    import math
    pts = [(0,0),(1,0),(1,1),(0,1)]
    w4 = [[math.dist(pts[i],pts[j]) for j in range(4)] for i in range(4)]
    r8 = kruskal(4, complete_graph(4, w4))
    check("K4 : 3 arêtes",                  len(r8.mst_edges), 3)
    check("K4 : poids = 3.0 (3 côtés)",    abs(r8.total_weight - 3.0) < 1e-9, True)

    # ── Test 9 : Union-Find seul ─────────────────────────────────────
    uf = UnionFind(5)
    check("UF : find initial",              uf.find(3), 3)
    uf.union(0,1); uf.union(1,2)
    check("UF : 0 et 2 connectés",         uf.connected(0,2), True)
    check("UF : 0 et 3 non connectés",     uf.connected(0,3), False)
    check("UF : n_components = 3",         uf.n_components, 3)
    check("UF : union redondante = False",  uf.union(0,2), False)

    print(f"\n  Résultat : {passed} réussi(s), {failed} échoué(s)\n")


# ══════════════════════════════════════════════════════════════════════
# Démonstration
# ══════════════════════════════════════════════════════════════════════

def demo() -> None:
    print("══ Démonstration Kruskal — graphe CLRS ══")
    edges = [
        (0,1,4),(0,7,8),(1,2,8),(1,7,11),(2,3,7),(2,5,4),(2,8,2),
        (3,4,9),(3,5,14),(4,5,10),(5,6,2),(6,7,1),(6,8,6),(7,8,7),
    ]
    result = kruskal(9, edges)
    result.print_summary()

    print("  Liste d'adjacence de l'ACM :")
    for v, neighbors in result.adjacency().items():
        if neighbors:
            print(f"    {v} → {[(u, round(w,1)) for u, w in neighbors]}")


if __name__ == "__main__":
    _run_tests()
    demo()
