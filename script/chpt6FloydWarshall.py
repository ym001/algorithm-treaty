"""
Algorithme de Floyd-Warshall
============================
Trouve les plus courts chemins entre toutes les paires de sommets
d'un graphe pondéré (avec ou sans cycles négatifs).

Complexité :
  - Temporelle : O(V³)
  - Spatiale   : O(V²)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# Alias lisible pour l'infini
INF = math.inf


# ─────────────────────────────────────────────
# Types de données
# ─────────────────────────────────────────────

@dataclass
class FloydResult:
    """Résultat de l'algorithme de Floyd-Warshall."""
    dist: list[list[float]]          # dist[i][j] = distance minimale i → j
    next_: list[list[Optional[int]]] # next_[i][j] = prochain sommet sur le chemin i → j
    n: int                           # nombre de sommets
    has_negative_cycle: bool = field(init=False)

    def __post_init__(self) -> None:
        # Un cycle négatif existe si dist[v][v] < 0 pour un sommet v
        self.has_negative_cycle = any(
            self.dist[v][v] < 0 for v in range(self.n)
        )

    def path(self, src: int, dst: int) -> Optional[list[int]]:
        """
        Reconstruit le chemin le plus court entre src et dst.
        Retourne None si aucun chemin n'existe.
        Lève ValueError si un cycle négatif est détecté sur le chemin.
        """
        if self.dist[src][dst] == INF:
            return None

        route: list[int] = [src]
        visited: set[int] = {src}
        cur = src

        while cur != dst:
            nxt = self.next_[cur][dst]
            if nxt is None:
                return None  # chemin brisé (ne devrait pas arriver)
            if nxt in visited:
                raise ValueError(
                    f"Cycle négatif détecté sur le chemin {src} → {dst}"
                )
            visited.add(nxt)
            route.append(nxt)
            cur = nxt

        return route

    def print_matrix(self, label: str = "Distances") -> None:
        """Affiche la matrice de distances de façon lisible."""
        print(f"\n{'─' * 40}")
        print(f"  {label} ({self.n}×{self.n})")
        print(f"{'─' * 40}")
        header = "     " + "  ".join(f"{j:>5}" for j in range(self.n))
        print(header)
        for i, row in enumerate(self.dist):
            cells = "  ".join(
                f"{'∞':>5}" if d == INF else f"{d:>5.1f}" for d in row
            )
            print(f"  {i:>2} | {cells}")
        print(f"{'─' * 40}")
        if self.has_negative_cycle:
            print("  ⚠  Cycle négatif détecté !")
        print()


# ─────────────────────────────────────────────
# Algorithme principal
# ─────────────────────────────────────────────

def floyd_warshall(
    n: int,
    edges: list[tuple[int, int, float]],
) -> FloydResult:
    """
    Execute l'algorithme de Floyd-Warshall.

    Paramètres
    ----------
    n     : nombre de sommets (indices 0 … n-1)
    edges : liste de (u, v, poids) – arêtes dirigées

    Retour
    ------
    FloydResult avec les matrices dist et next_
    """
    # ── Initialisation ──────────────────────────────────────────────
    dist:  list[list[float]]         = [[INF] * n for _ in range(n)]
    next_: list[list[Optional[int]]] = [[None] * n for _ in range(n)]

    for v in range(n):
        dist[v][v] = 0.0

    for u, v, w in edges:
        if w < dist[u][v]:          # garde l'arête la moins chère
            dist[u][v] = w
            next_[u][v] = v

    # ── Relaxation triple boucle ─────────────────────────────────────
    for k in range(n):
        dist_k = dist[k]            # cache de la ligne k → évite des lookups répétés
        for i in range(n):
            if dist[i][k] == INF:   # saut rapide : aucun chemin i → k
                continue
            dist_i  = dist[i]
            next_i  = next_[i]
            d_ik    = dist[i][k]
            for j in range(n):
                new_d = d_ik + dist_k[j]
                if new_d < dist_i[j]:
                    dist_i[j]  = new_d
                    next_i[j]  = next_[i][k]

    return FloydResult(dist=dist, next_=next_, n=n)


# ─────────────────────────────────────────────
# Helpers de construction de graphe
# ─────────────────────────────────────────────

def add_undirected(
    edges: list[tuple[int, int, float]],
    u: int, v: int, w: float,
) -> None:
    """Ajoute une arête non-dirigée (les deux sens)."""
    edges.append((u, v, w))
    edges.append((v, u, w))


# ─────────────────────────────────────────────
# Tests de vérification
# ─────────────────────────────────────────────

def _run_tests() -> None:
    import traceback

    passed = failed = 0

    def check(name: str, got, expected) -> None:
        nonlocal passed, failed
        if got == expected:
            print(f"  ✔  {name}")
            passed += 1
        else:
            print(f"  ✘  {name}")
            print(f"       attendu : {expected}")
            print(f"       obtenu  : {got}")
            failed += 1

    print("\n══ Tests Floyd-Warshall ══\n")

    # ── Test 1 : graphe classique (Cormen CLRS) ──────────────────────
    edges = [
        (0, 1, 3), (0, 2, 8), (0, 4, -4),
        (1, 3, 1), (1, 4, 7),
        (2, 1, 4),
        (3, 0, 2), (3, 2, -5),
        (4, 3, 6),
    ]
    r = floyd_warshall(5, edges)
    check("CLRS dist[0][2]", r.dist[0][2], -3.0)
    check("CLRS dist[0][3]", r.dist[0][3],  2.0)
    check("CLRS dist[3][4]", r.dist[3][4], -2.0)
    check("CLRS pas de cycle négatif", r.has_negative_cycle, False)

    # ── Test 2 : reconstruction de chemin ────────────────────────────
    path = r.path(3, 4)
    check("Chemin 3→4 premier sommet", path[0] if path else None, 3)
    check("Chemin 3→4 dernier sommet", path[-1] if path else None, 4)

    # ── Test 3 : graphe déconnecté ───────────────────────────────────
    r2 = floyd_warshall(3, [(0, 1, 5)])
    check("Déconnecté dist[0][2] = INF", r2.dist[0][2], INF)
    check("Déconnecté path(0,2) = None", r2.path(0, 2), None)

    # ── Test 4 : graphe à un seul sommet ─────────────────────────────
    r3 = floyd_warshall(1, [])
    check("1 sommet dist[0][0] = 0", r3.dist[0][0], 0.0)

    # ── Test 5 : cycle négatif ───────────────────────────────────────
    r4 = floyd_warshall(3, [(0, 1, 1), (1, 2, -2), (2, 0, -2)])
    check("Cycle négatif détecté", r4.has_negative_cycle, True)

    # ── Test 6 : graphe non-dirigé ───────────────────────────────────
    edges6: list[tuple[int, int, float]] = []
    add_undirected(edges6, 0, 1, 4)
    add_undirected(edges6, 0, 2, 1)
    add_undirected(edges6, 2, 1, 2)
    r5 = floyd_warshall(3, edges6)
    check("Non-dirigé dist[0][1] = 3", r5.dist[0][1], 3.0)
    check("Non-dirigé dist[1][0] = 3", r5.dist[1][0], 3.0)

    print(f"\n  Résultat : {passed} réussi(s), {failed} échoué(s)\n")


# ─────────────────────────────────────────────
# Démonstration
# ─────────────────────────────────────────────

def demo() -> None:
    print("══ Démonstration Floyd-Warshall ══")
    edges = [
        (0, 1, 3), (0, 2, 8), (0, 4, -4),
        (1, 3, 1), (1, 4, 7),
        (2, 1, 4),
        (3, 0, 2), (3, 2, -5),
        (4, 3, 6),
    ]
    result = floyd_warshall(5, edges)
    result.print_matrix("Distances minimales (graphe CLRS)")

    for src, dst in [(0, 2), (3, 4), (1, 0)]:
        try:
            p = result.path(src, dst)
            dist_val = result.dist[src][dst]
            if p is None:
                print(f"  {src} → {dst} : aucun chemin")
            else:
                print(f"  {src} → {dst} : {' → '.join(map(str, p))}  (coût {dist_val})")
        except ValueError as e:
            print(f"  {src} → {dst} : {e}")


if __name__ == "__main__":
    _run_tests()
    demo()
