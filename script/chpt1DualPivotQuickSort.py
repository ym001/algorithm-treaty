# -*- coding: utf-8 -*-
"""
Dual-Pivot QuickSort -- Implementation Python
=============================================
Algorithme de Vladimir Yaroslavskiy (2009).

Optimisations :
  - Pivots choisis par pseudo-mediane de 5 elements (reseau de tri 7 comp.)
  - Bascule InsertionSort sous un seuil (16 elements)
  - Dutch National Flag quand les deux pivots sont egaux -> O(n) sur tout-identique
  - Recursion sur la plus petite partition en premier -> profondeur O(log n)
  - Support key=, reverse= (signature identique a list.sort / sorted)

Complexite :
  Moyenne  O(n log n)   ~5/3 * n * ln(n) comparaisons
  Pire cas O(n**2)      tres rare avec la selection des pivots
  Espace   O(log n)     pile de recursion
"""

from __future__ import annotations
from typing import Any, Callable
import random
import time


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_INSERTION_THRESHOLD = 16   # seuil de bascule vers InsertionSort


# ---------------------------------------------------------------------------
# Utilitaires bas niveau
# ---------------------------------------------------------------------------

def _swap(a: list, i: int, j: int) -> None:
    a[i], a[j] = a[j], a[i]


def _insertion_sort(a: list, lo: int, hi: int, key: Callable) -> None:
    """Insertion sort stable sur a[lo..hi] inclus."""
    for i in range(lo + 1, hi + 1):
        x, kx = a[i], key(a[i])
        j = i - 1
        while j >= lo and key(a[j]) > kx:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = x


def _sort5(a: list, i0: int, i1: int, i2: int, i3: int, i4: int,
           key: Callable) -> None:
    """Reseau de tri optimal (7 comparaisons) pour 5 elements."""
    if key(a[i1]) < key(a[i0]): _swap(a, i0, i1)
    if key(a[i3]) < key(a[i2]): _swap(a, i2, i3)
    if key(a[i2]) < key(a[i0]): _swap(a, i0, i2); _swap(a, i1, i3)
    if key(a[i4]) < key(a[i1]): _swap(a, i1, i4)
    if key(a[i1]) < key(a[i0]): _swap(a, i0, i1)
    if key(a[i4]) < key(a[i3]): _swap(a, i3, i4)
    if key(a[i3]) < key(a[i1]): _swap(a, i1, i3); _swap(a, i2, i4)
    if key(a[i2]) < key(a[i1]): _swap(a, i1, i2)
    if key(a[i3]) < key(a[i2]): _swap(a, i2, i3)


def _dutch_flag(a: list, lo: int, hi: int, key: Callable,
                pk: Any) -> tuple[int, int]:
    """
    Partition Dutch National Flag en 3 zones :  [ <pk | =pk | >pk ]
    Retourne (lt, gt) tels que a[lt..gt] sont tous egaux a pk.
    """
    lt, k, gt = lo, lo, hi
    while k <= gt:
        kk = key(a[k])
        if   kk < pk: _swap(a, k, lt); lt += 1; k += 1
        elif kk > pk: _swap(a, k, gt); gt -= 1
        else:         k += 1
    return lt, gt


# ---------------------------------------------
# Coeur Dual-Pivot (3 zones + cas pivot unique)
# ---------------------------------------------

def _dps(a: list, lo: int, hi: int, key: Callable) -> None:
    """Recursion interne du Dual-Pivot QuickSort."""

    if hi - lo < _INSERTION_THRESHOLD:
        _insertion_sort(a, lo, hi, key)
        return

    # -- Selection des pivots par pseudo-mediane de 5 -----------------------
    s = (hi - lo) // 4
    _sort5(a, lo, lo + s, lo + 2*s, lo + 3*s, hi, key)
    # Apres _sort5 : a[lo] <= a[lo+s] <= a[lo+2s] <= a[lo+3s] <= a[hi]
    # On prend le 2e et 4e comme pivots, puis on les deplace aux extremes.
    _swap(a, lo, lo + s)
    _swap(a, hi, lo + 3*s)

    kp1, kp2 = key(a[lo]), key(a[hi])

    # -- Cas pivots identiques : Dutch Flag, aucune recursion inutile --------
    if kp1 == kp2:
        lt, gt = _dutch_flag(a, lo, hi, key, kp1)
        if lo < lt - 1: _dps(a, lo,     lt - 1, key)
        if gt + 1 < hi: _dps(a, gt + 1, hi,     key)
        return

    # -- Partition 3 zones de Yaroslavskiy ----------------------------------
    #
    #   lo        lt              k    gt          hi
    #   [p1] [ <p1 | p1 <= . <= p2 |????| >p2 ] [p2]
    #
    lt = lo + 1   # a[lo+1 .. lt-1]  est strictement < p1
    gt = hi - 1   # a[gt+1 .. hi-1]  est strictement > p2
    k  = lo + 1   # curseur courant

    while k <= gt:
        kk = key(a[k])
        if kk < kp1:
            _swap(a, k, lt); lt += 1; k += 1
        elif kk > kp2:
            _swap(a, k, gt); gt -= 1
            # Ne pas avancer k : on vient de recevoir un element inconnu
        else:
            k += 1

    # Replacer les pivots sentinelles a leur position definitive
    lt -= 1; _swap(a, lo, lt)   # p1 correctement place
    gt += 1; _swap(a, hi, gt)   # p2 correctement place

    # -- Recursion sur les 3 sous-tableaux ----------------------------------
    # Ordre : petite partition en premier => profondeur de pile O(log n)
    segments = [(lo, lt - 1), (lt + 1, gt - 1), (gt + 1, hi)]
    segments.sort(key=lambda seg: seg[1] - seg[0])

    for l, h in segments:
        if l < h:
            _dps(a, l, h, key)


# ---------------------------------------------------------------------------
# Wrapper pour l'ordre inverse
# ---------------------------------------------------------------------------

class _Rev:
    """Renverse l'ordre de comparaison sans copier le tableau."""
    __slots__ = ("v",)

    def __init__(self, v: Any) -> None:
        self.v = v

    def __lt__(self, o: "_Rev") -> bool: return self.v > o.v
    def __gt__(self, o: "_Rev") -> bool: return self.v < o.v
    def __le__(self, o: "_Rev") -> bool: return self.v >= o.v
    def __ge__(self, o: "_Rev") -> bool: return self.v <= o.v
    def __eq__(self, o: object)  -> bool:
        return isinstance(o, _Rev) and self.v == o.v


# ---------------------------------------------------------------------------
# Interface publique
# ---------------------------------------------------------------------------

def dual_pivot_sort(
    arr: list,
    *,
    key: Callable[[Any], Any] | None = None,
    reverse: bool = False,
) -> list:
    """
    Trie ``arr`` en place et retourne la liste.

    Compatible avec la signature de ``list.sort()`` / ``sorted()``.

    Parametres
    ----------
    arr     : liste a trier (modifiee en place)
    key     : fonction de cle optionnelle
    reverse : ordre decroissant si True

    Retour
    ------
    La meme liste ``arr``, triee.

    Exemples
    --------
    >>> dual_pivot_sort([3, 1, 4, 1, 5, 9, 2, 6])
    [1, 1, 2, 3, 4, 5, 6, 9]
    >>> dual_pivot_sort(['poire', 'pomme', 'kiwi'], key=len)
    ['kiwi', 'poire', 'pomme']
    >>> dual_pivot_sort([3, 1, 4, 1, 5], reverse=True)
    [5, 4, 3, 1, 1]
    """
    if len(arr) < 2:
        return arr

    _key: Callable = key if key is not None else (lambda x: x)

    if reverse:
        _base = _key
        _key  = lambda x: _Rev(_base(x))

    _dps(arr, 0, len(arr) - 1, _key)
    return arr


# ---------------------------------------------------------------------------
# Suite de verification
# ---------------------------------------------------------------------------

def _verify() -> None:
    """Batterie de tests contre sorted() de la bibliotheque standard."""

    def chk(arr: list, **kw) -> None:
        exp = sorted(arr, **kw)
        res = dual_pivot_sort(arr[:], **kw)
        assert res == exp, (
            "ECHEC\n"
            "  resultat : {}\n"
            "  attendu  : {}".format(res[:30], exp[:30])
        )

    rng = random.Random(0)

    # Cas limites
    for edge in [[], [0], [1, 2], [2, 1], [1] * 6]:
        chk(edge)

    # Structures particulieres
    chk(list(range(200)))
    chk(list(range(200, 0, -1)))
    chk(list(range(50)) * 2)

    # Aleatoire, tailles variees
    for n in [10, 100, 1_000, 10_000]:
        chk(rng.choices(range(n), k=n))
        chk(rng.choices(range(n), k=n), reverse=True)

    # Doublons (stress test de la partition Dutch Flag)
    chk([rng.randint(0, 3) for _ in range(10_000)])
    chk([rng.randint(0, 1) for _ in range(10_000)])
    chk([0] * 5_000 + [1] * 5_000)
    chk([7] * 10_000)

    # Cle personnalisee
    chk(["banana", "apple", "fig", "cherry", "kiwi"], key=len)
    chk(["banana", "apple", "fig", "cherry", "kiwi"], key=len, reverse=True)

    # Tuples
    chk([(rng.randint(0, 9), rng.randint(0, 9)) for _ in range(500)])

    print("[OK] Tous les tests passent.")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _benchmark(n: int = 100_000) -> None:
    """Compare dual_pivot_sort avec list.sort (Timsort C) sur 6 cas."""
    rng  = random.Random(42)
    base = rng.choices(range(n), k=n)

    cases = {
        "aleatoire"          : base[:],
        "trie"               : sorted(base),
        "inverse"            : sorted(base, reverse=True),
        "100 valeurs dist."  : rng.choices(range(100), k=n),
        "doublons extremes"  : rng.choices(range(3),   k=n),
        "tout identique"     : [7] * n,
    }

    print("\n{:<24} {:>12} {:>12}  {:>8}".format(
        "Cas", "dual_pivot", "list.sort", "ratio"))
    print("-" * 62)

    for label, arr in cases.items():
        t0   = time.perf_counter()
        dual_pivot_sort(arr[:])
        t_dp = time.perf_counter() - t0

        t0   = time.perf_counter()
        sorted(arr)
        t_ts = time.perf_counter() - t0

        ratio = t_dp / t_ts
        note  = "bon" if ratio < 5 else ("moyen" if ratio < 15 else "lent")
        print("{:<24} {:>10.1f}ms {:>10.1f}ms  {:>7.2f}x  {}".format(
            label, t_dp * 1000, t_ts * 1000, ratio, note))
    print()


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(100_000)

    print("=" * 46)
    print("   Dual-Pivot QuickSort -- Python")
    print("=" * 46)
    print()

    _verify()
    _benchmark()

    # Demonstrations
    print("--- Demonstrations ---")

    nums = [5, -3, 8, 0, 3, 1, 7, 2, -1, 4, 6]
    print("entiers      :", dual_pivot_sort(nums))

    mots = ["cerise", "abricot", "fraise", "banane", "datte", "kiwi"]
    print("longueur dec.:", dual_pivot_sort(mots, key=len, reverse=True))

    pers = [("Alice", 30), ("Bob", 25), ("Carol", 30), ("Dave", 25)]
    print("(age, nom)   :", dual_pivot_sort(pers, key=lambda p: (p[1], p[0])))
