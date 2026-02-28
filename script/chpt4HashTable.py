"""
============================================================
  HashTable — Table de hachage haute performance en Python
============================================================

Stratégie : Adressage ouvert + Robin Hood Hashing
  • Déplacement arrière à la suppression (pas de tombstones)
  • Facteur de charge max = 0.70 → resize × 2
  • Facteur de charge min = 0.15 → shrink ÷ 2  (taille ≥ MIN_CAPACITY)
  • Taille toujours une puissance de 2  → masquage de bits (i & mask)
  • Hash secondaire (Fibonacci mixing) pour réduire les collisions

Complexités :
  ┌─────────────┬───────────┬───────────┐
  │ Opération   │ Moy.      │ Pire      │
  ├─────────────┼───────────┼───────────┤
  │ insert      │  O(1)     │  O(n)     │
  │ search      │  O(1)     │  O(n)     │
  │ delete      │  O(1)     │  O(n)     │
  │ resize      │  O(n)     │  O(n)     │
  └─────────────┴───────────┴───────────┘
"""

from __future__ import annotations

import math
from typing import Any, Generic, Hashable, Iterator, Optional, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")

# ─────────────────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────────────────
_LOAD_MAX: float = 0.70
_LOAD_MIN: float = 0.15
_MIN_CAPACITY: int = 8          # puissance de 2, jamais en dessous
_EMPTY = object()               # sentinelle « case vide »

# Fibonacci mixing 64 bits (réduction des clusters)
_FIBO64 = 0x9E3779B97F4A7C15


# ─────────────────────────────────────────────────────────
#  Slot interne
# ─────────────────────────────────────────────────────────
class _Slot:
    """Case du tableau interne."""

    __slots__ = ("key", "value", "hash", "psl")

    def __init__(self, key: Any, value: Any, h: int, psl: int = 0) -> None:
        self.key = key
        self.value = value
        self.hash = h      # hash brut mis en cache
        self.psl = psl     # Probe Sequence Length (distance à la position idéale)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Slot(key={self.key!r}, psl={self.psl})"


# ─────────────────────────────────────────────────────────
#  HashTable
# ─────────────────────────────────────────────────────────
class HashTable(Generic[K, V]):
    """
    Table de hachage générique basée sur Robin Hood open-addressing.

    Paramètres
    ----------
    capacity : int
        Capacité initiale (arrondie à la prochaine puissance de 2 ≥ MIN_CAPACITY).

    Exemple
    -------
    >>> ht = HashTable()
    >>> ht["clé"] = 42
    >>> ht["clé"]
    42
    >>> del ht["clé"]
    >>> len(ht)
    0
    """

    # ── construction ──────────────────────────────────────

    def __init__(self, capacity: int = _MIN_CAPACITY) -> None:
        self._capacity: int = self._next_power2(max(capacity, _MIN_CAPACITY))
        self._mask: int = self._capacity - 1
        self._table: list[Optional[_Slot]] = [None] * self._capacity
        self._size: int = 0

    # ── API publique ──────────────────────────────────────

    def __setitem__(self, key: K, value: V) -> None:
        """Insertion ou mise à jour."""
        self._ensure_capacity_up()
        new = self._insert_raw(self._table, self._mask, key, value, self._hash(key))
        if new:
            self._size += 1

    def __getitem__(self, key: K) -> V:
        """Accès en lecture ; lève KeyError si absent."""
        slot = self._find_slot(key)
        if slot is None:
            raise KeyError(key)
        return slot.value

    def __delitem__(self, key: K) -> None:
        """Suppression avec backward-shift (pas de tombstone)."""
        idx = self._find_index(key)
        if idx is None:
            raise KeyError(key)
        self._delete_at(idx)
        self._size -= 1
        self._ensure_capacity_down()

    def __contains__(self, key: object) -> bool:
        return self._find_slot(key) is not None

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[K]:
        """Itère sur les clés."""
        for slot in self._table:
            if slot is not None:
                yield slot.key

    def __repr__(self) -> str:
        pairs = ", ".join(f"{s.key!r}: {s.value!r}" for s in self._table if s)
        return f"HashTable({{{pairs}}})"

    # ── méthodes dict-like ────────────────────────────────

    def get(self, key: K, default: Any = None) -> Any:
        slot = self._find_slot(key)
        return slot.value if slot is not None else default

    def keys(self) -> Iterator[K]:
        return iter(self)

    def values(self) -> Iterator[V]:
        for slot in self._table:
            if slot is not None:
                yield slot.value

    def items(self) -> Iterator[Tuple[K, V]]:
        for slot in self._table:
            if slot is not None:
                yield slot.key, slot.value

    def pop(self, key: K, *args: Any) -> Any:
        idx = self._find_index(key)
        if idx is None:
            if args:
                return args[0]
            raise KeyError(key)
        value = self._table[idx].value
        self._delete_at(idx)
        self._size -= 1
        self._ensure_capacity_down()
        return value

    def clear(self) -> None:
        self._capacity = _MIN_CAPACITY
        self._mask = self._capacity - 1
        self._table = [None] * self._capacity
        self._size = 0

    # ── métriques ─────────────────────────────────────────

    @property
    def load_factor(self) -> float:
        return self._size / self._capacity

    @property
    def capacity(self) -> int:
        return self._capacity

    def stats(self) -> dict:
        psls = [s.psl for s in self._table if s is not None]
        return {
            "size": self._size,
            "capacity": self._capacity,
            "load_factor": round(self.load_factor, 4),
            "max_psl": max(psls, default=0),
            "avg_psl": round(sum(psls) / len(psls), 4) if psls else 0.0,
        }

    # ── internals : hachage ───────────────────────────────

    @staticmethod
    def _hash(key: Any) -> int:
        """Hash avec Fibonacci mixing pour mieux distribuer les bits faibles."""
        h = hash(key)
        # mélange multiplicatif : très rapide, excellent avalanche sur les LSB
        h = (h * _FIBO64) & 0xFFFFFFFFFFFFFFFF
        return h

    # ── internals : insertion Robin Hood ──────────────────

    @staticmethod
    def _insert_raw(
        table: list, mask: int, key: Any, value: Any, h: int
    ) -> bool:
        """
        Insère (key, value) dans `table` via Robin Hood.
        Retourne True si c'est un nouvel élément, False si mise à jour.
        """
        idx = h & mask
        psl = 0
        incoming = _Slot(key, value, h, psl)

        while True:
            current = table[idx]

            if current is None:                       # case libre → poser
                table[idx] = incoming
                return True

            if current.hash == incoming.hash and current.key == incoming.key:
                current.value = incoming.value        # mise à jour
                return False

            # Robin Hood : on vole à celui qui est « riche » (psl faible)
            if current.psl < incoming.psl:
                table[idx], incoming = incoming, current

            incoming.psl += 1
            idx = (idx + 1) & mask

    # ── internals : recherche ─────────────────────────────

    def _find_slot(self, key: Any) -> Optional[_Slot]:
        idx = self._find_index(key)
        return self._table[idx] if idx is not None else None

    def _find_index(self, key: Any) -> Optional[int]:
        h = self._hash(key)
        idx = h & self._mask
        psl = 0

        while True:
            slot = self._table[idx]
            if slot is None:
                return None                           # clé absente (trou → stop)
            if slot.hash == h and slot.key == key:
                return idx                            # trouvé
            if slot.psl < psl:
                return None                           # impossible d'aller plus loin (RH inv.)
            psl += 1
            idx = (idx + 1) & self._mask

    # ── internals : suppression backward-shift ────────────

    def _delete_at(self, idx: int) -> None:
        """
        Backward shift : déplace les voisins vers la gauche
        jusqu'à rencontrer une case vide ou un slot à psl=0.
        Évite les tombstones tout en maintenant l'invariant Robin Hood.
        """
        mask = self._mask
        while True:
            nxt = (idx + 1) & mask
            neighbor = self._table[nxt]
            if neighbor is None or neighbor.psl == 0:
                self._table[idx] = None
                break
            self._table[idx] = neighbor
            neighbor.psl -= 1
            idx = nxt

    # ── internals : resize ────────────────────────────────

    def _resize(self, new_cap: int) -> None:
        new_cap = max(new_cap, _MIN_CAPACITY)
        new_mask = new_cap - 1
        new_table: list[Optional[_Slot]] = [None] * new_cap

        for slot in self._table:
            if slot is not None:
                slot.psl = 0                          # recalculé lors du re-insert
                self._insert_raw(new_table, new_mask, slot.key, slot.value, slot.hash)

        self._table = new_table
        self._capacity = new_cap
        self._mask = new_mask

    def _ensure_capacity_up(self) -> None:
        if (self._size + 1) / self._capacity > _LOAD_MAX:
            self._resize(self._capacity * 2)

    def _ensure_capacity_down(self) -> None:
        if (
            self._capacity > _MIN_CAPACITY
            and self._size / self._capacity < _LOAD_MIN
        ):
            self._resize(self._capacity // 2)

    # ── utilitaires ───────────────────────────────────────

    @staticmethod
    def _next_power2(n: int) -> int:
        """Plus petite puissance de 2 ≥ n."""
        return 1 << (n - 1).bit_length() if n > 1 else 1


# ─────────────────────────────────────────────────────────
#  Suite de tests
# ─────────────────────────────────────────────────────────
def _run_tests() -> None:
    import random, string, time

    OK = "\033[92m✓\033[0m"
    FAIL = "\033[91m✗\033[0m"
    errors = []

    def check(name: str, condition: bool) -> None:
        sym = OK if condition else FAIL
        print(f"  {sym}  {name}")
        if not condition:
            errors.append(name)

    print("\n══════════════════════════════════════════")
    print("  Tests HashTable — Robin Hood Hashing")
    print("══════════════════════════════════════════\n")

    # ── 1. opérations de base ──────────────────────────────
    print("▸ Opérations de base")
    ht: HashTable[str, int] = HashTable()
    ht["a"] = 1
    ht["b"] = 2
    ht["c"] = 3
    check("insert + getitem", ht["a"] == 1 and ht["b"] == 2 and ht["c"] == 3)
    check("len après 3 inserts", len(ht) == 3)
    ht["a"] = 99
    check("mise à jour (upsert)", ht["a"] == 99)
    check("contains True", "b" in ht)
    check("contains False", "z" not in ht)
    del ht["b"]
    check("delete + contains", "b" not in ht and len(ht) == 2)
    check("get existant", ht.get("a") == 99)
    check("get absent → default", ht.get("z", -1) == -1)

    # ── 2. KeyError ────────────────────────────────────────
    print("\n▸ Gestion des erreurs")
    try:
        _ = ht["z"]
        check("KeyError __getitem__", False)
    except KeyError:
        check("KeyError __getitem__", True)
    try:
        del ht["z"]
        check("KeyError __delitem__", False)
    except KeyError:
        check("KeyError __delitem__", True)

    # ── 3. itération ──────────────────────────────────────
    print("\n▸ Itération")
    ht2: HashTable[str, int] = HashTable()
    ref = {chr(65 + i): i for i in range(20)}
    for k, v in ref.items():
        ht2[k] = v
    check("keys()", set(ht2.keys()) == set(ref.keys()))
    check("values()", set(ht2.values()) == set(ref.values()))
    check("items()", dict(ht2.items()) == ref)

    # ── 4. pop ────────────────────────────────────────────
    print("\n▸ pop")
    ht3: HashTable[str, int] = HashTable()
    ht3["x"] = 10
    check("pop existant", ht3.pop("x") == 10)
    check("pop absent + default", ht3.pop("x", -1) == -1)
    try:
        ht3.pop("x")
        check("pop absent sans default → KeyError", False)
    except KeyError:
        check("pop absent sans default → KeyError", True)

    # ── 5. clear ──────────────────────────────────────────
    print("\n▸ clear")
    ht4: HashTable[int, int] = HashTable()
    for i in range(50):
        ht4[i] = i
    ht4.clear()
    check("clear → size=0", len(ht4) == 0)
    check("clear → capacity reset", ht4.capacity == _MIN_CAPACITY)

    # ── 6. resize automatique ─────────────────────────────
    print("\n▸ Resize dynamique")
    ht5: HashTable[int, int] = HashTable(8)
    for i in range(200):
        ht5[i] = i * 2
    check("intégrité après N inserts", all(ht5[i] == i * 2 for i in range(200)))
    check("load_factor ≤ 0.70", ht5.load_factor <= _LOAD_MAX)
    for i in range(180):
        del ht5[i]
    check("intégrité après N deletes", all(ht5[i] == i * 2 for i in range(180, 200)))
    check("load_factor ≥ min après shrink", ht5.load_factor >= _LOAD_MIN)

    # ── 7. types de clés variés ───────────────────────────
    print("\n▸ Types de clés")
    htk: HashTable = HashTable()
    htk[42] = "int"
    htk[3.14] = "float"
    htk[(1, 2)] = "tuple"
    htk[True] = "bool"            # True hash == 1 == hash(1) en Python
    check("int / float / tuple / bool", htk[3.14] == "float" and htk[(1, 2)] == "tuple")

    # ── 8. grande charge aléatoire ─────────────────────────
    print("\n▸ Charge aléatoire (10 000 éléments)")
    N = 10_000
    random.seed(0)
    keys = ["".join(random.choices(string.ascii_letters, k=8)) for _ in range(N)]
    vals = random.sample(range(10 * N), N)
    ref2 = dict(zip(keys, vals))

    t0 = time.perf_counter()
    htbig: HashTable[str, int] = HashTable()
    for k, v in zip(keys, vals):
        htbig[k] = v
    t_insert = time.perf_counter() - t0

    t0 = time.perf_counter()
    ok_search = all(htbig[k] == ref2[k] for k in ref2)
    t_search = time.perf_counter() - t0

    check(f"insert {N} clés ({t_insert*1000:.1f} ms)", True)
    check(f"search {N} clés ({t_search*1000:.1f} ms)", ok_search)

    s = htbig.stats()
    print(f"     stats → {s}")
    check("max_psl raisonnable (< 30)", s["max_psl"] < 30)

    # ── 9. comparaison dict Python ────────────────────────
    print("\n▸ Cohérence vs dict Python")
    ht_cmp: HashTable[int, int] = HashTable()
    d_cmp: dict[int, int] = {}
    ops = []
    rng = random.Random(42)
    for _ in range(5000):
        k = rng.randint(0, 200)
        if rng.random() < 0.6:
            v = rng.randint(0, 1000)
            ht_cmp[k] = v
            d_cmp[k] = v
            ops.append(("set", k, v))
        elif k in d_cmp:
            del ht_cmp[k]
            del d_cmp[k]
            ops.append(("del", k))

    match = all(ht_cmp.get(k) == d_cmp.get(k) for k in range(201))
    check("cohérence complète avec dict", match)
    check("len identique", len(ht_cmp) == len(d_cmp))

    # ── résultat ──────────────────────────────────────────
    print("\n══════════════════════════════════════════")
    if not errors:
        print(f"  \033[92mTous les tests passés ✓\033[0m")
    else:
        print(f"  \033[91m{len(errors)} test(s) échoué(s) : {errors}\033[0m")
    print("══════════════════════════════════════════\n")


# ─────────────────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    _run_tests()

    # Mini-démo
    print("── Démo rapide ─────────────────────────────")
    ht: HashTable[str, Any] = HashTable()
    for mot in ["alpha", "beta", "gamma", "delta", "epsilon"]:
        ht[mot] = len(mot)
    print(ht)
    print(f"load_factor = {ht.load_factor:.2f}, capacity = {ht.capacity}")
    print(f"stats       = {ht.stats()}")
