"""
Algorithme de Prim pour Graphes Denses
Implémentation optimisée O(V^2) avec matrice d'adjacence

Auteur: Implémentation pédagogique
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Set

class PrimGrapheDense:
    """
    Implémentation de l'algorithme de Prim pour graphes denses.
    Utilise une matrice d'adjacence et une complexité O(V^2).
    
    Optimal pour les graphes denses.
    """
    
    def __init__(self, num_vertices: int):
        """
        Initialise le graphe.
        
        Args:
            num_vertices: Nombre de sommets
        """
        self.V = num_vertices
        # Matrice d'adjacence (inf signifie pas de connexion)
        self.graph = [[float('inf')] * num_vertices for _ in range(num_vertices)]
        
        # Pas de boucles sur soi-même
        for i in range(num_vertices):
            self.graph[i][i] = 0
    
    def ajouter_arete(self, u: int, v: int, poids: float):
        """
        Ajoute une arête non-orientée au graphe.
        
        Args:
            u: Sommet source
            v: Sommet destination
            poids: Poids de l'arête
        """
        self.graph[u][v] = poids
        self.graph[v][u] = poids  # Graphe non-orienté
    
    def prim_mst(self, sommet_depart: int = 0, verbose: bool = True):
        """
        Calcule l'Arbre Couvrant Minimum avec l'algorithme de Prim.
        Version optimisée pour graphes denses: O(V^2).
        
        Args:
            sommet_depart: Sommet de départ (défaut: 0)
            verbose: Afficher les étapes détaillées
            
        Returns:
            Tuple (arêtes_MST, poids_total, historique)
        """
        # Initialisation
        dans_MST = [False] * self.V  # Sommets déjà dans le MST
        key = [float('inf')] * self.V  # Clé minimale pour atteindre chaque sommet
        parent = [-1] * self.V  # Parent de chaque sommet dans le MST
        
        # Démarrer du sommet_depart
        key[sommet_depart] = 0
        
        # Historique pour visualisation
        historique = []
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"ALGORITHME DE PRIM - Graphe Dense (O(V^2))")
            print(f"{'='*70}")
            print(f"Nombre de sommets: {self.V}")
            print(f"Sommet de départ: {sommet_depart}")
            print(f"{'='*70}\n")
        
        # Construire le MST avec V sommets
        for iteration in range(self.V):
            # Étape 1: Trouver le sommet avec la clé minimale parmi ceux pas encore dans MST
            min_key = float('inf')
            u = -1
            
            for v in range(self.V):
                if not dans_MST[v] and key[v] < min_key:
                    min_key = key[v]
                    u = v
            
            # Ajouter u au MST
            dans_MST[u] = True
            
            if verbose:
                print(f"\n--- Itération {iteration + 1} ---")
                print(f"Sommet sélectionné: {u} (clé = {key[u]})")
                if parent[u] != -1:
                    print(f"Arête ajoutée: {parent[u]} -- {u} (poids: {self.graph[parent[u]][u]})")
            
            # Sauvegarder l'état pour visualisation
            if parent[u] != -1:
                historique.append({
                    'iteration': iteration,
                    'sommet': u,
                    'arete': (parent[u], u),
                    'poids': self.graph[parent[u]][u],
                    'key': key.copy(),
                    'parent': parent.copy(),
                    'dans_MST': dans_MST.copy()
                })
            
            # Étape 2: Mettre à jour les clés des sommets adjacents
            for v in range(self.V):
                # Si v n'est pas dans MST et il existe une arête u-v
                # et le poids u-v est meilleur que la clé actuelle de v
                if (not dans_MST[v] and 
                    self.graph[u][v] != float('inf') and 
                    self.graph[u][v] < key[v]):
                    
                    ancienne_key = key[v]
                    key[v] = self.graph[u][v]
                    parent[v] = u
                    
                    if verbose and ancienne_key != float('inf'):
                        print(f"  Mise à jour: sommet {v}, clé {ancienne_key} -> {key[v]} (via {u})")
                    elif verbose:
                        print(f"  Découverte: sommet {v}, clé inf -> {key[v]} (via {u})")
        
        # Construire la liste des arêtes du MST
        aretes_MST = []
        poids_total = 0
        
        for v in range(self.V):
            if parent[v] != -1:
                aretes_MST.append((parent[v], v, self.graph[parent[v]][v]))
                poids_total += self.graph[parent[v]][v]
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"RÉSULTAT")
            print(f"{'='*70}")
            print(f"Arêtes du MST:")
            for u, v, w in aretes_MST:
                print(f"  {u} -- {v} : poids {w}")
            print(f"\nPoids total du MST: {poids_total}")
            print(f"{'='*70}\n")
        
        return aretes_MST, poids_total, historique
    
    def afficher_matrice(self):
        """Affiche la matrice d'adjacence."""
        print("\nMatrice d'adjacence (inf = pas de connexion):")
        print("     ", end="")
        for i in range(self.V):
            print(f"{i:6}", end="")
        print()
        
        for i in range(self.V):
            print(f"{i:3}: ", end="")
            for j in range(self.V):
                if self.graph[i][j] == float('inf'):
                    print(f"  {'inf':>4}", end="")
                else:
                    print(f"{self.graph[i][j]:6.1f}", end="")
            print()


def visualiser_mst(graphe: PrimGrapheDense, aretes_MST: List[Tuple], 
                   poids_total: float, titre: str = "Arbre Couvrant Minimum (Prim)"):
    """
    Visualise le graphe et son MST avec matplotlib et networkx.
    
    Args:
        graphe: Instance de PrimGrapheDense
        aretes_MST: Liste des arêtes du MST
        poids_total: Poids total du MST
        titre: Titre du graphique
    """
    # Créer un graphe NetworkX
    G = nx.Graph()
    
    # Ajouter tous les sommets
    for i in range(graphe.V):
        G.add_node(i)
    
    # Ajouter toutes les arêtes du graphe original
    toutes_aretes = []
    for i in range(graphe.V):
        for j in range(i + 1, graphe.V):
            if graphe.graph[i][j] != float('inf'):
                G.add_edge(i, j, weight=graphe.graph[i][j])
                toutes_aretes.append((i, j))
    
    # Arêtes du MST
    aretes_mst_set = {(min(u, v), max(u, v)) for u, v, _ in aretes_MST}
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=2)
    
    # Dessiner
    plt.figure(figsize=(14, 8))
    
    # Arêtes du graphe original (en gris clair)
    nx.draw_networkx_edges(G, pos, edgelist=toutes_aretes, 
                           width=1, alpha=0.3, edge_color='gray', 
                           style='dashed')
    
    # Arêtes du MST (en rouge épais)
    aretes_mst_list = [(u, v) for u, v, _ in aretes_MST]
    nx.draw_networkx_edges(G, pos, edgelist=aretes_mst_list, 
                           width=3, alpha=1, edge_color='red')
    
    # Sommets
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                           node_size=800, edgecolors='black', linewidths=2)
    
    # Labels des sommets
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
    
    # Labels des poids (toutes les arêtes)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    
    plt.title(f"{titre}\nPoids total: {poids_total}", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('prim_mst_dense.png', dpi=150, bbox_inches='tight')
    print("\n Graphique sauvegardé: prim_mst_dense.png")
    plt.close()


def exemple_simple():
    """Exemple simple avec un petit graphe."""
    print("\n" + "="*70)
    print("EXEMPLE 1: Graphe Simple (5 sommets)")
    print("="*70)
    
    # Créer un graphe avec 5 sommets
    g = PrimGrapheDense(5)
    
    # Ajouter des arêtes (sommet1, sommet2, poids)
    aretes = [
        (0, 1, 2),
        (0, 3, 6),
        (1, 2, 3),
        (1, 3, 8),
        (1, 4, 5),
        (2, 4, 7),
        (3, 4, 9)
    ]
    
    for u, v, w in aretes:
        g.ajouter_arete(u, v, w)
    
    g.afficher_matrice()
    
    # Exécuter Prim
    aretes_MST, poids_total, historique = g.prim_mst(sommet_depart=0, verbose=True)
    
    # Visualiser
    visualiser_mst(g, aretes_MST, poids_total, "Exemple 1: Graphe Simple")
    
    return g, aretes_MST, poids_total


def exemple_graphe_dense():
    """Exemple avec un graphe plus dense."""
    print("\n" + "="*70)
    print("EXEMPLE 2: Graphe Dense (6 sommets, très connecté)")
    print("="*70)
    
    # Créer un graphe avec 6 sommets
    g = PrimGrapheDense(6)
    
    # Graphe dense: beaucoup de connexions
    aretes = [
        (0, 1, 4), (0, 2, 3), (0, 3, 7),
        (1, 2, 2), (1, 4, 5), (1, 5, 6),
        (2, 3, 5), (2, 4, 1), (2, 5, 8),
        (3, 4, 9), (3, 5, 4),
        (4, 5, 3)
    ]
    
    for u, v, w in aretes:
        g.ajouter_arete(u, v, w)
    
    g.afficher_matrice()
    
    # Exécuter Prim
    aretes_MST, poids_total, historique = g.prim_mst(sommet_depart=0, verbose=True)
    
    # Visualiser
    visualiser_mst(g, aretes_MST, poids_total, "Exemple 2: Graphe Dense")
    
    return g, aretes_MST, poids_total


def exemple_comparaison_sommets_depart():
    """Compare les résultats selon le sommet de départ."""
    print("\n" + "="*70)
    print("EXEMPLE 3: Comparaison selon le sommet de départ")
    print("="*70)
    
    # Créer le même graphe
    g = PrimGrapheDense(5)
    aretes = [
        (0, 1, 2), (0, 3, 6), (1, 2, 3),
        (1, 3, 8), (1, 4, 5), (2, 4, 7), (3, 4, 9)
    ]
    for u, v, w in aretes:
        g.ajouter_arete(u, v, w)
    
    # Tester différents sommets de départ
    print("\nTest avec différents sommets de départ:")
    for depart in range(g.V):
        print(f"\n--- Départ du sommet {depart} ---")
        aretes_MST, poids_total, _ = g.prim_mst(sommet_depart=depart, verbose=False)
        print(f"Poids total: {poids_total}")
        print(f"Arêtes: {[(u, v) for u, v, _ in aretes_MST]}")
    
    print("\n Observation: Le poids total est identique quel que soit le sommet de départ!")
    print("   (Mais l'ordre de découverte des arêtes peut varier)")


def benchmark_complexite():
    """Démontre la complexité O(V^2) pour graphes denses."""
    import time
    
    print("\n" + "="*70)
    print("BENCHMARK: Complexité O(V^2) pour graphes denses")
    print("="*70)
    
    tailles = [10, 20, 30, 40, 50]
    temps = []
    
    print(f"\n{'Taille (V)':<12} {'Temps (ms)':<15} {'Ratio t/t_prev':<15}")
    print("-" * 42)
    
    temps_precedent = None
    
    for V in tailles:
        # Créer un graphe dense (presque complet)
        g = PrimGrapheDense(V)
        
        # Ajouter beaucoup d'arêtes (graphe dense)
        import random
        random.seed(42)
        for i in range(V):
            for j in range(i + 1, V):
                if random.random() < 0.7:  # 70% de connexions
                    g.ajouter_arete(i, j, random.randint(1, 100))
        
        # Mesurer le temps
        debut = time.time()
        g.prim_mst(verbose=False)
        fin = time.time()
        
        temps_ms = (fin - debut) * 1000
        temps.append(temps_ms)
        
        if temps_precedent:
            ratio = temps_ms / temps_precedent
            print(f"{V:<12} {temps_ms:<15.3f} {ratio:<15.2f}x")
        else:
            print(f"{V:<12} {temps_ms:<15.3f} {'--':<15}")
        
        temps_precedent = temps_ms
    
    print("\n Pour un graphe dense, Prim en O(V^2) est optimal!")
    print("   Le ratio devrait être environ (Vnew/Vold)^2 = 4x quand on double V")


def tableau_recap_complexite():
    """Affiche un tableau récapitulatif des complexités."""
    print("\n" + "="*70)
    print("TABLEAU RÉCAPITULATIF: Prim selon la densité du graphe")
    print("="*70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    ALGORITHME DE PRIM                               │
├─────────────────────┬──────────────────────┬────────────────────────┤
│  Type de graphe     │  Structure de données│  Complexité temporelle │
├─────────────────────┼──────────────────────┼────────────────────────┤
│  DENSE              │  Array simple        │  O(V^2)                │
│  (|E| ≈ V²)         │  + Matrice adjacence │  ← OPTIMAL             │
├─────────────────────┼──────────────────────┼────────────────────────┤
│  SPARSE             │  Binary Heap         │  O(E log V)            │
│  (|E| ≈ V)          │  + Liste adjacence   │  ← OPTIMAL             │
├─────────────────────┼──────────────────────┼────────────────────────┤
│  Très SPARSE        │  Fibonacci Heap      │  O(E + V log V)        │
│  (|E| << V²)        │  + Liste adjacence   │  (théorique)           │
└─────────────────────┴──────────────────────┴────────────────────────┘

EXEMPLE NUMÉRIQUE (V = 1000 sommets):

Graphe DENSE (|E| = 400,000 arêtes):
  - Array simple:    O(1,000^2)      = 1,000,000 ops   MEILLEUR
  - Binary Heap:     O(400k log 1k) = 4,000,000 ops

Graphe SPARSE (|E| = 5,000 arêtes):
  - Array simple:    O(1,000^2)      = 1,000,000 ops
  - Binary Heap:     O(5k log 1k)   = 50,000 ops      MEILLEUR

💡 CONCLUSION: Utilisez la version array simple (O(V^2)) pour les graphes DENSES!
""")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            ALGORITHME DE PRIM POUR GRAPHES DENSES                    ║
║                    Implémentation O(V^2)                              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Exécuter tous les exemples
    exemple_simple()
    
    print("\n" + "─"*70 + "\n")
    exemple_graphe_dense()
    
    print("\n" + "─"*70 + "\n")
    exemple_comparaison_sommets_depart()
    
    print("\n" + "─"*70 + "\n")
    benchmark_complexite()
    
    print("\n" + "─"*70 + "\n")
    tableau_recap_complexite()
    
    print("\n✓ Tous les exemples exécutés avec succès!")
    print("✓ Visualisations sauvegardées dans le répertoire courant")
