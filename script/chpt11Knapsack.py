def knapsack(n, W, poids, valeur):
    """
    Résout le problème du sac à dos 0-1
    
    n : nombre d'objets
    W : capacité maximale
    poids : liste des poids
    valeur : liste des valeurs
    """
    # Créer tableau (n+1) x (W+1)
    K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
    
    # Remplir le tableau
    for i in range(1, n + 1):
        for w in range(W + 1):
            if poids[i-1] > w:
                # Objet trop lourd
                K[i][w] = K[i-1][w]
            else:
                # Choisir le meilleur
                sans_i = K[i-1][w]
                avec_i = valeur[i-1] + K[i-1][w - poids[i-1]]
                K[i][w] = max(sans_i, avec_i)
    
    return K[n][W]

# Exemple
poids = [2, 3, 4]
valeur = [3, 4, 5]
W = 5
n = len(poids)

print(f"Valeur maximale : {knapsack(n, W, poids, valeur)}")
# Affiche : 7
