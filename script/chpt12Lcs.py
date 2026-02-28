def lcs_length(X, Y):
    """
    Calcule la longueur de la LCS entre X et Y
    """
    m = len(X)
    n = len(Y)
    
    # Créer tableau (m+1) x (n+1)
    L = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Remplir le tableau
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                # Match !
                L[i][j] = 1 + L[i-1][j-1]
            else:
                # Pas de match
                L[i][j] = max(L[i-1][j], L[i][j-1])
    
    return L[m][n]

# Exemple
X = "ABCDGH"
Y = "AEDFHR"
print(f"Longueur LCS : {lcs_length(X, Y)}")
# Affiche : 3
