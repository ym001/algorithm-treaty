import heapq
from collections import defaultdict

class NoeudHuffman:
    def __init__(self, symbole=None, freq=0, gauche=None, droite=None):
        self.symbole = symbole
        self.freq = freq
        self.gauche = gauche
        self.droite = droite
    
    # Pour la file de priorité
    def __lt__(self, autre):
        return self.freq < autre.freq

def calculer_frequences(texte):
    """Compte les fréquences des symboles"""
    freq = defaultdict(int)
    for symbole in texte:
        freq[symbole] += 1
    return freq

def construire_arbre_huffman(frequences):
    """Construit l'arbre de Huffman"""
    # File de priorité (min-heap)
    heap = []
    
    # Créer une feuille pour chaque symbole
    for symbole, freq in frequences.items():
        noeud = NoeudHuffman(symbole, freq)
        heapq.heappush(heap, noeud)
    
    # Construire l'arbre
    while len(heap) > 1:
        # Extraire les deux plus petits
        gauche = heapq.heappop(heap)
        droite = heapq.heappop(heap)
        
        # Créer un nœud parent
        parent = NoeudHuffman(
            symbole=None,
            freq=gauche.freq + droite.freq,
            gauche=gauche,
            droite=droite
        )
        
        # Réinsérer
        heapq.heappush(heap, parent)
    
    return heap[0]  # La racine

def generer_codes(noeud, code="", codes=None):
    """Génère les codes de Huffman"""
    if codes is None:
        codes = {}
    
    if noeud.symbole is not None:
        # C'est une feuille
        codes[noeud.symbole] = code if code else "0"
        return codes
    
    # Récursion
    if noeud.gauche:
        generer_codes(noeud.gauche, code + "0", codes)
    if noeud.droite:
        generer_codes(noeud.droite, code + "1", codes)
    
    return codes

def compresser_huffman(texte):
    """Compresse un texte avec Huffman"""
    # 1. Calculer fréquences
    freq = calculer_frequences(texte)
    
    # 2. Construire l'arbre
    arbre = construire_arbre_huffman(freq)
    
    # 3. Générer les codes
    codes = generer_codes(arbre)
    
    # 4. Encoder le texte
    texte_encode = ''.join(codes[symbole] for symbole in texte)
    
    return texte_encode, arbre, codes

def decompresser_huffman(texte_encode, arbre):
    """Décompresse un texte encodé"""
    texte = []
    noeud = arbre
    
    for bit in texte_encode:
        # Naviguer dans l'arbre
        if bit == '0':
            noeud = noeud.gauche
        else:
            noeud = noeud.droite
        
        # Si c'est une feuille, on a décodé un symbole
        if noeud.symbole is not None:
            texte.append(noeud.symbole)
            noeud = arbre  # Recommencer
    
    return ''.join(texte)

# Exemple d'utilisation
texte = "MISSISSIPPI"
print(f"Texte original : {texte}")
print(f"Taille : {len(texte)} caractères = {len(texte) * 8} bits (ASCII)")

# Compression
encode, arbre, codes = compresser_huffman(texte)

print("\nCodes de Huffman :")
for symbole, code in sorted(codes.items()):
    print(f"  {symbole} : {code}")

print(f"\nTexte encodé : {encode}")
print(f"Taille encodée : {len(encode)} bits")
print(f"Taux de compression : {len(encode) / (len(texte) * 8) * 100:.1f}%")

# Décompression
decompresse = decompresser_huffman(encode, arbre)
print(f"\nTexte décompressé : {decompresse}")
print(f"Identique ? {texte == decompresse}")
