"""
Script de test pour vérifier que les deux méthodes retournent bien
le masque final ET son enveloppe convexe
"""

import numpy as np
import matplotlib.pyplot as plt
from segmentation_avec_deux_methodes import lbp_clustering_segmentation, otsu_multi_channel_segmentation
import sys
import os

# Ajouter le dossier parent au path pour accéder au dataset
sys.path.append('..')

def test_both_methods():
    """Test des deux méthodes avec une image du dataset"""
    
    # Charger une image de test
    from skimage import io
    
    # Chemin vers une image du dataset
    dataset_path = "../dataset"
    img_path = os.path.join(dataset_path, "melanoma", "ISIC_0000030.jpg")
    
    if not os.path.exists(img_path):
        print(f"❌ Image non trouvée: {img_path}")
        return
    
    # Charger l'image
    img = io.imread(img_path)
    print(f"✅ Image chargée: {img.shape}")
    
    # Test de la méthode LBP
    print("\n=== TEST MÉTHODE LBP ===")
    try:
        mask_lbp, convex_hull_lbp, intermediate_lbp = lbp_clustering_segmentation(
            img, sigma=3.0, verbose=True
        )
        print(f"✅ LBP - Masque: {mask_lbp.shape}, Enveloppe: {convex_hull_lbp.shape}")
        print(f"   Types: {mask_lbp.dtype}, {convex_hull_lbp.dtype}")
        print(f"   Valeurs uniques masque: {np.unique(mask_lbp)}")
        print(f"   Valeurs uniques enveloppe: {np.unique(convex_hull_lbp)}")
    except Exception as e:
        print(f"❌ Erreur LBP: {e}")
        return
    
    # Test de la méthode Otsu
    print("\n=== TEST MÉTHODE OTSU ===")
    try:
        mask_otsu, convex_hull_otsu, intermediate_otsu = otsu_multi_channel_segmentation(
            img, verbose=True
        )
        print(f"✅ Otsu - Masque: {mask_otsu.shape}, Enveloppe: {convex_hull_otsu.shape}")
        print(f"    Types: {mask_otsu.dtype}, {convex_hull_otsu.dtype}")
        print(f"    Valeurs uniques masque: {np.unique(mask_otsu)}")
        print(f"    Valeurs uniques enveloppe: {np.unique(convex_hull_otsu)}")
    except Exception as e:
        print(f"❌ Erreur Otsu: {e}")
        return
    
    # Créer une visualisation comparative
    print("\n=== CRÉATION DE LA VISUALISATION ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Ligne 1: LBP
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Image originale")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_lbp, cmap='gray')
    axes[0, 1].set_title("LBP - Masque final")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(convex_hull_lbp, cmap='gray')
    axes[0, 2].set_title("LBP - Enveloppe convexe")
    axes[0, 2].axis('off')
    
    # Ligne 2: Otsu
    axes[1, 0].imshow(img)
    axes[1, 0].set_title("Image originale")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask_otsu, cmap='gray')
    axes[1, 1].set_title("Otsu - Masque final")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(convex_hull_otsu, cmap='gray')
    axes[1, 2].set_title("Otsu - Enveloppe convexe")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = "tests/test_convex_hull_comparison.png"
    os.makedirs("tests", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualisation sauvegardée: {output_path}")
    
    print("\n" + "="*60)
    print("🎉 TEST RÉUSSI - Les deux méthodes retournent bien:")
    print("   1. Le masque final")
    print("   2. L'enveloppe convexe du masque")
    print("   3. Les résultats intermédiaires")
    print("="*60)

if __name__ == "__main__":
    test_both_methods()