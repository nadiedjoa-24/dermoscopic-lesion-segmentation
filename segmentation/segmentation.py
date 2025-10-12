import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
# necessite scikit-image 
from skimage import io as skio
import IPython
from skimage.transform import rescale

def _to_float255(img: np.ndarray) -> np.ndarray:
    """
    Convertit une image RGB en float32 échelle 0..255, sans changer la gamme relative.
    - uint8  -> float32, 0..255 (copie vue)
    - uint16 -> float32, 0..255 (normalisation 65535 -> 255)
    - float  -> si max<=1 => 0..1 -> *255 ; sinon supposé déjà 0..255
    """
    if img.dtype == np.uint8:
        return img.astype(np.float32, copy=False)
    if img.dtype == np.uint16:
        return (img.astype(np.float32) / 65535.0) * 255.0
    if np.issubdtype(img.dtype, np.floating):
        imgf = img.astype(np.float32, copy=False)
        maxv = float(np.nanmax(imgf)) if imgf.size else 1.0
        if maxv <= 1.0001:  # image float normalisée
            imgf = imgf * 255.0
        return imgf
    # fallback générique
    return img.astype(np.float32)


#LUMINANCE BT.601
def luminance_bt601(img_rgb: np.ndarray, *, normalize: bool = False) -> np.ndarray:
    """
    Calcule la luminance Y selon ITU-R BT.601 (R,G,B pondérés 0.299/0.587/0.114).
    Entrée  : img_rgb (H,W,3) en uint8/uint16/float.
    Sortie  : Y en float32.
      - si normalize=False : Y est en échelle 0..255 (recommandé pour la suite LC).
      - si normalize=True  : Y est en échelle 0..1.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
        raise ValueError("img_rgb doit avoir la forme (H, W, 3)")

    img = _to_float255(img_rgb)  # float32, 0..255
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B  # BT.601

    if normalize:
        Y = Y / 255.0
    return Y.astype(np.float32, copy=False)


#LPB P=8, R=1, VERSION NUMPY
def lbp_p8_r1_numpy(Y: np.ndarray, *, pad_mode: str = "reflect", strict: bool = True) -> np.ndarray:
    """
    Calcule les Local Binary Patterns (LBP) pour P=8, R=1 sur une image de luminance Y.
    - Y : array float/uint (H,W), conseillé float32.
    - pad_mode : "reflect" recommandé (pas imposé par l'article).
    - strict : True -> signe '>' (article LC), False -> '>='.
    Retourne : codes LBP uint8 dans [0..255], shape (H,W).
    """
    if Y.ndim != 2:
        raise ValueError("Y doit être 2D (H, W).")
    Y = Y.astype(np.float32, copy=False)

    # Pad d'1 pixel pour pouvoir adresser les 8 voisins (choix d'implémentation)
    Yp = np.pad(Y, 1, mode=pad_mode)

    # Centre
    C = Yp[1:-1, 1:-1]

    # 8 voisins (ordre: N, NE, E, SE, S, SW, W, NW)
    N  = Yp[0:-2, 1:-1]
    NE = Yp[0:-2, 2:  ]
    E  = Yp[1:-1, 2:  ]
    SE = Yp[2:  , 2:  ]
    S  = Yp[2:  , 1:-1]
    SW = Yp[2:  , 0:-2]
    W  = Yp[1:-1, 0:-2]
    NW = Yp[0:-2, 0:-2]

    # Comparaison stricte '>' comme dans s(u)=1[u>0] (article LC)
    if strict:
        b0 = (N  > C).astype(np.uint8)
        b1 = (NE > C).astype(np.uint8)
        b2 = (E  > C).astype(np.uint8)
        b3 = (SE > C).astype(np.uint8)
        b4 = (S  > C).astype(np.uint8)
        b5 = (SW > C).astype(np.uint8)
        b6 = (W  > C).astype(np.uint8)
        b7 = (NW > C).astype(np.uint8)
    else:  # variante '>=' optionnelle
        b0 = (N  >= C).astype(np.uint8)
        b1 = (NE >= C).astype(np.uint8)
        b2 = (E  >= C).astype(np.uint8)
        b3 = (SE >= C).astype(np.uint8)
        b4 = (S  >= C).astype(np.uint8)
        b5 = (SW >= C).astype(np.uint8)
        b6 = (W  >= C).astype(np.uint8)
        b7 = (NW >= C).astype(np.uint8)

    # Assemblage des 8 bits -> code uint8
    lbp = (b0 << 0) | (b1 << 1) | (b2 << 2) | (b3 << 3) | \
          (b4 << 4) | (b5 << 5) | (b6 << 6) | (b7 << 7)

    return lbp.astype(np.uint8, copy=False)





#VERSION SKIMAGE, IDENTIQUE AU PAPIER 
def lbp_p8_r1_skimage(Y: np.ndarray) -> np.ndarray:
    """
    Requiert scikit-image: pip install scikit-image
    Utilise local_binary_pattern(P=8, R=1, method='default') -> seuil strict '>'.
    """
    from skimage.feature import local_binary_pattern
    Y = Y.astype(np.float32, copy=False)
    codes = local_binary_pattern(Y, P=8, R=1, method='default')
    # skimage renvoie float; cast en uint8 pour des codes [0..255]
    return codes.astype(np.uint8)


# BINARISATION PAR SOUS-ENSEMBLE DE MOTIFS
def binarize_lbp_patterns(lbp_codes: np.ndarray) -> np.ndarray:
    """
    Binarise les codes LBP selon le critère de l'article :
    - LBP = 0 et puissances de 2 (motifs lisses) -> 0
    - Tous les autres codes (textures non lisses) -> 1
    Les 1 se concentrent dans la lésion.
    """
    # Puissances de 2 : [1, 2, 4, 8, 16, 32, 64, 128]
    powers_of_2 = np.array([2**i for i in range(8)], dtype=np.uint8)
    
    # Créer un masque pour les motifs à garder à 0
    mask_smooth = np.isin(lbp_codes, np.concatenate([[0], powers_of_2]))
    
    # Binarisation : motifs lisses = 0, autres = 1
    binary_lbp = (~mask_smooth).astype(np.uint8)
    
    return binary_lbp


# LISSAGE GAUSSIEN
def gaussian_smoothing(binary_img: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    Applique un filtre gaussien 2D sur l'image binaire.
    Paramètres par défaut : sigma=3, noyau 13x13
    Retourne une image L ∈ [0,1]
    """
    from scipy.ndimage import gaussian_filter
    
    # Convertir en float pour le filtrage
    img_float = binary_img.astype(np.float32)
    
    # Appliquer le filtre gaussien
    smoothed = gaussian_filter(img_float, sigma=sigma)
    
    # Normaliser entre 0 et 1
    if smoothed.max() > 0:
        smoothed = smoothed / smoothed.max()
    
    return smoothed.astype(np.float32)


# CONVERSION VERS CIE L*A*B*
def rgb_to_lab_ab(pseudo_rgb: np.ndarray) -> np.ndarray:
    """
    Convertit un pseudo-RGB [L, Y, L] vers l'espace CIE L*a*b*
    et retourne seulement les composantes (a*, b*)
    """
    from skimage.color import rgb2lab
    
    # Normaliser le pseudo-RGB entre 0 et 1 pour rgb2lab
    pseudo_rgb_norm = pseudo_rgb / 255.0
    
    # Conversion vers L*a*b*
    lab = rgb2lab(pseudo_rgb_norm)
    
    # Extraire seulement a* et b*
    ab = lab[..., 1:]  # Shape: (H, W, 2)
    
    return ab


# CLUSTERING K-MEANS++
def kmeans_clustering(ab_features: np.ndarray, k: int = 2, n_init: int = 3, max_iter: int = 100) -> tuple:
    """
    Applique k-means++ sur les caractéristiques (a*, b*)
    Retourne les labels et les centres des clusters
    """
    from sklearn.cluster import KMeans
    
    # Reshape pour sklearn : (n_pixels, 2)
    h, w = ab_features.shape[:2]
    features_reshaped = ab_features.reshape(-1, 2)
    
    # K-means++
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=n_init,
        max_iter=max_iter,
        random_state=42  # Pour la reproductibilité
    )
    
    labels = kmeans.fit_predict(features_reshaped)
    centers = kmeans.cluster_centers_
    
    # Reshape les labels vers la forme originale
    labels_2d = labels.reshape(h, w)
    
    return labels_2d, centers


# SÉLECTION DU CLUSTER LÉSION (CRITÈRE PINKNESS)
def select_lesion_cluster(labels: np.ndarray, ab_features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Sélectionne automatiquement le cluster correspondant à la lésion
    selon le critère de 'pinkness' : max(a*, 0) - min(b*, 0)
    """
    n_clusters = len(centers)
    pinkness_scores = []
    
    for i in range(n_clusters):
        # Masque pour ce cluster
        cluster_mask = (labels == i)
        
        if np.sum(cluster_mask) == 0:
            pinkness_scores.append(-np.inf)
            continue
            
        # Extraire les valeurs a* et b* pour ce cluster
        a_values = ab_features[cluster_mask, 0]
        b_values = ab_features[cluster_mask, 1]
        
        # Critère pinkness : moyenne de max(a*, 0) - min(b*, 0)
        pinkness = np.mean(np.maximum(a_values, 0) - np.minimum(b_values, 0))
        pinkness_scores.append(pinkness)
    
    # Sélectionner le cluster avec le score maximum
    lesion_cluster_id = np.argmax(pinkness_scores)
    
    # Créer le masque binaire (1 = lésion, 0 = peau)
    lesion_mask = (labels == lesion_cluster_id).astype(np.uint8)
    
    return lesion_mask


# POST-TRAITEMENT MORPHOLOGIQUE
def morphological_postprocessing(mask: np.ndarray, disk_size: int = 3) -> np.ndarray:
    """
    Applique un post-traitement morphologique :
    - Ouverture (érosion + dilatation)
    - Remplissage des trous
    - Conservation de la plus grande composante connexe
    """
    from skimage.morphology import disk, opening, remove_small_holes
    from skimage.measure import label
    from scipy.ndimage import binary_fill_holes
    
    # Ouverture morphologique
    selem = disk(disk_size)
    opened = opening(mask, selem)
    
    # Remplissage des trous
    filled = binary_fill_holes(opened).astype(np.uint8)
    
    # Garder seulement la plus grande composante connexe
    labeled = label(filled)
    if labeled.max() == 0:
        return filled
    
    # Compter les pixels de chaque composante
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # Ignorer le background
    
    # Garder la plus grande composante
    largest_component = np.argmax(component_sizes)
    final_mask = (labeled == largest_component).astype(np.uint8)
    
    return final_mask


# FONCTION PRINCIPALE DE SEGMENTATION LBP CLUSTERING
def lbp_clustering_segmentation(img_rgb: np.ndarray, 
                              sigma: float = 3.0,
                              disk_size: int = 3,
                              verbose: bool = False) -> tuple:
    """
    Pipeline complet de segmentation par LBP Clustering
    
    Paramètres:
    - img_rgb: Image RGB d'entrée (H, W, 3)
    - sigma: Paramètre du filtre gaussien (défaut: 3.0)
    - disk_size: Taille de l'élément structurant pour post-traitement (défaut: 3)
    - verbose: Affichage des étapes intermédiaires
    
    Retourne:
    - mask: Masque binaire final (1 = lésion, 0 = peau)
    - intermediate_results: Dictionnaire avec les résultats intermédiaires
    """
    
    if verbose:
        print("Étape 1: Conversion en luminance BT.601...")
    
    # 1. Luminance Y (BT.601)
    Y = luminance_bt601(img_rgb, normalize=False)
    
    if verbose:
        print("Étape 2: Calcul des LBP (P=8, R=1)...")
    
    # 2. LBP P=8, R=1
    lbp_codes = lbp_p8_r1_numpy(Y)
    
    if verbose:
        print("Étape 3: Binarisation par sous-ensemble de motifs...")
    
    # 3. Binarisation par sous-ensemble de motifs
    binary_lbp = binarize_lbp_patterns(lbp_codes)
    
    if verbose:
        print("Étape 4: Lissage gaussien...")
    
    # 4. Lissage gaussien → image L
    L = gaussian_smoothing(binary_lbp, sigma=sigma)
    
    if verbose:
        print("Étape 5: Formation du pseudo-RGB et conversion Lab...")
    
    # 5. Empilement [L, Y, L] et conversion vers CIE L*a*b*
    pseudo_rgb = np.stack([L * 255, Y, L * 255], axis=-1)
    ab_features = rgb_to_lab_ab(pseudo_rgb)
    
    if verbose:
        print("Étape 6: Clustering k-means++ (K=2)...")
    
    # 6. Clustering k-means++
    labels, centers = kmeans_clustering(ab_features, k=2, n_init=3, max_iter=100)
    
    if verbose:
        print("Étape 7: Sélection du cluster lésion (critère pinkness)...")
    
    # 7. Sélection automatique du cluster "lésion"
    lesion_mask = select_lesion_cluster(labels, ab_features, centers)
    
    if verbose:
        print("Étape 8: Post-traitement morphologique...")
    
    # 8. Post-traitement morphologique
    final_mask = morphological_postprocessing(lesion_mask, disk_size=disk_size)
    
    # Résultats intermédiaires pour debug/visualisation
    intermediate_results = {
        'luminance': Y,
        'lbp_codes': lbp_codes,
        'binary_lbp': binary_lbp,
        'smoothed_L': L,
        'pseudo_rgb': pseudo_rgb,
        'ab_features': ab_features,
        'cluster_labels': labels,
        'cluster_centers': centers,
        'raw_lesion_mask': lesion_mask
    }
    
    if verbose:
        print("Segmentation terminée !")
    
    return final_mask, intermediate_results


# ========== MÉTRIQUES D'ÉVALUATION ==========

def compute_segmentation_metrics(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> dict:
    """
    Calcule les métriques standard de segmentation
    
    Paramètres:
    - predicted_mask: Masque prédit (0 ou 1)
    - ground_truth_mask: Masque de vérité terrain (0 ou 1)
    
    Retourne:
    - dict avec les métriques: dice, jaccard, accuracy, sensitivity, specificity
    """
    
    # Conversion en booléen et aplatissement
    pred = predicted_mask.astype(bool).flatten()
    gt = ground_truth_mask.astype(bool).flatten()
    
    # Calcul des true positives, false positives, etc.
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    tn = np.sum(~pred & ~gt)
    
    # Éviter la division par zéro
    epsilon = 1e-7
    
    # Dice Coefficient (F1-score)
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    
    # Jaccard Index (IoU)
    jaccard = tp / (tp + fp + fn + epsilon)
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    
    # Sensitivity (Recall)
    sensitivity = tp / (tp + fn + epsilon)
    
    # Specificity
    specificity = tn / (tn + fp + epsilon)
    
    # Precision
    precision = tp / (tp + fp + epsilon)
    
    return {
        'dice': dice,
        'jaccard': jaccard,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def load_ground_truth_mask(mask_path: str) -> np.ndarray:
    """
    Charge un masque de vérité terrain et le convertit en binaire
    """
    mask = skio.imread(mask_path)
    
    # Si l'image est en couleur, prendre un seul canal
    if mask.ndim == 3:
        mask = mask[..., 0]
    
    # Binariser (seuil à 127 pour uint8)
    binary_mask = (mask > 127).astype(np.uint8)
    
    return binary_mask


def evaluate_single_image(img_path: str, gt_mask_path: str, verbose: bool = False) -> dict:
    """
    Évalue la segmentation sur une seule image
    
    Paramètres:
    - img_path: Chemin vers l'image RGB
    - gt_mask_path: Chemin vers le masque de vérité terrain
    - verbose: Affichage détaillé
    
    Retourne:
    - dict avec les métriques et les résultats
    """
    
    # Charger l'image
    img = skio.imread(img_path)
    
    # Charger le masque de vérité terrain
    gt_mask = load_ground_truth_mask(gt_mask_path)
    
    if verbose:
        print(f"Traitement de {os.path.basename(img_path)}...")
        print(f"Taille image: {img.shape}")
        print(f"Taille GT mask: {gt_mask.shape}")
    
    # Segmentation LBP Clustering
    pred_mask, intermediate = lbp_clustering_segmentation(img, verbose=verbose)
    
    # Redimensionner si nécessaire
    if pred_mask.shape != gt_mask.shape:
        from skimage.transform import resize
        pred_mask = resize(pred_mask, gt_mask.shape, 
                          preserve_range=True, anti_aliasing=False) > 0.5
        pred_mask = pred_mask.astype(np.uint8)
    
    # Calcul des métriques
    metrics = compute_segmentation_metrics(pred_mask, gt_mask)
    
    if verbose:
        print(f"Dice: {metrics['dice']:.4f}")
        print(f"Jaccard: {metrics['jaccard']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    return {
        'image_path': img_path,
        'gt_mask_path': gt_mask_path,
        'predicted_mask': pred_mask,
        'ground_truth_mask': gt_mask,
        'metrics': metrics,
        'intermediate_results': intermediate
    }


# ========== VISUALISATION ==========

def visualize_segmentation_result(result: dict, save_path: str = None):
    """
    Visualise les résultats de segmentation pour une image
    """
    img = skio.imread(result['image_path'])
    pred_mask = result['predicted_mask']
    gt_mask = result['ground_truth_mask']
    metrics = result['metrics']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Image originale
    axes[0].imshow(img)
    axes[0].set_title('Image originale')
    axes[0].axis('off')
    
    # Masque prédit
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('Segmentation prédite')
    axes[1].axis('off')
    
    # Masque vérité terrain
    axes[2].imshow(gt_mask, cmap='gray')
    axes[2].set_title('Vérité terrain')
    axes[2].axis('off')
    
    # Overlay comparaison
    overlay = np.zeros((*img.shape[:2], 3))
    overlay[gt_mask == 1] = [0, 1, 0]  # Vert pour GT
    overlay[pred_mask == 1] = [1, 0, 0]  # Rouge pour prédiction
    overlay[(gt_mask == 1) & (pred_mask == 1)] = [1, 1, 0]  # Jaune pour intersection
    
    axes[3].imshow(img)
    axes[3].imshow(overlay, alpha=0.3)
    axes[3].set_title(f'Overlay (Dice: {metrics["dice"]:.3f})')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_metrics_diagram(results: list, save_path: str = None):
    """
    Crée le diagramme demandé : photos en abscisse, scores en ordonnée
    """
    # Extraire les noms d'images et les métriques
    image_names = [os.path.basename(r['image_path']).replace('.jpg', '') for r in results]
    
    metrics_names = ['dice', 'jaccard', 'accuracy', 'sensitivity', 'specificity']
    metrics_data = {name: [r['metrics'][name] for r in results] for name in metrics_names}
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(image_names))
    width = 0.15
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Barres pour chaque métrique
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        ax.bar(x_pos + i * width, values, width, 
               label=metric_name.capitalize(), color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Images')
    ax.set_ylabel('Score (0-1)')
    ax.set_title('Évaluation de la segmentation LBP Clustering')
    ax.set_xticks(x_pos + width * 2)
    ax.set_xticklabels(image_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Afficher aussi les valeurs numériques
    print("\n=== RÉSULTATS NUMÉRIQUES ===")
    for i, result in enumerate(results):
        img_name = os.path.basename(result['image_path'])
        metrics = result['metrics']
        print(f"\n{img_name}:")
        for metric_name in metrics_names:
            print(f"  {metric_name.capitalize():12}: {metrics[metric_name]:.4f}")


def evaluate_dataset(dataset_dir: str, image_pattern: str = "*.jpg", 
                     mask_pattern: str = "*Segmentation.png", verbose: bool = True) -> list:
    """
    Évalue la méthode sur un dataset complet
    
    Paramètres:
    - dataset_dir: Répertoire contenant les images et masques
    - image_pattern: Pattern pour les images (ex: "*.jpg")
    - mask_pattern: Pattern pour les masques (ex: "*Segmentation.png")
    
    Retourne:
    - Liste des résultats pour chaque image
    """
    import glob
    
    # Trouver toutes les images
    image_paths = glob.glob(os.path.join(dataset_dir, image_pattern))
    image_paths.sort()
    
    results = []
    
    for img_path in image_paths:
        # Construire le chemin du masque correspondant
        base_name = os.path.basename(img_path).replace('.jpg', '')
        mask_path = os.path.join(dataset_dir, f"{base_name}_Segmentation.png")
        
        if not os.path.exists(mask_path):
            print(f"Attention: Masque non trouvé pour {img_path}")
            continue
        
        try:
            # Évaluer cette image
            result = evaluate_single_image(img_path, mask_path, verbose=verbose)
            results.append(result)
            
            if verbose:
                print(f"✓ {base_name} traité avec succès")
                
        except Exception as e:
            print(f"✗ Erreur lors du traitement de {base_name}: {e}")
            continue
    
    return results


# ========== FONCTION PRINCIPALE ==========

def main_evaluation():
    """
    Fonction principale pour lancer l'évaluation complète
    """
    # Répertoire du dataset
    dataset_dir = "dataset"
    
    print("=== ÉVALUATION DE LA MÉTHODE LBP CLUSTERING ===")
    print(f"Dataset: {dataset_dir}")
    
    # Évaluation sur le dataset
    results = evaluate_dataset(dataset_dir, verbose=True)
    
    if not results:
        print("Aucun résultat obtenu. Vérifiez le dataset.")
        return
    
    print(f"\n{len(results)} image(s) traitée(s) avec succès.")
    
    # Créer le diagramme des scores
    plot_metrics_diagram(results, save_path="segmentation_scores.png")
    
    # Visualiser chaque résultat
    for i, result in enumerate(results):
        print(f"\nVisualisation {i+1}/{len(results)}:")
        visualize_segmentation_result(result, 
                                    save_path=f"segmentation_result_{i+1}.png")
    
    return results


# Test sur l'image exemple
if __name__ == "__main__":
    main_evaluation()
