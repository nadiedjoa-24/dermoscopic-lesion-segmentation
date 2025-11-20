import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
# necessite scikit-image 
from skimage import io as skio
from skimage.morphology import convex_hull_image
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
def luminance_bt601(img_rgb: np.ndarray) -> np.ndarray:
    """
    Calcule la luminance Y selon ITU-R BT.601 (R,G,B pondérés 0.299/0.587/0.114).
    Entrée  : img_rgb (H,W,3) en uint8/uint16/float.
    Sortie  : Y en float32, échelle 0..255 (comme attendu par LBP).
    """
    if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
        raise ValueError("img_rgb doit avoir la forme (H, W, 3)")

    img = _to_float255(img_rgb)  # float32, 0..255
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B  # BT.601
    
    return Y.astype(np.float32, copy=False)


#LPB P=8, R=1, VERSION NUMPY A LA MAIN
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
    L'idée est que les 1 sont censés être concentrés dans la lésion.
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


# ========================================================================
# =                    MÉTHODE DE SEGMENTATION OTSU                     =
# ========================================================================

def otsu_threshold_segmentation(img_rgb: np.ndarray, 
                               channel: str = 'red',
                               disk_size: int = 3,
                               verbose: bool = False) -> tuple:
    """
    Méthode de segmentation basée sur le seuillage d'Otsu sur un canal de couleur
    
    Paramètres:
    - img_rgb: Image RGB d'entrée (H, W, 3)
    - channel: Canal de couleur ('red', 'green', 'blue', 'gray')
    - disk_size: Taille de l'élément structurant pour post-traitement (défaut: 3)
    - verbose: Affichage des étapes intermédiaires
    
    Retourne:
    - mask: Masque binaire final (1 = lésion, 0 = peau)
    - intermediate_results: Dictionnaire avec les résultats intermédiaires
    """
    
    if verbose:
        print(f"Méthode Otsu sur canal {channel}...")
    
    # Conversion en format float32
    img_float = _to_float255(img_rgb) / 255.0  # Normaliser entre 0 et 1
    
    # Extraction du canal spécifié
    if channel == 'red':
        channel_img = img_float[:, :, 0]
    elif channel == 'green':
        channel_img = img_float[:, :, 1]
    elif channel == 'blue':
        channel_img = img_float[:, :, 2]
    elif channel == 'gray':
        # Conversion en niveaux de gris avec les poids standard
        channel_img = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
    else:
        raise ValueError(f"Canal '{channel}' non supporté. Utilisez 'red', 'green', 'blue' ou 'gray'.")
    
    if verbose:
        print(f"Canal {channel} extrait...")
    
    # Calcul du seuil d'Otsu
    thresh = otsu_threshold_calculation(channel_img)
    
    if verbose:
        print(f"Seuil d'Otsu calculé: {thresh:.2f}")
    
    # Application du seuillage (lésions généralement plus sombres)
    if channel in ['red', 'green', 'blue']:
        binary_mask = (channel_img < thresh).astype(np.uint8)
    else:  # gray
        binary_mask = (channel_img < thresh).astype(np.uint8)
    
    if verbose:
        print("Seuillage appliqué...")
    
    # Post-traitement morphologique adapté
    final_mask = otsu_morphological_postprocessing(binary_mask, disk_size)
    
    if verbose:
        print("Post-traitement morphologique terminé...")
    
    # Résultats intermédiaires
    intermediate_results = {
        'channel_image': channel_img,
        'threshold_value': thresh,
        'binary_mask': binary_mask,
        'final_mask': final_mask
    }
    
    if verbose:
        print("Segmentation Otsu terminée !")
    
    return final_mask, intermediate_results


def otsu_threshold_calculation(img: np.ndarray) -> float:
    """
    Calcule le seuil optimal d'Otsu pour une image en niveaux de gris
    
    Paramètres:
    - img: Image en niveaux de gris (H, W)
    
    Retourne:
    - thresh: Seuil optimal d'Otsu
    """
    # Normaliser l'image entre 0 et 255
    img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    
    # Calculer l'histogramme
    hist, bins = np.histogram(img_norm, bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    
    # Normaliser l'histogramme
    hist_norm = hist / hist.sum()
    
    # Variables pour la variance inter-classe
    variance_max = 0
    threshold_best = 0
    
    # Calculer la variance inter-classe pour chaque seuil possible
    for t in range(256):
        # Probabilités des classes
        w0 = hist_norm[:t].sum()  # Classe 0 (sombre)
        w1 = hist_norm[t:].sum()  # Classe 1 (clair)
        
        if w0 == 0 or w1 == 0:
            continue
        
        # Moyennes des classes
        mu0 = (hist_norm[:t] * np.arange(t)).sum() / w0 if w0 > 0 else 0
        mu1 = (hist_norm[t:] * np.arange(t, 256)).sum() / w1 if w1 > 0 else 0
        
        # Variance inter-classe
        variance_between = w0 * w1 * (mu0 - mu1) ** 2
        
        if variance_between > variance_max:
            variance_max = variance_between
            threshold_best = t
    
    # Reconvertir le seuil dans l'espace original
    thresh_original = (threshold_best / 255.0) * (img.max() - img.min()) + img.min()
    
    return thresh_original


def otsu_morphological_postprocessing(mask: np.ndarray, disk_size: int = 3) -> np.ndarray:
    """
    Post-traitement morphologique spécialisé pour la méthode Otsu
    Inspiré de la fonction postProc du notebook
    
    Paramètres:
    - mask: Masque binaire d'entrée
    - disk_size: Taille de base pour les éléments structurants
    
    Retourne:
    - final_mask: Masque après post-traitement
    """
    from skimage.morphology import disk, binary_opening
    from skimage.morphology import convex_hull_image
    
    h, w = mask.shape
    
    # Calculer la taille adaptative basée sur la taille de l'image
    # Comme dans le notebook: radius = int(0.01 * max(h, w))
    adaptive_radius = max(1, int(0.01 * max(h, w)))
    
    # Combiner avec le disk_size paramétrable
    final_radius = max(disk_size, adaptive_radius)
    
    # Ouverture morphologique pour enlever le bruit
    selem = disk(final_radius)
    opened = binary_opening(mask, selem)
    
    # Enveloppe convexe pour régulariser la forme
    final_mask = convex_hull_image(opened).astype(np.uint8)
    
    return final_mask


# ========================================================================


# POST-TRAITEMENT MORPHOLOGIQUE
def morphological_postprocessing(mask: np.ndarray, disk_size: int = 3) -> np.ndarray:
    """
    Applique un post-traitement morphologique :
    - Ouverture (érosion + dilatation)
    - Remplissage des trous
    - Conservation de TOUTES les composantes connexes (pas seulement la plus grande)
    """
    from skimage.morphology import disk, opening, remove_small_holes
    from skimage.measure import label
    from scipy.ndimage import binary_fill_holes
    
    # Ouverture morphologique
    selem = disk(disk_size)
    opened = opening(mask, selem)
    
    # Remplissage des trous
    filled = binary_fill_holes(opened).astype(np.uint8)
    
    # MODIFICATION: Garder TOUTES les composantes connexes
    # au lieu de seulement la plus grande
    return filled


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
    - convex_hull_mask: Enveloppe convexe du masque final
    - intermediate_results: Dictionnaire avec les résultats intermédiaires
    """
    
    if verbose:
        print("Étape 1: Conversion en luminance BT.601...")
    
    # 1. Luminance Y (BT.601)
    Y = luminance_bt601(img_rgb)
    
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
    
    # Calcul de l'enveloppe convexe du masque final
    convex_hull_mask = convex_hull_image(final_mask).astype(np.uint8)
    
    # Ajout de l'enveloppe convexe aux résultats intermédiaires
    intermediate_results['convex_hull_mask'] = convex_hull_mask
    
    return final_mask, convex_hull_mask, intermediate_results


# FONCTION PRINCIPALE DE SEGMENTATION OTSU MULTI-CANAL
def otsu_multi_channel_segmentation(img_rgb: np.ndarray,
                                  channels: list = ['red', 'green', 'blue', 'gray'],
                                  disk_size: int = 3,
                                  selection_method: str = 'best_dice',
                                  verbose: bool = False) -> tuple:
    """
    Pipeline complet de segmentation par seuillage d'Otsu multi-canal avec prétraitement
    
    Paramètres:
    - img_rgb: Image RGB d'entrée (H, W, 3) 
    - channels: Liste des canaux à tester ['red', 'green', 'blue', 'gray']
    - disk_size: Taille de l'élément structurant pour post-traitement (défaut: 3)
    - selection_method: Méthode de sélection du meilleur canal ('best_dice', 'largest_area', 'manual')
    - verbose: Affichage des étapes intermédiaires
    
    Retourne:
    - best_mask: Meilleur masque binaire final (1 = lésion, 0 = peau)
    - convex_hull_mask: Enveloppe convexe du meilleur masque
    - all_results: Dictionnaire avec tous les résultats par canal
    """
    
    if verbose:
        print("=== SEGMENTATION OTSU MULTI-CANAL ===")
        print(f"Canaux à tester: {channels}")
        print(f"Méthode de sélection: {selection_method}")
    
    # 1. Prétraitement de l'image d'entrée
    if verbose:
        print("\nÉtape 1: Prétraitement de l'image...")
    
    # Normalisation et conversion si nécessaire
    img_preprocessed = preprocess_image(img_rgb, verbose=verbose)
    
    # 2. Segmentation sur chaque canal
    all_results = {}
    channel_scores = {}
    
    if verbose:
        print(f"\nÉtape 2: Segmentation sur {len(channels)} canaux...")
    
    for channel in channels:
        if verbose:
            print(f"\n--- Test du canal {channel} ---")
        
        try:
            # Segmentation Otsu sur ce canal
            mask, intermediate = otsu_threshold_segmentation(
                img_preprocessed, 
                channel=channel, 
                disk_size=disk_size, 
                verbose=verbose
            )
            
            # Calcul de métriques pour évaluation
            area_ratio = np.sum(mask) / mask.size
            mean_threshold = intermediate['threshold_value']
            
            # Score composite (à ajuster selon les besoins)
            composite_score = area_ratio * (1.0 - abs(mean_threshold - 0.5))
            
            all_results[channel] = {
                'mask': mask,
                'intermediate': intermediate,
                'area_ratio': area_ratio,
                'threshold': mean_threshold,
                'composite_score': composite_score
            }
            
            channel_scores[channel] = composite_score
            
            if verbose:
                print(f"Canal {channel}: seuil={mean_threshold:.3f}, aire={area_ratio:.3f}, score={composite_score:.3f}")
                
        except Exception as e:
            if verbose:
                print(f"Erreur avec le canal {channel}: {e}")
            all_results[channel] = None
            channel_scores[channel] = -1.0
    
    # 3. Sélection du meilleur canal
    if verbose:
        print(f"\nÉtape 3: Sélection du meilleur canal...")
    
    best_channel, best_mask = select_best_channel(
        all_results, 
        channel_scores, 
        method=selection_method,
        verbose=verbose
    )
    
    if verbose:
        print(f"Meilleur canal sélectionné: {best_channel}")
        print("=== SEGMENTATION OTSU TERMINÉE ===")
    
    # Calcul de l'enveloppe convexe du masque final
    convex_hull_mask = convex_hull_image(best_mask).astype(np.uint8)
    
    return best_mask, convex_hull_mask, {
        'best_channel': best_channel,
        'all_results': all_results,
        'channel_scores': channel_scores,
        'preprocessed_image': img_preprocessed,
        'convex_hull_mask': convex_hull_mask
    }


def preprocess_image(img_rgb: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Prétraitement de l'image d'entrée pour la segmentation Otsu
    
    Paramètres:
    - img_rgb: Image RGB d'entrée
    - verbose: Affichage des informations
    
    Retourne:
    - img_processed: Image prétraitée
    """
    
    # Conversion en float32 avec normalisation
    img_processed = _to_float255(img_rgb)
    
    if verbose:
        print(f"Image convertie: shape={img_processed.shape}, dtype={img_processed.dtype}")
        print(f"Plage de valeurs: [{img_processed.min():.1f}, {img_processed.max():.1f}]")
    
    # Optionnel: redimensionnement si l'image est très grande
    h, w = img_processed.shape[:2]
    max_size = 800
    
    if max(h, w) > max_size:
        if verbose:
            print(f"Redimensionnement de {h}x{w} vers une taille maximale de {max_size}")
        
        scale_factor = max_size / max(h, w)
        img_processed = rescale(
            img_processed, 
            scale_factor, 
            channel_axis=2, 
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.float32)
        
        if verbose:
            new_h, new_w = img_processed.shape[:2]
            print(f"Nouvelle taille: {new_h}x{new_w}")
    
    return img_processed


def select_best_channel(all_results: dict, 
                       channel_scores: dict, 
                       method: str = 'best_dice',
                       verbose: bool = False) -> tuple:
    """
    Sélectionne le meilleur canal selon la méthode spécifiée
    
    Paramètres:
    - all_results: Dictionnaire avec tous les résultats
    - channel_scores: Scores de chaque canal
    - method: Méthode de sélection ('best_dice', 'largest_area', 'composite_score')
    - verbose: Affichage des informations
    
    Retourne:
    - best_channel: Nom du meilleur canal
    - best_mask: Masque correspondant au meilleur canal
    """
    
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    
    if not valid_results:
        raise ValueError("Aucun canal n'a produit de résultat valide")
    
    if method == 'composite_score':
        # Utilise le score composite calculé
        best_channel = max(channel_scores.keys(), key=lambda k: channel_scores[k])
        
    elif method == 'largest_area':
        # Sélectionne le canal avec la plus grande aire de lésion
        areas = {k: v['area_ratio'] for k, v in valid_results.items()}
        best_channel = max(areas.keys(), key=lambda k: areas[k])
        
    elif method == 'middle_threshold':
        # Sélectionne le canal avec le seuil le plus proche de 0.5
        thresholds = {k: abs(v['threshold'] - 0.5) for k, v in valid_results.items()}
        best_channel = min(thresholds.keys(), key=lambda k: thresholds[k])
        
    else:  # 'best_dice' par défaut (nécessiterait un masque de référence)
        # En l'absence de masque de référence, utilise le score composite
        best_channel = max(channel_scores.keys(), key=lambda k: channel_scores[k])
    
    if verbose:
        print(f"Méthode de sélection: {method}")
        for channel in valid_results.keys():
            score = channel_scores.get(channel, 0)
            area = valid_results[channel]['area_ratio']
            thresh = valid_results[channel]['threshold']
            marker = " ← SÉLECTIONNÉ" if channel == best_channel else ""
            print(f"  {channel}: score={score:.3f}, aire={area:.3f}, seuil={thresh:.3f}{marker}")
    
    best_mask = valid_results[best_channel]['mask']
    
    return best_channel, best_mask


# ========== MÉTRIQUES D'ÉVALUATION ==========

def compute_segmentation_metrics(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> dict:
    """
    Calcule le coefficient de Dice pour l'évaluation de segmentation
    
    Paramètres:
    - predicted_mask: Masque prédit (0 ou 1)
    - ground_truth_mask: Masque de vérité terrain (0 ou 1)
    
    Retourne:
    - dict avec la métrique: dice
    """
    
    # Conversion en booléen et aplatissement
    pred = predicted_mask.astype(bool).flatten()
    gt = ground_truth_mask.astype(bool).flatten()
    
    # Calcul des true positives, false positives, false negatives
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    
    # Éviter la division par zéro
    epsilon = 1e-7
    
    # Dice Coefficient (F1-score)
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    
    return {
        'dice': dice
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
