#!/usr/bin/env python3
"""
Script de test pour la méthode LBP Clustering
Permet de tester et visualiser les résultats étape par étape
Sauvegarde tous les résultats dans le dossier tests
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# Importer toutes les fonctions de segmentation
from segmentation import (
    lbp_clustering_segmentation, 
    load_ground_truth_mask, 
    compute_segmentation_metrics,
    luminance_bt601,
    lbp_p8_r1_numpy,
    binarize_lbp_patterns,
    gaussian_smoothing,
    rgb_to_lab_ab,
    kmeans_clustering,
    select_lesion_cluster,
    morphological_postprocessing
)

# Importer skimage
from skimage import io as skio

# Configuration des répertoires
TEST_DIR = 'tests'  # Répertoire pour sauvegarder les résultats (dans le même dossier)

def adaptive_sigma(img_shape: tuple, base_sigma: float = 3.0, reference_size: int = 1000) -> float:
    """
    Calcule un sigma adaptatif basé sur la taille de l'image
    
    Paramètres:
    - img_shape: Forme de l'image (H, W, C) ou (H, W)
    - base_sigma: Sigma de référence (défaut: 3.0 comme dans l'article)
    - reference_size: Taille de référence pour laquelle base_sigma était optimisé
    
    Retourne:
    - sigma adapté à la taille de l'image
    """
    # Prendre la dimension la plus grande (hauteur ou largeur)
    max_dim = max(img_shape[:2])
    
    # Facteur d'échelle basé sur la racine carrée (pour préserver les proportions)
    scale_factor = np.sqrt(max_dim / reference_size)
    
    # Sigma adaptatif avec bornes min/max pour éviter les extrêmes
    adaptive_sigma_val = base_sigma * scale_factor
    
    # Limiter sigma entre 1.0 et 8.0 pour éviter les valeurs aberrantes
    adaptive_sigma_val = np.clip(adaptive_sigma_val, 1.0, 8.0)
    
    return adaptive_sigma_val


def find_optimal_sigma_for_image(img: np.ndarray, gt_mask: np.ndarray, 
                                sigma_range: tuple = (1.0, 6.0), 
                                n_samples: int = 11) -> tuple:
    """
    Trouve le sigma optimal pour une image donnée par recherche en grille
    
    Paramètres:
    - img: Image RGB
    - gt_mask: Masque de vérité terrain
    - sigma_range: Plage de sigma à tester (min, max)
    - n_samples: Nombre d'échantillons à tester
    
    Retourne:
    - (sigma_optimal, score_dice_max)
    """
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_samples)
    best_score = 0
    best_sigma = sigma_range[0]
    
    for sigma in sigmas:
        mask, _ = lbp_clustering_segmentation(img, sigma=sigma, verbose=False)
        
        # Redimensionner si nécessaire
        if mask.shape != gt_mask.shape:
            from skimage.transform import resize
            mask = resize(mask, gt_mask.shape, preserve_range=True, anti_aliasing=False) > 0.5
            mask = mask.astype(np.uint8)
        
        metrics = compute_segmentation_metrics(mask, gt_mask)
        dice_score = metrics['dice']
        
        if dice_score > best_score:
            best_score = dice_score
            best_sigma = sigma
    
    return best_sigma, best_score

def find_all_images(dataset_dir: str) -> list:
    """
    Trouve toutes les images dans le dataset (exclut les masques de segmentation)
    """
    image_paths = []
    
    # Parcourir tous les sous-dossiers (melanoma, nevus)
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            # Prendre seulement les .jpg qui ne finissent pas par "Segmentation"
            if file.endswith('.jpg') and not file.endswith('_Segmentation.jpg'):
                image_paths.append(os.path.join(root, file))
    
    return sorted(image_paths)


def get_corresponding_mask(img_path: str) -> str:
    """
    Trouve le masque correspondant à une image
    """
    base_name = os.path.basename(img_path).replace('.jpg', '')
    mask_name = f"{base_name}_Segmentation.png"
    mask_path = os.path.join(os.path.dirname(img_path), mask_name)
    return mask_path


def test_single_image_detailed(img_path: str, gt_mask_path: str = None, save_detailed: bool = True, save_comparison: bool = True, use_adaptive_sigma: bool = True):
    """
    Test détaillé sur une seule image avec visualisation des étapes
    """
    print(f"Traitement de {os.path.basename(img_path)}...")
    
    # Charger l'image
    img = skio.imread(img_path)
    
    # Choisir le sigma adaptatif ou fixe
    if use_adaptive_sigma:
        sigma = adaptive_sigma(img.shape)
        print(f"  Sigma adaptatif: {sigma:.2f} (taille: {img.shape[:2]})")
    else:
        sigma = 3.0
        print(f"  Sigma fixe: {sigma}")
    
    # Appliquer la segmentation avec résultats intermédiaires
    mask, intermediate = lbp_clustering_segmentation(img, sigma=sigma, verbose=False)
    
    # Sauvegarder les étapes détaillées seulement pour quelques images
    if save_detailed:
        # Visualisation des étapes intermédiaires
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Pipeline LBP Clustering - {os.path.basename(img_path)}', fontsize=16)
        
        # Image originale
        axes[0,0].imshow(img)
        axes[0,0].set_title('1. Image originale')
        axes[0,0].axis('off')
        
        # Luminance
        axes[0,1].imshow(intermediate['luminance'], cmap='gray')
        axes[0,1].set_title('2. Luminance BT.601')
        axes[0,1].axis('off')
        
        # Codes LBP
        axes[0,2].imshow(intermediate['lbp_codes'], cmap='hot')
        axes[0,2].set_title('3. Codes LBP (P=8, R=1)')
        axes[0,2].axis('off')
        
        # LBP binarisé
        axes[1,0].imshow(intermediate['binary_lbp'], cmap='gray')
        axes[1,0].set_title('4. LBP Binarisé')
        axes[1,0].axis('off')
        
        # Image L lissée
        axes[1,1].imshow(intermediate['smoothed_L'], cmap='gray')
        axes[1,1].set_title('5. Image L (lissage gaussien)')
        axes[1,1].axis('off')
        
        # Caractéristiques a*b*
        ab_viz = np.zeros((*intermediate['ab_features'].shape[:2], 3))
        ab_viz[..., 0] = (intermediate['ab_features'][..., 0] + 100) / 200  # a* normalisé
        ab_viz[..., 1] = (intermediate['ab_features'][..., 1] + 100) / 200  # b* normalisé
        axes[1,2].imshow(ab_viz)
        axes[1,2].set_title('6. Espace a*b* (visualisation)')
        axes[1,2].axis('off')
        
        # Labels des clusters
        axes[2,0].imshow(intermediate['cluster_labels'], cmap='tab10')
        axes[2,0].set_title('7. Clusters k-means++')
        axes[2,0].axis('off')
        
        # Masque lésion brut
        axes[2,1].imshow(intermediate['raw_lesion_mask'], cmap='gray')
        axes[2,1].set_title('8. Masque lésion (brut)')
        axes[2,1].axis('off')
        
        # Masque final
        axes[2,2].imshow(mask, cmap='gray')
        axes[2,2].set_title('9. Masque final (post-traitement)')
        axes[2,2].axis('off')
        
        plt.tight_layout()
        
        # Sauvegarder dans tests/
        base_name = os.path.basename(img_path).replace('.jpg', '')
        save_path = os.path.join(TEST_DIR, f'pipeline_{base_name}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()  # Fermer pour économiser la mémoire
    
    # Si on a un ground truth, calculer les métriques
    metrics = None
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt_mask = load_ground_truth_mask(gt_mask_path)
        
        # Redimensionner si nécessaire
        if mask.shape != gt_mask.shape:
            from skimage.transform import resize
            mask = resize(mask, gt_mask.shape, preserve_range=True, anti_aliasing=False) > 0.5
            mask = mask.astype(np.uint8)
        
        metrics = compute_segmentation_metrics(mask, gt_mask)
        
        # Créer une visualisation de comparaison compacte SEULEMENT pour certaines images
        if save_comparison:
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            
            axes[0].imshow(img)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Segmentation')
            axes[1].axis('off')
            
            axes[2].imshow(gt_mask, cmap='gray')
            axes[2].set_title('Vérité terrain')
            axes[2].axis('off')
            
            # Overlay de comparaison
            overlay = np.zeros((*img.shape[:2], 3))
            overlay[gt_mask == 1] = [0, 1, 0]      # Vert: vérité terrain
            overlay[mask == 1] = [1, 0, 0]         # Rouge: prédiction
            overlay[(gt_mask == 1) & (mask == 1)] = [1, 1, 0]  # Jaune: intersection
            
            axes[3].imshow(img)
            axes[3].imshow(overlay, alpha=0.4)
            axes[3].set_title(f'Dice: {metrics["dice"]:.3f}')
            axes[3].axis('off')
            
            plt.tight_layout()
            
            # Sauvegarder dans tests/
            base_name = os.path.basename(img_path).replace('.jpg', '')
            save_path = os.path.join(TEST_DIR, f'comparison_{base_name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    return mask, metrics


def test_parameter_sensitivity():
    """
    Teste la sensibilité aux paramètres et évalue l'approche adaptative
    """
    print("=== TEST DE SENSIBILITÉ AUX PARAMÈTRES ===")
    
    # Prendre seulement 3 images représentatives pour le test de paramètres
    test_images = [
        "../dataset/melanoma/ISIC_0000030.jpg",
        "../dataset/nevus/ISIC_0000000.jpg", 
        "../dataset/melanoma/ISIC_0000140.jpg"
    ]
    
    # Test différents paramètres sigma
    sigmas = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    results_by_image = {}
    adaptive_results = {}
    optimal_results = {}
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
            
        print(f"Test sur {os.path.basename(img_path)}...")
        
        img = skio.imread(img_path)
        gt_path = get_corresponding_mask(img_path)
        
        if not os.path.exists(gt_path):
            continue
            
        gt_mask = load_ground_truth_mask(gt_path)
        
        print(f"  Taille image: {img.shape}")
        
        # 1. Test des sigmas fixes
        results = []
        for sigma in sigmas:
            mask, _ = lbp_clustering_segmentation(img, sigma=sigma, verbose=False)
            
            # Redimensionner si nécessaire
            if mask.shape != gt_mask.shape:
                from skimage.transform import resize
                mask = resize(mask, gt_mask.shape, preserve_range=True, anti_aliasing=False) > 0.5
                mask = mask.astype(np.uint8)
            
            metrics = compute_segmentation_metrics(mask, gt_mask)
            results.append(metrics['dice'])
        
        results_by_image[os.path.basename(img_path)] = results
        
        # 2. Test du sigma adaptatif
        sigma_adaptive = adaptive_sigma(img.shape)
        mask_adaptive, _ = lbp_clustering_segmentation(img, sigma=sigma_adaptive, verbose=False)
        
        if mask_adaptive.shape != gt_mask.shape:
            from skimage.transform import resize
            mask_adaptive = resize(mask_adaptive, gt_mask.shape, preserve_range=True, anti_aliasing=False) > 0.5
            mask_adaptive = mask_adaptive.astype(np.uint8)
        
        metrics_adaptive = compute_segmentation_metrics(mask_adaptive, gt_mask)
        adaptive_results[os.path.basename(img_path)] = {
            'sigma': sigma_adaptive,
            'dice': metrics_adaptive['dice']
        }
        
        # 3. Recherche du sigma optimal
        optimal_sigma, optimal_dice = find_optimal_sigma_for_image(img, gt_mask)
        optimal_results[os.path.basename(img_path)] = {
            'sigma': optimal_sigma,
            'dice': optimal_dice
        }
        
        print(f"  Sigma adaptatif: {sigma_adaptive:.2f} -> Dice: {metrics_adaptive['dice']:.4f}")
        print(f"  Sigma optimal:   {optimal_sigma:.2f} -> Dice: {optimal_dice:.4f}")
    
    # Graphique des résultats
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique 1: Sensibilité aux paramètres fixes
    for img_name, results in results_by_image.items():
        ax1.plot(sigmas, results, 'o-', linewidth=2, markersize=6, label=img_name)
        
        # Marquer le meilleur score pour cette image
        best_idx = np.argmax(results)
        ax1.plot(sigmas[best_idx], results[best_idx], 's', markersize=8, 
                markerfacecolor='white', markeredgecolor=ax1.lines[-1].get_color(), 
                markeredgewidth=2)
        
        # Ajouter les marqueurs pour sigma adaptatif et optimal
        img_basename = img_name
        if img_basename in adaptive_results:
            ax1.axvline(x=adaptive_results[img_basename]['sigma'], 
                       color=ax1.lines[-1].get_color(), linestyle='--', alpha=0.7,
                       label=f'Adaptatif {img_name}')
    
    ax1.set_xlabel('Valeur de sigma (lissage gaussien)')
    ax1.set_ylabel('Score Dice')
    ax1.set_title('Sensibilité au paramètre sigma')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.legend()
    
    # Graphique 2: Comparaison des approches
    images = list(adaptive_results.keys())
    x_pos = np.arange(len(images))
    
    # Scores avec sigma=3 (référence)
    ref_scores = []
    adaptive_scores = []
    optimal_scores = []
    
    for img_name in images:
        # Score avec sigma=3 (index 2 dans sigmas)
        ref_scores.append(results_by_image[img_name][2])  # sigma=3.0
        adaptive_scores.append(adaptive_results[img_name]['dice'])
        optimal_scores.append(optimal_results[img_name]['dice'])
    
    width = 0.25
    ax2.bar(x_pos - width, ref_scores, width, label='Sigma=3 (référence)', alpha=0.8, color='gray')
    ax2.bar(x_pos, adaptive_scores, width, label='Sigma adaptatif', alpha=0.8, color='blue')
    ax2.bar(x_pos + width, optimal_scores, width, label='Sigma optimal', alpha=0.8, color='green')
    
    ax2.set_xlabel('Images')
    ax2.set_ylabel('Score Dice')
    ax2.set_title('Comparaison des stratégies de sigma')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([img.replace('.jpg', '') for img in images], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    save_path = os.path.join(TEST_DIR, 'parameter_sensitivity.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculer et afficher les statistiques
    if not results_by_image:
        print("⚠️ Aucune image testée pour la sensibilité aux paramètres")
        return 3.0, True  # Valeurs par défaut
        
    if results_by_image:
        avg_results = np.mean(list(results_by_image.values()), axis=0)
        best_sigma_idx = np.argmax(avg_results)
        best_sigma = sigmas[best_sigma_idx]
        
        avg_adaptive = np.mean([r['dice'] for r in adaptive_results.values()])
        avg_optimal = np.mean([r['dice'] for r in optimal_results.values()])
        
        print(f"\nRÉSUMÉ:")
        print(f"Meilleur sigma fixe en moyenne: {best_sigma} (Dice moyen: {avg_results[best_sigma_idx]:.4f})")
        print(f"Performance sigma adaptatif:    Dice moyen: {avg_adaptive:.4f}")
        print(f"Performance sigma optimal:      Dice moyen: {avg_optimal:.4f}")
        
        # Recommandation
        if avg_adaptive > avg_results[best_sigma_idx]:
            print("✅ Le sigma adaptatif améliore les performances !")
        else:
            print("⚠️  Le sigma fixe reste meilleur pour ces images")
            
        return best_sigma, avg_adaptive > avg_results[best_sigma_idx]


def evaluate_full_dataset():
    """
    Évalue la méthode sur tout le dataset de manière efficace
    """
    print("=== ÉVALUATION COMPLÈTE DU DATASET ===")
    
    # Trouver toutes les images
    image_paths = find_all_images("../dataset")
    print(f"Trouvé {len(image_paths)} images à traiter")
    
    results = []
    successful_count = 0
    
    for i, img_path in enumerate(image_paths):
        gt_path = get_corresponding_mask(img_path)
        
        if not os.path.exists(gt_path):
            print(f"⚠️  Masque manquant pour {os.path.basename(img_path)}")
            continue
        
        try:
            # Traitement détaillé seulement pour les 3 premières images
            save_detailed = (i < 3)
            # Comparaisons visuelles seulement pour les 3 premières images
            save_comparison = (i < 3)
            # Utiliser le sigma adaptatif
            use_adaptive = True
            
            mask, metrics = test_single_image_detailed(img_path, gt_path, save_detailed, save_comparison, use_adaptive)
            
            if metrics:
                result = {
                    'image_name': os.path.basename(img_path),
                    'category': os.path.basename(os.path.dirname(img_path)),
                    'metrics': metrics
                }
                results.append(result)
                successful_count += 1
                
                # Affichage compact des résultats
                print(f"✓ {result['image_name']:20} ({result['category']:8}) - Dice: {metrics['dice']:.3f}")
            
        except Exception as e:
            print(f"✗ Erreur sur {os.path.basename(img_path)}: {e}")
            continue
    
    print(f"\n{successful_count}/{len(image_paths)} images traitées avec succès")
    
    return results


def create_summary_plots(results: list):
    """
    Crée les graphiques de synthèse pour tous les résultats
    """
    if not results:
        print("Aucun résultat à visualiser")
        return
    
    # Données pour les graphiques
    image_names = [r['image_name'].replace('.jpg', '') for r in results]
    categories = [r['category'] for r in results]
    
    # Seulement la métrique Dice
    dice_scores = [r['metrics']['dice'] for r in results]
    
    # 1. Graphique principal : seulement Dice
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x_pos = np.arange(len(image_names))
    colors_cat = ['red' if cat == 'melanoma' else 'blue' for cat in categories]
    
    bars = ax.bar(x_pos, dice_scores, color=colors_cat, alpha=0.7)
    
    # Ajouter les valeurs sur les barres
    for bar, score in zip(bars, dice_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Images')
    ax.set_ylabel('Score Dice (0-1)')
    ax.set_title('Évaluation LBP Clustering - Score Dice (Rouge=Melanoma, Bleu=Nevus)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(image_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Ligne de score moyen
    mean_dice = np.mean(dice_scores)
    ax.axhline(y=mean_dice, color='green', linestyle='--', 
               label=f'Moyenne: {mean_dice:.3f}')
    
    # Ligne de seuil de validation à 0.9
    ax.axhline(y=0.9, color='red', linestyle='-', linewidth=2,
               label='Seuil validation: 0.900')
    
    ax.legend()
    
    # Colorier le fond selon la catégorie
    for i, cat in enumerate(categories):
        color = 'lightcoral' if cat == 'melanoma' else 'lightblue'
        ax.axvspan(i-0.4, i+0.4, alpha=0.1, color=color)
    
    plt.tight_layout()
    save_path = os.path.join(TEST_DIR, 'dice_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Graphique category_comparison - Simple avec moyennes et étendues
    melanoma_results = [r for r in results if r['category'] == 'melanoma']
    nevus_results = [r for r in results if r['category'] == 'nevus']
    
    # Un seul graphique simple et clair
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculer les données
    melanoma_dice = [r['metrics']['dice'] for r in melanoma_results]
    nevus_dice = [r['metrics']['dice'] for r in nevus_results]
    
    categories = ['Melanoma', 'Nevus']
    means = [np.mean(melanoma_dice), np.mean(nevus_dice)]
    mins = [np.min(melanoma_dice), np.min(nevus_dice)]
    maxs = [np.max(melanoma_dice), np.max(nevus_dice)]
    
    # Position et couleurs (MÊMES que dice_summary)
    x_pos = np.arange(len(categories))
    colors = ['red', 'blue']  # Exactement comme dice_summary
    
    # Barres pour les moyennes
    bars = ax.bar(x_pos, means, color=colors, alpha=0.7, width=0.6)
    
    # Valeurs moyennes au-dessus des barres (bien espacées)
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2., mean + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    # Étendue avec traits verticaux et valeurs min/max
    for i, (minimum, maximum, color) in enumerate(zip(mins, maxs, colors)):
        # Trait vertical pour l'étendue
        ax.plot([i, i], [minimum, maximum], color='black', linewidth=3, alpha=0.8)
        
        # Point et valeur min (en bas, décalés pour éviter conflit)
        ax.plot(i, minimum, 'o', color='black', markersize=8)
        ax.text(i - 0.15, minimum - 0.03, f'Min: {minimum:.3f}', 
                ha='center', va='top', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                         edgecolor='black', alpha=0.9))
        
        # Point et valeur max (en haut, décalés pour éviter conflit)
        ax.plot(i, maximum, 'o', color='black', markersize=8)
        ax.text(i + 0.15, maximum + 0.03, f'Max: {maximum:.3f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                         edgecolor='black', alpha=0.9))
    
    # Seuil de validation à 0.9
    ax.axhline(y=0.9, color='red', linestyle='-', linewidth=2,
               label='Seuil validation: 0.900')
    
    # Mise en forme
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
    ax.set_ylabel('Score Dice', fontsize=14, fontweight='bold')
    ax.set_title('Moyennes par catégorie avec étendue (Min-Max)', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12)
    
    # Statistiques dans un coin
    total_images = len(melanoma_dice) + len(nevus_dice)
    validated_count = len([d for d in melanoma_dice + nevus_dice if d >= 0.9])
    
    stats_text = f'Total: {total_images} images\nValidées (≥0.9): {validated_count}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(TEST_DIR, 'category_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mean_dice, len(results)


def print_summary_statistics(results: list):
    """
Affiche un résumé statistique des résultats
    """
    if not results:
        return
    
    print("\n" + "="*60)
    print("RÉSUMÉ STATISTIQUE")
    print("="*60)
    
    # Statistiques globales - seulement Dice
    all_dice = [r['metrics']['dice'] for r in results]
    
    print(f"Nombre total d'images traitées: {len(results)}")
    print(f"\nMÉTRIQUE GLOBALE:")
    print(f"  Dice moyen: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    
    # Statistiques par catégorie
    melanoma_results = [r for r in results if r['category'] == 'melanoma']
    nevus_results = [r for r in results if r['category'] == 'nevus']
    
    if melanoma_results:
        mel_dice = [r['metrics']['dice'] for r in melanoma_results]
        print(f"\nMELANOMA ({len(melanoma_results)} images):")
        print(f"  Dice moyen: {np.mean(mel_dice):.4f} ± {np.std(mel_dice):.4f}")
        print(f"  Min/Max   : {np.min(mel_dice):.4f} / {np.max(mel_dice):.4f}")
    
    if nevus_results:
        nev_dice = [r['metrics']['dice'] for r in nevus_results]
        print(f"\nNEVUS ({len(nevus_results)} images):")
        print(f"  Dice moyen: {np.mean(nev_dice):.4f} ± {np.std(nev_dice):.4f}")
        print(f"  Min/Max   : {np.min(nev_dice):.4f} / {np.max(nev_dice):.4f}")
    
    # Top 5 et Bottom 5
    sorted_results = sorted(results, key=lambda x: x['metrics']['dice'], reverse=True)
    
    print(f"\nTOP 5 MEILLEURES PERFORMANCES:")
    for i, r in enumerate(sorted_results[:5]):
        print(f"  {i+1}. {r['image_name']:20} ({r['category']:8}) - Dice: {r['metrics']['dice']:.4f}")
    
    print(f"\nTOP 5 MOINS BONNES PERFORMANCES:")
    for i, r in enumerate(sorted_results[-5:]):
        print(f"  {i+1}. {r['image_name']:20} ({r['category']:8}) - Dice: {r['metrics']['dice']:.4f}")


def main():
    """
    Fonction principale de test optimisée pour le grand dataset
    """
    print("=== TEST DE LA MÉTHODE LBP CLUSTERING ===")
    print(f"Sauvegarde dans: {TEST_DIR}")
    
    # 1. Évaluation complète du dataset
    print("\n1. Évaluation complète du dataset...")
    results = evaluate_full_dataset()
    
    if not results:
        print("Aucun résultat obtenu. Vérifiez le dataset.")
        return
    
    # 2. Test de sensibilité aux paramètres (sur échantillon)
    print("\n2. Test de sensibilité aux paramètres et validation sigma adaptatif...")
    best_sigma_fixed, adaptive_is_better = test_parameter_sensitivity()
    
    # 3. Création des graphiques de synthèse
    print("\n3. Création des graphiques de synthèse...")
    mean_dice, total_images = create_summary_plots(results)
    
    # 4. Affichage des statistiques
    print_summary_statistics(results)
    
    print("\n" + "="*60)
    print("TESTS TERMINÉS")
    print("="*60)
    print(f"Résultats sauvegardés dans: {TEST_DIR}")
    print(f"Score Dice moyen: {mean_dice:.4f}")
    print(f"Images traitées: {total_images}")
    
    if adaptive_is_better:
        print("✅ Sigma adaptatif utilisé avec succès !")
    else:
        print(f"⚠️  Sigma fixe ({best_sigma_fixed}) pourrait être meilleur")
    
    print("\nFichiers générés:")
    print("- pipeline_XXX.png : Détails pour les 3 premières images")
    print("- comparison_XXX.png : Comparaisons pour les 3 premières images")
    print("- dice_summary.png : Scores Dice pour toutes les images")
    print("- category_comparison.png : Comparaison Melanoma vs Nevus")
    print("- parameter_sensitivity.png : Analyse des paramètres + sigma adaptatif")


if __name__ == "__main__":
    main()