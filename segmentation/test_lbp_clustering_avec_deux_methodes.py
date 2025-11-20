#!/usr/bin/env python3
"""
Comparaison LBP Clustering vs Otsu multi-canal sur tout le dataset.
- Calcule le Dice pour CHAQUE image et CHAQUE méthode
- Sauvegarde un CSV de comparaison et plusieurs figures de synthèse
- Affiche un résumé (moyennes, écarts-types, #victoires)
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from skimage import io as skio
from skimage.transform import resize

# ===== Imports méthodes & utilitaires =====
from segmentation_avec_deux_methodes import (
    lbp_clustering_segmentation,                  # méthode 1 (LBP clustering)  [:contentReference[oaicite:4]{index=4}]
    otsu_multi_channel_segmentation,              # méthode 2 (Otsu multi-canal) [:contentReference[oaicite:5]{index=5}]
    load_ground_truth_mask,                       # lecture masque GT            [:contentReference[oaicite:6]{index=6}]
    compute_segmentation_metrics                  # Dice                         [:contentReference[oaicite:7]{index=7}]
)

TEST_DIR = "tests"  # dossier de sortie
os.makedirs(TEST_DIR, exist_ok=True)

# ---------- Dataset helpers (identiques à ta logique existante) ----------
def find_all_images(dataset_dir: str) -> list:
    paths = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith(".jpg") and not f.endswith("_Segmentation.jpg"):
                paths.append(os.path.join(root, f))
    return sorted(paths)

def get_corresponding_mask(img_path: str) -> str:
    base = os.path.basename(img_path).replace(".jpg", "")
    return os.path.join(os.path.dirname(img_path), f"{base}_Segmentation.png")

# ---------- Évaluation par image ----------
def run_both_methods_on_image(img_path: str, gt_path: str,
                              lbp_sigma: float = 3.0,
                              lbp_disk: int = 3,
                              otsu_channels=('red','green','blue','gray'),
                              otsu_disk: int = 3):
    """Retourne (dice_lbp, dice_otsu, infos) pour une image."""
    img = skio.imread(img_path)
    gt  = load_ground_truth_mask(gt_path)

    # --- Méthode 1 : LBP clustering ---
    try:
        mask_lbp, convex_hull_lbp, _ = lbp_clustering_segmentation(img, sigma=lbp_sigma, disk_size=lbp_disk, verbose=False)
        if mask_lbp.shape != gt.shape:
            mask_lbp = resize(mask_lbp, gt.shape, preserve_range=True, anti_aliasing=False) > 0.5
            mask_lbp = mask_lbp.astype(np.uint8)
        dice_lbp = compute_segmentation_metrics(mask_lbp, gt)['dice']
    except Exception as e:
        dice_lbp = 0.0
        mask_lbp = None
        print(f"   ⚠️ LBP échec sur {os.path.basename(img_path)} : {e}")

    # --- Méthode 2 : Otsu multi-canal ---
    try:
        mask_otsu, convex_hull_otsu, _ = otsu_multi_channel_segmentation(
            img, channels=list(otsu_channels), disk_size=otsu_disk,
            selection_method='composite_score', verbose=False
        )
        if mask_otsu.shape != gt.shape:
            mask_otsu = resize(mask_otsu, gt.shape, preserve_range=True, anti_aliasing=False) > 0.5
            mask_otsu = mask_otsu.astype(np.uint8)
        dice_otsu = compute_segmentation_metrics(mask_otsu, gt)['dice']
    except Exception as e:
        dice_otsu = 0.0
        mask_otsu = None
        print(f"   ⚠️ Otsu échec sur {os.path.basename(img_path)} : {e}")

    return dice_lbp, dice_otsu, dict(mask_lbp=mask_lbp, mask_otsu=mask_otsu, gt=gt, img=img)

# ---------- Évaluation dataset ----------
def evaluate_dataset_compare(dataset_root: str) -> list:
    print("=== ÉVALUATION COMPARATIVE LBP vs OTSU ===")
    dataset_root = os.path.abspath(dataset_root)
    print(f"Dataset: {dataset_root}")

    images = find_all_images(dataset_root)
    print(f"Trouvé {len(images)} images")

    results = []
    wins_lbp = 0
    wins_otsu = 0

    for i, img_path in enumerate(images):
        gt_path = get_corresponding_mask(img_path)
        if not os.path.exists(gt_path):
            print(f"   ⚠️ GT manquant pour {os.path.basename(img_path)} — sauté")
            continue

        print(f"[{i+1}/{len(images)}] {os.path.basename(img_path)}")
        d_lbp, d_otsu, _ = run_both_methods_on_image(img_path, gt_path)

        if d_lbp > d_otsu: wins_lbp += 1
        elif d_otsu > d_lbp: wins_otsu += 1

        results.append({
            "image_name": os.path.basename(img_path),
            "category": os.path.basename(os.path.dirname(img_path)),
            "dice_lbp": float(d_lbp),
            "dice_otsu": float(d_otsu),
            "winner": "LBP" if d_lbp > d_otsu else ("OTSU" if d_otsu > d_lbp else "TIE")
        })

    print(f"\nVictoires — LBP: {wins_lbp} | OTSU: {wins_otsu} | Égalités: {len(results) - wins_lbp - wins_otsu}")
    return results

# ---------- Sauvegarde CSV ----------
def save_results_csv(results: list, out_csv: str):
    fieldnames = ["image_name","category","dice_lbp","dice_otsu","winner"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"CSV sauvegardé: {out_csv}")

# ---------- Visualisations ----------
def plot_per_image_bars(results: list):
    names = [r["image_name"].replace(".jpg","") for r in results]
    dice_lbp = [r["dice_lbp"] for r in results]
    dice_otsu = [r["dice_otsu"] for r in results]

    x = np.arange(len(names))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(12, len(names)*0.35), 7))
    ax.bar(x - width/2, dice_lbp, width, label="LBP")
    ax.bar(x + width/2, dice_otsu, width, label="OTSU")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.05)
    ax.set_title("Dice par image — LBP vs Otsu")
    ax.grid(True, alpha=0.3)
    ax.legend()

    for i,(dl,do) in enumerate(zip(dice_lbp,dice_otsu)):
        ax.text(i - width/2, dl+0.01, f"{dl:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, do+0.01, f"{do:.2f}", ha="center", va="bottom", fontsize=8)

    path = os.path.join(TEST_DIR, "dice_compare_per_image.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure: {path}")

def plot_cdf(results: list):
    lbp = np.sort([r["dice_lbp"] for r in results])
    otsu = np.sort([r["dice_otsu"] for r in results])
    x_lbp = np.linspace(0,1,len(lbp),endpoint=False)
    x_otsu = np.linspace(0,1,len(otsu),endpoint=False)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(lbp, np.arange(1,len(lbp)+1)/len(lbp), label="LBP")
    ax.plot(otsu, np.arange(1,len(otsu)+1)/len(otsu), label="OTSU")
    ax.set_xlabel("Dice")
    ax.set_ylabel("CDF")
    ax.set_title("CDF des scores Dice")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = os.path.join(TEST_DIR, "dice_cdf.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure: {path}")

def plot_means_by_category(results: list):
    cats = ["melanoma","nevus"]
    means_lbp, stds_lbp, means_otsu, stds_otsu = [],[],[],[]

    for c in cats:
        vals_lbp = [r["dice_lbp"] for r in results if r["category"]==c]
        vals_otsu = [r["dice_otsu"] for r in results if r["category"]==c]
        if len(vals_lbp)==0: vals_lbp=[0.0]
        if len(vals_otsu)==0: vals_otsu=[0.0]
        means_lbp.append(np.mean(vals_lbp)); stds_lbp.append(np.std(vals_lbp))
        means_otsu.append(np.mean(vals_otsu)); stds_otsu.append(np.std(vals_otsu))

    x = np.arange(len(cats)); width = 0.35
    fig, ax = plt.subplots(figsize=(9,6))
    ax.bar(x - width/2, means_lbp, width, yerr=stds_lbp, capsize=6, label="LBP")
    ax.bar(x + width/2, means_otsu, width, yerr=stds_otsu, capsize=6, label="OTSU")
    ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in cats])
    ax.set_ylabel("Dice moyen ± écart-type")
    ax.set_ylim(0, 1.05)
    ax.set_title("Moyennes par catégorie")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = os.path.join(TEST_DIR, "dice_means_by_category.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure: {path}")

def plot_scatter(results: list):
    lbp = [r["dice_lbp"] for r in results]
    otsu = [r["dice_otsu"] for r in results]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(lbp, otsu, s=30)
    ax.plot([0,1],[0,1])  # ligne y=x
    ax.set_xlabel("Dice LBP")
    ax.set_ylabel("Dice OTSU")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.grid(True, alpha=0.3)
    ax.set_title("LBP vs Otsu — par image")

    path = os.path.join(TEST_DIR, "dice_scatter_lbp_vs_otsu.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure: {path}")

# ---------- Résumé console ----------
def print_summary(results: list):
    if not results:
        print("Aucun résultat.")
        return

    dl = np.array([r["dice_lbp"]  for r in results])
    do = np.array([r["dice_otsu"] for r in results])

    print("\n" + "="*60)
    print("RÉSUMÉ GLOBAL")
    print("="*60)
    print(f"Images évaluées: {len(results)}")
    print(f"LBP  : mean={dl.mean():.4f}  std={dl.std():.4f}")
    print(f"OTSU : mean={do.mean():.4f}  std={do.std():.4f}")
    wins_lbp = np.sum(dl > do); wins_otsu = np.sum(do > dl); ties = np.sum(np.isclose(dl,do,atol=1e-6))
    print(f"Victoires — LBP: {wins_lbp} | OTSU: {wins_otsu} | Égalités: {ties}")

    # Top / Bottom (sur la meilleure des deux)
    best = sorted(results, key=lambda r: max(r["dice_lbp"], r["dice_otsu"]), reverse=True)
    print("\nTOP 5 (meilleure méthode par image) :")
    for i, r in enumerate(best[:5]):
        print(f"  {i+1}. {r['image_name']:20} ({r['category']})  maxDice={max(r['dice_lbp'],r['dice_otsu']):.4f}")

    worst = list(reversed(best))[:5]
    print("\nBOTTOM 5 :")
    for i, r in enumerate(worst):
        print(f"  {i+1}. {r['image_name']:20} ({r['category']})  maxDice={max(r['dice_lbp'],r['dice_otsu']):.4f}")

# ---------- Main ----------
def main():
    # racine dataset = ../dataset (relatif à ce script)
    repo_root   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(repo_root, "dataset")

    results = evaluate_dataset_compare(dataset_dir)
    if not results:
        print("Aucun résultat obtenu — vérifie le chemin du dataset et les masques GT.")
        return

    # CSV
    save_results_csv(results, os.path.join(TEST_DIR, "dice_comparison.csv"))

    # Figures
    plot_per_image_bars(results)
    plot_cdf(results)
    plot_means_by_category(results)
    plot_scatter(results)

    # Résumé console
    print_summary(results)

    print("\nFichiers générés dans 'tests/':")
    print("- dice_comparison.csv")
    print("- dice_compare_per_image.png")
    print("- dice_cdf.png")
    print("- dice_means_by_category.png")
    print("- dice_scatter_lbp_vs_otsu.png")

if __name__ == "__main__":
    main()
