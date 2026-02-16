# crop.py
import numpy as np
import cv2

def isolate_dermato_circle_adaptive(
    img,
    thresh_circle=60,          # seuil pour trouver le disque clair
    shrink_factor=0.9,         # pour enlever un peu le bord sombre
    crop=True,
    border_dark_thresh=30,     # seuil "noir" pour les bords
    border_ratio_trigger=0.2,  # % de pixels noirs sur les bords pour déclencher le cercle
    border_width_ratio=0.05    # largeur relative de la bande de bord (5%)
):
    """
    img : image RGB (np.ndarray, HxWx3)

    Retourne : 
        - img_out : image RGB avec le cercle isolé (et éventuellement croppée)
        - mask_out : masque 0/255 correspondant au cercle
    """

    h, w = img.shape[:2]
    # ICI on suppose que l'image est RGB
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ---- 0) Vérifier s'il y a vraiment un cadre noir ----
    bw = int(min(h, w) * border_width_ratio)
    border_mask = np.zeros_like(gray, dtype=bool)
    border_mask[:bw, :] = True      # haut
    border_mask[-bw:, :] = True     # bas
    border_mask[:, :bw] = True      # gauche
    border_mask[:, -bw:] = True     # droite

    dark_border = (gray < border_dark_thresh) & border_mask
    dark_ratio = dark_border.sum() / border_mask.sum()

    # Si peu de pixels vraiment noirs sur les bords -> pas de cercle
    if dark_ratio < border_ratio_trigger:
        mask_full = np.ones((h, w), dtype=np.uint8) * 255
        return img.copy(), mask_full

    # ---- 1) Détection du disque clair ----
    _, mask = cv2.threshold(gray, thresh_circle, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        mask_full = np.ones((h, w), dtype=np.uint8) * 255
        return img.copy(), mask_full

    areas = stats[1:, cv2.CC_STAT_AREA]
    main_label = 1 + np.argmax(areas)
    mask_big = (labels == main_label).astype(np.uint8) * 255

    ys, xs = np.where(mask_big == 255)
    cx = int(xs.mean())
    cy = int(ys.mean())
    r = int(np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).max()))
    r = int(r * shrink_factor)

    Y, X = np.ogrid[:h, :w]
    circle_mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= r * r
    circle_mask = (circle_mask * 255).astype(np.uint8)

    output = img.copy()
    # On met le fond en blanc
    output[circle_mask == 0] = 255

    if crop:
        ys, xs = np.where(circle_mask == 255)
        top, bottom = ys.min(), ys.max() + 1
        left, right = xs.min(), xs.max() + 1
        output = output[top:bottom, left:right]
        circle_mask = circle_mask[top:bottom, left:right]

    return output, circle_mask
