import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# default input image (you can change this path)
DATASET_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'dataset', 'melanoma', 'ISIC_0000046.jpg'))
# output base folder
DATASET_PATH2 = os.path.normpath(os.path.join(BASE_DIR, '..', 'test'))


def get_v_channel(img_bgr: np.ndarray) -> np.ndarray:
    """Convertit une image BGR en HSV et retourne le canal V (uint8).
    """
    if img_bgr is None:
        raise ValueError('Image fournie est None')
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2]


def generate_threshold_masks(v_channel: np.ndarray):
    """Génère une liste de masques binaires T_i (0/255) en ignorant les doublons consécutifs.
    Retourne une liste de masques uint8.
    """
    masks = []
    prev = None
    for i in range(1, 256):
        _, t = cv2.threshold(v_channel, i, 255, cv2.THRESH_BINARY)
        if prev is None or not np.array_equal(t, prev):
            masks.append(t)
            prev = t
    return masks


def detect_hair_candidates(ti: np.ndarray, kernel_size: int = 5, lambda_: float = 0.2, max_radius: int = 30) -> np.ndarray:
    """Pour un masque binaire Ti (0/255), retourne Gi_mask (0/255) isolant les gaps/poils.
    Implémente closing/opening, skeletonisation, distance transforms et reconstruction par disques.
    """
    if ti is None:
        return None

    ti_bin = (ti > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # ouverture puis fermeture pour obtenir Omega_oc (zones potentiellement foncées)
    opened = cv2.morphologyEx(ti_bin, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    omega_oc = closed.copy()

    # fermeture puis ouverture pour Omega_co
    closed2 = cv2.morphologyEx(ti_bin, cv2.MORPH_CLOSE, kernel)
    omega_co = cv2.morphologyEx(closed2, cv2.MORPH_OPEN, kernel)

    # squelette de Omega_oc
    skeleton = skeletonize(omega_oc.astype(bool)).astype(np.uint8)

    # distance transform (sur 0/255 images)
    dt_oc = cv2.distanceTransform((omega_oc * 255).astype(np.uint8), cv2.DIST_L2, 3)
    dt_co = cv2.distanceTransform((omega_co * 255).astype(np.uint8), cv2.DIST_L2, 3)

    # fusion
    rho = (1 - lambda_) * dt_co + lambda_ * dt_oc

    # reconstruction par disques centrés sur le squelette
    Gi_mask = np.zeros_like(ti_bin, dtype=np.uint8)
    points = np.column_stack(np.where(skeleton == 1))
    h, w = ti_bin.shape
    for (y, x) in points:
        r = int(rho[y, x])
        # limiter le rayon pour éviter de relier des régions distantes
        if r <= 0:
            continue
        r = min(r, max_radius)
        # si le disque est trop grand (proportionnellement), ignorer
        if r > max(3, int(min(h, w) * 0.05)):
            continue
        cv2.circle(Gi_mask, (x, y), r, 1, -1)

    # enlever les régions internes déjà dans omega_oc
    Gi_mask = cv2.subtract(Gi_mask, omega_oc)

    return (Gi_mask * 255).astype(np.uint8)


def build_global_mask(masks_list, kernel_size=5, lambda_=0.2):
    """Applique detect_hair_candidates sur chaque masque et fusionne les Gi.
    Retourne M (0/255 uint8).
    """
    M = np.zeros_like(masks_list[0], dtype=np.uint8)
    for ti in masks_list:
        gi = detect_hair_candidates(ti, kernel_size=kernel_size, lambda_=lambda_)
        if gi is not None:
            M = cv2.bitwise_or(M, gi)
    return M


def detect_hairs_by_blackhat(v_channel: np.ndarray, rect_size=(9, 1), thresh: int = 10) -> np.ndarray:
    """Détecte les poils sombres fins via un black-hat morphologique appliqué au canal V.
    rect_size: taille du structuring element (large, mince) pour accentuer les lignes.
    thresh: seuil pour binariser le résultat du blackhat.
    Retourne masque 0/255 uint8.
    """
    if v_channel is None:
        return None
    # kernel allongé pour capter lignes fines (poils)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, rect_size)
    bh = cv2.morphologyEx(v_channel, cv2.MORPH_BLACKHAT, k)
    # légère égalisation/blur pour réduire le bruit
    bh_blur = cv2.GaussianBlur(bh, (3, 3), 0)
    _, mask = cv2.threshold(bh_blur, thresh, 255, cv2.THRESH_BINARY)
    # optionnel: ouvrir/fermer pour affiner
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
    return mask.astype(np.uint8)


def detect_hairs_by_blackhat_simple(img_bgr: np.ndarray = None, v_channel: np.ndarray = None, rect_size=(31, 1), thresh: int = 12, dilate_iter: int = 1) -> np.ndarray:
    """Méthode simple: blackhat -> threshold -> skeleton -> dilate.
    Retourne un masque 0/255 uint8 adapté à l'inpainting des poils fins.
    Fournir soit img_bgr soit v_channel.
    """
    if v_channel is None:
        if img_bgr is None:
            return None
        v_channel = get_v_channel(img_bgr)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, rect_size)
    bh = cv2.morphologyEx(v_channel, cv2.MORPH_BLACKHAT, k)
    bh = cv2.GaussianBlur(bh, (3, 3), 0)
    _, mask = cv2.threshold(bh, thresh, 255, cv2.THRESH_BINARY)
    # small clean
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    # skeletonize then dilate slightly to get thin hair regions
    sk = skeletonize((mask > 0)).astype(np.uint8)
    sk = (sk * 255).astype(np.uint8)
    dil = cv2.dilate(sk, kernel_small, iterations=dilate_iter)
    return dil


def filter_false_positives(M: np.ndarray, img_shape, min_area_ratio: float = 0.01, max_area_ratio: float = 0.05, delta_max: float = None, delta_avg: float = None) -> np.ndarray:
    """Filtre les composantes connexes du masque M en appliquant des critères de taille et formes.
    Retourne M_f (0/255 uint8).
    """
    if M is None:
        return None

    h, w = img_shape[:2]
    area_thresh = max(1, int(min_area_ratio * h * w))
    diag = np.sqrt(h * h + w * w)
    if delta_max is None:
        delta_max = max(15, 0.02 * diag)
    if delta_avg is None:
        delta_avg = max(7, 0.01 * diag)

    labelled = label(M > 0)
    M_f = np.zeros_like(M, dtype=np.uint8)

    for region in regionprops(labelled):
        if region.area < area_thresh:
            continue
        # rejeter les composantes trop grandes (probables faux positifs)
        if region.area > max(1, int(max_area_ratio * h * w)):
            continue
        # extraire composante
        minr, minc, maxr, maxc = region.bbox
        comp = (labelled[minr:maxr, minc:maxc] == region.label).astype(np.uint8)

        # squelette
        sk = skeletonize(comp.astype(bool)).astype(np.uint8)
        sk_coords = np.column_stack(np.where(sk == 1))
        if sk_coords.shape[0] < 2:
            continue

    # approx d_max par bbox diagonal
        dy = maxr - minr
        dx = maxc - minc
        d_max = np.hypot(dy, dx)

        # skeleton length et endpoints
        sk_len = sk.sum()
        # voisins 8-connect pour trouver endpoints
        neigh = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
        endpoints = np.logical_and(sk == 1, neigh == 2)  # pixel lui-même + 1 voisin => neigh==2
        num_endpoints = max(1, int(endpoints.sum()))
        d_avg = sk_len / num_endpoints

        if (d_max < delta_max) or (d_avg < delta_avg):
            # filtrer (ne pas ajouter)
            continue

        # sinon réintégrer dans M_f (avec l'offset)
        M_f[minr:maxr, minc:maxc] = np.where(comp == 1, 255, M_f[minr:maxr, minc:maxc])

    return M_f


def inpaint_hairs(img_bgr: np.ndarray, mask: np.ndarray, inpaint_radius: int = 3, method: str = 'telea') -> np.ndarray:
    """Applique une dilatation légère au masque puis inpainting sur l'image couleur.
    method: 'telea' or 'ns' (Navier-Stokes)
    """
    if mask is None or img_bgr is None:
        raise ValueError('Image ou masque manquant pour inpainting')

    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1)
    inpaint_mask = (dil * 255).astype(np.uint8)

    flags = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
    result = cv2.inpaint(img_bgr, inpaint_mask, inpaint_radius, flags)
    return result


if __name__ == '__main__':
    # Script de démonstration : charge l'image, exécute la pipeline et sauvegarde les sorties
    im1 = cv2.imread(DATASET_PATH, cv2.IMREAD_COLOR)
    if im1 is None:
        print('Image non trouvée:', DATASET_PATH)
        raise SystemExit(1)
    print('Image chargée, shape =', im1.shape)

    v = get_v_channel(im1)
    masks = generate_threshold_masks(v)
    print('Nombre de masques T_i générés (sans doublons consécutifs) :', len(masks))

    M = build_global_mask(masks, kernel_size=5, lambda_=0.2)
    print('Masque global M calculé')

    # ajouter détection black-hat pour capturer les poils fins
    bh_mask = detect_hairs_by_blackhat(v, rect_size=(11, 1), thresh=12)
    if bh_mask is not None:
        M = cv2.bitwise_or(M, bh_mask)

    # détection simple blackhat->skeleton (capture très fines lignes)
    simple_bh = detect_hairs_by_blackhat_simple(im1, v_channel=v, rect_size=(31, 1), thresh=14, dilate_iter=1)
    if simple_bh is not None:
        M = cv2.bitwise_or(M, simple_bh)

    # filtrage plus permissif pour préserver les poils fins
    M_f = filter_false_positives(M, im1.shape, min_area_ratio=0.0001, max_area_ratio=0.02)
    print('Masque filtré M_f obtenu')

    # inpainting
    result = inpaint_hairs(im1, M_f, inpaint_radius=3, method='telea')
    print('Inpainting terminé')

    # sauvegarde
    out_dir = DATASET_PATH2
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, 'original.jpg'), im1)
    cv2.imwrite(os.path.join(out_dir, 'v_channel.png'), v)
    cv2.imwrite(os.path.join(out_dir, 'mask_M.png'), M)
    cv2.imwrite(os.path.join(out_dir, 'mask_Mf.png'), M_f)
    cv2.imwrite(os.path.join(out_dir, 'inpaint_result.png'), result)
    print('Fichiers sauvegardés dans', out_dir)

