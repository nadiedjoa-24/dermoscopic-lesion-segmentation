# dualrazor.py
import numpy as np
import cv2
from skimage import morphology, filters

from scipy import interpolate


def compute_mask(canal, struct, thresh):
    L = [cv2.morphologyEx(canal, cv2.MORPH_CLOSE, x) for x in struct]
    maxL = np.maximum.reduce(L)
    G = abs(canal - maxL)
    M = np.where(G > thresh, 255, 0).astype(np.uint8)
    return M 


def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    image: 2D
    mask: 2D bool, True = pixel manquant (poil)
    """
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


def hair_removal(img_rgb):
    """
    img_rgb : image RGB (np.ndarray, HxWx3)
    retourne : image RGB avec poils supprimés
    """

    # Lissage léger
    im3 = cv2.GaussianBlur(img_rgb, (5, 5), 1)

    # éléments structurants
    Ma = np.array([0,1,1,1,1,1,1,1,1,1,1,1,0], dtype=np.uint8)
    Mb = np.zeros((9, 9), dtype=np.uint8)
    for i in range(1, 8):
        Mb[i, i] = 1   
    Mc = np.array([[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0]], dtype=np.uint8)
    struct_list = [Ma, Mb, Mc]

    # canaux
    R = im3[:,:,0]
    G = im3[:,:,1]
    B = im3[:,:,2]

    # masques de poils
    Mr = compute_mask(R, struct_list, 250)
    Mg = compute_mask(G, struct_list, 250)
    Mbm = compute_mask(B, struct_list, 250)
    # fusion
    Mfinal = (Mr + Mg + Mbm) > 0   # booléen

    # interpolation par canal
    newR = interpolate_missing_pixels(R, Mfinal)
    newG = interpolate_missing_pixels(G, Mfinal)
    newB = interpolate_missing_pixels(B, Mfinal)

    # filtre médian pour lisser
    square = morphology.rectangle(20, 20)
    Rf = filters.median(newR, square)
    Gf = filters.median(newG, square)
    Bf = filters.median(newB, square)

    final_img = np.copy(img_rgb)
    final_img[:,:,0] = Rf
    final_img[:,:,1] = Gf
    final_img[:,:,2] = Bf

    return final_img
