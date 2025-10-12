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

im=skio.imread('dataset/ISIC_0000030.jpg')
im_luminance = luminance_bt601(im)
print(im_luminance)




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
