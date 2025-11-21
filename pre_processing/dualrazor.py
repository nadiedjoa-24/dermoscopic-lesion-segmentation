import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage import morphology, filters 
from scipy import interpolate
import skimage
from skimage import io



BASEDIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.normpath(os.path.join(BASEDIR,'..','dataset','nevus','ISIC_0000095.jpg'))
SAVE_PATH = os.path.normpath(os.path.join(BASEDIR, 'test'))

# ISIC_0000140.jpg ; ISIC_0000145.jpg ; ISIC_0000150.jpg ; ISIC_0000146.jpg

im_orig = io.imread(IMG_PATH)


im1 = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
im2 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

im3 = cv2.GaussianBlur(im2, (5,5), 1)

#je definis les élements structurels que j'vais utiliser

Ma = np.array([0,1,1,1,1,1,1,1,1,1,1,1,0], dtype=np.uint8)
Mb = np.zeros((9, 9), dtype=np.uint8)
for i in range(1, 8):  # lignes de 1 à 7
    Mb[i, i] = 1   

Mc = np.array([[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0]], dtype = np.uint8)
M = [Ma, Mb, Mc]


#j'extraie les canaux R,G,B de mon image

R = im3[:,:,0]
G = im3[:,:,1]
B = im3[:,:,2]


def compute_mask(canal, struct, thresh):
    L = [cv2.morphologyEx(canal, cv2.MORPH_CLOSE, x) for x in struct]
    maxL = np.maximum.reduce(L)
    G = abs(canal - maxL)
    M = np.where(G > thresh, 255, 0).astype(np.uint8) #np.where : renvoie un tableau de la meme taille que G, avec comme element 255 si la condition est vérifié, 0 sinon
    return M 

Mr = compute_mask(R, M, 250)
Mg = compute_mask(G, M, 250)
Mb = compute_mask(B, M, 250)
Mfinal = Mr + Mg + Mb

# 
def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h)) #crée une grille de coordonnée (x,y) pour chq pixel

    known_x = xx[~mask]     #mask = true => pixel manquant ; mask = false => pixel connue
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values #on remplace les pixels manquant avec leurs valeurs interpolés

    return interp_image

newR = interpolate_missing_pixels(R, Mr > 0)
newB = interpolate_missing_pixels(B, Mb > 0)
newG = interpolate_missing_pixels(G, Mg > 0)

square = morphology.rectangle(20, 20)
Rf = skimage.filters.median(newR, square)
Gf = skimage.filters.median(newG, square)
Bf = skimage.filters.median(newB, square)

final_img = np.copy(im2)
final_img[:,:,0]=Rf
final_img[:,:,1]=Gf
final_img[:,:,2]=Bf


# affiche Mfinal et Mfinal2 côte à côte
plt.figure(figsize=(10, 5))
Mdisp1 = np.clip(Mfinal, 0, 255).astype(np.uint8)
#Mdisp2 = np.clip(Mfinall, 0, 255).astype(np.uint8)

plt.subplot(1, 2, 1)
plt.imshow(im_orig, cmap='gray', vmin=0, vmax=255)
plt.title('Image Originale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(final_img, cmap='gray', vmin=0, vmax=255)
plt.title('Image sans poils')
plt.axis('off')

plt.tight_layout()
plt.show()




