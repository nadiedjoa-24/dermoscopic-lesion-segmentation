import numpy as np
import cv2
from skimage.morphology import skeletonize

'''Je charge ici une image, en l'occurence ISIC_0000030.jpg'''
import os

# construire un chemin absolu vers le dossier dataset, basé sur ce fichier
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.normpath(os.path.join(BASE_DIR,'..', 'dataset', 'melanoma' , 'ISIC_0000146.jpg'))
DATASET_PATH2 = os.path.normpath(os.path.join(BASE_DIR, 'test'))
im1 = cv2.imread(DATASET_PATH, cv2.IMREAD_COLOR)
if im1 is not None:
    im2 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
else:
    im2 = None

if im1 is None:
    print("image pas chargée")
else:
    print(im1.shape[:2])

'''la fonction cadre noire permet d'enlever le cadre noir qui apparait autour d'une melanome 
lorsqu'on le capture, permet de réduire le nombre de pixel noir et donc son impact sur l'algo utilisé'''

def cadre_noire(img):
    #Calcul de lightness
    
    pixel_noir = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            max_rgb = float(np.max(img[i, j]))
            min_rgb = float(np.min(img[i,j]))
            lightness = (max_rgb +min_rgb)/2.0

            if lightness < 20:
                pixel_noir[i,j] = lightness
            else:
                pixel_noir[i,j] = 255
            

    print(pixel_noir)


    cv2.imwrite("masquenoir.jpg", pixel_noir.astype(np.uint8))
    #Image/masque binaire, compare chaque pixel à 20

    def cherche_retrait_indes(mask_noir = pixel_noir, axis=0):
        if axis == 0:
            start, end = 0, img.shape[0]
            taille = img.shape[0]
        else:
            start, end = 0, img.shape[1]
            taille = img.shape[1]


        for i in range(taille): 
            if axis == 0:
                ligne = mask_noir[i, :]
            else:
                ligne = mask_noir[:, i] #en realité colonne mais on dit ligne pour que ça soit plus simple
            
            k = 0
            for j in range(len(ligne)):
                if ligne[j] != 255:
                    k = k+1
                

            if k/(len(ligne)) < 0.6:
                start = i
                break
        
        print(start)

        for i in reversed(range(taille)): #on commence du bas ou de la droite selon la valeur de axis
            if axis == 0:
                ligne = mask_noir[i, :]
            else:
                ligne = mask_noir[:, i]

            k = 0
            for j in range(len(ligne)):
                if ligne[j] != 255:
                    k = k+1

            if k/(len(ligne)) < 0.6:
                end = i+1
                break

        print(end)

        return start,end
    
    top, bottom = cherche_retrait_indes(pixel_noir, axis=0)
    left, right = cherche_retrait_indes(pixel_noir, axis=1)

    cropped_img = img[top:bottom, left:right]
    return cropped_img

'im1_coupé = cadre_noire(im1)'
'cv2.imwrite("image_coupé.png", im1_coupé)'

''' hair removal''' " on implémente un algorithme qui permet d'effacer les poils sur l'image"

def luminance(im=None): #renvoie le canal V (= Luminance ici) du domaine HSV de notre image

    if im is None:
        raise ValueError("il n'y a pas d'image")

    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    v_channel = im_hsv[:, :, 2]
    return v_channel

def genere_masque(v_channel): #crée les 256 couches binaires à partir de la luminance
    #
    if v_channel is None or len(v_channel.shape) != 2:
        raise ValueError("L'entrée doit être une image 2D (canal V).")

    thresholds = []

    for i in range(256):
        _, t_i = cv2.threshold(v_channel, i, 255, cv2.THRESH_BINARY)
        thresholds.append(t_i)

    return thresholds

v = luminance(im1)
t = genere_masque(v)


chemin_original = os.path.join(DATASET_PATH2, "original.jpg")
chemin_modif = os.path.join(DATASET_PATH2, "modif.jpg")
cv2.imwrite(chemin_modif, t[100])

def detect_hairs_from_ti(ti: np.ndarray, kernel_size: int = 5, lambda_: float = 0.2) -> np.ndarray:
#detecte les poils dans le masque ti, via closing, skeleton, distance transform et reconstruction par disque
    if ti is None or not isinstance(ti, np.ndarray):
        raise ValueError("Entrée invalide pour Ti.")
    
    # Conversion en binaire 0/1, ne pas garder binaire 0;255
    ti_bin = (ti > 0).astype(np.uint8)

    # Structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)) #disque de rayon kernel_size/2 où on va appliquer les opérations morphologiques

    # Morphological closing pour poils sombres 
    opened = cv2.morphologyEx(ti_bin, cv2.MORPH_OPEN, kernel) #érosion (réduction) + dilatation (grossissement) => enlève les points blancs dans l'image
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel) #dilatation + érosion => enlève points noirs hors de l'image
    
    omega_oc = closed.copy()

    # Morphological opening pour Ω_co (pour DT plus généreux) 
    closed2 = cv2.morphologyEx(ti_bin, cv2.MORPH_CLOSE, kernel)
    omega_co = cv2.morphologyEx(closed2, cv2.MORPH_OPEN, kernel)
    chemin_close = os.path.join(DATASET_PATH2, "close.jpg")
    cv2.imwrite(chemin_close, closed2 * 255)
    chemin_open = os.path.join(DATASET_PATH2, "open.jpg")
    cv2.imwrite(chemin_open, omega_co * 255)

    # Skeletonisation de Ω_oc 
    skeleton = skeletonize(omega_oc.astype(bool)).astype(np.uint8)  # 0/1
    chemin_open = os.path.join(DATASET_PATH2, "skeleton.jpg")
    cv2.imwrite(chemin_open, skeleton * 255)

    # Distance Transform sur Ω_oc et Ω_co 
    dt_oc = cv2.distanceTransform((omega_oc * 255).astype(np.uint8), cv2.DIST_L2, 3)
    dt_co = cv2.distanceTransform((omega_co * 255).astype(np.uint8), cv2.DIST_L2, 3)

    # Fusion des DTs pour obtenir ρ(x) 
    rho = (1 - lambda_) * dt_co + lambda_ * dt_oc

    # Reconstruction par disques centrés sur squelette 
    Gi_mask = np.zeros_like(ti_bin, dtype=np.uint8)

    skeleton_points = np.column_stack(np.where(skeleton == 1))
    h, w = ti_bin.shape

    for y, x in skeleton_points:
        r = int(rho[y, x])
        if r > 0:
            cv2.circle(Gi_mask, (x, y), r, 1, -1)  # remplissage disque

    #  Éliminer les régions internes à la peau : G_i = D \ Ω_oc
    Gi_mask = cv2.subtract(Gi_mask, omega_oc)

    # Format final en 0/255
    Gi_mask = (Gi_mask * 255).astype(np.uint8)

    return Gi_mask

h = detect_hairs_from_ti(t[100])
chemin_open = os.path.join(DATASET_PATH2, "final.jpg")
cv2.imwrite(chemin_open, h)


