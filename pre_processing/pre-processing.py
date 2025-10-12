import numpy as np
import cv2
from skimage.morphology import skeletonize

'''Je charge ici une image, en l'occurence ISIC_0000030.jpg'''
import os

# construire un chemin absolu vers le dossier dataset, basé sur ce fichier
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'dataset', 'ISIC_0000030.jpg'))

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

''' hair removal'''

def luminance(im=None):

    if im is None:
        raise ValueError("il n'y a pas d'image")

    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    v_channel = im_hsv[:, :, 2]
    return v_channel

def genere_masque(v_channel):
    #
    if v_channel is None or len(v_channel.shape) != 2:
        raise ValueError("L'entrée doit être une image 2D (canal V).")

    thresholds = []

    for i in range(256):
        _, t_i = cv2.threshold(v_channel, i, 255, cv2.THRESH_BINARY)
        thresholds.append(t_i)

    return thresholds

def detect_gaps_in_ti(ti , kernel_size: int = 5):
    """
    Applique un closing sur le masque Ti pour combler les poils sombres,
    puis extrait le squelette des fragments de poils détectés.

    Paramètres :
        ti_mask (np.ndarray) : Image binaire Ti (uint8, valeurs 0 et 255)
        kernel_size (int) : Taille du disque structurant pour le closing

    Retour :
        gap_skeleton (np.ndarray) : Image binaire (0 ou 1) contenant le squelette des gaps
    """
    if ti is None:
        raise ValueError("Le masque Ti est None.")

    # s'assurer que ti est un tableau numpy
    if not isinstance(ti, np.ndarray):
        raise ValueError("Le masque Ti doit être un numpy.ndarray.")

    # convertir en binaire uint8 (0 ou 1)
    ti_bin = (ti > 0).astype(np.uint8)

    # Structuring element en disque
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Opening puis Closing pour lisser / combler
    opened = cv2.morphologyEx(ti_bin, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    omega_oc = closed.copy()

    # Skeletonisation attend un tableau booléen
    omega_bool = omega_oc.astype(bool)
    skeleton = skeletonize(omega_bool).astype(np.uint8)  # 0/1

    # Extraire les fragments de squelette qui ne sont pas dans Ti
    gap_skeleton = cv2.subtract(skeleton, ti_bin)

    # Retour en uint8 0/255 pour être visible et sauvegardable
    gap_skeleton_255 = (gap_skeleton * 255).astype(np.uint8)
    return gap_skeleton_255

v = luminance(im1)
t = genere_masque(v)
print(len(t))
# s'assurer que le dossier test existe (placé au même niveau que dataset)
TEST_DIR = os.path.normpath(os.path.join(BASE_DIR,'test'))
os.makedirs(TEST_DIR, exist_ok=True)


# préparer l'image à écrire: convertir en uint8 si nécessaire
t0 = detect_gaps_in_ti(t[120], 5)

if t0 is None:
    raise ValueError('t[0] est None, impossible de sauvegarder')

if t0.dtype == np.bool_:
    t0 = (t0.astype(np.uint8) * 255)
elif t0.dtype != np.uint8:
    t0 = np.clip(t0, 0, 255).astype(np.uint8)

out_path = os.path.join(TEST_DIR, 'image_coupé.png')
ok = cv2.imwrite(out_path, t0)
print(f"Écriture de {out_path} :", ok)

