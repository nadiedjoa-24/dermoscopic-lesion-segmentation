import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import label, find_objects
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

'''Je charge ici une image, en l'occurence ISIC_0000030.jpg'''
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVEPATH = os.path.normpath(os.path.join(BASE_DIR,'test'))

# construire un chemin absolu vers le dossier dataset, basé sur ce fichier

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


def detect_hairs_from_ti(ti: np.ndarray, kernel_size: int = 5, lambda_: float = 0.4) -> np.ndarray:
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



def filter_false_positives(M: np.ndarray,
                           min_component_ratio: float = 0.01,
                           mu: float = 0.02,
                           tau_min: int = 5,
                           tau_max: int = 30,
                           delta_max: float = 20,
                           delta_avg: float = 6) -> np.ndarray:
    """
    Filtre les faux positifs du masque M en gardant uniquement les composants ressemblant à des poils.

    Paramètres :
        M (np.ndarray) : Masque binaire des poils (0/255)
        min_component_ratio (float) : Seuil minimal de taille composante (en % de l’image)
        mu, tau_min, tau_max : Paramètres de l'élagage
        delta_max, delta_avg : Seuils pour rejeter les composantes non filiformes

    Retour :
        Mf (np.ndarray) : Masque final avec les vrais poils uniquement (0/255)
    """
    h, w = M.shape
    min_component_size = min_component_ratio * h * w #poils < surface = min_component_size supprimé

    # Binarisation (0 ou 1)
    M_bin = (M > 0).astype(np.uint8)

    # Labeliser les composantes connexes 8-connectées
    num_labels, labels = cv2.connectedComponents(M_bin, connectivity=8)

    # Image squelette final
    final_skeleton = np.zeros_like(M_bin, dtype=np.uint8)

    for label_id in range(1, num_labels):  # On ignore le label 0 (fond)
        component_mask = (labels == label_id).astype(np.uint8)
        

        # Vérifier la taille
        if np.sum(component_mask) < min_component_size:
            continue  # trop petit → ignoré

        # Skeletonisation
        skeleton = skeletonize(component_mask.astype(bool)).astype(np.uint8)

        # Calcul de la longueur du contour ∂C_i
        contour = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if len(contour) == 0:
            continue
        perimeter = cv2.arcLength(contour[0], closed=True)

        # Calcul τ
        tau = int(max(tau_min, min(mu * perimeter, tau_max)))

        # Pruning du squelette (en supprimant les branches courtes)
        # → ici, on supprime tous les pixels dont le voisinage 8-connexes < τ pixels
        pruned = np.zeros_like(skeleton)
        coords = np.column_stack(np.where(skeleton == 1))
        for y, x in coords:
            neighborhood = skeleton[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            if np.sum(neighborhood) >= tau:
                pruned[y, x] = 1

        if np.sum(pruned) == 0:
            continue  # squelette vide

        # Détection des jonctions (pixels avec ≥ 3 voisins)
        junctions = []
        for y, x in np.column_stack(np.where(pruned == 1)):
            neighborhood = pruned[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            if np.sum(neighborhood) >= 4:  # 3 voisins + le pixel central
                junctions.append((y, x))

        # Calcul des métriques géométriques
        if len(junctions) >= 2:
            distances = pdist(junctions)  # distances entre tous les couples
            d_max = np.max(distances)
        else:
            d_max = 0

        if len(junctions) > 0:
            d_avg = np.sum(pruned) / len(junctions)
        else:
            d_avg = 0

        # Filtrage
        if d_max < delta_max or d_avg < delta_avg:
            continue  # pas un poil : trop fragmenté ou irrégulier

        # Ajouter au squelette final
        final_skeleton += pruned

    # Reconstruction par disques autour du squelette filtré
    Mf = np.zeros_like(M_bin, dtype=np.uint8)
    dt_M = cv2.distanceTransform(M_bin * 255, cv2.DIST_L2, 3)
    coords = np.column_stack(np.where(final_skeleton == 1))

    for y, x in coords:
        r = int(dt_M[y, x])
        if r > 0:
            cv2.circle(Mf, (x, y), r, 1, -1)

    return (Mf * 255).astype(np.uint8)


def remove_hairs_by_inpainting(image: np.ndarray,
                                hair_mask: np.ndarray,
                                dilation_kernel_size: int = 3,
                                method: str = "telea") -> np.ndarray:
    """
    Supprime les poils de l'image à partir du masque binaire M_f via inpainting.

    Paramètres :
        image (np.ndarray) : Image couleur (BGR) d'origine
        hair_mask (np.ndarray) : Masque des poils (M_f), en binaire (0/255)
        dilation_kernel_size (int) : Taille du noyau pour dilatation isotrope
        method (str) : Méthode d'inpainting : "telea" ou "ns" (Navier-Stokes)

    Retour :
        image_clean (np.ndarray) : Image sans poils
    """
    if hair_mask is None or image is None:
        raise ValueError("Image ou masque non fourni.")

    if image.shape[:2] != hair_mask.shape:
        raise ValueError("Taille de l'image et du masque incompatibles.")

    # Dilatation isotrope 3x3 pour couvrir pénombres
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_kernel_size, dilation_kernel_size))
    dilated_mask = cv2.dilate(hair_mask, kernel, iterations=1)

    # Choix de la méthode d’inpainting
    inpaint_flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS

    # Inpainting (OpenCV attend un masque en 8-bit avec valeurs 0 ou 255)
    image_clean = cv2.inpaint(image, dilated_mask, inpaintRadius=3, flags=inpaint_flag)

    return image_clean



"""t0 = t[200]
cv2.imwrite(os.path.join(DATASET_PATH2, "masquet200.jpg"), t0)

Gi = detect_hairs_from_ti(t0)
cv2.imwrite(os.path.join(DATASET_PATH2, "poilss.jpg"), Gi)

Mf = filter_false_positives(Gi)
cv2.imwrite(os.path.join(DATASET_PATH2, "fauxpositif.jpg"), Mf)"""




'''# Étape 1 : Générer tous les masques G_i
G_list = []
for i, ti in enumerate(t[190:210]):
    Gi = detect_hairs_from_ti(ti, kernel_size=5, lambda_=0.2)
    G_list.append(Gi)

# Étape 2 : Fusionner tous les masques G_i → masque global M
M = np.zeros_like(G_list[0], dtype=np.uint8)
for Gi in G_list:
    M = cv2.bitwise_or(M, Gi)

cv2.imwrite(os.path.join(DATASET_PATH2, "fusion.jpg"), M)

Mf = filter_false_positives(M)
cv2.imwrite(os.path.join(DATASET_PATH2, "masque_filtré_Mf.jpg"), Mf)"""

# Sauvegarde (optionnel pour debug)
cv2.imwrite(os.path.join(DATASET_PATH2, "masque_global_M.jpg"), M)

# Étape 3 : Filtrage des faux positifs → M_f
Mf = filter_false_positives(M)
cv2.imwrite(os.path.join(DATASET_PATH2, "masque_filtré_Mf.jpg"), Mf)

# Étape 4 : Inpainting sur l’image originale
image_sans_poils = remove_hairs_by_inpainting(im1, Mf, dilation_kernel_size=3, method="telea")
cv2.imwrite(os.path.join(DATASET_PATH2, "image_sans_poils.jpg"), image_sans_poils)'''


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.normpath(os.path.join(BASE_DIR,'..', 'dataset', 'melanoma' , 'ISIC_0000140.jpg'))
    im1 = cv2.imread(DATASET_PATH, cv2.IMREAD_COLOR)
    if im1 is not None:
        im2 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    else:
        im2 = None

    SAVEPATH = os.path.normpath(os.path.join(BASE_DIR,'test'))



    im_crop = cadre_noire(im1)

    if im1 is None:
        raise ValueError("im1 is None")
    if im_crop is None:
        raise ValueError("im_crop is None")


    DATASET_PATH2 = os.path.normpath(os.path.join(BASE_DIR,'..', 'dataset', 'melanoma' , 'ISIC_0000046.jpg'))
    im3 = cv2.imread(DATASET_PATH2, cv2.IMREAD_COLOR)
    if im3 is not None:
        im4 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
    else:
        im4 = None
    
    v = luminance(im3)
    t = genere_masque(v)

    t1 = t[200]

    G_list = []
    for i, ti in enumerate(t[195:205]):
        Gi = detect_hairs_from_ti(ti, kernel_size=5, lambda_=0.2)
        G_list.append(Gi)

    

    # Étape 2 : Fusionner tous les masques G_i → masque global M
    M = np.zeros_like(G_list[0], dtype=np.uint8)
    for Gi in G_list:
        M = cv2.bitwise_or(M, Gi)

    Mfiltré = filter_false_positives(M)

    





    # Convert BGR -> RGB pour l'affichage des images couleur
    img1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB) if im1 is not None else None
    img_crop_rgb = cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB) if im_crop is not None else None

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Ligne du haut : images liées au crop
    axes[0, 0].imshow(img1_rgb)
    axes[0, 0].set_title("Original (im1)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_crop_rgb)
    axes[0, 1].set_title("Cropped (im_crop)")
    axes[0, 1].axis("off")

    # Troisième emplacement en haut vide
    axes[0, 2].axis("off")

    # Ligne du bas : masks liés aux hair detection
    axes[1, 0].imshow(im4, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("image originale im3")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(M, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Masque combiné")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(Mfiltré, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title("M — filtered hair mask (Mf)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()
