# main.py
import os
from skimage import io

from crop import isolate_dermato_circle_adaptive
from dualrazor import hair_removal

BASEDIR = os.path.dirname(os.path.abspath(__file__))

# ⚠ Vérifie bien le nom du fichier, il y avait un espace avant ".jpg" dans ton message
IMG_PATH = os.path.normpath(
    os.path.join(BASEDIR, '..', 'dataset', 'melanoma', 'ISIC_0000140.jpg')
)

SAVE_PATH = os.path.normpath(os.path.join(BASEDIR, 'test'))
os.makedirs(SAVE_PATH, exist_ok=True)

if __name__ == "__main__":
    # 1) Chargement de l'image (en RGB avec skimage)
    img_rgb = io.imread(IMG_PATH)

    # 2) Crop du cadre noir / cercle dermatoscope
    img_cropped, mask = isolate_dermato_circle_adaptive(img_rgb, crop=True)

    # 3) Hair removal sur l'image croppée
    img_no_hair = hair_removal(img_cropped)

    # 4) Sauvegarde des résultats
    out_cropped_path = os.path.join(SAVE_PATH, "ISIC_0000146_cropped.png")
    out_clean_path   = os.path.join(SAVE_PATH, "ISIC_0000146_nohair.png")

    io.imsave(out_cropped_path, img_cropped)
    io.imsave(out_clean_path, img_no_hair)

    print("Images sauvegardées :")
    print(" -", out_cropped_path)
    print(" -", out_clean_path)
