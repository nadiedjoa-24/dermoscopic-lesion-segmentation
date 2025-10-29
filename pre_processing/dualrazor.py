import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.normpath(os.path.join(BASEDIR,'..','dataset','melanoma','ISIC_0000140.jpg'))
SAVE_PATH = os.path.normpath(os.path.join(BASEDIR, 'test'))

im1 = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
im2 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

#je definis les élements structurels que j'vais utiliser

Ma = np.array([0,1,1,1,1,1,1,1,1,1,1,1,0], dtype=np.uint8)
Mb = np.array([[0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]], dtype = np.uint8)
Mc = np.array([[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0]], dtype = np.uint8)

M = [Ma, Mb, Mc]

#j'extraie les canaux R,G,B de mon image

R = im2[:,:,0]
G = im2[:,:,1]
B = im2[:,:,2]

def compute_mask(canal, struct, thresh):
    L = [cv2.morphologyEx(canal, cv2.MORPH_CLOSE, x) for x in struct]
    maxL = np.maximum.reduce(L)
    G = cv2.subtract(canal, maxL)
    M = np.where(G > thresh, 255, 0).astype(np.uint8) #np.where : renvoie un tableau de la meme taille que G, avec comme element 255 si la condition est vérifié, 0 sinon
    return M

Mr = compute_mask(R, M, 10)
Mg = compute_mask(G, M, 10)
Mb = compute_mask(B, M, 10)

Mfinal = cv2.merge([Mr, Mg, Mb])

plt.imshow(cv2.cvtColor(Mfinal, cv2.COLOR_BGR2RGB))
plt.show()

