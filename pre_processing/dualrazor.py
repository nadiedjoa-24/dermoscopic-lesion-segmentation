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


Md = np.zeros((9, 9), dtype=np.uint8)
for i in range(1, 8):  # lignes de 1 à 7
    Md[i, i] = 1   

Mc = np.array([[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0]], dtype = np.uint8)

M = [Ma, Md, Mc]

#j'extraie les canaux R,G,B de mon image

R = im2[:,:,0]
G = im2[:,:,1]
B = im2[:,:,2]

def compute_mask(canal, struct, thresh):
    L = [cv2.morphologyEx(canal, cv2.MORPH_CLOSE, x) for x in struct]
    maxL = np.maximum.reduce(L)
    G = abs(canal - maxL)
    M = np.where(G > thresh, 255, 0).astype(np.uint8) #np.where : renvoie un tableau de la meme taille que G, avec comme element 255 si la condition est vérifié, 0 sinon
    return M

Mr = compute_mask(R, M, 250)
Mg = compute_mask(G, M, 250)
Mb = compute_mask(B, M, 250)


Mfinal = cv2.merge([Mr, Mg, Mb])


test = cv2.morphologyEx(R, cv2.MORPH_CLOSE, Ma)
test1 = cv2.morphologyEx(R, cv2.MORPH_CLOSE, Md)
test3 = cv2.morphologyEx(R, cv2.MORPH_CLOSE, Mc)
maxL = np.maximum.reduce([test, test1, test3])
G = abs(R - maxL)

'''
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(test, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('test (R, Ma)')
axes[0].axis('off')

axes[1].imshow(test1, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('test1 (B, Mb)')
axes[1].axis('off')

axes[2].imshow(test3, cmap='gray', vmin=0, vmax=255)
axes[2].set_title('test3 (G, Mc)')
axes[2].axis('off')

plt.tight_layout()
plt.show()
'''

# afficher Mfinal
plt.figure(figsize=(8, 8))
plt.imshow(Mfinal)
plt.title('Mfinal')
plt.axis('off')
plt.tight_layout()
plt.show()

