import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'test'))
DEBUG_DIR = os.path.join(TEST_DIR, 'debug')
os.makedirs(DEBUG_DIR, exist_ok=True)

paths = {
    'original': os.path.join(TEST_DIR, 'original.jpg'),
    'v_channel': os.path.join(TEST_DIR, 'v_channel.png'),
    'M': os.path.join(TEST_DIR, 'mask_M.png'),
    'Mf': os.path.join(TEST_DIR, 'mask_Mf.png'),
    'inpaint': os.path.join(TEST_DIR, 'inpaint_result.png')
}

for k, p in paths.items():
    print(f"{k}: {p} ->", 'FOUND' if os.path.exists(p) else 'MISSING')

# load images (some may be missing)
orig = cv2.imread(paths['original'])
M = cv2.imread(paths['M'], cv2.IMREAD_GRAYSCALE)
Mf = cv2.imread(paths['Mf'], cv2.IMREAD_GRAYSCALE)
inpaint = cv2.imread(paths['inpaint'])

h, w = orig.shape[:2]
print('Original shape:', (h, w))

def stats_mask(mask, name):
    if mask is None:
        print(f"{name}: None")
        return None
    total = mask.size
    nonzero = int(np.count_nonzero(mask))
    pct = nonzero / total
    print(f"{name}: nonzero={nonzero} pixels ({pct:.6f} fraction)")

    # connected components
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask>0).astype('uint8'), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if numLabels>1 else np.array([])
    print(f"  components (excluding background): {len(areas)}")
    if len(areas)>0:
        idx = np.argsort(areas)[::-1]
        top = idx[:10]
        for rank, i in enumerate(top):
            a = areas[i]
            bbox = stats[1 + i, cv2.CC_STAT_LEFT], stats[1 + i, cv2.CC_STAT_TOP], stats[1 + i, cv2.CC_STAT_WIDTH], stats[1 + i, cv2.CC_STAT_HEIGHT]
            print(f"   #{rank+1}: area={a}, bbox={bbox}")
    return locals()

sM = stats_mask(M, 'M')
sMf = stats_mask(Mf, 'Mf')

# overlay masks
def overlay_mask_on_image(img, mask, color=(0,0,255), alpha=0.6):
    if img is None or mask is None:
        return None
    overlay = img.copy()
    m3 = np.zeros_like(img)
    m3[:, :, 0] = (mask>0).astype('uint8') * color[0]
    m3[:, :, 1] = (mask>0).astype('uint8') * color[1]
    m3[:, :, 2] = (mask>0).astype('uint8') * color[2]
    cv2.addWeighted(m3, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

ovM = overlay_mask_on_image(orig, M, color=(0,0,255), alpha=0.6)
ovMf = overlay_mask_on_image(orig, Mf, color=(0,255,0), alpha=0.6)

if ovM is not None:
    cv2.imwrite(os.path.join(DEBUG_DIR, 'overlay_M.png'), ovM)
if ovMf is not None:
    cv2.imwrite(os.path.join(DEBUG_DIR, 'overlay_Mf.png'), ovMf)

# save zooms around largest components of Mf
if Mf is not None:
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats((Mf>0).astype('uint8'), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if numLabels>1 else np.array([])
    if len(areas)>0:
        idx = np.argsort(areas)[::-1]
        for rank, i in enumerate(idx[:6]):
            left = stats[1 + i, cv2.CC_STAT_LEFT]
            top = stats[1 + i, cv2.CC_STAT_TOP]
            width = stats[1 + i, cv2.CC_STAT_WIDTH]
            height = stats[1 + i, cv2.CC_STAT_HEIGHT]
            pad = 10
            x0 = max(0, left - pad)
            y0 = max(0, top - pad)
            x1 = min(w, left + width + pad)
            y1 = min(h, top + height + pad)
            crop_orig = orig[y0:y1, x0:x1]
            crop_mask = Mf[y0:y1, x0:x1]
            # overlay crop
            ov = overlay_mask_on_image(crop_orig, crop_mask, color=(0,255,0), alpha=0.6)
            cv2.imwrite(os.path.join(DEBUG_DIR, f'zoom_Mf_{rank+1}.png'), ov)

print('Debug images saved to', DEBUG_DIR)
print('Done')
