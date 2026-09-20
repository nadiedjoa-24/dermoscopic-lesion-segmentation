"""LBP Clustering segmentation, after Pereira et al. (2020).

The method assumes lesion skin is texturally busier than healthy skin. Local
Binary Patterns with P=8 neighbours at radius R=1 label each pixel by the sign
pattern of its neighbourhood; the flat patterns (code 0 and the eight powers of
two, i.e. a single neighbour brighter than the centre) mark smooth skin, and
everything else marks texture.

Binarising on that subset gives a texture map, which is smoothed into a
continuous field L, stacked with the luminance Y as a pseudo-RGB image
``[L, Y, L]``, and converted to CIE L*a*b*. Clustering the (a*, b*) chromaticity
of that synthetic image separates lesion from skin, and the *pinkness* score
``max(a*, 0) - min(b*, 0)`` picks the lesion cluster without supervision.

See ``docs/references.md``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.color import rgb2lab
from skimage.transform import resize
from sklearn.cluster import KMeans

from .base import CALC_SIZE, SegmentationResult, downscale, upscale_mask

RANDOM_STATE = 42
# BT.601 luminance weights.
LUMA_WEIGHTS = (0.299, 0.587, 0.114)


def luminance_bt601(image: np.ndarray) -> np.ndarray:
    """Luminance Y on a 0..255 scale, as the LBP stage expects."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("expected an (H, W, 3) RGB image")

    scaled = image.astype(np.float32)
    if scaled.max() <= 1.0001:
        scaled = scaled * 255.0

    weights = np.array(LUMA_WEIGHTS, dtype=np.float32)
    return (scaled[..., :3] @ weights).astype(np.float32)


def local_binary_pattern_p8r1(luminance: np.ndarray, *, pad_mode: str = "reflect") -> np.ndarray:
    """LBP codes for P=8, R=1, written out rather than called from scikit-image.

    The eight neighbours are compared to the centre with a strict ``>``, matching
    the paper's ``s(u) = 1[u > 0]``, and the bits are packed clockwise from north.
    """
    if luminance.ndim != 2:
        raise ValueError("expected a 2D luminance image")

    padded = np.pad(luminance.astype(np.float32), 1, mode=pad_mode)
    centre = padded[1:-1, 1:-1]

    neighbours = (
        padded[0:-2, 1:-1],  # N
        padded[0:-2, 2:],    # NE
        padded[1:-1, 2:],    # E
        padded[2:, 2:],      # SE
        padded[2:, 1:-1],    # S
        padded[2:, 0:-2],    # SW
        padded[1:-1, 0:-2],  # W
        padded[0:-2, 0:-2],  # NW
    )

    codes = np.zeros(centre.shape, dtype=np.uint8)
    for bit, neighbour in enumerate(neighbours):
        codes |= ((neighbour > centre).astype(np.uint8) << bit)

    return codes


def binarize_patterns(codes: np.ndarray) -> np.ndarray:
    """Mark textured pixels: 0 for the flat pattern subset, 1 for everything else."""
    flat_patterns = np.concatenate([[0], [2**bit for bit in range(8)]]).astype(np.uint8)
    return (~np.isin(codes, flat_patterns)).astype(np.uint8)


def pinkness(a_star: np.ndarray, b_star: np.ndarray) -> float:
    """Paper's cluster score: reddish and not yellowish."""
    return float(np.mean(np.maximum(a_star, 0) - np.minimum(b_star, 0)))


def segment(
    image: np.ndarray,
    *,
    sigma: float = 3.0,
    disk_size: int = 3,
    valid_mask: np.ndarray | None = None,
) -> SegmentationResult:
    """Segment a preprocessed RGB image with LBP clustering and the pinkness rule.

    Args:
        image: Preprocessed RGB image.
        sigma: Width of the Gaussian that turns the texture map into the field L.
        disk_size: Structuring element radius for the opening.
        valid_mask: Boolean mask of pixels carrying real image data. Frame
            removal whitens the corners outside the dermoscope disc, and in the
            pseudo-RGB those white pixels turn pure green, an extreme in (a*, b*).
            Left in, they capture one of the two clusters outright and k-means
            separates disc from corners instead of lesion from skin. Restricting
            the fit to the valid pixels is what keeps the paper's criterion
            meaningful on cropped images.
    """
    original_shape = image.shape[:2]
    small = downscale(image)

    if valid_mask is None:
        valid_small = np.ones(CALC_SIZE, dtype=bool)
    else:
        valid_small = (
            resize(valid_mask.astype(float), CALC_SIZE, order=0, mode="edge", anti_aliasing=False)
            > 0.5
        )

    luminance = luminance_bt601(small)
    codes = local_binary_pattern_p8r1(luminance)
    texture = binarize_patterns(codes)

    smoothed = gaussian_filter(texture.astype(np.float32), sigma=sigma)
    if smoothed.max() > 0:
        smoothed = smoothed / smoothed.max()

    pseudo_rgb = np.stack([smoothed * 255.0, luminance, smoothed * 255.0], axis=-1)
    chromaticity = rgb2lab(pseudo_rgb / 255.0)[..., 1:]

    features = chromaticity.reshape(-1, 2)
    kmeans = KMeans(n_clusters=2, init="k-means++", n_init=3, max_iter=100, random_state=RANDOM_STATE)
    kmeans.fit(features[valid_small.reshape(-1)])
    labels = kmeans.predict(features).reshape(CALC_SIZE)

    scores = []
    for cluster in range(2):
        members = (labels == cluster) & valid_small
        if not members.any():
            scores.append(-np.inf)
            continue
        scores.append(pinkness(chromaticity[members, 0], chromaticity[members, 1]))

    lesion_cluster = int(np.argmax(scores))
    raw = ((labels == lesion_cluster) & valid_small).astype(np.uint8)
    opened = morphology.opening(raw, morphology.disk(disk_size))

    mask = upscale_mask(opened, original_shape)

    return SegmentationResult(
        mask=mask,
        steps={
            "Downscaled input": small,
            "Luminance Y (BT.601)": luminance,
            "LBP codes (P=8, R=1)": codes,
            "Texture patterns (binarised)": texture,
            f"Gaussian smoothing (sigma={sigma})": smoothed,
            "Pseudo-RGB [L, Y, L]": pseudo_rgb.astype(np.uint8),
            "Chromaticity a*": chromaticity[..., 0],
            "Chromaticity b*": chromaticity[..., 1],
            "K-means clusters": labels,
            "Lesion cluster (pinkness)": raw,
            "Morphological opening": opened,
        },
        info={"pinkness_scores": scores, "lesion_cluster": lesion_cluster, "calc_size": CALC_SIZE},
    )
