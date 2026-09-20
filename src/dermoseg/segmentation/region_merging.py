"""Region-merging segmentation, after Celebi et al. (2008).

The image is over-segmented into homogeneous regions, then the regions making
up the lesion are selected by a score. Two over-segmentation backends are
available:

``srm``
    Statistical Region Merging (Nock and Nielsen), implemented here from the
    paper: pixel pairs are visited in order of increasing intensity difference
    and merged through a union-find structure whenever their region means
    differ by less than a statistical bound that tightens as regions grow.
    This is the algorithm Celebi et al. applied to dermoscopy.

``felzenszwalb``
    Felzenszwalb's graph-based segmentation, as a faster reference point. It is
    a different algorithm and is reported separately, never as SRM.

Region selection reuses the observation from Zortea et al. (2017) that a band
along the image border is mostly skin, which gives a reference value without
any training.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.segmentation import felzenszwalb

from .base import CALC_SIZE, SegmentationResult, downscale, upscale_mask

# Region-selection weights: contrast dominates, centrality and colour assist.
WEIGHT_CONTRAST = 0.5
WEIGHT_CENTRALITY = 0.3
WEIGHT_SATURATION = 0.2
SATURATION_GAIN = 2.0
# A region within this fraction of the best score is kept if it is also
# meaningfully darker or more saturated than skin.
SCORE_KEEP_RATIO = 0.60
MIN_CONTRAST = 0.005
MIN_SATURATION_DIFF = 0.1
SKIN_PERCENTILE = 70


def statistical_region_merging(gray: np.ndarray, *, scale: float = 32.0) -> np.ndarray:
    """Statistical Region Merging on a grayscale image in [0, 1].

    Args:
        gray: 2D image, float in [0, 1].
        scale: The Q parameter. Smaller Q merges more, giving fewer regions.

    Returns:
        Label map with compact labels 0..K-1.
    """
    height, width = gray.shape
    n_pixels = height * width
    values = gray.reshape(-1)

    parent = np.arange(n_pixels, dtype=np.int32)
    rank = np.zeros(n_pixels, dtype=np.int16)
    size = np.ones(n_pixels, dtype=np.int32)
    mean = values.copy()

    # Merging bound b(R) from Nock and Nielsen, with g = 1 for [0, 1] intensities.
    log_delta = 2.0 * np.log(6.0 * n_pixels)

    def bound(region_size: int) -> float:
        return (1.0 / (2.0 * scale * region_size)) * (np.log(1.0 + region_size) + log_delta)

    def find_root(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:  # path compression
            parent[node], node = root, parent[node]
        return root

    # 4-connected neighbour pairs, built without a Python loop.
    index = np.arange(n_pixels, dtype=np.int32).reshape(height, width)
    horizontal = np.stack([index[:, :-1].ravel(), index[:, 1:].ravel()], axis=1)
    vertical = np.stack([index[:-1, :].ravel(), index[1:, :].ravel()], axis=1)
    pairs = np.concatenate([horizontal, vertical], axis=0)

    # SRM's ordering: merge the most similar neighbours first.
    pairs = pairs[np.argsort(np.abs(values[pairs[:, 0]] - values[pairs[:, 1]]))]

    for left, right in pairs:
        root_left, root_right = find_root(left), find_root(right)
        if root_left == root_right:
            continue

        squared_gap = (mean[root_left] - mean[root_right]) ** 2
        if squared_gap >= bound(size[root_left]) + bound(size[root_right]):
            continue

        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        elif rank[root_left] == rank[root_right]:
            rank[root_left] += 1

        merged_size = size[root_left] + size[root_right]
        merged_mean = (
            mean[root_left] * size[root_left] + mean[root_right] * size[root_right]
        ) / merged_size

        parent[root_right] = root_left
        size[root_left], mean[root_left] = merged_size, merged_mean

    roots = np.array([find_root(i) for i in range(n_pixels)], dtype=np.int32)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.reshape(height, width)


def felzenszwalb_regions(image: np.ndarray, *, scale: float = 32.0) -> np.ndarray:
    """Felzenszwalb over-segmentation, parameterised to be comparable to SRM."""
    return felzenszwalb(image, scale=max(10, scale * 5), sigma=0, min_size=max(20, int(scale * 2)))


def score_regions(labels: np.ndarray, image: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Score every region on darkness, centrality and saturation.

    Skin reference values are read off a one-pixel band along the image border,
    which after frame removal and cropping is almost entirely healthy skin.

    Returns:
        The per-region scores, and a score map for visualisation.
    """
    # Blue carries the strongest melanin contrast of the three channels.
    if image.ndim == 3 and image.shape[-1] >= 3:
        intensity = image[..., 2]
        saturation = rgb2hsv(image)[..., 1]
    else:
        intensity = image if image.ndim == 2 else rgb2gray(image)
        saturation = np.zeros_like(intensity)

    def border_values(array: np.ndarray) -> np.ndarray:
        return np.concatenate([array[0, :], array[-1, :], array[:, 0], array[:, -1]])

    skin_intensity = np.percentile(border_values(intensity), SKIN_PERCENTILE)
    skin_saturation = np.median(border_values(saturation))

    height, width = intensity.shape
    centre_y, centre_x = height // 2, width // 2
    saturation_means = {
        int(region): float(saturation[labels == region].mean()) for region in np.unique(labels)
    }

    scores: list[dict] = []
    score_map = np.zeros_like(intensity, dtype=float)

    for region in regionprops(labels + 1, intensity_image=intensity):
        region_id = region.label - 1

        contrast = skin_intensity - region.mean_intensity

        centroid_y, centroid_x = region.centroid
        distance = np.hypot(centroid_y - centre_y, centroid_x - centre_x) / (height / 1.8)
        centrality = 1.0 - min(distance, 1.0)

        saturation_diff = saturation_means[region_id] - skin_saturation
        saturation_bonus = max(0.0, saturation_diff) * SATURATION_GAIN

        score = (
            WEIGHT_CONTRAST * contrast
            + WEIGHT_CENTRALITY * centrality
            + WEIGHT_SATURATION * saturation_bonus
        )
        score_map[labels == region_id] = score
        scores.append(
            {
                "id": region_id,
                "score": score,
                "contrast": contrast,
                "saturation_diff": saturation_diff,
            }
        )

    return scores, score_map


def select_lesion_regions(labels: np.ndarray, scores: list[dict]) -> np.ndarray:
    """Merge the best-scoring region with any close runner-up.

    Lesions are frequently split across several regions by the over-segmentation,
    so taking only the maximum truncates them. A runner-up joins the lesion when
    it scores near the best *and* is genuinely darker or more colourful than skin,
    the second condition stopping the mask from creeping into shaded skin.
    """
    mask = np.zeros(labels.shape, dtype=bool)
    if not scores:
        return mask.astype(np.uint8)

    best = max(entry["score"] for entry in scores)
    for entry in scores:
        near_best = entry["score"] > best * SCORE_KEEP_RATIO
        significant = (
            entry["contrast"] > MIN_CONTRAST or entry["saturation_diff"] > MIN_SATURATION_DIFF
        )
        if entry["score"] == best or (near_best and significant):
            mask[labels == entry["id"]] = True

    return mask.astype(np.uint8)


def segment(
    image: np.ndarray,
    *,
    scale: float = 32.0,
    gaussian_sigma: float = 1.0,
    backend: str = "srm",
    disk_size: int = 3,
) -> SegmentationResult:
    """Segment a preprocessed RGB image by over-segmentation and region selection.

    Args:
        image: Preprocessed RGB image.
        scale: The Q parameter of SRM, or its Felzenszwalb equivalent.
        gaussian_sigma: Pre-smoothing, 0 to disable.
        backend: ``"srm"`` for Statistical Region Merging, ``"felzenszwalb"`` otherwise.
        disk_size: Structuring element radius for the closing.
    """
    if backend not in {"srm", "felzenszwalb"}:
        raise ValueError(f"unknown backend {backend!r}, expected 'srm' or 'felzenszwalb'")

    original_shape = image.shape[:2]
    small = downscale(image)

    if backend == "srm":
        # Smooth the grayscale, not the colour image: a scalar sigma on an
        # (H, W, 3) array also blurs across the channel axis, which mixes red
        # into blue and measurably degrades the region scoring.
        smoothed = rgb2gray(small)
        if gaussian_sigma:
            smoothed = gaussian(smoothed, sigma=gaussian_sigma, mode="reflect")
        labels = statistical_region_merging(smoothed, scale=scale)
    else:
        smoothed = gaussian_filter(small, sigma=gaussian_sigma) if gaussian_sigma else small
        labels = felzenszwalb_regions(smoothed, scale=scale)

    scores, score_map = score_regions(labels, small)
    selected = select_lesion_regions(labels, scores)
    closed = morphology.binary_closing(selected, morphology.disk(disk_size))

    mask = upscale_mask(closed, original_shape)

    return SegmentationResult(
        mask=mask,
        steps={
            "Downscaled input": small,
            f"Gaussian smoothing (sigma={gaussian_sigma})": smoothed,
            f"Over-segmentation ({backend})": labels,
            "Blue channel": small[..., 2],
            "Saturation": rgb2hsv(small)[..., 1],
            "Region score map": score_map,
            "Selected regions": selected,
            "Morphological closing": closed,
        },
        info={
            "backend": backend,
            "n_regions": int(len(np.unique(labels))),
            "scale": scale,
            "calc_size": CALC_SIZE,
        },
    )
