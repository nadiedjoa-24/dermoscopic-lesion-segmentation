"""Shared post-processing applied to every segmenter's raw mask.

Keeping these steps out of the segmenters means the three methods are compared
on equal terms: only the segmentation itself differs.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage import morphology
from skimage.measure import label, regionprops

# A region this much smaller than the main one is kept only if it is nearby.
SATELLITE_AREA_RATIO = 0.03
SATELLITE_DISTANCE_RADII = 2.5
# Above this coverage a mask is assumed to be inverted (skin selected, not lesion).
INVERSION_COVERAGE = 0.60
VALID_DISC_RATIO = 0.95


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill dark spots enclosed in the lesion (hair remnants, specular highlights)."""
    return binary_fill_holes(mask).astype(np.uint8)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Drop everything outside the dermoscope disc, and undo an inverted mask.

    The inversion guard matters because every segmenter here picks a cluster or
    region by a heuristic; on low-contrast images that heuristic can select the
    skin instead of the lesion, producing the exact complement of the answer.
    A mask covering more than :data:`INVERSION_COVERAGE` of the disc is assumed
    to have made that mistake.

    This is a heuristic, not a principled step: it rescues failures rather than
    preventing them, and it is reported as part of the pipeline for that reason.
    """
    mask = np.asarray(mask)
    if mask.sum() == 0:
        return mask.astype(np.uint8)

    height, width = mask.shape
    grid_y, grid_x = np.ogrid[:height, :width]
    centre_y, centre_x = height // 2, width // 2
    inside_disc = (grid_x - centre_x) ** 2 + (grid_y - centre_y) ** 2 <= (
        min(height, width) // 2 * VALID_DISC_RATIO
    ) ** 2

    cleaned = mask.copy()
    cleaned[~inside_disc] = 0

    if cleaned.mean() > INVERSION_COVERAGE:
        cleaned = 1 - cleaned
        cleaned[~inside_disc] = 0

    return cleaned.astype(np.uint8)


def smart_convex_hull(mask: np.ndarray) -> np.ndarray:
    """Convex hull of the lesion, ignoring distant specks of noise.

    A plain convex hull is fragile: one isolated false-positive pixel in a
    corner stretches the hull across the whole image. This keeps the largest
    component plus any satellite that is either large relative to it or close
    to it, and takes the hull of that.
    """
    mask = np.asarray(mask)
    if mask.sum() == 0:
        return mask.astype(np.uint8)

    labelled = label(mask)
    if labelled.max() == 0:
        return mask.astype(np.uint8)

    regions = regionprops(labelled)
    main = max(regions, key=lambda region: region.area)
    main_centroid = np.array(main.centroid)
    main_radius = np.sqrt(main.area / np.pi)

    kept = labelled == main.label
    for region in regions:
        if region.label == main.label:
            continue
        large_enough = region.area / main.area > SATELLITE_AREA_RATIO
        close_enough = (
            np.linalg.norm(np.array(region.centroid) - main_centroid)
            < main_radius * SATELLITE_DISTANCE_RADII
        )
        if large_enough or close_enough:
            kept |= labelled == region.label

    return morphology.convex_hull_image(kept).astype(np.uint8)
