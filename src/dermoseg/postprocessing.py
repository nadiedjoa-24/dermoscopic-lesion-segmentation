"""Shared post-processing applied to every segmenter's raw mask.

Keeping these steps out of the segmenters means the three methods are compared
on equal terms: only the segmentation itself differs.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage import morphology
from skimage.measure import label, regionprops

# A satellite closer than this fraction of the image diagonal is kept.
# Expressed against the diagonal rather than the lesion's own radius: on a
# large lesion, a radius-relative reach can exceed the image itself, so the
# "distant speck" it was meant to exclude becomes unreachable: every pixel in
# the image counts as "close enough". Measured against the 20 images here, the
# farthest corner from the lesion sits at 51-58% of the diagonal regardless of
# lesion size, so 0.25 keeps genuine nearby fragments while remaining unable to
# reach a far corner on any of them.
#
# An earlier version of this filter also kept a satellite regardless of
# distance when it was large relative to the lesion (more than 3% of its
# area), meant to catch a genuine second lobe of the lesion. In practice a
# region that size is just as often an unrelated artefact placed anywhere in
# the image (a patch of scaly skin texture picked up by LBP in a corner, for
# instance), and being large gave it no more reason to belong to the lesion
# than being small. Distance alone decides now.
SATELLITE_DISTANCE_FRACTION = 0.25


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill dark spots enclosed in the lesion (hair remnants, specular highlights)."""
    return binary_fill_holes(mask).astype(np.uint8)


def clean_mask(mask: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Drop everything outside the valid image area.

    An earlier version also inverted any mask covering more than 60% of the
    valid area, on the theory that it had selected skin instead of lesion. On
    this dataset no segmenter output ever came near that coverage, while two
    real lesions exceed it, so the guard could only ever turn a correct mask
    into its complement. It was removed.

    Args:
        mask: The raw binary mask.
        valid: Boolean mask of the pixels that carry real image data, as
            returned by :func:`~dermoseg.preprocessing.preprocess_with_mask`.
            Frame removal whitens the corners outside the dermoscope disc on
            images that have a frame, and leaves the whole image valid on
            images that do not. Using that real area, rather than guessing a
            disc from the image's aspect ratio, keeps lesion pixels that a
            guessed disc would cut. Defaults to the whole image when omitted,
            for callers that have no preprocessing mask to give.
    """
    mask = np.asarray(mask)
    if valid is None:
        return mask.astype(np.uint8)

    cleaned = mask.copy()
    cleaned[~valid] = 0
    return cleaned.astype(np.uint8)


def smart_convex_hull(mask: np.ndarray) -> np.ndarray:
    """Convex hull of the lesion, ignoring distant specks of noise.

    A plain convex hull is fragile: one isolated false-positive pixel in a
    corner stretches the hull across the whole image. This keeps the largest
    component plus any satellite close enough to it, and takes the hull of
    that.

    "Close" is measured against the image diagonal, not the lesion's own
    radius: a radius-relative reach grows with the lesion, and on a large
    lesion it can exceed the image itself, at which point every pixel counts
    as close and the filter stops filtering anything.
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
    satellite_reach = np.hypot(*mask.shape) * SATELLITE_DISTANCE_FRACTION

    kept = labelled == main.label
    for region in regions:
        if region.label == main.label:
            continue
        distance = np.linalg.norm(np.array(region.centroid) - main_centroid)
        if distance < satellite_reach:
            kept |= labelled == region.label

    return morphology.convex_hull_image(kept).astype(np.uint8)
