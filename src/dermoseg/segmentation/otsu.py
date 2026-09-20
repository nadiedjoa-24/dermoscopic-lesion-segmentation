"""Multi-channel Otsu thresholding, refined by a Chan-Vese active contour.

Otsu's threshold is computed independently on R, G and B. A lesion is darker
than skin in all three, but the blue channel carries the strongest melanin
contrast, so the channels are combined as ``(R and G) or B``: a pixel is lesion
when red *and* green agree, or when blue alone is confident.

The resulting mask has ragged borders, so it seeds a Chan-Vese contour that
relaxes onto the actual intensity boundary.
"""

from __future__ import annotations

import numpy as np
from skimage import color, filters, morphology, segmentation

from .base import CALC_SIZE, SegmentationResult, downscale, upscale_mask

CHAN_VESE_ITERATIONS = 5


def segment(image: np.ndarray, *, disk_size: int = 3) -> SegmentationResult:
    """Segment a preprocessed RGB image with multi-channel Otsu + Chan-Vese."""
    original_shape = image.shape[:2]
    small = downscale(image)

    if small.ndim == 3:
        red, green, blue = small[..., 0], small[..., 1], small[..., 2]
    else:
        red = green = blue = small

    thresholds, channel_masks = {}, {}
    for name, channel in (("red", red), ("green", green), ("blue", blue)):
        thresholds[name] = float(filters.threshold_otsu(channel))
        channel_masks[name] = channel < thresholds[name]

    combined = (channel_masks["red"] & channel_masks["green"]) | channel_masks["blue"]

    footprint = morphology.disk(disk_size)
    morphed = morphology.closing(morphology.opening(combined, footprint), footprint)

    contour = segmentation.morphological_chan_vese(
        color.rgb2gray(small),
        num_iter=CHAN_VESE_ITERATIONS,
        init_level_set=morphed,
        smoothing=1,
    )

    mask = upscale_mask(contour, original_shape)

    return SegmentationResult(
        mask=mask,
        steps={
            "Downscaled input": small,
            "Red channel": red,
            "Green channel": green,
            "Blue channel": blue,
            "Otsu on red": channel_masks["red"],
            "Otsu on green": channel_masks["green"],
            "Otsu on blue": channel_masks["blue"],
            "Combined (R & G) | B": combined,
            "Morphological cleanup": morphed,
            "Chan-Vese contour": contour,
        },
        info={"thresholds": thresholds, "calc_size": CALC_SIZE},
    )
