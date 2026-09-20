"""Common contract for the segmentation methods."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage.transform import resize

# Every method computes on a downscaled copy. The lesion border is a
# large-scale structure, so 256x256 loses no useful detail, keeps the three
# methods comparable, and brings the per-image cost down to about a second.
CALC_SIZE = (256, 256)


@dataclass
class SegmentationResult:
    """A binary lesion mask plus the intermediate images that produced it.

    ``steps`` maps a human-readable stage name to the image at that stage, in
    pipeline order. Segmenters record it; :mod:`dermoseg.visualization` renders
    it. No segmenter draws anything itself.
    """

    mask: np.ndarray
    steps: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def downscale(image: np.ndarray) -> np.ndarray:
    """Resize an image to :data:`CALC_SIZE` as float in [0, 1]."""
    return resize(image, CALC_SIZE, anti_aliasing=True)


def upscale_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask back to ``shape`` without introducing new values."""
    restored = resize(mask.astype(float), shape, order=0, mode="edge", anti_aliasing=False)
    return (restored > 0.5).astype(np.uint8)
