"""Preprocessing of dermoscopic images: frame removal and hair removal.

Dermoscopes image the skin through a circular lens, so many ISIC images carry a
black corner frame that no intensity-based segmenter can tell from a dark
lesion. Body hair is the other recurring artefact: dark elongated structures
crossing the lesion border.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import interpolate
from skimage import filters, morphology

# Hair detection thresholds, from the DullRazor-style tuning of this project.
HAIR_STRUCTURE_LENGTH = 50
HAIR_STRUCTURE_WIDTH = 10
HAIR_RESPONSE_THRESHOLD = 40
HAIR_DILATION_SIZE = 20
# Fraction of the valid area the (dilated, eroded) hair mask must cover for
# removal to run. Recalibrated against this dataset after the coverage gate's
# formula was fixed to actually count pixels (see remove_hair): the twenty
# images split into five clearly hairy ones between 4.5% and 19.2% coverage,
# and the rest under 3%, with nothing in between.
HAIR_COVERAGE_THRESHOLD = 0.03


def isolate_dermoscope_circle(
    image: np.ndarray,
    *,
    circle_threshold: int = 60,
    shrink_factor: float = 0.9,
    crop: bool = True,
    border_dark_threshold: int = 30,
    border_ratio_trigger: float = 0.2,
    border_width_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect the bright dermoscope disc and discard the dark frame around it.

    Images without a dark frame are returned untouched, so the function is safe
    to apply to the whole dataset.

    Args:
        image: RGB image, shape (H, W, 3).
        circle_threshold: Intensity above which a pixel belongs to the lit disc.
        shrink_factor: Radius shrinkage, to stay clear of the blurred rim.
        crop: Whether to crop to the disc's bounding box.
        border_dark_threshold: Intensity below which a border pixel counts as black.
        border_ratio_trigger: Fraction of black border pixels needed to assume a frame.
        border_width_ratio: Width of the inspected border band, relative to the image.

    Returns:
        The image with the frame whitened (and cropped), and the 0/255 disc mask.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Is there actually a dark frame? Inspect a band along the four edges.
    band = int(min(height, width) * border_width_ratio)
    border = np.zeros_like(gray, dtype=bool)
    border[:band, :] = border[-band:, :] = True
    border[:, :band] = border[:, -band:] = True

    dark_ratio = ((gray < border_dark_threshold) & border).sum() / border.sum()
    full_mask = np.full((height, width), 255, dtype=np.uint8)
    if dark_ratio < border_ratio_trigger:
        return image.copy(), full_mask

    # Largest bright connected component = the lit disc.
    _, binary = cv2.threshold(gray, circle_threshold, 255, cv2.THRESH_BINARY)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_labels <= 1:
        return image.copy(), full_mask

    main_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(labels == main_label)

    centre_x, centre_y = int(xs.mean()), int(ys.mean())
    radius = int(np.sqrt(((xs - centre_x) ** 2 + (ys - centre_y) ** 2).max()) * shrink_factor)

    grid_y, grid_x = np.ogrid[:height, :width]
    disc = ((grid_x - centre_x) ** 2 + (grid_y - centre_y) ** 2) <= radius**2
    disc = (disc * 255).astype(np.uint8)

    output = image.copy()
    output[disc == 0] = 255  # white, not black, so the frame cannot pass for a lesion

    if crop:
        ys, xs = np.where(disc == 255)
        output = output[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        disc = disc[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    return output, disc


def _diagonal_footprint(length: int, width: int, *, anti: bool = False) -> np.ndarray:
    """A diagonal-line structuring element, the counterpart of the axis-aligned rectangles.

    Built from the diagonal of a square array sized so the diagonal's own
    length matches ``length``, then thickened to ``width`` by dilation.
    ``anti`` flips it to the other diagonal (135 degrees instead of 45).
    """
    side = max(int(round(length / np.sqrt(2))), 1)
    line = np.eye(side, dtype=bool)
    if anti:
        line = np.fliplr(line)
    if width > 1:
        line = morphology.binary_dilation(line, morphology.disk(width // 2))
    return line


def _directional_closing_max(channel: np.ndarray) -> np.ndarray:
    """Maximum of four grayscale closings, at 0, 45, 90 and 135 degrees.

    Hairs are thin and elongated, so a closing with a structuring element longer
    than the hair width erases them. A hair running horizontally or vertically
    is caught by the matching axis-aligned element, but one running diagonally
    is caught well by neither: it benefits from a horizontal or vertical
    element only through the width margin, not the length, the same way a
    horizontal element barely helps against a vertical hair. The two diagonal
    elements close that gap, matching the range of orientations DullRazor
    itself uses.
    """
    horizontal = np.ones((HAIR_STRUCTURE_LENGTH, HAIR_STRUCTURE_WIDTH), dtype=bool)
    vertical = np.ones((HAIR_STRUCTURE_WIDTH, HAIR_STRUCTURE_LENGTH), dtype=bool)
    diagonal = _diagonal_footprint(HAIR_STRUCTURE_LENGTH, HAIR_STRUCTURE_WIDTH)
    anti_diagonal = _diagonal_footprint(HAIR_STRUCTURE_LENGTH, HAIR_STRUCTURE_WIDTH, anti=True)

    closed_horizontal = morphology.closing(channel, footprint=horizontal)
    closed_vertical = morphology.closing(channel, footprint=vertical)
    closed_diagonal = morphology.closing(channel, footprint=diagonal)
    closed_anti_diagonal = morphology.closing(channel, footprint=anti_diagonal)

    return np.maximum.reduce(
        [closed_horizontal, closed_vertical, closed_diagonal, closed_anti_diagonal]
    )


def detect_hair(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-channel hair mask and the closing response of the red channel.

    A pixel is hair where the closing removed a noticeable amount of intensity,
    i.e. where it was darker than its elongated neighbourhood.
    """
    responses = [_directional_closing_max(image[..., c]) for c in range(3)]
    dilation = np.ones((HAIR_DILATION_SIZE, HAIR_DILATION_SIZE), dtype=bool)

    masks = []
    for channel_index, response in enumerate(responses):
        difference = np.abs(image[..., channel_index].astype(float) - response)
        masks.append(morphology.binary_dilation(difference > HAIR_RESPONSE_THRESHOLD, dilation))

    return np.stack(masks, axis=-1), responses[0]


def _inpaint_channel(channel: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Fill the pixels flagged in ``missing`` by nearest-neighbour interpolation."""
    height, width = channel.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    values = interpolate.griddata(
        (grid_x[~missing], grid_y[~missing]),
        channel[~missing],
        (grid_x[missing], grid_y[missing]),
        method="nearest",
        fill_value=0,
    )

    filled = channel.copy()
    filled[grid_y[missing], grid_x[missing]] = values
    return filled


def remove_hair(image: np.ndarray, disc_mask: np.ndarray) -> np.ndarray:
    """Detect hairs and inpaint them from the surrounding skin.

    Images whose hair coverage is below :data:`HAIR_COVERAGE_THRESHOLD` are
    returned untouched: inpainting a hairless image only blurs the lesion border.

    Args:
        image: RGB image, shape (H, W, 3).
        disc_mask: Boolean mask of the valid (non-frame) area.

    Returns:
        The RGB image with hairs removed, or the input if there are too few hairs.
    """
    hair_masks, red_response = detect_hair(image)

    # hair_masks is already a thresholded, dilated pixel map (20x20 square),
    # so eroding it by a disc keeps only areas thick enough to be a real,
    # extended patch of hair rather than a handful of stray flagged pixels: a
    # thin hair line, on its own, is nowhere near wide enough to survive this
    # erosion, no matter how long it runs. What survives is genuinely close to
    # what the inpainting below will touch, so its fraction of the valid area
    # is a meaningful coverage measure.
    coverage = morphology.erosion(hair_masks[..., 0], footprint=morphology.disk(12))
    if coverage.sum() / max(disc_mask.sum(), 1) < HAIR_COVERAGE_THRESHOLD:
        return image

    smoothing = np.ones((HAIR_DILATION_SIZE, HAIR_DILATION_SIZE), dtype=bool)
    result = image.copy()
    for channel_index in range(3):
        inpainted = _inpaint_channel(image[..., channel_index], hair_masks[..., channel_index])
        result[..., channel_index] = filters.median(inpainted, smoothing)

    return result


def preprocess_with_mask(
    image: np.ndarray,
    *,
    remove_frame: bool = True,
    remove_hairs: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full preprocessing pipeline, returning the image and its valid area.

    The valid area matters downstream: frame removal whitens the corners outside
    the dermoscope disc, and those synthetic pixels are not skin. A segmenter
    that clusters over the whole frame would let them form their own cluster,
    so methods that fit a global colour model must be told to ignore them.

    Returns:
        The preprocessed RGB image, and a boolean mask of the pixels that carry
        real image data.
    """
    processed = image.copy()

    if remove_frame:
        processed, disc_mask = isolate_dermoscope_circle(processed, crop=True)
    else:
        disc_mask = np.full(processed.shape[:2], 255, dtype=np.uint8)

    valid = disc_mask > 0
    if remove_hairs:
        processed = remove_hair(processed, valid)

    return processed, valid


def preprocess(
    image: np.ndarray,
    *,
    remove_frame: bool = True,
    remove_hairs: bool = True,
) -> np.ndarray:
    """Run the full preprocessing pipeline on an RGB image.

    Convenience wrapper around :func:`preprocess_with_mask` for callers that do
    not need the valid-area mask.
    """
    return preprocess_with_mask(
        image, remove_frame=remove_frame, remove_hairs=remove_hairs
    )[0]
