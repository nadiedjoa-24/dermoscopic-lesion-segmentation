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
# removal to run (see remove_hair). Five of the twenty images clear it. Only
# two of them are genuinely hairy (ISIC_0000042 and ISIC_0000095) and one has
# a few thin hairs (ISIC_0000146); on the other two (ISIC_0000140 and
# ISIC_0000142) the detector is responding to the lesion's own dark texture,
# not to hair. That is a known limitation of this detector, stated in the README.
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
    disc = _detect_disc(
        image,
        circle_threshold=circle_threshold,
        shrink_factor=shrink_factor,
        border_dark_threshold=border_dark_threshold,
        border_ratio_trigger=border_ratio_trigger,
        border_width_ratio=border_width_ratio,
    )
    if disc is None:
        return image.copy(), np.full(image.shape[:2], 255, dtype=np.uint8)

    output = image.copy()
    output[~disc] = 255  # white, not black, so the frame cannot pass for a lesion
    disc = (disc * 255).astype(np.uint8)

    if crop:
        box = _bounding_box(disc > 0)
        output, disc = output[box], disc[box]

    return output, disc


def dermoscope_crop_box(image: np.ndarray) -> tuple[slice, slice]:
    """The rows and columns :func:`isolate_dermoscope_circle` keeps when it crops.

    Anything pixel-aligned with the original image, the ground-truth mask above
    all, must be cropped with this same box to stay aligned with the
    preprocessed image. Resizing it to the cropped shape instead would stretch
    it: on this dataset's framed images, even a perfect segmentation would then
    score only 0.915 to 0.965 Dice. Images without a frame keep every pixel.
    """
    disc = _detect_disc(image)
    if disc is None:
        return slice(None), slice(None)
    return _bounding_box(disc)


def _detect_disc(
    image: np.ndarray,
    *,
    circle_threshold: int = 60,
    shrink_factor: float = 0.9,
    border_dark_threshold: int = 30,
    border_ratio_trigger: float = 0.2,
    border_width_ratio: float = 0.05,
) -> np.ndarray | None:
    """Boolean mask of the lit dermoscope disc, or None if there is no dark frame."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Is there actually a dark frame? Inspect a band along the four edges.
    band = int(min(height, width) * border_width_ratio)
    border = np.zeros_like(gray, dtype=bool)
    border[:band, :] = border[-band:, :] = True
    border[:, :band] = border[:, -band:] = True

    dark_ratio = ((gray < border_dark_threshold) & border).sum() / border.sum()
    if dark_ratio < border_ratio_trigger:
        return None

    # Largest bright connected component = the lit disc.
    _, binary = cv2.threshold(gray, circle_threshold, 255, cv2.THRESH_BINARY)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_labels <= 1:
        return None

    main_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(labels == main_label)

    centre_x, centre_y = int(xs.mean()), int(ys.mean())
    radius = int(np.sqrt(((xs - centre_x) ** 2 + (ys - centre_y) ** 2).max()) * shrink_factor)

    grid_y, grid_x = np.ogrid[:height, :width]
    return ((grid_x - centre_x) ** 2 + (grid_y - centre_y) ** 2) <= radius**2


def _bounding_box(mask: np.ndarray) -> tuple[slice, slice]:
    """The smallest rows-by-columns box containing every True pixel of ``mask``."""
    ys, xs = np.where(mask)
    return slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)


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

    A closing erases any dark structure its structuring element cannot fit
    inside. Every element here is 10 px wide, so any hair thinner than that is
    erased by every one of them, whatever its orientation: two elements already
    catch every thin hair, and adding the two diagonals does not catch more
    hair. What the maximum over more elements does add is dark regions wide
    enough to hold some of the elements but not all of them, which in practice
    means the dark, textured interior of the lesion itself. That is why this
    detector also fires inside lesions (see HAIR_COVERAGE_THRESHOLD).
    """
    vertical = np.ones((HAIR_STRUCTURE_LENGTH, HAIR_STRUCTURE_WIDTH), dtype=bool)
    horizontal = np.ones((HAIR_STRUCTURE_WIDTH, HAIR_STRUCTURE_LENGTH), dtype=bool)
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

    A pixel is flagged where the closing raised its intensity noticeably, i.e.
    where it was darker than its elongated neighbourhood. That is true of hair,
    but also of the darker parts of a textured lesion, so the mask is best read
    as "thin or textured dark structure" rather than hair alone.
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
