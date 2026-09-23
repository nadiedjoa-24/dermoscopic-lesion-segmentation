"""Preprocessing: frame removal geometry and directional hair detection."""

import numpy as np

from dermoseg import preprocessing
from dermoseg.preprocessing import (
    _diagonal_footprint,
    _directional_closing_max,
    dermoscope_crop_box,
    isolate_dermoscope_circle,
)


def test_diagonal_footprint_runs_corner_to_corner():
    footprint = _diagonal_footprint(length=40, width=1)
    # A width-1 footprint is exactly the identity diagonal.
    assert np.array_equal(footprint, np.eye(footprint.shape[0], dtype=bool))


def test_anti_diagonal_footprint_is_the_mirror():
    footprint = _diagonal_footprint(length=40, width=1)
    anti = _diagonal_footprint(length=40, width=1, anti=True)
    assert np.array_equal(anti, np.fliplr(footprint))


def test_diagonal_footprint_spans_the_requested_length_diagonally():
    """The diagonal element must actually reach `length` corner to corner.

    An axis-aligned element of this length spans it along a single row or
    column. The diagonal counterpart has to span the *same* physical length,
    along its own diagonal, which means a smaller bounding box (by a factor of
    ~sqrt(2)) than an axis-aligned element of the same nominal length would
    have, not the same bounding box, which would make it longer than intended.
    """
    footprint = _diagonal_footprint(length=100, width=1)
    diagonal_reach = np.hypot(*footprint.shape)
    assert abs(diagonal_reach - 100) < 2  # within one pixel's rounding


def test_directional_closing_is_the_max_of_all_four_orientations():
    """All four closings must run and enter the pixelwise maximum.

    Pins the detector's current definition, so dropping an orientation shows
    up as a mismatch rather than a subtle score change.
    """
    from skimage import morphology

    rng = np.random.default_rng(0)
    channel = rng.integers(0, 256, size=(70, 70)).astype(float)

    vertical = np.ones((50, 10), dtype=bool)
    horizontal = np.ones((10, 50), dtype=bool)
    diagonal = _diagonal_footprint(50, 10)
    anti_diagonal = _diagonal_footprint(50, 10, anti=True)

    expected = np.maximum.reduce(
        [morphology.closing(channel, footprint=fp) for fp in (horizontal, vertical, diagonal, anti_diagonal)]
    )

    assert np.array_equal(_directional_closing_max(channel), expected)


def test_isolate_dermoscope_circle_leaves_unframed_images_untouched():
    image = np.full((100, 120, 3), 200, dtype=np.uint8)
    output, mask = isolate_dermoscope_circle(image)
    assert output.shape == image.shape
    assert (mask > 0).all()
    assert dermoscope_crop_box(image) == (slice(None), slice(None))


def test_dermoscope_crop_box_is_the_crop_frame_removal_applies():
    """A framed image is cropped; the crop box must reproduce that exact crop.

    Everything aligned with the original image (the ground truth) is cropped
    with this box, so a mismatch would silently misalign it.
    """
    height, width = 200, 300
    grid_y, grid_x = np.ogrid[:height, :width]
    lit_disc = (grid_x - 150) ** 2 + (grid_y - 100) ** 2 <= 90**2
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[lit_disc] = 200

    cropped, _ = isolate_dermoscope_circle(image, crop=True)
    rows, columns = dermoscope_crop_box(image)

    assert cropped.shape[:2] != image.shape[:2]  # the frame really was cropped
    assert np.array_equal(image[rows, columns].shape, cropped.shape)
    whitened, _ = isolate_dermoscope_circle(image, crop=False)
    assert np.array_equal(whitened[rows, columns], cropped)


def test_remove_hair_coverage_is_a_pixel_fraction_of_the_dilated_mask(monkeypatch):
    """Regression guard on the coverage gate.

    It must erode ``detect_hair``'s already-dilated mask and measure the
    fraction of the disc that survives, not sum a raw, unthresholded
    intensity difference (a completely different scale, barely tracking how
    much hair is actually present). Eroding the *dilated* mask matters too: a
    thin hair line, however long, is nowhere near wide enough to survive a
    disc(12) erosion on its own, only a genuinely extended patch is.
    """
    size = 100
    image = np.full((size, size, 3), 150, dtype=np.uint8)
    disc_mask = np.ones((size, size), dtype=bool)
    response = np.zeros((size, size))  # unused by the fixed gate, kept for the call signature

    calls = []
    monkeypatch.setattr(
        preprocessing, "_inpaint_channel", lambda channel, missing: calls.append(1) or channel
    )

    # No hair at all: the gate must skip.
    no_hair = np.zeros((size, size, 3), dtype=bool)
    monkeypatch.setattr(preprocessing, "detect_hair", lambda img: (no_hair, response))
    preprocessing.remove_hair(image, disc_mask)
    assert calls == []

    # A thin, full-width line: real pixels are flagged, but a 1-pixel-thin
    # line cannot survive a disc(12) erosion, so it must still skip.
    thin_line = np.zeros((size, size, 3), dtype=bool)
    thin_line[size // 2, :, :] = True
    monkeypatch.setattr(preprocessing, "detect_hair", lambda img: (thin_line, response))
    preprocessing.remove_hair(image, disc_mask)
    assert calls == []

    # A large, solid block: genuinely extended, well over a disc(12) across,
    # so it must survive erosion and trigger removal.
    thick_block = np.zeros((size, size, 3), dtype=bool)
    thick_block[10:-10, 10:-10, :] = True
    monkeypatch.setattr(preprocessing, "detect_hair", lambda img: (thick_block, response))
    preprocessing.remove_hair(image, disc_mask)
    assert len(calls) == 3  # once per RGB channel
