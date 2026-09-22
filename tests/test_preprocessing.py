"""Preprocessing: frame removal geometry and directional hair detection."""

import numpy as np

from dermoseg import preprocessing
from dermoseg.preprocessing import (
    _diagonal_footprint,
    _directional_closing_max,
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
    have — not the same bounding box, which would make it longer than intended.
    """
    footprint = _diagonal_footprint(length=100, width=1)
    diagonal_reach = np.hypot(*footprint.shape)
    assert abs(diagonal_reach - 100) < 2  # within one pixel's rounding


def test_directional_closing_is_the_max_of_all_four_orientations():
    """A regression guard on the fix itself: all four closings must run and count.

    Recomputes the same four closings independently and checks
    _directional_closing_max against their pixelwise maximum, so dropping an
    orientation back out of the max (the bug this test guards against) shows
    up as a mismatch rather than a subtle score change.
    """
    from skimage import morphology

    rng = np.random.default_rng(0)
    channel = rng.integers(0, 256, size=(70, 70)).astype(float)

    horizontal = np.ones((50, 10), dtype=bool)
    vertical = np.ones((10, 50), dtype=bool)
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


def test_remove_hair_coverage_is_a_pixel_fraction_not_a_mean_intensity(monkeypatch):
    """Regression guard on the coverage gate.

    It must count pixels whose response clears ``HAIR_RESPONSE_THRESHOLD``
    and measure the fraction of the disc they cover, not sum the raw,
    unthresholded intensity difference: that sum lives on a completely
    different scale and barely tracks how much hair is actually present (a
    uniform, sub-threshold difference across the whole image summed to more
    than a real, dense hair patch ever could).
    """
    size = 100
    image = np.full((size, size, 3), 150, dtype=np.uint8)
    disc_mask = np.ones((size, size), dtype=bool)
    hair_masks = np.zeros((size, size, 3), dtype=bool)

    calls = []
    monkeypatch.setattr(
        preprocessing, "_inpaint_channel", lambda channel, missing: calls.append(1) or channel
    )

    # Every pixel differs by 30, under the 40-response threshold: no pixel is
    # hair, so the correct covered fraction is exactly zero.
    faint_response = np.full((size, size), 150.0 - 30.0)
    monkeypatch.setattr(preprocessing, "detect_hair", lambda img: (hair_masks, faint_response))
    preprocessing.remove_hair(image, disc_mask)
    assert calls == []

    # A dense, above-threshold patch covering most of the image: real hair,
    # and it must survive erosion above the 20% coverage threshold.
    thick_response = np.full((size, size), 150.0)
    thick_response[6:-6, 6:-6] = 150.0 - 60.0
    monkeypatch.setattr(preprocessing, "detect_hair", lambda img: (hair_masks, thick_response))
    preprocessing.remove_hair(image, disc_mask)
    assert len(calls) == 3  # once per RGB channel
