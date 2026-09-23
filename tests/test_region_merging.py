"""Region merging: the SRM operator and the border-sampled skin reference."""

import numpy as np
import pytest

from dermoseg.segmentation.region_merging import (
    score_regions,
    segment,
    statistical_region_merging,
)


def test_statistical_region_merging_separates_two_flat_regions():
    gray = np.zeros((40, 40))
    gray[:, 20:] = 1.0  # a hard step, no noise

    labels = statistical_region_merging(gray, scale=32.0)

    left_labels = set(np.unique(labels[:, :20]))
    right_labels = set(np.unique(labels[:, 20:]))
    assert left_labels.isdisjoint(right_labels)


def test_score_regions_ignores_frame_corners_in_the_skin_reference():
    """A regression guard on the fix: whitened corners must not skew the border sample.

    Four corners are whitened brighter than either the lesion or the real
    skin, sitting on the very border band ``score_regions`` samples. Passing
    the valid mask should drop them from the reference; without it, the
    "skin" reference is pulled toward the corners and the lesion looks far
    darker by comparison than it really is.
    """
    size = 20
    image = np.zeros((size, size, 3))
    image[..., 2] = 150.0  # skin, blue channel
    image[6:14, 6:14, 2] = 30.0  # lesion, away from the border

    corners = ((0, 4), (size - 4, size))
    for r0, r1 in corners:
        for c0, c1 in corners:
            image[r0:r1, c0:c1, 2] = 255.0

    labels = np.zeros((size, size), dtype=int)
    labels[6:14, 6:14] = 1

    valid = np.ones((size, size), dtype=bool)
    for r0, r1 in corners:
        for c0, c1 in corners:
            valid[r0:r1, c0:c1] = False

    contaminated, _ = score_regions(labels, image)
    clean, _ = score_regions(labels, image, valid)

    contrast_contaminated = next(entry["contrast"] for entry in contaminated if entry["id"] == 1)
    contrast_clean = next(entry["contrast"] for entry in clean if entry["id"] == 1)

    assert contrast_contaminated > contrast_clean
    assert contrast_clean == pytest.approx(150.0 - 30.0, abs=1.0)


def test_segment_returns_a_binary_mask_matching_the_input_shape():
    rng = np.random.default_rng(0)
    image = (rng.random((120, 160, 3)) * 20 + 180).astype(np.uint8)
    image[40:80, 60:100] = 30

    result = segment(image, scale=25.0, gaussian_sigma=1.0, backend="srm")

    assert result.mask.shape == (120, 160)
    assert set(np.unique(result.mask)) <= {0, 1}


def test_felzenszwalb_smoothing_does_not_mix_colour_channels():
    """A scalar sigma on an (H, W, 3) array also blurs across the channel axis.

    On a pure red image, that leaks red into green and blue. Each channel has
    to be smoothed on its own.
    """
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 0] = 255

    result = segment(image, scale=25.0, gaussian_sigma=2.0, backend="felzenszwalb")
    smoothed = result.steps["Gaussian smoothing (sigma=2.0)"]

    assert np.allclose(smoothed[..., 1], 0.0)
    assert np.allclose(smoothed[..., 2], 0.0)


def test_segment_rejects_an_unknown_backend():
    with pytest.raises(ValueError):
        segment(np.zeros((10, 10, 3), dtype=np.uint8), backend="not-a-backend")
