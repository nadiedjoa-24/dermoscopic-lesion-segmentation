"""Multi-channel Otsu: threshold selection and the valid-mask guard."""

import numpy as np

from dermoseg.segmentation.otsu import _otsu_threshold, segment


def test_otsu_threshold_ignores_pixels_outside_the_valid_mask():
    """A synthetic frame corner must not pull the threshold away from the real boundary.

    Without the valid mask, Otsu sees three clusters (lesion, skin, corner) and
    picks the split that separates the corner from everything else, lumping
    lesion and skin into the same class. With the corner excluded, it falls
    back to the real lesion/skin boundary.
    """
    rng = np.random.default_rng(0)
    channel = np.empty((100, 100))
    channel[:30, :] = rng.normal(30, 4, size=(30, 100))  # lesion
    channel[30:60, :] = rng.normal(90, 4, size=(30, 100))  # skin
    channel[60:, :] = rng.normal(250, 4, size=(40, 100))  # whitened frame corner

    all_valid = np.ones(channel.shape, dtype=bool)
    valid = all_valid.copy()
    valid[60:, :] = False

    contaminated = _otsu_threshold(channel, all_valid)
    clean = _otsu_threshold(channel, valid)

    assert contaminated > 90.0  # split lands between skin and the corner, not lesion and skin
    assert 30.0 < clean < 90.0  # split correctly lands between lesion and skin


def test_otsu_threshold_falls_back_to_the_full_channel_when_valid_is_degenerate():
    channel = np.zeros((10, 10))
    channel[:5, :] = 100.0
    empty_valid = np.zeros((10, 10), dtype=bool)
    full_valid = np.ones((10, 10), dtype=bool)

    assert _otsu_threshold(channel, empty_valid) == _otsu_threshold(channel, full_valid)


def test_segment_returns_a_binary_mask_matching_the_input_shape():
    rng = np.random.default_rng(0)
    image = (rng.random((120, 160, 3)) * 40 + 180).astype(np.uint8)  # bright "skin"
    image[40:80, 60:100] = 30  # dark "lesion"

    result = segment(image)

    assert result.mask.shape == (120, 160)
    assert set(np.unique(result.mask)) <= {0, 1}


def test_segment_accepts_a_valid_mask_without_changing_the_output_contract():
    rng = np.random.default_rng(0)
    image = (rng.random((120, 160, 3)) * 40 + 180).astype(np.uint8)
    image[40:80, 60:100] = 30
    valid_mask = np.ones((120, 160), dtype=bool)
    valid_mask[:10, :10] = False

    result = segment(image, valid_mask=valid_mask)

    assert result.mask.shape == (120, 160)
    assert set(np.unique(result.mask)) <= {0, 1}
