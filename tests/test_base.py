"""Scaling helpers shared by the three segmenters."""

import numpy as np

from dermoseg.segmentation.base import CALC_SIZE, downscale, downscale_mask, upscale_mask


def test_downscale_normalises_to_the_calc_size():
    image = (np.random.default_rng(0).random((400, 300, 3)) * 255).astype(np.uint8)
    small = downscale(image)
    assert small.shape[:2] == CALC_SIZE
    assert 0.0 <= small.min() and small.max() <= 1.0


def test_upscale_keeps_a_mask_binary():
    """Nearest-neighbour, so no intermediate values sneak into a binary mask."""
    mask = np.zeros(CALC_SIZE, dtype=np.uint8)
    mask[60:200, 60:200] = 1
    restored = upscale_mask(mask, (500, 700))
    assert restored.shape == (500, 700)
    assert set(np.unique(restored)) <= {0, 1}
    assert abs(restored.mean() - mask.mean()) < 0.01


def test_downscale_mask_stays_binary_and_matches_calc_size():
    mask = np.zeros((500, 700), dtype=bool)
    mask[100:400, 100:400] = True
    small = downscale_mask(mask)
    assert small.shape == CALC_SIZE
    assert small.dtype == np.bool_
    assert abs(small.mean() - mask.mean()) < 0.01
