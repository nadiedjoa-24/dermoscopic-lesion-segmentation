"""Dice coefficient, including the cases the docstring singles out."""

import numpy as np

from dermoseg.metrics import dice_score


def test_identical_masks_score_one():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    assert dice_score(mask, mask) == 1.0


def test_disjoint_masks_score_zero():
    left = np.zeros((20, 20), dtype=np.uint8)
    left[0:5, 0:5] = 1
    right = np.zeros((20, 20), dtype=np.uint8)
    right[15:20, 15:20] = 1
    assert dice_score(left, right) == 0.0


def test_two_empty_masks_score_one():
    """Predicting "no lesion" where there is none is a perfect answer."""
    empty = np.zeros((10, 10), dtype=np.uint8)
    assert dice_score(empty, empty) == 1.0


def test_one_empty_mask_scores_zero():
    truth = np.zeros((10, 10), dtype=np.uint8)
    truth[2:8, 2:8] = 1
    assert dice_score(np.zeros((10, 10), dtype=np.uint8), truth) == 0.0


def test_known_overlap():
    """10 predicted, 10 true, 5 shared: 2*5 / (10+10) = 0.5."""
    predicted = np.zeros((10, 10), dtype=np.uint8)
    predicted[0, 0:10] = 1
    truth = np.zeros((10, 10), dtype=np.uint8)
    truth[0, 5:10] = 1
    truth[1, 0:5] = 1
    assert dice_score(predicted, truth) == 0.5


def test_ground_truth_is_resized_to_the_prediction():
    """Predictions come back from a downscaled computation, so shapes differ."""
    truth = np.zeros((100, 100), dtype=np.uint8)
    truth[25:75, 25:75] = 1
    predicted = np.zeros((50, 50), dtype=np.uint8)
    predicted[12:38, 12:38] = 1
    assert dice_score(predicted, truth) > 0.95


def test_binarises_non_binary_input():
    """Anything above zero is lesion, whatever the dtype or scale."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    assert dice_score(mask * 255, mask) == 1.0
