"""Segmentation quality metrics."""

from __future__ import annotations

import numpy as np
from skimage.transform import resize


def dice_score(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Dice similarity coefficient between two binary masks.

    The ground truth is resized to the shape of the prediction with
    nearest-neighbour interpolation when the two differ, which happens
    whenever the prediction comes from a downscaled computation.

    Two empty masks score 1.0: predicting "no lesion" where there is none
    is a perfect answer, not an undefined one.
    """
    pred = np.asarray(predicted) > 0
    truth = np.asarray(ground_truth) > 0

    if pred.shape != truth.shape:
        truth = resize(truth, pred.shape, order=0, mode="edge", anti_aliasing=False) > 0

    total = pred.sum() + truth.sum()
    if total == 0:
        return 1.0

    return float(2.0 * np.logical_and(pred, truth).sum() / total)
