"""The pipeline's scoring: masks must be compared to an aligned ground truth."""

import numpy as np

from dermoseg import preprocessing
from dermoseg.data import Sample
from dermoseg.pipeline import run_all_methods
from dermoseg.segmentation.base import SegmentationResult


def test_a_perfect_mask_scores_one_on_a_framed_image(monkeypatch):
    """Frame removal crops a framed image, so the ground truth must be cropped too.

    Before this was fixed, the full-size ground truth was *resized* to the
    cropped shape instead, stretching it: a perfect segmentation of a framed
    image scored well below 1.0 (0.915 to 0.965 on this dataset's framed images).
    """
    height, width = 200, 300
    grid_y, grid_x = np.ogrid[:height, :width]
    lit_disc = (grid_x - 150) ** 2 + (grid_y - 100) ** 2 <= 90**2
    lesion = (grid_x - 170) ** 2 + (grid_y - 90) ** 2 <= 25**2

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[lit_disc] = 200
    image[lesion] = 60
    sample = Sample(
        name="framed.jpg", category="nevus", image=image, ground_truth=lesion.astype(np.uint8)
    )

    # Hair removal is irrelevant here; keep the synthetic lesion untouched.
    monkeypatch.setattr(preprocessing, "remove_hair", lambda image, disc_mask: image)

    def threshold_the_lesion(image, valid):
        return SegmentationResult(mask=(image[..., 0] < 100).astype(np.uint8))

    outcome = run_all_methods(sample, methods={"Perfect": threshold_the_lesion})

    assert outcome.preprocessed.shape[:2] != image.shape[:2]  # the frame was cropped
    assert outcome.ground_truth.shape == outcome.preprocessed.shape[:2]
    assert outcome.scores["Perfect"] == 1.0
