"""End-to-end pipeline: preprocess, segment with each method, score.

This is the single place where the three methods are run and compared, so the
notebook and the benchmark script cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Sample
from .metrics import dice_score
from .postprocessing import clean_mask, fill_holes, smart_convex_hull
from .preprocessing import dermoscope_crop_box, preprocess_with_mask
from .segmentation import lbp, otsu, region_merging

# Method name -> callable taking a preprocessed RGB image and its valid-area
# mask, and returning a SegmentationResult. The parameters are the ones tuned during the project.
METHODS = {
    "Otsu": lambda image, valid: otsu.segment(image, valid_mask=valid),
    "LBP": lambda image, valid: lbp.segment(image, sigma=3.0, valid_mask=valid),
    # SRM's region_merging.segment also accepts valid_mask (see its docstring
    # for why it is deliberately not passed here).
    "SRM": lambda image, valid: region_merging.segment(
        image, scale=25.0, gaussian_sigma=2.0, backend="srm"
    ),
}


@dataclass
class PipelineOutcome:
    """Everything produced for one image."""

    name: str
    category: str
    preprocessed: np.ndarray
    # The ground truth cropped exactly like ``preprocessed``, so the two stay
    # pixel-aligned. This, not ``Sample.ground_truth``, is what masks are
    # scored and drawn against.
    ground_truth: np.ndarray
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    hulls: dict[str, np.ndarray] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    results: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        """Flatten into one record for a results table."""
        return {"Image": self.name, "Category": self.category, **self.scores}


def finalize_mask(raw_mask: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Shared post-processing, applied identically to every method.

    Order matters: the inversion guard in :func:`~dermoseg.postprocessing.clean_mask`
    has to run before the holes are filled. On an inverted mask the lesion *is*
    the hole, so filling first would erase the very region the guard needs to
    recover, leaving an empty mask.

    ``valid`` is the real valid-area mask from preprocessing, passed through so
    ``clean_mask`` clips to the actual dermoscope disc instead of guessing one
    from the image's aspect ratio.
    """
    return fill_holes(clean_mask(raw_mask, valid))


def run_all_methods(
    sample: Sample,
    *,
    methods=None,
    do_preprocess: bool = True,
    keep_results: bool = False,
) -> PipelineOutcome:
    """Run every method on one sample and score it against the ground truth.

    Args:
        sample: The image and its ground truth.
        methods: Mapping of name to a callable taking the preprocessed image
            and its valid-area mask, defaulting to :data:`METHODS`.
        do_preprocess: Whether to run frame and hair removal first.
        keep_results: Whether to keep the intermediate images for plotting.
            Off by default so a batch run stays light on memory.
    """
    methods = METHODS if methods is None else methods
    if do_preprocess:
        image, valid_mask = preprocess_with_mask(sample.image)
        # Frame removal crops framed images; the ground truth has to lose the
        # same rows and columns, not be resized to the cropped shape.
        ground_truth = sample.ground_truth[dermoscope_crop_box(sample.image)]
    else:
        image = sample.image
        valid_mask = np.ones(image.shape[:2], dtype=bool)
        ground_truth = sample.ground_truth

    outcome = PipelineOutcome(
        name=sample.name, category=sample.category, preprocessed=image, ground_truth=ground_truth
    )

    for name, segmenter in methods.items():
        result = segmenter(image, valid_mask)

        mask = finalize_mask(result.mask, valid_mask)
        hull = smart_convex_hull(mask)

        outcome.masks[name] = mask
        outcome.hulls[name] = hull
        outcome.scores[name] = dice_score(mask, ground_truth)
        outcome.scores[f"{name}_Hull"] = dice_score(hull, ground_truth)
        if keep_results:
            outcome.results[name] = result

    return outcome
