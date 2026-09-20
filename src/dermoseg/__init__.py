"""Classical computer-vision segmentation of dermoscopic skin lesions.

Typical use::

    from dermoseg.data import load_sample
    from dermoseg.pipeline import run_all_methods

    sample = load_sample("data/melanoma/ISIC_0000030.jpg")
    outcome = run_all_methods(sample)
    print(outcome.scores)
"""

from . import data, metrics, postprocessing, preprocessing, segmentation, visualization
from .pipeline import METHODS, run_all_methods

__all__ = [
    "data",
    "metrics",
    "postprocessing",
    "preprocessing",
    "segmentation",
    "visualization",
    "run_all_methods",
    "METHODS",
]
