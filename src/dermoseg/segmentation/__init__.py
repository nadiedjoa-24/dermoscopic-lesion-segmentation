"""The three segmentation methods compared in this project."""

from . import lbp, otsu, region_merging
from .base import CALC_SIZE, SegmentationResult

__all__ = ["otsu", "lbp", "region_merging", "SegmentationResult", "CALC_SIZE"]
