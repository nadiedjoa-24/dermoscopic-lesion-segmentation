"""Dataset access.

The dataset ships with the repository under ``data/``, laid out as one folder
per diagnosis, each image paired with the ISIC ground-truth mask::

    data/
      melanoma/ISIC_0000030.jpg
                ISIC_0000030_Segmentation.png
      nevus/...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage import io

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
CATEGORIES = ("melanoma", "nevus")


@dataclass(frozen=True)
class Sample:
    """One dermoscopic image with its ground-truth lesion mask."""

    name: str
    category: str
    image: np.ndarray
    ground_truth: np.ndarray


def ground_truth_path(image_path: Path) -> Path | None:
    """Locate the ISIC segmentation mask paired with ``image_path``."""
    for suffix in ("_Segmentation.png", "_segmentation.png"):
        candidate = image_path.with_name(image_path.stem + suffix)
        if candidate.exists():
            return candidate
    return None


def load_binary_mask(path: Path) -> np.ndarray:
    """Read a ground-truth PNG as a 0/1 uint8 array."""
    mask = io.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 127).astype(np.uint8)


def load_sample(image_path: Path | str) -> Sample:
    """Load one image and its ground truth."""
    image_path = Path(image_path)
    mask_path = ground_truth_path(image_path)
    if mask_path is None:
        raise FileNotFoundError(f"no ground-truth mask next to {image_path}")

    return Sample(
        name=image_path.name,
        category=image_path.parent.name,
        image=io.imread(image_path),
        ground_truth=load_binary_mask(mask_path),
    )


def iter_samples(root: Path | str = DATA_ROOT, categories=CATEGORIES):
    """Yield every :class:`Sample` in the dataset, ordered by category then name."""
    root = Path(root)
    for category in categories:
        folder = root / category
        if not folder.is_dir():
            continue
        for image_path in sorted(folder.glob("*.jpg")):
            if ground_truth_path(image_path) is not None:
                yield load_sample(image_path)
