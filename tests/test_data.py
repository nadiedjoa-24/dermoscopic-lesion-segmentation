"""Dataset access, exercised on tiny synthetic files rather than the real dataset."""

import numpy as np
from skimage import io

from dermoseg.data import CATEGORIES, ground_truth_path, iter_samples, load_sample


def _write_sample(root, category, name, size=(20, 20)):
    folder = root / category
    folder.mkdir(parents=True, exist_ok=True)
    image = np.zeros((*size, 3), dtype=np.uint8)
    image[5:15, 5:15] = 200
    mask = np.zeros(size, dtype=np.uint8)
    mask[5:15, 5:15] = 255
    io.imsave(folder / f"{name}.jpg", image, check_contrast=False)
    io.imsave(folder / f"{name}_Segmentation.png", mask, check_contrast=False)
    return folder / f"{name}.jpg"


def test_ground_truth_path_finds_the_matching_mask(tmp_path):
    image_path = _write_sample(tmp_path, "melanoma", "ISIC_0000001")
    mask_path = ground_truth_path(image_path)
    assert mask_path is not None and mask_path.exists()


def test_ground_truth_path_returns_none_when_no_mask_exists(tmp_path):
    lone_image = tmp_path / "not_a_real_sample.jpg"
    lone_image.touch()
    assert ground_truth_path(lone_image) is None


def test_load_sample_pairs_the_image_with_a_binary_mask(tmp_path):
    image_path = _write_sample(tmp_path, "melanoma", "ISIC_0000001")
    sample = load_sample(image_path)

    assert sample.category == "melanoma"
    assert sample.image.ndim == 3
    assert set(np.unique(sample.ground_truth)) <= {0, 1}
    assert sample.ground_truth.shape == sample.image.shape[:2]


def test_iter_samples_covers_every_image_that_has_a_mask(tmp_path):
    _write_sample(tmp_path, "melanoma", "ISIC_0000001")
    _write_sample(tmp_path, "melanoma", "ISIC_0000002")
    _write_sample(tmp_path, "nevus", "ISIC_0000003")
    (tmp_path / "nevus" / "ISIC_0000004.jpg").write_bytes(b"")  # no mask: must be skipped

    samples = list(iter_samples(tmp_path))

    assert len(samples) == 3
    assert {sample.category for sample in samples} == set(CATEGORIES)
