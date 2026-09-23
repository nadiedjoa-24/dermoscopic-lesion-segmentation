"""Post-processing guards.

These are heuristics, so they are exactly the code that quietly stops working
when something upstream changes. Each test here pins one claim the docstrings
make: ``clean_mask`` clipping to the real valid area instead of a guessed disc
and never inverting a large lesion, and the convex hull's satellite reach being
bounded by the image, for every satellite, instead of growing with the lesion.
"""

import numpy as np

from dermoseg.pipeline import finalize_mask
from dermoseg.postprocessing import clean_mask, fill_holes, smart_convex_hull

SIZE = 100


def disc_mask(size: int = SIZE) -> np.ndarray:
    """A plausible valid area: the dermoscope disc, as preprocessing would give it."""
    grid_y, grid_x = np.ogrid[:size, :size]
    centre = size // 2
    radius = size // 2 * 0.95
    return (grid_x - centre) ** 2 + (grid_y - centre) ** 2 <= radius**2


def blob(centre, radius, size: int = SIZE) -> np.ndarray:
    grid_y, grid_x = np.ogrid[:size, :size]
    return (
        (grid_x - centre[1]) ** 2 + (grid_y - centre[0]) ** 2 <= radius**2
    ).astype(np.uint8)


def test_pixels_outside_the_valid_area_are_dropped():
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[0, 0] = 1  # outside the disc
    mask |= blob((SIZE // 2, SIZE // 2), 15)
    cleaned = clean_mask(mask, disc_mask())
    assert cleaned[0, 0] == 0
    assert cleaned.sum() > 0


def test_no_valid_mask_means_no_clipping():
    """Without preprocessing info, clean_mask must not invent a disc to cut.

    This is the first fix: the old code always clipped to a circle guessed
    from the image's aspect ratio, even on images that preprocessing found to
    have no frame at all. A lesion that pokes into what would have been that
    guessed circle's corners must survive when no real valid area is given.
    """
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[0:10, 0:10] = 1  # squarely in a corner a guessed disc would cut
    cleaned = clean_mask(mask)
    assert cleaned.sum() == mask.sum()


def test_a_mask_inside_the_valid_area_is_left_alone():
    valid = disc_mask()
    mask = blob((SIZE // 2, SIZE // 2), 25)
    cleaned = clean_mask(mask, valid)
    assert cleaned.sum() == (mask.astype(bool) & valid).sum()


def test_a_large_lesion_is_never_inverted():
    """A lesion covering most of the valid area must come through intact.

    An earlier version flipped any mask covering more than 60% of the valid
    area, assuming it had selected skin instead of lesion. Two ground-truth
    lesions in this dataset are that large, so the guard turned a perfect
    segmentation of them into its complement, while never once rescuing a
    real segmenter output. It was removed; this pins that.
    """
    valid = disc_mask()
    large_lesion = blob((SIZE // 2, SIZE // 2), 40)
    assert (large_lesion.astype(bool) & valid).sum() / valid.sum() > 0.6

    finalized = finalize_mask(large_lesion, valid)
    assert np.array_equal(finalized.astype(bool), large_lesion.astype(bool) & valid)


def test_empty_mask_stays_empty():
    empty = np.zeros((SIZE, SIZE), dtype=np.uint8)
    assert clean_mask(empty, disc_mask()).sum() == 0
    assert smart_convex_hull(empty).sum() == 0


def test_fill_holes_closes_hair_remnants():
    mask = blob((SIZE // 2, SIZE // 2), 20)
    holed = mask & (1 - blob((SIZE // 2, SIZE // 2), 4))
    assert fill_holes(holed).sum() == mask.sum()


def test_convex_hull_ignores_a_distant_speck():
    """One false-positive pixel in a corner must not stretch the hull."""
    mask = blob((SIZE // 2, SIZE // 2), 15)
    with_speck = mask.copy()
    with_speck[2, 2] = 1

    hull = smart_convex_hull(with_speck)
    assert hull[2, 2] == 0
    assert hull.sum() < 2 * mask.sum()


def test_convex_hull_keeps_a_nearby_satellite():
    """A fragment close to the lesion is part of it, and the hull bridges them."""
    main = blob((SIZE // 2, SIZE // 2), 15)
    satellite = blob((SIZE // 2, SIZE // 2 + 22), 4)
    hull = smart_convex_hull(main | satellite)
    assert hull[SIZE // 2, SIZE // 2 + 22] == 1
    assert hull.sum() > main.sum()


def test_convex_hull_reach_does_not_exceed_the_image_on_a_large_lesion():
    """The satellite reach must stay bounded by the image.

    A radius-relative reach (the previous design) grows with the lesion, so on
    a large enough lesion it exceeds the image itself: every pixel, including
    a far corner, counts as "close enough" and the filter stops filtering
    anything. Here the lesion is large enough to have triggered exactly that
    with the old 2.5x-radius rule; the diagonal-relative rule must still
    reject a speck in the far corner.
    """
    big = 800
    main = blob((big // 2, big // 2), 300, size=big)
    with_corner_speck = main.copy()
    with_corner_speck[5, 5] = 1

    hull = smart_convex_hull(with_corner_speck)
    assert hull[5, 5] == 0


def test_convex_hull_rejects_a_large_but_distant_region():
    """Size grants no exemption from the distance rule.

    A region used to be kept regardless of distance if it was large relative
    to the lesion (meant to catch a genuine second lobe of the same lesion).
    In practice a large, far, unrelated blob (a texture artefact elsewhere in
    the image) satisfied that just as well and was glued into the hull no
    matter how far away it was. Distance alone decides now, however large the
    other region is.
    """
    big = 800
    main = blob((big // 2, big // 2), 100, size=big)  # small lesion
    far_but_large = blob((50, 50), 90, size=big)  # bigger than the lesion, in a far corner

    hull = smart_convex_hull(main | far_but_large)
    assert hull[50, 50] == 0
    assert hull.sum() < 2 * main.sum()
