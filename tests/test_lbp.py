"""The hand-written LBP operator and the pinkness criterion.

Written out from Pereira et al. rather than called from scikit-image, so it is
checked against the paper's definition on patterns built by hand. scikit-image
packs its bits from a different neighbour and in the other direction, so its
codes are not comparable to these and are not used as a reference here.
"""

import numpy as np

from dermoseg.segmentation.lbp import (
    binarize_patterns,
    local_binary_pattern_p8r1,
    luminance_bt601,
    pinkness,
)


def test_uniform_image_has_no_texture():
    """Every neighbour equals the centre, and the comparison is a strict >."""
    codes = local_binary_pattern_p8r1(np.full((9, 9), 120.0, dtype=np.float32))
    assert (codes == 0).all()
    assert (binarize_patterns(codes) == 0).all()


def test_single_bright_neighbour_sets_one_bit():
    """Bits are packed clockwise from north, so a bright north sets bit 0."""
    image = np.zeros((5, 5), dtype=np.float32)
    image[1, 2] = 255.0  # directly north of the centre
    codes = local_binary_pattern_p8r1(image)
    assert codes[2, 2] == 0b00000001


def test_single_bright_neighbour_counts_as_flat():
    """One neighbour brighter than the centre is the paper's flat subset."""
    image = np.zeros((5, 5), dtype=np.float32)
    image[1, 2] = 255.0
    codes = local_binary_pattern_p8r1(image)
    assert binarize_patterns(codes)[2, 2] == 0


def test_all_neighbours_brighter_counts_as_textured():
    image = np.full((5, 5), 255.0, dtype=np.float32)
    image[2, 2] = 0.0
    codes = local_binary_pattern_p8r1(image)
    assert codes[2, 2] == 0b11111111
    assert binarize_patterns(codes)[2, 2] == 1


def test_checkerboard_flags_the_dark_pixels():
    """A dark checkerboard pixel has four brighter orthogonal neighbours.

    That is code 0b01010101, well outside the flat subset. Its bright
    counterpart has no neighbour above it and stays at 0, so exactly half the
    board is marked. The comparison is one-sided by design: the operator asks
    "is this pixel a local minimum", not "is this edge busy".
    """
    grid_y, grid_x = np.indices((9, 9))
    board = ((grid_x + grid_y) % 2 * 255).astype(np.float32)
    codes = local_binary_pattern_p8r1(board)
    textured = binarize_patterns(codes)

    dark = board == 0
    inner = np.zeros_like(dark)
    inner[1:-1, 1:-1] = True

    assert codes[4, 4] == 0b01010101 if dark[4, 4] else codes[4, 4] == 0
    assert textured[inner & dark].all()
    assert not textured[inner & ~dark].any()


def test_rejects_a_colour_image():
    try:
        local_binary_pattern_p8r1(np.zeros((5, 5, 3), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("expected a ValueError on a 3D input")


def test_luminance_rescales_float_images_to_0_255():
    """The LBP stage expects 0..255, but downscaled images come back in [0, 1]."""
    white = np.ones((4, 4, 3), dtype=np.float32)
    assert np.allclose(luminance_bt601(white), 255.0)
    assert np.allclose(luminance_bt601(white * 255.0), 255.0)


def test_pinkness_scores_red_above_yellow():
    reddish = pinkness(np.full(10, 20.0), np.zeros(10))
    yellowish = pinkness(np.zeros(10), np.full(10, 30.0))
    assert reddish > yellowish
