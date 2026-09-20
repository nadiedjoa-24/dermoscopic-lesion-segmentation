"""Figures.

Every plot in the project lives here. The segmenters return their intermediate
images in :attr:`~dermoseg.segmentation.base.SegmentationResult.steps` and draw
nothing themselves, so a batch run over the dataset costs no rendering time.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

# Pastel for the raw mask, saturated for the same method after convex hull.
METHOD_COLORS = {
    "Otsu": "#ffb3b3",
    "Otsu_Hull": "#cc0000",
    "LBP": "#b3ffb3",
    "LBP_Hull": "#009900",
    "SRM": "#b3b3ff",
    "SRM_Hull": "#0000cc",
}
ACCEPTANCE_THRESHOLD = 0.9


def _show(axis, image, title, cmap=None):
    """Draw one panel, choosing a sensible colormap for 2D data."""
    image = np.asarray(image)
    if cmap is None and image.ndim == 2:
        cmap = "gray"
    if image.ndim == 3 and np.issubdtype(image.dtype, np.floating):
        # Filtering can push a float image a hair outside [0, 1]; imshow would
        # clip it anyway and warn about it.
        image = np.clip(image, 0.0, 1.0)
    axis.imshow(image, cmap=cmap)
    axis.set_title(title, fontsize=10)
    axis.axis("off")


def plot_dataset_overview(samples, *, columns=5):
    """A contact sheet of the dataset: each image above its ground-truth mask."""
    samples = list(samples)
    rows = math.ceil(len(samples) / columns)
    figure, axes = plt.subplots(rows * 2, columns, figsize=(3 * columns, 3.2 * rows * 2))
    axes = np.atleast_2d(axes)

    for index, sample in enumerate(samples):
        row, column = divmod(index, columns)
        _show(axes[row * 2][column], sample.image, f"{sample.name}\n{sample.category}")
        _show(axes[row * 2 + 1][column], sample.ground_truth, "ground truth")

    for index in range(len(samples), rows * columns):
        row, column = divmod(index, columns)
        axes[row * 2][column].axis("off")
        axes[row * 2 + 1][column].axis("off")

    figure.tight_layout()
    return figure


def plot_preprocessing(original, processed, *, figsize=(12, 5)):
    """Before and after frame removal and hair removal."""
    figure, axes = plt.subplots(1, 2, figsize=figsize)
    _show(axes[0], original, "Original")
    _show(axes[1], processed, "Preprocessed (frame and hair removed)")
    figure.tight_layout()
    return figure


def plot_pipeline(result, *, title, original=None, columns=4, figsize_per_panel=(4.0, 3.4)):
    """Every intermediate stage of one segmentation, in pipeline order.

    This is the figure that makes a classical pipeline readable: each panel is
    one operation, so a failure can be traced to the stage that caused it.
    """
    panels = []
    if original is not None:
        panels.append(("Original (preprocessed)", original))
    panels.extend(result.steps.items())
    panels.append(("Final mask", result.mask))

    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * figsize_per_panel[0], rows * figsize_per_panel[1]),
    )
    axes = np.atleast_1d(axes).ravel()

    for index, (name, image) in enumerate(panels):
        cmap = "nipy_spectral" if "segmentation" in name or "clusters" in name else None
        if "score map" in name:
            cmap = "hot"
        if name == "Saturation":
            cmap = "viridis"
        _show(axes[index], image, f"{index + 1}. {name}", cmap=cmap)

    for axis in axes[len(panels) :]:
        axis.axis("off")

    figure.suptitle(title, fontsize=16)
    figure.tight_layout()
    return figure


def plot_comparison(image, ground_truth, masks, hulls=None, scores=None, *, name=""):
    """Ground truth against every method's mask, and optionally its convex hull.

    Args:
        image: The preprocessed image.
        ground_truth: Reference mask.
        masks: Mapping of method name to binary mask.
        hulls: Optional mapping of method name to convex hull.
        scores: Optional mapping of method name to Dice score, shown in the titles.
        name: Image name, used in the figure title.
    """
    methods = list(masks)
    rows = 2 if hulls else 1
    figure, axes = plt.subplots(
        rows, len(methods) + 2, figsize=(4 * (len(methods) + 2), 4.2 * rows), squeeze=False
    )

    _show(axes[0][0], image, "Preprocessed")
    _show(axes[0][1], ground_truth, "Ground truth")
    for column, method in enumerate(methods, start=2):
        suffix = f"\nDice {scores[method]:.3f}" if scores and method in scores else ""
        _show(axes[0][column], masks[method], f"{method}{suffix}")

    if hulls:
        _show(axes[1][0], image, "Preprocessed")
        _show(axes[1][1], ground_truth, "Ground truth")
        for column, method in enumerate(methods, start=2):
            key = f"{method}_Hull"
            suffix = f"\nDice {scores[key]:.3f}" if scores and key in scores else ""
            _show(axes[1][column], hulls[method], f"{method} + convex hull{suffix}")

    figure.suptitle(f"Segmentation comparison {name}".strip(), fontsize=15)
    figure.tight_layout()
    return figure


def plot_dice_per_image(frame, title, axis):
    """Grouped bars: one group per image, one bar per method."""
    if frame.empty:
        axis.text(0.5, 0.5, "No data", ha="center")
        axis.set_title(title)
        return

    frame = frame.sort_values("SRM_Hull", ascending=False)
    methods = list(METHOD_COLORS)
    positions = np.arange(len(frame))
    width = 0.12

    for offset, method in enumerate(methods):
        axis.bar(
            positions + (offset - (len(methods) - 1) / 2) * width,
            frame[method],
            width,
            label=method.replace("_Hull", " + hull"),
            color=METHOD_COLORS[method],
        )

    axis.axhline(
        y=ACCEPTANCE_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Acceptance threshold ({ACCEPTANCE_THRESHOLD})",
    )
    axis.set_ylabel("Dice score")
    axis.set_title(title, fontsize=14, fontweight="bold")
    axis.set_xticks(positions)
    axis.set_xticklabels(frame["Image"], rotation=90, fontsize=8)
    axis.set_ylim(0, 1.1)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10)
    axis.grid(axis="y", linestyle="--", alpha=0.7)


def plot_summary_stats(frame, title, axis):
    """Mean Dice per method, with min-max whiskers."""
    if frame.empty:
        axis.text(0.5, 0.5, "No data", ha="center")
        axis.set_title(title)
        return

    methods = list(METHOD_COLORS)
    means = frame[methods].mean()
    positions = np.arange(len(methods))

    bars = axis.bar(
        positions, means, color=[METHOD_COLORS[m] for m in methods], alpha=0.85, label="Mean"
    )
    axis.errorbar(
        positions,
        means,
        yerr=[means - frame[methods].min(), frame[methods].max() - means],
        fmt="none",
        ecolor="black",
        capsize=5,
        elinewidth=2,
        label="Min-max",
    )

    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    axis.set_ylabel("Dice score")
    axis.set_title(title, fontsize=14, fontweight="bold")
    axis.set_xticks(positions)
    axis.set_xticklabels([m.replace("_Hull", " + hull") for m in methods], rotation=45, ha="right")
    axis.set_ylim(0, 1.1)
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.legend()
