"""Segment a single image and save the detailed pipeline figures.

Usage::

    python scripts/segment_image.py data/melanoma/ISIC_0000030.jpg
    python scripts/segment_image.py data/nevus/ISIC_0000008.jpg --out reports/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dermoseg.data import load_sample  # noqa: E402
from dermoseg.pipeline import run_all_methods  # noqa: E402
from dermoseg.visualization import plot_comparison, plot_pipeline, plot_preprocessing  # noqa: E402

TITLES = {
    "Otsu": "Multi-channel Otsu pipeline",
    "LBP": "LBP clustering pipeline",
    "SRM": "Statistical Region Merging pipeline",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a dataset image.")
    parser.add_argument(
        "--out", type=Path, default=REPOSITORY_ROOT / "reports" / "figures", help="Output folder."
    )
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)

    sample = load_sample(arguments.image)
    outcome = run_all_methods(sample, keep_results=True)

    written = []

    figure = plot_preprocessing(sample.image, outcome.preprocessed)
    path = arguments.out / "preprocessing.jpg"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    for method, result in outcome.results.items():
        figure = plot_pipeline(
            result, title=TITLES[method], original=outcome.preprocessed
        )
        path = arguments.out / f"{method.lower()}_pipeline.jpg"
        figure.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(figure)
        written.append(path)

    figure = plot_comparison(
        outcome.preprocessed,
        outcome.ground_truth,
        outcome.masks,
        hulls=outcome.hulls,
        scores=outcome.scores,
        name=sample.name,
    )
    path = arguments.out / "method_comparison.jpg"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    for method in outcome.masks:
        print(
            f"{method:<6} Dice {outcome.scores[method]:.3f}"
            f"   with hull {outcome.scores[f'{method}_Hull']:.3f}"
        )
    print("\nFigures written:")
    for path in written:
        print(" -", path.relative_to(REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
