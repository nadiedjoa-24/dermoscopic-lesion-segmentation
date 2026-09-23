"""Run every method on the whole dataset and write the report.

Usage::

    python scripts/run_benchmark.py

Outputs, all under ``reports/``:

- ``per_image_results.csv``      one row per image, one column per method
- ``segmentation_report.pdf``    one page per image, ground truth beside each mask
- ``figures/dice_per_image.pdf`` grouped bars, melanoma and nevus
- ``figures/dice_summary.pdf``   mean Dice with min-max whiskers
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dermoseg.data import DATA_ROOT, iter_samples  # noqa: E402
from dermoseg.pipeline import METHODS, run_all_methods  # noqa: E402
from dermoseg.visualization import plot_comparison, plot_dice_per_image, plot_summary_stats  # noqa: E402

REPORTS = REPOSITORY_ROOT / "reports"
FIGURES = REPORTS / "figures"


def run(data_root: Path, reports: Path) -> pd.DataFrame:
    """Segment every sample, write the per-image PDF, return the results table."""
    figures = reports / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    rows = []
    report_pdf = reports / "segmentation_report.pdf"
    started = time.time()

    with PdfPages(report_pdf) as pdf:
        for index, sample in enumerate(iter_samples(data_root), start=1):
            outcome = run_all_methods(sample)
            rows.append(outcome.as_row())

            figure = plot_comparison(
                outcome.preprocessed,
                outcome.ground_truth,
                outcome.masks,
                hulls=outcome.hulls,
                scores=outcome.scores,
                name=f"{sample.name} ({sample.category})",
            )
            pdf.savefig(figure)
            plt.close(figure)

            summary = "  ".join(f"{m}:{outcome.scores[m]:.2f}" for m in METHODS)
            print(f"[{index:2d}] {sample.name:<20} {summary}")

    frame = pd.DataFrame(rows)
    frame.to_csv(reports / "per_image_results.csv", index=False)
    print(f"\nProcessed {len(frame)} images in {time.time() - started:.0f}s -> {report_pdf}")
    return frame


def plot_results(frame: pd.DataFrame, figures: Path) -> None:
    """Write the per-image and summary Dice figures."""
    melanoma = frame[frame["Category"] == "melanoma"]
    nevus = frame[frame["Category"] == "nevus"]

    figure, axes = plt.subplots(2, 1, figsize=(20, 18))
    plot_dice_per_image(melanoma, "Melanoma: Dice score per image", axes[0])
    plot_dice_per_image(nevus, "Nevus: Dice score per image", axes[1])
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.1, hspace=0.4)
    figure.savefig(figures / "dice_per_image.pdf", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_summary_stats(melanoma, "Melanoma: summary", axes[0])
    plot_summary_stats(nevus, "Nevus: summary", axes[1])
    figure.tight_layout()
    figure.savefig(figures / "dice_summary.pdf", dpi=200, bbox_inches="tight")
    plt.close(figure)


def print_summary(frame: pd.DataFrame) -> None:
    """Print the mean Dice table, overall and per category."""
    columns = [name for method in METHODS for name in (method, f"{method}_Hull")]

    print("\nMean Dice per category")
    print(frame.groupby("Category")[columns].mean().round(3).to_string())
    print("\nMean Dice overall")
    print(frame[columns].mean().round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_ROOT, help="Dataset root folder.")
    parser.add_argument("--reports", type=Path, default=REPORTS, help="Output folder.")
    arguments = parser.parse_args()

    frame = run(arguments.data, arguments.reports)
    plot_results(frame, arguments.reports / "figures")
    print_summary(frame)


if __name__ == "__main__":
    main()
