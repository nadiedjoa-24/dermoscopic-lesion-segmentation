# Dermoscopic Lesion Segmentation

**Classical computer vision, no deep learning, for delineating skin lesions in dermoscopic images.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

![Comparison of the three methods against the ground truth](reports/figures/method_comparison.jpg)

Three published segmentation methods, reimplemented from their papers and
benchmarked head-to-head on twenty ISIC images against expert ground truth.

---

## Why classical computer vision

A U-Net would score higher. That is not what this project is for.

- **No training data required.** Every method here is unsupervised. It runs on
  twenty images; a segmentation network does not.
- **Every step is inspectable.** When a mask is wrong, the pipeline figures show
  *which operation* broke it. That is hard to get from a network's feature maps,
  and it matters in a clinical setting where a wrong border propagates into
  every downstream measurement of asymmetry, border and colour.
- **Reimplementing a paper is the point.** The LBP and SRM methods are written
  out from their publications, not called from a library.

## Dataset

Twenty dermoscopic images from the [ISIC Archive](https://www.isic-archive.com/):
ten melanomas and ten nevi, each paired with an expert-drawn segmentation mask.

```
data/melanoma/ISIC_0000030.jpg
              ISIC_0000030_Segmentation.png
data/nevus/...
```

Images are redistributed here for reproducibility under the terms of the ISIC
Archive; the archive remains the authoritative source.

## Pipeline

### Preprocessing

**Frame removal.** A dermoscope images the skin through a circular lens, so many
ISIC images have black corners. No intensity-based method can tell a black
corner from a dark lesion, so the lit disc is detected and the outside is
whitened and cropped.

**Hair removal.** Following DullRazor, hairs are found by grayscale closing with
elongated structuring elements — a closing with an element wider than a hair
erases it, so the difference reveals it — then inpainted from surrounding skin.
Images with little hair are left untouched, since inpainting a clean image only
blurs the border being sought.

### The three methods

| Method | Idea | Source |
|---|---|---|
| **Multi-channel Otsu** | Otsu's threshold on R, G and B independently, combined as `(R and G) or B`, then refined by a Chan-Vese active contour. | Garnavi et al. (2009) |
| **LBP Clustering** | LBP (P=8, R=1) marks textured pixels; the smoothed texture field is stacked with luminance into a pseudo-RGB `[L, Y, L]`, and the lesion cluster is picked in L\*a\*b\* by the *pinkness* score `max(a*,0) − min(b*,0)`. | Pereira et al. (2020) |
| **Statistical Region Merging** | Union-find merging of neighbouring pixels ordered by intensity difference, under a bound that tightens as regions grow; lesion regions are then selected by darkness, centrality and saturation. | Celebi et al. (2008) |

Felzenszwalb's graph-based segmentation is also available as a faster
over-segmentation backend. It is a **different algorithm** and is always
reported as such, never as SRM.

### Post-processing

Identical for all three, so only the segmentation differs: hole filling, an
inversion guard for masks that selected skin instead of lesion, and a convex
hull that ignores distant specks of noise.

## Results

Mean Dice over the twenty images, computed by `scripts/run_benchmark.py`:

| Method | Dice (mask) | Dice (+ convex hull) |
|---|---|---|
| Multi-channel Otsu | 0.844 | **0.897** |
| LBP Clustering | **0.871** | 0.884 |
| Statistical Region Merging | 0.817 | 0.866 |

Per category:

| Method | Melanoma | Melanoma + hull | Nevus | Nevus + hull |
|---|---|---|---|---|
| Multi-channel Otsu | 0.823 | 0.877 | 0.866 | 0.917 |
| LBP Clustering | 0.850 | 0.896 | 0.893 | 0.872 |
| Statistical Region Merging | 0.782 | 0.845 | 0.852 | 0.887 |

Per-image scores are in [`reports/per_image_results.csv`](reports/per_image_results.csv),
and one report page per image in [`reports/segmentation_report.pdf`](reports/segmentation_report.pdf).

**What the numbers say.**

- **LBP Clustering produces the best raw masks** (0.871). Its texture assumption
  holds well on these pigmented, structurally busy lesions.
- **The convex hull helps, but not everywhere.** It lifts Otsu by 5 points and
  SRM by 5, because both fragment the lesion and the hull glues the pieces back
  together. It barely helps LBP (+1 point), whose masks are already connected,
  and on nevi it actively hurts it (0.893 down to 0.872). Convexity is an
  assumption about lesion shape, and a concave border breaks it.
- **Melanomas are harder than nevi** for all three methods — the clinically
  expected direction, since irregular, poorly defined borders are exactly what
  the ABCD rule looks for.

## Installation

```bash
git clone https://github.com/nadiedjoa-24/dermoscopic-lesion-segmentation.git
cd dermoscopic-lesion-segmentation
pip install -r requirements.txt
```

## Reproducing the results

```bash
# Full benchmark: CSV, per-image PDF report and summary figures (~9 min)
python scripts/run_benchmark.py

# One image, with the detailed pipeline figure for each method
python scripts/segment_image.py data/melanoma/ISIC_0000140.jpg
```

Every method is seeded, so the numbers above reproduce exactly.

The narrative walkthrough is in
[`notebooks/01_method_comparison.ipynb`](notebooks/01_method_comparison.ipynb),
which imports `dermoseg` rather than redefining it.

## Using the package

```python
from dermoseg.data import load_sample
from dermoseg.pipeline import run_all_methods

sample = load_sample("data/melanoma/ISIC_0000140.jpg")
outcome = run_all_methods(sample)

print(outcome.scores)   # {'Otsu': 0.911, 'Otsu_Hull': 0.930, 'LBP': 0.915, ...}
print(outcome.masks["LBP"].shape)
```

## Project structure

```
src/dermoseg/
├── data.py              dataset access and ground-truth loading
├── preprocessing.py     frame removal, hair removal
├── postprocessing.py    hole filling, inversion guard, smart convex hull
├── metrics.py           Dice coefficient
├── pipeline.py          run every method on one sample and score it
├── visualization.py     every figure in the project
└── segmentation/
    ├── base.py            shared contract and scaling helpers
    ├── otsu.py            multi-channel Otsu + Chan-Vese
    ├── lbp.py             LBP clustering with the pinkness criterion
    └── region_merging.py  SRM (and Felzenszwalb) + region selection

scripts/     run_benchmark.py, segment_image.py
notebooks/   01_method_comparison.ipynb
data/        20 ISIC images with ground-truth masks
reports/     benchmark outputs and figures
docs/        research_paper.pdf (+ LaTeX source), references.md
```

Segmenters return their intermediate images in a `SegmentationResult` and draw
nothing themselves, so a batch run costs no rendering time and every figure
lives in one module.

## Limitations

Stated plainly, because they bound what these numbers mean:

- **Twenty images** is a demonstration, not an evaluation. No confidence
  interval here would be meaningful.
- **Parameters were tuned on these same twenty images.** There is no held-out
  set, so the scores are optimistic.
- **The inversion guard is a rescue, not a fix.** It flips masks that covered
  more than 60% of the disc, which recovers a failed cluster choice rather than
  preventing it.
- **Low-contrast lesions defeat all three methods.** ISIC_0000030 and
  ISIC_0000049 score below 0.69 everywhere: no intensity or texture cue
  separates lesion from skin. These are the cases where a learned model would
  win, and they mark the honest limit of this approach.

## References

Full citations in [`docs/references.md`](docs/references.md). The methods come
from Celebi et al. (2008), Pereira et al. (2020), Garnavi et al. (2009),
Zortea et al. (2017) and Lee et al. (1997).

The full write-up is in [`docs/research_paper.pdf`](docs/research_paper.pdf):
method derivations, the failure analysis, and per-image scores. Its tables are
generated from the benchmark CSV, so the report and the code cannot disagree.

## Authors

Théophile Nadiedjoa ([@nadiedjoa-24](https://github.com/nadiedjoa-24)) and
Agshay Nadanakumar ([@agshayn](https://github.com/agshayn)) — Télécom Paris,
2025.

Licensed under the [MIT License](LICENSE).
