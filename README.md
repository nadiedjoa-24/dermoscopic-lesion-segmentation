# Dermoscopic Lesion Segmentation

**Classical computer vision, no deep learning, for delineating skin lesions in dermoscopic images.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/nadiedjoa-24/dermoscopic-lesion-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/nadiedjoa-24/dermoscopic-lesion-segmentation/actions/workflows/ci.yml)

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
elongated structuring elements, then inpainted from surrounding skin. A closing
with an element wider than a hair erases it, so the difference reveals it.
Images with little hair are left untouched, since inpainting a clean image only
blurs the border being sought. The detector has a known weakness, stated under
[Limitations](#limitations): it also responds to a lesion's own dark texture.

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

Identical for all three, so only the segmentation differs: clipping to the
real image area, hole filling, and a convex hull that ignores distant specks
of noise.

## Results

Mean Dice over the twenty images, computed by `scripts/run_benchmark.py`:

| Method | Dice (mask) | Dice (+ convex hull) |
|---|---|---|
| Multi-channel Otsu | 0.832 | 0.887 |
| LBP Clustering | **0.841** | **0.901** |
| Statistical Region Merging | 0.820 | 0.869 |

Per category:

| Method | Melanoma | Melanoma + hull | Nevus | Nevus + hull |
|---|---|---|---|---|
| Multi-channel Otsu | 0.815 | 0.872 | 0.849 | 0.902 |
| LBP Clustering | 0.842 | 0.895 | 0.841 | 0.908 |
| Statistical Region Merging | 0.790 | 0.849 | 0.850 | 0.888 |

Per-image scores are in [`reports/per_image_results.csv`](reports/per_image_results.csv),
and one report page per image in [`reports/segmentation_report.pdf`](reports/segmentation_report.pdf).

**What the numbers say.**

- **LBP Clustering produces the best raw masks** (0.841), narrowly ahead of
  Otsu (0.832) and SRM (0.820). With the convex hull applied, LBP is also the
  best method overall (0.901). On twenty images these gaps are small, so the
  ranking is indicative rather than definitive.
- **The convex hull helps all three methods by a similar amount** (roughly
  +0.049 to +0.060 Dice on average), but not on every image: it lowers LBP's
  score on `ISIC_0000001`, `ISIC_0000145` and `ISIC_0000080`, by up to 0.027.
  Convexity is an assumption about lesion shape, not a guarantee of improvement.
- **Melanoma against nevus shows no reliable difference here.** Melanomas
  score lower on average in five of the six columns, the clinically expected
  direction, but with ten images per class none of these gaps is
  statistically significant (Mann-Whitney p above 0.4 in every column).
- **An audit of the pipeline found and fixed several silent bugs**, the most
  important being that the three framed images were scored against a
  stretched ground truth. The notebook (section 7) lists them all and
  demonstrates the main ones; the numbers above are measured after every fix.

## Installation

```bash
git clone https://github.com/nadiedjoa-24/dermoscopic-lesion-segmentation.git
cd dermoscopic-lesion-segmentation
pip install -e .
```

That installs `dermoseg` with its dependencies, so the package imports from
anywhere. The scripts and the notebook also add `src/` to the path
themselves, so they run from a bare clone either way.

## Reproducing the results

```bash
# Full benchmark: CSV, per-image PDF report and summary figures
# (around 15-20 min, almost all of it in preprocessing)
python scripts/run_benchmark.py

# One image, with the detailed pipeline figure for each method
python scripts/segment_image.py data/melanoma/ISIC_0000140.jpg
```

LBP and SRM are seeded and reproduce exactly. Otsu's Chan-Vese refinement
(`skimage.segmentation.morphological_chan_vese`) has a small run-to-run
non-determinism of its own, unrelated to this codebase: it can flip a
handful of boundary pixels between runs on the same input, moving its Dice
score by up to about 0.0002. It never changes a number at the precision
reported here.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Forty-nine tests, in a few seconds, almost entirely on synthetic in-memory
arrays: Otsu, SRM and the pipeline are only run on small synthetic images to
check their contracts, and the real dataset is never touched. The exception is
`data.py`'s four tests, which write and read back a handful of tiny synthetic
files. They pin the parts that are heuristics rather than
mathematics, because those are what break quietly:

- **Dice** on its stated edge cases, including two empty masks scoring 1.0 and
  the resize path taken when a downscaled prediction meets a full-size mask.
- **Clipping to the valid area**, and the fact that a lesion covering most of
  it comes through post-processing intact. An earlier inversion guard would
  have flipped it into its complement; it was removed.
- **The convex hull's satellite rule**: a speck in a corner must not stretch
  the hull, and its reach cannot exceed the image regardless of the lesion's
  or the satellite's own size.
- **The LBP operator** against patterns built by hand, since it is written from
  the paper and packs its bits clockwise from north. scikit-image starts
  elsewhere and turns the other way, so its codes are not a usable reference.
- **The hair-removal footprints**, both their geometry and that all four
  orientations actually enter the pixelwise maximum.
- **The hair-removal coverage gate**, which has to threshold the response into
  hair/not-hair pixels before measuring the fraction of the valid area they cover,
  not sum the raw intensity difference: that sum lives on a different scale
  and barely tracks how much hair is actually present.
- **Otsu's per-channel threshold and SRM's border-sampled skin reference**,
  which ignore the whitened frame corners when given a valid mask, the same
  way LBP's clustering already did, instead of letting that synthetic cluster
  pull the statistic away from the real lesion/skin boundary. (The pipeline
  passes that mask to Otsu but not to SRM; the notebook's section 7 says why.)
- **Scoring on a framed image**: frame removal crops it, so the ground truth
  must be cropped with the exact same box. The test scores a perfect mask on
  a synthetic framed image and requires exactly 1.0, which the old resizing
  failed.
- **Colour smoothing** for the Felzenszwalb backend, which must blur each
  channel on its own rather than mix red into blue.

The narrative walkthrough is in
[`notebooks/01_method_comparison.ipynb`](notebooks/01_method_comparison.ipynb),
which imports `dermoseg` rather than redefining it.

## Using the package

```python
from dermoseg.data import load_sample
from dermoseg.pipeline import run_all_methods

sample = load_sample("data/melanoma/ISIC_0000140.jpg")
outcome = run_all_methods(sample)

print(outcome.scores)   # {'Otsu': 0.877, 'Otsu_Hull': 0.931, 'LBP': 0.895, ...}
# Otsu's exact figure can differ by ~0.0002 from reports/per_image_results.csv:
# its Chan-Vese refinement has the small run-to-run non-determinism described
# under "Reproducing the results" above. LBP and SRM are seeded and match exactly.
print(outcome.masks["LBP"].shape)
```

## Project structure

```
src/dermoseg/
├── data.py              dataset access and ground-truth loading
├── preprocessing.py     frame removal, hair removal
├── postprocessing.py    valid-area clipping, hole filling, smart convex hull
├── metrics.py           Dice coefficient
├── pipeline.py          run every method on one sample and score it
├── visualization.py     every figure in the project
└── segmentation/
    ├── base.py            shared contract and scaling helpers
    ├── otsu.py            multi-channel Otsu + Chan-Vese
    ├── lbp.py             LBP clustering with the pinkness criterion
    └── region_merging.py  SRM (and Felzenszwalb) + region selection

tests/       49 unit tests on the metric, the guards, the LBP operator,
             the preprocessing helpers, the three segmenters and the scoring
scripts/     run_benchmark.py, segment_image.py
notebooks/   01_method_comparison.ipynb
data/        20 ISIC images with ground-truth masks
reports/     benchmark outputs and figures
docs/        research_paper.pdf (+ LaTeX source and table generator), references.md
pyproject.toml
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
- **The hair detector also detects lesion texture.** It flags any dark
  structure narrower than its structuring elements, and the textured interior
  of a lesion qualifies. On two hairless images (ISIC_0000140 and
  ISIC_0000142) it passes the coverage gate on lesion texture alone and
  inpaints parts of the lesion, which slightly lowers their scores and leaves
  the coloured shards visible in the figure at the top of this page. A
  detector that tells hair from lesion texture is left as future work.
- **Two lesions defeat all three methods.** ISIC_0000024 and ISIC_0000049
  score below 0.70 on their raw masks. The first is low-contrast; the second
  is also the largest lesion in the set, and which of the two explains its
  failure is not established. These are the cases where a learned model would
  most likely do better.

## References

Full citations in [`docs/references.md`](docs/references.md). The methods come
from Celebi et al. (2008), Pereira et al. (2020), Garnavi et al. (2009),
Zortea et al. (2017) and Lee et al. (1997).

The full write-up is in [`docs/research_paper.pdf`](docs/research_paper.pdf):
method derivations, the failure analysis, and per-image scores. Its tables are
written out of `reports/per_image_results.csv` by a generator script rather than
typed by hand, so the report quotes the same numbers the benchmark produced.

## Authors

Théophile Nadiedjoa ([@nadiedjoa-24](https://github.com/nadiedjoa-24)) and
Agshay Nadanakumar ([@agshayn](https://github.com/agshayn)), Télécom Paris,
2025.

Licensed under the [MIT License](LICENSE).
